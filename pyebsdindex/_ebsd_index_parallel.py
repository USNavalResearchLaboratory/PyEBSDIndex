# This software was developed by employees of the US Naval Research Laboratory (NRL), an
# agency of the Federal Government. Pursuant to title 17 section 105 of the United States
# Code, works of NRL employees are not subject to copyright protection, and this software
# is in the public domain. PyEBSDIndex is an experimental system. NRL assumes no
# responsibility whatsoever for its use by other parties, and makes no guarantees,
# expressed or implied, about its quality, reliability, or any other characteristic. We
# would appreciate acknowledgment if the software is used. To the extent that NRL may hold
# copyright in countries other than the United States, you are hereby granted the
# non-exclusive irrevocable and unconditional right to print, publish, prepare derivative
# works and distribute this software, in any medium, or authorize others to do so on your
# behalf, on a royalty-free basis throughout the world. You may improve, modify, and
# create derivative works of the software or any portion of the software, and you may copy
# and distribute such modifications or works. Modified works should carry a notice stating
# that you changed the software and should note the date and nature of any such change.
# Please explicitly acknowledge the US Naval Research Laboratory as the original source.
# This software can be redistributed and/or modified freely provided that any derivative
# works bear some notice that they are derived from it, and any modified versions bear
# some notice that they have been modified.
#
# Author: David Rowenhorst;
# The US Naval Research Laboratory Date: 21 Aug 2020

"""Setup and handling of Hough indexing runs of EBSD patterns in
parallel.
"""


import os
import logging
import sys
import time
from timeit import default_timer as timer

import numpy as np
import h5py
import ray

from pyebsdindex import ebsd_pattern, _pyopencl_installed
from pyebsdindex._ebsd_index_single import EBSDIndexer, index_pats

if _pyopencl_installed:
    from pyebsdindex.opencl import band_detect_cl as band_detect
else:
    from pyebsdindex import band_detect as band_detect


def index_pats_distributed(
    patsin=None,
    filename=None,
    phaselist=["FCC"],
    vendor=None,
    PC=None,
    sampleTilt=70.0,
    camElev=5.3,
    bandDetectPlan=None,
    nRho=90,
    nTheta=180,
    tSigma=None,
    rSigma=None,
    rhoMaskFrac=0.1,
    nBands=9,
    patstart=0,
    npats=-1,
    chunksize=528,
    ncpu=-1,
    return_indexer_obj=False,
    ebsd_indexer_obj=None,
    keep_log=False,
    gpu_id=None,
):
    """Index EBSD patterns in parallel.

    Parameters
    ----------
    patsin : numpy.ndarray, optional
        EBSD patterns in an array of shape (n points, n pattern
        rows, n pattern columns). If not given, these are read from
        ``filename``.
    filename : str, optional
        Name of file with EBSD patterns. If not given, ``patsin`` must
        be passed.
    phaselist : list of str, optional
        Options are ``"FCC"`` and ``"BCC"``. Default is ``["FCC"]``.
    vendor : str, optional
        Which vendor convention to use for the pattern center (PC) and
        the returned orientations. The available options are ``"EDAX"``
        (default), ``"BRUKER"``, ``"OXFORD"``, ``"EMSOFT"``,
        ``"KIKUCHIPY"``.
    PC : list, optional
        Pattern center (PCx, PCy, PCz) in the :attr:`indexer.vendor` or
        ``vendor`` convention. For EDAX TSL, this is (x*, y*, z*),
        defined in fractions of pattern width with respect to the lower
        left corner of the detector. If not passed, this is set to (x*,
        y*, z*) = (0.471659, 0.675044, 0.630139). If
        ``vendor="EMSOFT"``, the PC must be four numbers, the final
        number being the pixel size.
    sampleTilt : float, optional
        Sample tilt towards the detector in degrees. Default is 70
        degrees. Unused if ``ebsd_indexer_obj`` is passed.
    camElev : float, optional
        Camera elevation in degrees. Default is 5.3 degrees. Unused
        if ``ebsd_indexer_obj`` is passed.
    bandDetectPlan : pyebsdindex.band_detect.BandDetect, optional
        Collection of parameters using in band detection. Unused if
        ``ebsd_indexer_obj`` is passed.
    nRho : int, optional
        Default is 90 degrees. Unused if ``ebsd_indexer_obj`` is
        passed.
    nTheta : int, optional
        Default is 180 degrees. Unused if ``ebsd_indexer_obj`` is
        passed.
    tSigma : float, optional
        Unused if ``ebsd_indexer_obj`` is passed.
    rSigma : float, optional
        Unused if ``ebsd_indexer_obj`` is passed.
    rhoMaskFrac : float, optional
        Default is 0.1. Unused if ``ebsd_indexer_obj`` is passed.
    nBands : int, optional
        Number of detected bands to use in triplet voting. Default
        is 9. Unused if ``ebsd_indexer_obj`` is passed.
    patstart : int, optional
        Starting index of the patterns to index. Default is ``0``.
    npats : int, optional
        Number of patterns to index. Default is ``-1``, which will
        index up to the final pattern in ``patsin``.
    chunksize : int, optional
        Default is 528.
    ncpu : int, optional
        Number of CPUs to use. Default value is ``-1``, meaning all
        available CPUs will be used.
    return_indexer_obj : bool, optional
        Whether to return the EBSD indexer. Default is ``False``.
    ebsd_indexer_obj : EBSDIndexer, optional
        EBSD indexer. If not given, many of the above parameters must be
        passed. Otherwise, these parameters are retrieved from this
        indexer.
    keep_log : bool, optional
        Whether to keep the log. Default is ``False``.
    gpu_id : int, optional
        ID of GPU to use if :mod:`pyopencl` is installed.

    Returns
    -------
    indxData : numpy.ndarray
        Complex numpy array (or array of structured data), that is
        [nphases + 1, npoints]. The data is stored for each phase used
        in indexing and the ``indxData[-1]`` layer uses the best guess
        on which is the most likely phase, based on the fit, and number
        of bands matched for each phase. Each data entry contains the
        orientation expressed as a quaternion (quat) (using the
        convention of ``vendor`` or :attr:`indexer.vendor`), Pattern
        Quality (pq), Confidence Metric (cm), Phase ID (phase), Fit
        (fit) and Number of Bands Matched (nmatch). There are some other
        metrics reported, but these are mostly for debugging purposes.
    bandData : numpy.ndarray
        Band identification data from the Radon transform.
    indexer : EBSDIndexer
        EBSD indexer, returned if ``return_indexer_obj=True``.

    Notes
    -----
    Requires :mod:`ray[default]`. See the :doc:`installation guide
    </user/installation>` for details.
    """
    pats = None
    if patsin is None:
        pdim = None
    else:
        if isinstance(patsin, ebsd_pattern.EBSDPatterns):
            pats = patsin.patterns
        if type(patsin) is np.ndarray:
            pats = patsin
        if isinstance(patsin, h5py.Dataset):
            shp = patsin.shape
            if len(shp) == 3:
                pats = patsin
            if len(shp) == 2:  # just read off disk now.
                pats = patsin[()]
                pats = pats.reshape(1, shp[0], shp[1])

        if pats is None:
            print("Unrecognized input data type")
            return
        pdim = pats.shape[-2:]

    # run a test flight to make sure all parameters are set properly before being sent off to the cluster
    if ebsd_indexer_obj is None:
        indexer = EBSDIndexer(
            filename=filename,
            phaselist=phaselist,
            vendor=vendor,
            PC=PC,
            sampleTilt=sampleTilt,
            camElev=camElev,
            bandDetectPlan=bandDetectPlan,
            nRho=nRho,
            nTheta=nTheta,
            tSigma=tSigma,
            rSigma=rSigma,
            rhoMaskFrac=rhoMaskFrac,
            nBands=nBands,
            patDim=pdim,
            gpu_id=gpu_id,
        )
    else:
        indexer = ebsd_indexer_obj

    if filename is not None:
        indexer.update_file(filename)
    else:
        indexer.update_file(patDim=pats.shape[-2:])

    # Differentiate between getting a file to index or an array.
    # Need to index one pattern to make sure the indexer object is fully
    # initiated before placing in shared memory store.
    mode = "memorymode"
    if pats is None:
        mode = "filemode"
        temp, temp2, indexer = index_pats(
            npats=1, return_indexer_obj=True, ebsd_indexer_obj=indexer
        )

    if mode == "filemode":
        npatsTotal = indexer.fID.nPatterns
    else:
        pshape = pats.shape
        if len(pshape) == 2:
            npatsTotal = 1
            pats = pats.reshape([1, pshape[0], pshape[1]])
        else:
            npatsTotal = pshape[0]
        temp, temp2, indexer = index_pats(
            pats[0, :, :],
            npats=1,
            return_indexer_obj=True,
            ebsd_indexer_obj=indexer,
        )

    if patstart < 0:
        patstart = npatsTotal - patstart
    if npats <= 0:
        npats = npatsTotal - patstart

    # Now set up the cluster with the indexer
    n_cpu_nodes = int(os.cpu_count())
    # int(sum([ r['Resources']['CPU'] for r in ray.nodes()]))
    if ncpu != -1:
        n_cpu_nodes = int(ncpu)

    ngpu = None
    if gpu_id is not None:
        ngpu = np.atleast_1d(gpu_id).shape[0]

    try:
        clparam = band_detect.getopenclparam()
        if clparam is None:
            ngpu = 0
            ngpupnode = 0
        else:
            if ngpu is None:
                ngpu = len(clparam.gpu)
            ngpupnode = ngpu / n_cpu_nodes
    except:
        ngpu = 0
        ngpupnode = 0

    ray.shutdown()

    print("num cpu/gpu:", n_cpu_nodes, ngpu)
    # ray.init(num_cpus=n_cpu_nodes,num_gpus=ngpu,_system_config={"maximum_gcs_destroyed_actor_cached_count": n_cpu_nodes})
    # Need to append path for installs from source ... otherwise the ray
    # workers do not know where to find the PyEBSDIndex module.
    ray.init(
        num_cpus=n_cpu_nodes,
        num_gpus=ngpu,
        _node_ip_address="0.0.0.0",
        runtime_env={"env_vars": {"PYTHONPATH": os.path.dirname(os.path.dirname(__file__))}},
        logging_level=logging.WARNING,
    )  # Supress INFO messages from ray.

    # Place indexer obj in shared memory store so all workers can use it - this is read only.
    remote_indexer = ray.put(indexer)
    # Get the function that will collect opencl parameters - if opencl
    # is not installed, this is None, and the program will automatically
    # fall back to CPU only calculation.
    clparamfunction = band_detect.getopenclparam
    # Set up the jobs
    njobs = (np.ceil(npats / chunksize)).astype(np.compat.long)
    # p_indx_start = [i*chunksize+patStart for i in range(njobs)]
    # p_indx_end = [(i+1)*chunksize+patStart for i in range(njobs)]
    # p_indx_end[-1] = npats+patStart
    p_indx_start_end = [
        [i * chunksize + patstart, (i + 1) * chunksize + patstart, chunksize]
        for i in range(njobs)
    ]
    p_indx_start_end[-1][1] = npats + patstart
    p_indx_start_end[-1][2] = p_indx_start_end[-1][1] - p_indx_start_end[-1][0]

    if njobs < n_cpu_nodes:
        n_cpu_nodes = njobs

    nPhases = len(indexer.phaseLib)
    dataout = np.zeros((nPhases + 1, npats), dtype=indexer.dataTemplate)
    banddataout = np.zeros(
        (npats, indexer.bandDetectPlan.nBands), dtype=indexer.bandDetectPlan.dataType
    )
    ndone = 0
    nsubmit = 0
    tic0 = timer()
    npatsdone = 0.0

    if keep_log is True:
        newline = "\n"
    else:
        newline = "\r"
    if mode == "filemode":
        # Send out the first batch
        workers = []
        jobs = []
        timers = []
        jobs_indx = []
        chunkave = 0.0
        for i in range(n_cpu_nodes):
            job_pstart_end = p_indx_start_end.pop(0)
            workers.append( # make a new Ray Actor that can call the indexer defined in shared memory.
                # These actors are read/write, thus can initialize the GPU queues
                IndexerRay.options(num_cpus=1, num_gpus=ngpupnode).remote(
                    i, clparamfunction, gpu_id=gpu_id
                )
            )
            jobs.append(
                workers[i].index_chunk_ray.remote(
                    pats=None,
                    indexer=remote_indexer,
                    patstart=job_pstart_end[0],
                    npats=job_pstart_end[2],
                )
            )
            nsubmit += 1
            timers.append(timer())
            #time.sleep(0.01)
            jobs_indx.append(job_pstart_end[:])

        while ndone < njobs:
            # toc = timer()
            wrker, busy = ray.wait(jobs, num_returns=1, timeout=60.0)

            # print("waittime: ",timer() - toc)
            if len(wrker) > 0: # trying to catch a hung worker.  Rare, but it happens
                jid = jobs.index(wrker[0])
            else:
                print('hang with ', ndone, 'out of ', njobs)
                jid = jobs.index(busy[0])
                wrker.append(busy[0])
                ray.kill(busy[0])
            try:
                wrkdataout, wrkbanddata, indxstr, indxend, rate = ray.get(wrker[0])
            except:
                # print('a death has occured')
                indxstr = jobs_indx[jid][0]
                indxend = jobs_indx[jid][1]
                rate = [-1, -1]
            if rate[0] >= 0:  # Job finished as expected

                ticp = timers[jid]
                dataout[:, indxstr - patstart : indxend - patstart] = wrkdataout
                banddataout[indxstr - patstart : indxend - patstart, :] = wrkbanddata
                npatsdone += rate[1]
                ndone += 1

                ratetemp = n_cpu_nodes * (rate[1]) / (timer() - ticp)
                chunkave += ratetemp
                totalave = npatsdone / (timer() - tic0)
                # print('Completed: ',str(indxstr),' -- ',str(indxend), '  ', npatsdone/(timer()-tic) )

                toc0 = timer() - tic0
                if keep_log is False:
                    print("", end="\r")
                    time.sleep(0.00001)
                print(
                    "Completed: ",
                    str(indxstr),
                    " -- ",
                    str(indxend),
                    "  PPS:",
                    "{:.0f}".format(ratetemp)
                    + ";"
                    + "{:.0f}".format(chunkave / ndone)
                    + ";"
                    + "{:.0f}".format(totalave),
                    "  ",
                    "{:.0f}".format((ndone / njobs) * 100) + "%",
                    "{:.0f};".format(toc0)
                    + "{:.0f}".format((njobs - ndone) / ndone * toc0)
                    + " running;remaining(s)",
                    end=newline,
                )

                if len(p_indx_start_end) > 0:
                    job_pstart_end = p_indx_start_end.pop(0)
                    jobs[jid] = workers[jid].index_chunk_ray.remote(
                        pats=None,
                        indexer=remote_indexer,
                        patstart=job_pstart_end[0],
                        npats=job_pstart_end[2],
                    )
                    nsubmit += 1
                    timers[jid] = timer()
                    jobs_indx[jid] = job_pstart_end[:]
                else:
                    del jobs[jid]
                    del workers[jid]
                    del timers[jid]
                    del jobs_indx[jid]
            else:
                # Something bad happened. Put the job back on the queue
                # and kill this worker.
                p_indx_start_end.append([indxstr, indxend, indxend - indxstr])
                del jobs[jid]
                del workers[jid]
                del timers[jid]
                del jobs_indx[jid]
                n_cpu_nodes -= 1
                if len(workers) < 1:  # Rare case that we have killed all workers...
                    job_pstart_end = p_indx_start_end.pop(0)
                    workers.append(
                        IndexerRay.options(num_cpus=1, num_gpus=ngpupnode).remote(
                            jid, clparamfunction, gpu_id
                        )
                    )
                    jobs.append(
                        workers[0].index_chunk_ray.remote(
                            pats=None,
                            indexer=remote_indexer,
                            patstart=job_pstart_end[0],
                            npats=job_pstart_end[2],
                        )
                    )
                    nsubmit += 1
                    timers.append(timer())
                    time.sleep(0.01)
                    jobs_indx.append(job_pstart_end[:])
                    n_cpu_nodes += 1

    if mode == "memorymode":
        workers = []
        jobs = []
        timers = []
        jobs_indx = []
        chunkave = 0.0
        for i in range(n_cpu_nodes):
            job_pstart_end = p_indx_start_end.pop(0)
            workers.append(
                IndexerRay.options(num_cpus=1, num_gpus=ngpupnode).remote(
                    i, clparamfunction, gpu_id
                )
            )
            jobs.append(
                workers[i].index_chunk_ray.remote(
                    pats=pats[job_pstart_end[0] : job_pstart_end[1], :, :],
                    indexer=remote_indexer,
                    patstart=job_pstart_end[0],
                    npats=job_pstart_end[2],
                )
            )
            nsubmit += 1
            timers.append(timer())
            jobs_indx.append(job_pstart_end)
            time.sleep(0.01)

        # workers = [index_chunk.remote(pats = None, indexer = remote_indexer, patStart = p_indx_start[i], patEnd = p_indx_end[i]) for i in range(n_cpu_nodes)]
        # nsubmit += n_cpu_nodes

        while ndone < njobs:
            # toc = timer()
            wrker, busy = ray.wait(jobs, num_returns=1, timeout=None)
            jid = jobs.index(wrker[0])
            # print("waittime: ",timer() - toc)
            if len(wrker) > 0:
                jid = jobs.index(wrker[0])
            else:
                print('hang with ', ndone, 'out of ', njobs)
                jid = jobs.index(busy[0])
                wrker.append(busy[0])
                ray.kill(busy[0])
            try:
                wrkdataout, wrkbanddata, indxstr, indxend, rate = ray.get(wrker[0])
            except:
                indxstr = jobs_indx[jid][0]
                indxend = jobs_indx[jid][1]
                rate = [-1, -1]
            if rate[0] >= 0:
                ticp = timers[jid]
                dataout[:, indxstr - patstart : indxend - patstart] = wrkdataout
                banddataout[indxstr - patstart : indxend - patstart, :] = wrkbanddata
                npatsdone += rate[1]
                ratetemp = n_cpu_nodes * (rate[1]) / (timer() - ticp)
                chunkave += ratetemp
                totalave = npatsdone / (timer() - tic0)
                # print('Completed: ',str(indxstr),' -- ',str(indxend), '  ', npatsdone/(timer()-tic) )
                ndone += 1
                toc0 = timer() - tic0
                if keep_log is False:
                    print("", end="\r")
                    time.sleep(0.0001)
                print(
                    "Completed: ",
                    str(indxstr),
                    " -- ",
                    str(indxend),
                    "  PPS:",
                    "{:.0f}".format(ratetemp)
                    + ";"
                    + "{:.0f}".format(chunkave / ndone)
                    + ";"
                    + "{:.0f}".format(totalave),
                    "  ",
                    "{:.0f}".format((ndone / njobs) * 100) + "%",
                    "{:.0f};".format(toc0)
                    + "{:.0f}".format((njobs - ndone) / ndone * toc0)
                    + " running;remaining(s)",
                    end=newline,
                )

                if len(p_indx_start_end) > 0:
                    job_pstart_end = p_indx_start_end.pop(0)
                    jobs[jid] = workers[jid].index_chunk_ray.remote(
                        pats=pats[job_pstart_end[0] : job_pstart_end[1], :, :],
                        indexer=remote_indexer,
                        patstart=job_pstart_end[0],
                        npats=job_pstart_end[2],
                    )
                    nsubmit += 1
                    timers[jid] = timer()
                    jobs_indx[jid] = job_pstart_end
                else:
                    del jobs[jid]
                    del workers[jid]
                    del timers[jid]
                    del jobs_indx[jid]
            else:
                # Something bad happened.  Put the job back on the queue
                # and kill this worker.
                p_indx_start_end.append([indxstr, indxend, indxend - indxstr])
                del jobs[jid]
                del workers[jid]
                del timers[jid]
                del jobs_indx[jid]
                n_cpu_nodes -= 1
                if len(workers) < 1:  # Rare case that we have killed all workers...
                    job_pstart_end = p_indx_start_end.pop(0)
                    workers.append(
                        IndexerRay.options(num_cpus=1, num_gpus=ngpupnode).remote(
                            jid, clparamfunction, gpu_id
                        )
                    )
                    jobs.append(
                        workers[0].index_chunk_ray.remote(
                            pats=pats[job_pstart_end[0] : job_pstart_end[1], :, :],
                            indexer=remote_indexer,
                            patstart=job_pstart_end[0],
                            npats=job_pstart_end[2],
                        )
                    )
                    nsubmit += 1
                    timers.append(timer())
                    jobs_indx.append(job_pstart_end)
                    n_cpu_nodes += 1

        del jobs
        del workers
        del timers
        # # send out the first batch
        # workers = [index_chunk_ray.remote(pats=pats[p_indx_start[i]:p_indx_end[i],:,:],indexer=remote_indexer,patStart=p_indx_start[i],patEnd=p_indx_end[i]) for i
        #            in range(n_cpu_nodes)]
        # nsubmit += n_cpu_nodes
        #
        # while ndone < njobs:
        #   wrker,busy = ray.wait(workers,num_returns=1,timeout=None)
        #   wrkdataout,indxstr,indxend, rate = ray.get(wrker[0])
        #   dataout[indxstr:indxend] = wrkdataout
        #   print('Completed: ',str(indxstr),' -- ',str(indxend))
        #   workers.remove(wrker[0])
        #   ndone += 1
        #
        #   if nsubmit < njobs:
        #     workers.append(index_chunk_ray.remote(pats=pats[p_indx_start[nsubmit]:p_indx_end[nsubmit],:,:],indexer=remote_indexer,patStart=p_indx_start[nsubmit],
        #                                       patEnd=p_indx_end[nsubmit]))
        #     nsubmit += 1

    ray.shutdown()
    if return_indexer_obj:
        return dataout, banddataout, indexer
    else:
        return dataout, banddataout


@ray.remote(num_cpus=1, num_gpus=1)
class IndexerRay:
    def __init__(self, actorid=0, clparammodule=None, gpu_id=None):
        # sys.path.append(path.dirname(path.dirname(__file__)))  # do this to help Ray find the program files
        # import openclparam # do this to help Ray find the program files
        # device, context, queue, program, mf
        # self.dataout = None
        # self.indxstart = None
        # self.indxend = None
        # self.rate = None
        self.actorID = actorid
        self.openCLParams = None
        self.useGPU = False
        if clparammodule is not None:
            try:
                if (
                    sys.platform != "darwin"
                ):  # linux with NVIDIA (unsure if it is the os or GPU type) is slow to make a
                    self.openCLParams = clparammodule()
                else:  # MacOS handles GPU memory conflicts much better when the context is destroyed between each
                    # run, and has very low overhead for making the context.
                    # pass
                    self.openCLParams = clparammodule()
                    # self.openCLParams.gpu_id = 0
                    # self.openCLParams.gpu_id = 1
                    self.openCLParams.gpu_id = self.actorID % self.openCLParams.ngpu
                if gpu_id is None:
                    gpu_id = np.arange(self.openCLParams.ngpu)
                gpu_list = np.atleast_1d(gpu_id)
                ngpu = gpu_list.shape[0]
                self.openCLParams.gpu_id = gpu_list[self.actorID % ngpu]
                self.openCLParams.get_queue()
                self.useGPU = True
            except:
                self.openCLParams = None

    def index_chunk_ray(self, pats=None, indexer=None, patstart=0, npats=-1):
        try:
            # print(type(self.openCLParams.ctx))
            tic = timer()
            dataout, banddata, indxstart, npatsout = indexer.index_pats(
                patsin=pats,
                patstart=patstart,
                npats=npats,
                clparams=self.openCLParams,
                chunksize=-1,
            )
            rate = np.array([timer() - tic, npatsout])
            return dataout, banddata, indxstart, indxstart + npatsout, rate
        except:
            indxstart = patstart
            indxend = patstart + npats
            return None, None, indxstart, indxend, [-1, -1]
