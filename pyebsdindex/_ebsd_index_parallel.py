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
import platform
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

RAYIPADDRESS = '127.0.0.1'
OSPLATFORM  = platform.system()
if OSPLATFORM  == 'Darwin':
    RAYIPADDRESS = '0.0.0.0' # the localhost address does not work on macOS when on a VPN

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
    chunksize=0,
    ncpu=-1,
    return_indexer_obj=False,
    ebsd_indexer_obj=None,
    keep_log=False,
    gpu_id=None,
    verbose = 1
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
        If not set. we will make a guess based on the resources available.
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
    starttime = timer()
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
    inputmode = "memorymode"
    if pats is None:
        inputmode = "filemode"
        temp, temp2, indexer = index_pats(
            npats=1, return_indexer_obj=True, ebsd_indexer_obj=indexer
        )

    if inputmode == "filemode":
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
    cudagpuvis = '0'
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
            cudagpuvis = ''
            for cdgpu in range(len(clparam.gpu)):
                cudagpuvis += str(cdgpu)+','

            #ngpupnode = ngpu / n_cpu_nodes
    except:
        ngpu = 0
        ngpupnode = 0

    if indexer.bandDetectPlan.useCPU == True:
        ngpu = 0


    if ngpu > 0:
        ngpupro = max(12, ngpu*2)  # number of processes that will serve data to the gpu
        if n_cpu_nodes < 8:
            ngpupro = 8
        if n_cpu_nodes < 2:
            ngpupro = 2
        #if OSPLATFORM == 'Linux':
        #    ngpupro = 2

        n_cpu_per_gpu = max(min(1.0, n_cpu_nodes-ngpu), 0.5/ngpu)

        ngpuwrker = ngpupro * ngpu

        ngpu_per_wrker =  1.0/ngpuwrker - 1.0e-6 # fraction of a GPU to give to each worker (band finding worker)
        ncpugpu_per_wrker = n_cpu_per_gpu/ngpuwrker - 1.0e-6 # fraction of a cpu to allocate to each gpu worker

        # amount of cpu to allocate to each cpu worker (indexing worker)
        ncpucpu_per_worker = (n_cpu_nodes - ncpugpu_per_wrker * ngpuwrker)/n_cpu_nodes


        if chunksize <= 0:
            chunksize = __optimizegpuchunk__(indexer, ngpupro, gpu_id, clparam)
    else:  # no gpus detected.
        ngpu_per_wrker = 0
        usegpu = False
        ngpu_per_wrker = 0
        ngpuwrker = n_cpu_nodes
        ncpucpu_per_worker = 0.5 - 1.0e-6
        ncpugpu_per_wrker = 0.5 - 1.0e-6
        if chunksize <= 0:
            chunksize = 1000
    ncpuwrker = n_cpu_nodes

    ray.shutdown()

    print("num cpu/gpu, and number of patterns per iteration:", n_cpu_nodes, ngpu, chunksize)
    # ray.init(num_cpus=n_cpu_nodes,num_gpus=ngpu,_system_config={"maximum_gcs_destroyed_actor_cached_count": n_cpu_nodes})
    # Need to append path for installs from source ... otherwise the ray
    # workers do not know where to find the PyEBSDIndex module.
    ray.init(
        num_cpus=int(np.round(n_cpu_nodes)),
        num_gpus=ngpu*ngpuwrker,
        _node_ip_address=RAYIPADDRESS, #"0.0.0.0",
        runtime_env={"env_vars":
                      {"PYTHONPATH": os.path.dirname(os.path.dirname(__file__)),
                      "CUDA_VISIBLE_DEVICES":cudagpuvis }},
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

    p_indx_start_end = [
        [i * chunksize + patstart, (i + 1) * chunksize + patstart, chunksize]
        for i in range(njobs)
    ]
    p_indx_start_end[-1][1] = npats + patstart
    p_indx_start_end[-1][2] = p_indx_start_end[-1][1] - p_indx_start_end[-1][0]

    gpujobs = []
    cpujobs = []
    jid = 1
    for jb in p_indx_start_end:
        gpujobs.append(CPUGPUJob(jid, jb[0], jb[1]))
        jid += 1


    if njobs < ncpuwrker:
        ncpuwrker = njobs
    if njobs < ngpuwrker:
        ngpuwrker = njobs

    nPhases = len(indexer.phaseLib)
    dataout = np.zeros((nPhases + 1, npats), dtype=indexer.dataTemplate)
    banddataout = np.zeros(
        (npats, indexer.bandDetectPlan.nBands), dtype=indexer.bandDetectPlan.dataType
    )
    bandnormsout = np.zeros((npats, indexer.bandDetectPlan.nBands, 3), dtype=np.float32)


    ncpudone = 0
    ngpudone = 0
    ncpusubmit = 0
    ngpusubmit = 0
    tic0 = timer()
    ncpupatsdone = 0.0

    if keep_log is True:
        if OSPLATFORM != 'Windows':
            newline = "\n"
        else:
            newline = "\r\n"
    else:
        newline = "\r"

    # Send out the first batch
    gpuworkers = []
    gputask = []
    gtaskindex = []
    cpuworkers = []
    cputask = []
    ctaskindex = []
    npatdone = 0.0
    chunkave = 0.0

    #print(ngpuwrker, ncpugpu_per_wrker, ngpu_per_wrker)
    #print(ncpuwrker, ncpucpu_per_worker)
    gpu_launched = 0
    cpu_launched = 0
    while ncpudone < njobs:
        #for i in range(ngpuwrker):

        while (gpu_launched < ngpuwrker) and (len(gpujobs) > 0):
            i = len(gpuworkers)
            gpuworkers.append( # make a new Ray Actor that can call the indexer defined in shared memory.
                # These actors are read/write, thus can initialize the GPU queues
                #GPUWorker.options(num_cpus=ncpugpu_per_wrker, num_gpus=ngpu_per_wrker).remote(
                GPUWorker.options(num_cpus=ncpugpu_per_wrker, num_gpus=ngpu_per_wrker).remote(
                    actorid=i, clparammodule=clparamfunction, gpu_id=gpu_id, cudavis = cudagpuvis
                )
            )
            gjob = gpujobs.pop(0)
            if inputmode == "filemode":
                gputask.append(
                    gpuworkers[i].findbands.remote(gjob,
                        pats=None,
                        indexer=remote_indexer
                    )
                )
            else:
                gputask.append(
                    gpuworkers[i].findbands.remote(gjob,
                        pats = pats[gjob.pstart:gjob.pend, :, :],
                        indexer=remote_indexer,
                    )
                )
            gtaskindex.append(gjob)
            gpu_launched += 1

        # initiate the CPU workers.
        #print(len(gpuworkers), len(gputask))
        #for i in range(ncpuwrker):
        if (cpu_launched < ncpuwrker) and (ncpudone < njobs):
            i = len(cpuworkers)
            cpuworkers.append(  # make a new Ray Actor that can call the indexer defined in shared memory.
                # These actors are read/write, thus can initialize the GPU queues
                CPUWorker.options(num_cpus=ncpucpu_per_worker).remote(i))
                #CPUWorker.options(num_cpus=1.0, num_gpus=0).remote(i))
            cputask.append(cpuworkers[i].indexpoles.remote(None, None, None,indexer=remote_indexer))
            ctaskindex.append(None)
            cpu_launched += 1
        #print(len(cpuworkers))



        if ngpudone < njobs: # check if gpu is done
            donewrker, busy = ray.wait(gputask,num_returns = len(gputask),  timeout=0.01)
            #if len(wrker) > 0:  # trying to catch a hung worker.  Rare, but it happens
            #print(len(donewrker))
            #else:
                #print('hung gpu process')
                #jid = gputask.index(busy[0])
                #wrker.append(busy[0])
                #ray.kill(gputask[jid])
            for wrker in donewrker:
                jid = gputask.index(wrker)
                try:
                    message, (banddata, bandnorm, gjob) = ray.get(wrker)

                    if message == 'Done':
                        banddataout[gjob.pstart - patstart: gjob.pend - patstart, :] = banddata
                        bandnormsout[gjob.pstart - patstart: gjob.pend - patstart, :,:] = bandnorm

                        cpujobs.append(CPUGPUJob(gjob.jobid, gjob.pstart, gjob.pend, extime=gjob.extime))
                        ngpudone += 1

                        if len(gpujobs) > 0: # still more gpu work to do
                            gjob = gpujobs.pop(0)
                            if inputmode == "filemode":
                                gputask[jid] = gpuworkers[jid].findbands.remote(gjob,
                                        pats=None,
                                        indexer=remote_indexer
                                )
                            else:
                                gputask[jid] = gpuworkers[jid].findbands.remote(gjob,
                                    pats=pats[gjob.pstart:gjob.pend, :, :],
                                    indexer=remote_indexer,
                               )
                            gtaskindex[jid] = gjob
                            ngpusubmit += 1
                        else: # no more gpu tasks to submit
                            #del gpuworkers[jid]
                            del gpuworkers[jid]
                            del gputask[jid]
                            del gtaskindex[jid]
                    else:
                        raise Exception("Error in GPU processing patterns: ", gtaskindex[jid].pstart, gtaskindex[jid].pend)


                except:
                    gjob = gtaskindex[jid]
                    print('A GPU death has occured', gjob.pstart, gjob.pend)
                    del gpuworkers[jid]
                    del gputask[jid]
                    del gtaskindex[jid]
                    gpujobs.append(gjob)
                    if len(gpuworkers) == 0:
                        if inputmode == "filemode":
                            gputask.append(
                                gpuworkers[0].findbands.remote(gjob,
                                   pats=None,
                                   indexer=remote_indexer
                                )
                            )
                        else:
                            gputask.append(
                                gpuworkers[0].findbands.remote(gjob,
                                   pats=pats[gjob.pstart:gjob.pend, :, :],
                                   indexer=remote_indexer,
                                )
                            )
                        gtaskindex.append(gjob)
            # toc = timer()
            if (ngpudone >= njobs) and (verbose >1 ):
                print('\n GPU Done')
        if ncpudone < njobs:
            donewrker, busy = ray.wait(cputask, num_returns = len(cputask),  timeout=0.01)
            for wrker in donewrker:
                jid = cputask.index(wrker)
                try:
                    message, (indexdata, cjob) = ray.get(wrker)
                    if message == 'Done':
                        dataout[:, cjob.pstart - patstart: cjob.pend - patstart] = indexdata
                        ncpudone += 1
                        chunkave += cjob.rate
                        npatdone += cjob.npat
                        currenttime = timer() - starttime
                        #print(cjob.rate * n_cpu_nodes)
                        if verbose > 0:
                            print(
                                "Completed: ",
                                str(cjob.pstart),
                                " -- ",
                                str(cjob.pend),
                                "  PPS:",
                                "{:.0f}".format(cjob.rate*ncpuwrker)
                                + ";"
                                + "{:.0f}".format(chunkave / ncpudone * ncpuwrker)
                                + ";"
                                + "{:.0f}".format(npatdone/currenttime),
                                "  ",
                                "{:.0f}".format((ncpudone / njobs) * 100) + "%",
                                "{:.0f};".format(currenttime)
                                + "{:.0f}".format((njobs - ncpudone) / ncpudone * currenttime)
                                + " running;remaining(s)",
                                end=newline,
                            )
                        #time.sleep(0.001)
                    if message != 'Error':
                        if ncpudone == njobs:
                            cpuworkers[jid] = None
                            cputask[jid] = None
                            ctaskindex[jid] = None
                            #del cpuworkers[jid]
                            #del cputask[jid]
                            #del ctaskindex[jid]
                        elif len(cpujobs) > 0:
                            cjob = cpujobs.pop(0)
                            banddata = banddataout[cjob.pstart - patstart: cjob.pend - patstart, :]
                            bandnorm = bandnormsout[cjob.pstart - patstart: cjob.pend - patstart, :, :]
                            cputask[jid] = cpuworkers[jid].indexpoles.remote(cjob,banddata, bandnorm, indexer=remote_indexer)
                            ctaskindex[jid] = cjob


                        else: # there should be more to do, but waiting for work.
                            cputask[jid] = cpuworkers[jid].indexpoles.remote(None, None, None)
                            ctaskindex[jid] = None

                    else:
                        raise Exception("Error in indexing bands: ", ctaskindex[jid].pstart,ctaskindex[jid].pend)


                except Exception as e:
                    print(e)
                    cjob = ctaskindex[jid]
                    print('A CPU death has occured', cjob.pstart,cjob.pend)
                    del cpuworkers[jid]
                    del cputask[jid]
                    del ctaskindex[jid]
                    cpujobs.append(cjob)
                    if len(cpuworkers) == 0:
                        cpuworkers.append(  # make a new Ray Actor that can call the indexer defined in shared memory.
                            # These actors are read/write, thus can initialize the GPU queues
                            CPUWorker.options(num_cpus=1, num_gpus=0).remote(i))
                        cputask.append(cpuworkers[0].indexpoles.remote(None, None, None))
                        ctaskindex.append(None)

    ray.shutdown()
    print('\n')
    if return_indexer_obj:
        return dataout, banddataout, indexer
    else:
        return dataout, banddataout

def __optimizegpuchunk__(indexer, ngpupro, gpu_id, clparam):


    gpulist = []
    # test for GPU presence
    if clparam is None:
        return 1000

    if clparam.ngpu == 0:
        return 1000

    if gpu_id is None:
        for g in clparam.gpu:
            gpulist.append(g)
    else:
        temp = np.atleast_1d(gpu_id)

        for g in temp:
            gpulist.append(clparam.gpu[g])
    ngpu = len(gpulist)

    if ngpu == 0:
        return 1000

    gmem = 1e99
    for g in gpulist:

        if g.global_mem_size < gmem:
            gmem = g.global_mem_size

    gmem = 0.8*gmem # Build a margin in.
    #print('Global Mem:', gmem)
    #ncpu_per_gpu = max(1, np.ceil(n_cpu_nodes/ngpu))

    #print('Ncpu/gpu:', ncpu_per_gpu)
    patdim = indexer.bandDetectPlan.patDim
    rdndim = np.array([indexer.bandDetectPlan.nTheta+2*indexer.bandDetectPlan.padding[1],
                       indexer.bandDetectPlan.nRho+2*indexer.bandDetectPlan.padding[0]])
    memperpat = 4.0*float(patdim[0] * patdim[1] + 6.0 * rdndim[0] * rdndim[1])# rough estimate

    #print('Mem/pat:', memperpat)
    chunkguess = (float(gmem)/float(ngpupro)) / memperpat

    #print('chunkguess:', chunkguess)


    #print('cheatguess:', chunkguess)
    chunk = int(max(2, np.floor(chunkguess/16))*16) # ideally should be a multiple of 16
    #print('chunk:', chunk)
    #check for powers of two: it is adventageous to be near those.
    twocheck = np.log2(float(chunk))
    if np.abs((twocheck) - np.round(twocheck)) < 0.2:
        chunk = int(2**int(np.round(twocheck)))

    # finally - I am unsure how to check for integrated graphics that report system memory, so I am going
    # throw an arbitrary cap on this:
    chunk = min(2048, chunk)

    return chunk


@ray.remote(num_cpus=1, num_gpus=1)
class GPUWorker:
    def __init__(self, actorid=0, clparammodule=None, gpu_id=None, cudavis = '0'):
        # sys.path.append(path.dirname(path.dirname(__file__)))  # do this to help Ray find the program files
        # import openclparam # do this to help Ray find the program files
        # device, context, queue, program, mf
        # self.dataout = None
        # self.indxstart = None
        # self.indxend = None
        # self.rate = None
        os.environ["CUDA_VISIBLE_DEVICES"] = cudavis
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
                self.openCLParams.get_context()
                #self.openCLParams.get_queue()
                self.useGPU = True
            except:
                self.openCLParams = None

    def findbands(self, gpujob, pats=None, xyloc=None, PC = None, indexer=None):
        if gpujob is None:
            return 'Bored', (None, None, None)
        try:
            # print(type(self.openCLParams.ctx))
            gpujob._starttime()
            if PC is None:
                PC = indexer.PC


            if self.openCLParams is not None:
                self.openCLParams.get_queue()
            pats, xyloc = indexer._getpats(
                patsin=pats,
                patstart=gpujob.pstart,
                npats=gpujob.npat,
                xyloc=xyloc)

            npoints = pats.shape[0]

            banddata, bandnorm = indexer._detectbands(pats, PC,
                                                   xyloc=xyloc,
                                                   clparams=self.openCLParams,
                                                   chunksize=-1)
            if self.openCLParams is not None:
                self.openCLParams.queue.finish()
                self.openCLParams.queue = None
            gpujob._endtime()
            return 'Done', (banddata, bandnorm, gpujob)
        except:
            gpujob.rate = None
            return "Error", (None, None, gpujob)
@ray.remote(num_cpus=1, num_gpus=0)
class CPUWorker:
    def __init__(self, actorid=0):
        self.actorID = actorid

    def indexpoles(self, cpujob, banddata, bandnorm, indexer=None):
        if cpujob is None:
            return 'Bored', (None, None)
        try:
            # print(type(self.openCLParams.ctx))

            cpujob._starttime()

            indxData = indexer._indexbandsphase(banddata, bandnorm, verbose=0)

            cpujob._endtime()
            return "Done", (indxData, cpujob)
        except Exception as e:
            print(e)
            cpujob.rate = None
            return "Error", (None, cpujob)
class CPUGPUJob:
    def __init__(self,jobid, pstart, pend, extime=0.0):
        self.jobid = jobid
        self.pstart = pstart
        self.pend = pend
        self.npat = pend - pstart
        self.starttime = 0.0
        self.endtime = 0.0
        self.extime = extime
        self.rate = 0.0
    def _starttime(self):
        self.starttime = timer()
    def _endtime(self):
        self.endtime = timer()
        self.extime += self.endtime - self.starttime
        self.rate = self.npat/(self.extime + 1e-12)

