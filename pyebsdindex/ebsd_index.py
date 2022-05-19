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

"""Setup and handling of Hough indexing runs of EBSD patterns."""

from os import path
import multiprocessing

import sys
import time
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import ray
import h5py

from pyebsdindex import (
    band_vote,
    ebsd_pattern,
    rotlib,
    tripletlib
)
from pyebsdindex.EBSDImage import IPFcolor

try:
    from pyebsdindex.opencl import band_detect_cl as band_detect
except ImportError:
    from pyebsdindex import band_detect as band_detect


# if sys.platform == 'darwin':
#   if ray.__version__ < '1.1.0':  # this fixes an issue when running locally on a VPN
#     ray.services.get_node_ip_address = lambda: '127.0.0.1'
#   else:
#     ray._private.services.get_node_ip_address = lambda: '127.0.0.1'

RADEG = 180.0 / np.pi


def index_pats(patsIn=None,filename=None,filenameout=None,phaselist=['FCC'], \
               vendor=None,PC=None,sampleTilt=70.0,camElev=5.3, \
               bandDetectPlan=None,nRho=90,nTheta=180,tSigma=None,rSigma=None,rhoMaskFrac=0.1,nBands=9, \
               backgroundSub=False,patstart=0,npats=-1, \
               return_indexer_obj=False,ebsd_indexer_obj=None,clparams=None,verbose=0, chunksize = 528, gpu_id=None):

  pats = None
  if patsIn is None:
    pdim = None
  else:
    if isinstance(patsIn,ebsd_pattern.EBSDPatterns):
      pats = patsIn.patterns
    if type(patsIn) is np.ndarray:
      pats = patsIn
    if isinstance(patsIn, h5py.Dataset):
      shp = patsIn.shape
      if len(shp) == 3:
        pats = patsIn
      if len(shp) == 2:  # just read off disk now.
        pats = patsIn[()]
        pats = pats.reshape(1, shp[0], shp[1])

    if pats is None:
      print('Unrecognized input data type')
      return
    pdim = pats.shape[-2:]


  if ebsd_indexer_obj == None:
    indexer = EBSDIndexer(filename=filename,phaselist=phaselist, \
                          vendor=vendor,PC=PC,sampleTilt=sampleTilt,camElev=camElev, \
                          bandDetectPlan=bandDetectPlan, \
                          nRho=nRho,nTheta=nTheta,tSigma=tSigma,rSigma=rSigma,rhoMaskFrac=rhoMaskFrac,nBands=nBands,
                          patDim=pdim, gpu_id=gpu_id)
  else:
    indexer = ebsd_indexer_obj

  if filename is not None:
    indexer.update_file(filename)
  if pats is not None:
    if not np.all(indexer.bandDetectPlan.patDim == np.array(pdim)):
      indexer.update_file(patDim=pats.shape[-2:])

  if backgroundSub == True:
    indexer.bandDetectPlan.collect_background(fileobj=indexer.fID,patsIn=pats,nsample=1000)

  dataout,banddata, indxstart,indxend = indexer.index_pats(patsin=pats,patstart=patstart,npats=npats, \
                                                 clparams=clparams,verbose=verbose, chunksize = chunksize)

  if return_indexer_obj == False:
    return dataout, banddata
  else:
    return dataout,banddata, indexer

def index_pats_distributed(patsIn=None,filename=None,filenameout=None,phaselist=['FCC'], \
                           vendor=None,PC=None,sampleTilt=70.0,camElev=5.3, \
                           peakDetectPlan=None,nRho=90,nTheta=180,tSigma=None,rSigma=None,rhoMaskFrac=0.1,nBands=9,
                           patstart=0,npats=-1,chunksize=528,ncpu=-1,
                           return_indexer_obj=False,ebsd_indexer_obj=None,keep_log=False, gpu_id=None):
  pats = None
  if patsIn is None:
    pdim = None
  else:
    if isinstance(patsIn, ebsd_pattern.EBSDPatterns):
      pats = patsIn.patterns
    if type(patsIn) is np.ndarray:
      pats = patsIn
    if isinstance(patsIn, h5py.Dataset):
      shp = patsIn.shape
      if len(shp) == 3:
        pats = patsIn
      if len(shp) == 2:  # just read off disk now.
        pats = patsIn[()]
        pats = pats.reshape(1, shp[0], shp[1])

    if pats is None:
      print('Unrecognized input data type')
      return
    pdim = pats.shape[-2:]


  # run a test flight to make sure all parameters are set properly before being sent off to the cluster
  if ebsd_indexer_obj == None:
    indexer = EBSDIndexer(filename=filename,phaselist=phaselist, \
                          vendor=vendor,PC=PC,sampleTilt=sampleTilt,camElev=camElev, \
                          bandDetectPlan=peakDetectPlan, \
                          nRho=nRho,nTheta=nTheta,tSigma=tSigma,rSigma=rSigma,rhoMaskFrac=rhoMaskFrac,nBands=nBands,
                          patDim=pdim, gpu_id=gpu_id)
  else:
    indexer = ebsd_indexer_obj

  if filename is not None:
    indexer.update_file(filename)
  else:
    indexer.update_file(patDim=pats.shape[-2:])

  # differentiate between getting a file to index or an array
  # Need to index one pattern to make sure the indexer object is fully initiated before
  #   placing in shared memory store.
  mode = 'memorymode'
  if pats is None:
    mode = 'filemode'
    temp,temp2, indexer = index_pats(patstart=0,npats=1,return_indexer_obj=True,ebsd_indexer_obj=indexer)

  if mode == 'filemode':
    npatsTotal = indexer.fID.nPatterns
  else:
    pshape = pats.shape
    if len(pshape) == 2:
      npatsTotal = 1
      pats = pats.reshape([1,pshape[0],pshape[1]])
    else:
      npatsTotal = pshape[0]
    temp,temp2, indexer = index_pats(pats[0,:,:],patstart=0,npats=1,return_indexer_obj=True,ebsd_indexer_obj=indexer)

  if patstart < 0:
    patstart = npatsTotal - patstart
  if npats <= 0:
    npats = npatsTotal - patstart

  # now set up the cluster with the indexer

  n_cpu_nodes = int(multiprocessing.cpu_count())  # int(sum([ r['Resources']['CPU'] for r in ray.nodes()]))
  if ncpu != -1:
    n_cpu_nodes = ncpu

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
  #ray.init(num_cpus=n_cpu_nodes,num_gpus=ngpu,_system_config={"maximum_gcs_destroyed_actor_cached_count": n_cpu_nodes})
  ray.init(num_cpus=n_cpu_nodes,num_gpus=ngpu, _node_ip_address="0.0.0.0")

  # place indexer obj in shared memory store so all workers can use it.
  remote_indexer = ray.put(indexer)
  clparamfunction = band_detect.getopenclparam
  # set up the jobs
  njobs = (np.ceil(npats / chunksize)).astype(np.long)
  # p_indx_start = [i*chunksize+patStart for i in range(njobs)]
  # p_indx_end = [(i+1)*chunksize+patStart for i in range(njobs)]
  # p_indx_end[-1] = npats+patStart
  p_indx_start_end = [[i * chunksize + patstart,(i + 1) * chunksize + patstart,chunksize] for i in range(njobs)]
  p_indx_start_end[-1][1] = npats + patstart
  p_indx_start_end[-1][2] = p_indx_start_end[-1][1] - p_indx_start_end[-1][0]

  if njobs < n_cpu_nodes:
    n_cpu_nodes = njobs

  nPhases = len(indexer.phaseLib)
  dataout = np.zeros((nPhases + 1,npats),dtype=indexer.dataTemplate)
  banddataout = np.zeros((npats, indexer.bandDetectPlan.nBands),dtype=indexer.bandDetectPlan.dataType)
  ndone = 0
  nsubmit = 0
  tic0 = timer()
  npatsdone = 0.0

  if keep_log is True:
    newline = '\n'
  else:
    newline = '\r'
  if mode == 'filemode':
    # send out the first batch
    workers = []
    jobs = []
    timers = []
    jobs_indx = []
    chunkave = 0.0
    for i in range(n_cpu_nodes):
      job_pstart_end = p_indx_start_end.pop(0)
      workers.append(IndexerRay.options(num_cpus=1,num_gpus=ngpupnode).remote(i, clparamfunction, gpu_id=gpu_id))
      jobs.append(workers[i].index_chunk_ray.remote(pats=None,indexer=remote_indexer, \
                                                    patstart=job_pstart_end[0],npats=job_pstart_end[2]))
      nsubmit += 1
      timers.append(timer())
      time.sleep(0.01)
      jobs_indx.append(job_pstart_end[:])

    while ndone < njobs:
      # toc = timer()
      wrker,busy = ray.wait(jobs,num_returns=1,timeout=None)

      # print("waittime: ",timer() - toc)
      jid = jobs.index(wrker[0])
      try:
        wrkdataout,wrkbanddata, indxstr,indxend,rate = ray.get(wrker[0])
      except:
        # print('a death has occured')
        indxstr = jobs_indx[jid][0]
        indxend = jobs_indx[jid][1]
        rate = [-1,-1]
      if rate[0] >= 0:  # job finished as expected

        ticp = timers[jid]
        dataout[:,indxstr - patstart:indxend - patstart] = wrkdataout
        banddataout[indxstr - patstart:indxend - patstart, :] = wrkbanddata
        npatsdone += rate[1]
        ndone += 1

        ratetemp = n_cpu_nodes * (rate[1]) / (timer() - ticp)
        chunkave += ratetemp
        totalave = npatsdone / (timer() - tic0)
        # print('Completed: ',str(indxstr),' -- ',str(indxend), '  ', npatsdone/(timer()-tic) )

        toc0 = timer() - tic0
        if keep_log is False:
          print('                                                                                             ',
                end='\r')
          time.sleep(0.00001)
        print('Completed: ',str(indxstr),' -- ',str(indxend),'  PPS:',"{:.0f}".format(ratetemp) + ';' +
              "{:.0f}".format(chunkave / ndone) + ';' + "{:.0f}".format(totalave),
              '  ',"{:.0f}".format((ndone / njobs) * 100) + '%',
              "{:.0f};".format(toc0) + "{:.0f}".format((njobs - ndone) / ndone * toc0) + ' running;remaining(s)',
              end=newline)

        if len(p_indx_start_end) > 0:
          job_pstart_end = p_indx_start_end.pop(0)
          jobs[jid] = workers[jid].index_chunk_ray.remote(pats=None,indexer=remote_indexer,
                                                          patstart=job_pstart_end[0],npats=job_pstart_end[2])
          nsubmit += 1
          timers[jid] = timer()
          jobs_indx[jid] = job_pstart_end[:]
        else:
          del jobs[jid]
          del workers[jid]
          del timers[jid]
          del jobs_indx[jid]

      else:  # something bad happened.  Put the job back on the queue and kill this worker
        p_indx_start_end.append([indxstr,indxend,indxend - indxstr])
        del jobs[jid]
        del workers[jid]
        del timers[jid]
        del jobs_indx[jid]
        n_cpu_nodes -= 1
        if len(workers) < 1:  # rare case that we have killed all workers...
          job_pstart_end = p_indx_start_end.pop(0)
          workers.append(IndexerRay.options(num_cpus=1,num_gpus=ngpupnode).remote(jid,clparamfunction,gpu_id))
          jobs.append(workers[0].index_chunk_ray.remote(pats=None,indexer=remote_indexer, \
                                                        patstart=job_pstart_end[0],npats=job_pstart_end[2]))
          nsubmit += 1
          timers.append(timer())
          time.sleep(0.01)
          jobs_indx.append(job_pstart_end[:])
          n_cpu_nodes += 1

  if mode == 'memorymode':
    workers = []
    jobs = []
    timers = []
    jobs_indx = []
    chunkave = 0.0
    for i in range(n_cpu_nodes):
      job_pstart_end = p_indx_start_end.pop(0)
      workers.append(IndexerRay.options(num_cpus=1,num_gpus=ngpupnode).remote(i,clparamfunction, gpu_id))
      jobs.append(workers[i].index_chunk_ray.remote(pats=pats[job_pstart_end[0]:job_pstart_end[1],:,:],
                                                    indexer=remote_indexer, \
                                                    patstart=job_pstart_end[0],npats=job_pstart_end[2]))
      nsubmit += 1
      timers.append(timer())
      jobs_indx.append(job_pstart_end)
      time.sleep(0.01)

    # workers = [index_chunk.remote(pats = None, indexer = remote_indexer, patStart = p_indx_start[i], patEnd = p_indx_end[i]) for i in range(n_cpu_nodes)]
    # nsubmit += n_cpu_nodes

    while ndone < njobs:
      toc = timer()
      wrker,busy = ray.wait(jobs,num_returns=1,timeout=None)
      jid = jobs.index(wrker[0])
      # print("waittime: ",timer() - toc)
      try:
        wrkdataout, wrkbanddata, indxstr, indxend, rate = ray.get(wrker[0])
      except:
        indxstr = jobs_indx[jid][0]
        indxend = jobs_indx[jid][1]
        rate = [-1,-1]
      if rate[0] >= 0:
        ticp = timers[jid]
        dataout[:,indxstr - patstart:indxend - patstart] = wrkdataout
        banddataout[indxstr - patstart:indxend - patstart, :] = wrkbanddata
        npatsdone += rate[1]
        ratetemp = n_cpu_nodes * (rate[1]) / (timer() - ticp)
        chunkave += ratetemp
        totalave = npatsdone / (timer() - tic0)
        # print('Completed: ',str(indxstr),' -- ',str(indxend), '  ', npatsdone/(timer()-tic) )
        ndone += 1
        toc0 = timer() - tic0
        if keep_log is False:
          print('                                                                                             ',
                end='\r')
          time.sleep(0.0001)
        print('Completed: ',str(indxstr),' -- ',str(indxend),'  PPS:',"{:.0f}".format(ratetemp) + ';' +
              "{:.0f}".format(chunkave / ndone) + ';' + "{:.0f}".format(totalave),
              '  ',"{:.0f}".format((ndone / njobs) * 100) + '%',
              "{:.0f};".format(toc0) + "{:.0f}".format((njobs - ndone) / ndone * toc0) + ' running;remaining(s)',
              end=newline)

        if len(p_indx_start_end) > 0:
          job_pstart_end = p_indx_start_end.pop(0)
          jobs[jid] = workers[jid].index_chunk_ray.remote(pats=pats[job_pstart_end[0]:job_pstart_end[1],:,:]
                                                          ,indexer=remote_indexer,
                                                          patstart=job_pstart_end[0],npats=job_pstart_end[2])
          nsubmit += 1
          timers[jid] = timer()
          jobs_indx[jid] = job_pstart_end
        else:
          del jobs[jid]
          del workers[jid]
          del timers[jid]
          del jobs_indx[jid]
      else:  # something bad happened.  Put the job back on the queue and kill this worker
        p_indx_start_end.append([indxstr,indxend,indxend - indxstr])
        del jobs[jid]
        del workers[jid]
        del timers[jid]
        del jobs_indx[jid]
        n_cpu_nodes -= 1
        if len(workers) < 1:  # rare case that we have killed all workers...
          job_pstart_end = p_indx_start_end.pop(0)
          workers.append(IndexerRay.options(num_cpus=1,num_gpus=ngpupnode).remote(jid, clparamfunction, gpu_id))
          jobs.append(workers[0].index_chunk_ray.remote(pats=pats[job_pstart_end[0]:job_pstart_end[1],:,:],
                                                        indexer=remote_indexer, \
                                                        patstart=job_pstart_end[0],npats=job_pstart_end[2]))
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
    return dataout,banddataout,indexer
  else:
    return dataout,banddataout

@ray.remote(num_cpus=1,num_gpus=1)
class IndexerRay():
  def __init__(self,actorid=0, clparammodule=None, gpu_id=None):
    sys.path.append((path.dirname(path.dirname(__file__))))  # do this to help Ray find the program files
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
        if sys.platform != 'darwin':  # linux with NVIDIA (unsure if it is the os or GPU type) is slow to make a
          self.openCLParams = clparammodule()


        else:  # MacOS handles GPU memory conflicts much better when the context is destroyed between each
          # run, and has very low overhead for making the context.
          #pass
          self.openCLParams = clparammodule()
          #self.openCLParams.gpu_id = 0
          #self.openCLParams.gpu_id = 1
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

  def index_chunk_ray(self,pats=None,indexer=None,patstart=0,npats=-1):
    try:
      #print(type(self.openCLParams.ctx))
      tic = timer()
      dataout,banddata, indxstart,npatsout = indexer.index_pats(patsin=pats,patstart=patstart,npats=npats,
                                                      clparams=self.openCLParams, chunksize = -1)
      rate = np.array([timer() - tic,npatsout])
      return dataout,banddata, indxstart,indxstart + npatsout,rate
    except:
      indxstart = patstart
      indxend = patstart + npats
      return None,None, indxstart,indxend,[-1,-1]


class EBSDIndexer:
    """Setup of Hough indexing of EBSD patterns."""
    def __init__(
        self,
        filename=None,
        phaselist=['FCC'],
        vendor=None,
        PC=None,
        sampleTilt=70.0,
        camElev=5.3,
        bandDetectPlan=None,
        nRho=90,
        nTheta=180,
        tSigma=1.0,
        rSigma=1.2,
        rhoMaskFrac=0.15,
        nBands=9,
        patDim=None,
        **kwargs
    ):
        """Create an EBSD indexer.

        Parameters
        ----------
        filename : str, optional
            Name of file with EBSD patterns.
        phaselist : list of str, optional
            Options are "FCC" and "BCC". Default is ["FCC"].
        vendor : str, optional
            This string determines the pattern center (PC) convention to
            use. The available options are "EDAX" (default), "BRUKER",
            "OXFORD", "EMSOFT", "KIKUCHIPY" (equivalent to "BRUKER").
        PC : list, optional
            Pattern center (PC) x*, y*, z* in EDAX TSL's convention,
            defined in fractions of pattern width with respect to the
            lower left corner of the detector. If not passed, this is
            set to (x*, y*, z*) = (0.471659, 0.675044, 0.630139).
        sampleTilt : float, optional
            Sample tilt towards the detector in degrees. Default is 70
            degrees. Unused if `ebsd_indexer_obj` is passed.
        camElev : float, optional
            Camera elevation in degrees. Default is 5.3 degrees. Unused
            if `ebsd_indexer_obj` is passed.
        bandDetectPlan : pyebsdindex.band_detect.BandDetect, optional
            Collection of parameters using in band detection. Unused if
            `ebsd_indexer_obj` is passed.
        nRho : int, optional
            Default is 90 degrees. Unused if `ebsd_indexer_obj` is passed.
        nTheta : int, optional
            Default is 180 degrees. Unused if `ebsd_indexer_obj` is passed.
        tSigma : float, optional
            Unused if `ebsd_indexer_obj` is passed.
        rSigma : float, optional
            Unused if `ebsd_indexer_obj` is passed.
        rhoMaskFrac : float, optional
            Default is 0.1. Unused if `ebsd_indexer_obj` is passed.
        nBands : int, optional
            Number of detected bands to use in triplet voting. Default
            is 9. Unused if `ebsd_indexer_obj` is passed.
        patDim : int, optional
            Number of dimensions of pattern array.
        kwargs
            Keyword arguments passed on to `BandDetect`.

        """
        self.filein = filename
        if self.filein is not None:
            self.fID = ebsd_pattern.get_pattern_file_obj(self.filein)
        else:
            self.fID = None

        # self.fileout = filenameout
        # if self.fileout is None:
        #     self.fileout = str.lower(Path(self.filein).stem)+'.ang'

        self.phaselist = phaselist
        self.phaseLib = []
        for ph in self.phaselist:
            self.phaseLib.append(band_vote.BandVote(tripletlib.triplib(libType=ph)))

        self.vendor = 'EDAX'
        if vendor is None:
            if self.fID is not None:
                self.vendor = self.fID.vendor
        else:
            self.vendor = vendor

        if PC is None:
            self.PC = np.array([0.471659, 0.675044, 0.630139])  # a default value
        else:
            self.PC = np.asarray(PC)

        self.PCcorrectMethod = None
        self.PCcorrectParam = None

        self.sampleTilt = sampleTilt
        self.camElev = camElev

        if bandDetectPlan is None:
            self.bandDetectPlan = band_detect.BandDetect(
              nRho=nRho, nTheta=nTheta, tSigma=tSigma, rSigma=rSigma, rhoMaskFrac=rhoMaskFrac, nBands=nBands,
              **kwargs)
        else:
            self.bandDetectPlan = bandDetectPlan

        if self.fID is not None:
            self.bandDetectPlan.band_detect_setup(patDim=[self.fID.patternW, self.fID.patternH])
        else:
            if patDim is not None:
                self.bandDetectPlan.band_detect_setup(patDim=patDim)

        self.dataTemplate = np.dtype([
            ('quat', np.float64, 4),
            ('iq', np.float32),
            ('pq', np.float32),
            ('cm', np.float32),
            ('phase', np.int32),
            ('fit', np.float32),
            ('nmatch', np.int32),
            ('matchattempts', np.int32, 2),
            ('totvotes', np.int32)
        ])

    def update_file(self, filename=None, patDim=np.array([120, 120], dtype=np.int32)):
        if filename is None:
            self.filein = None
            self.bandDetectPlan.band_detect_setup(patDim=patDim)
        else:
            self.filein = filename
            self.fID = ebsd_pattern.get_pattern_file_obj(self.filein)
            self.bandDetectPlan.band_detect_setup(patDim=[self.fID.patternW, self.fID.patternH])

    def index_pats(
        self,
        patsin=None,
        patstart=0,
        npats=-1,
        clparams=None,
        PC=[None, None, None],
        verbose=0,
        chunksize=528
    ):
        """Hough indexing of EBSD patterns.

        Parameters
        ----------
        patsin : numpy.ndarray, optional
            EBSD patterns in an array of shape (n points, n pattern
            rows, n pattern columns). If not given, these are read from
            `self.filename`.
        patstart : int, optional
            Starting index of the patterns to index. Default is 0.
        npats : int, optional
            Number of patterns to index. Default is -1, which will index
            up to the final pattern in `patsin`.
        clparams : list, optional
            OpenCL parameters passed to pyopencl.
        PC : list, optional
            Pattern center (PC) (PCx, PCy, PCz) in `self.vendor`'s
            convention (default is "EDAX"). If not given, this is read
            from `self.PC`.
        verbose : int, optional
            0 - no output, 1 - timings, 2 - timings and the Hough
            transform of the first pattern with detected bands
            highlighted.
        chunksize : int, optional
            Default is 528.

        Returns
        -------
        indxData : numpy.ndarray
            Complex numpy array (or array of structured data), that is
            [nphases + 1, npoints]. The data is stored for each phase
            used in indexing and the `indxData[-1]` layer uses the best
            guess on which is the most likely phase, based on the fit,
            and number of bands matched for each phase. Each data entry
            contains the orientation expressed as a quaternion (quat)
            (using `self.vendor`'s convention), Pattern Quality (pq),
            Confidence Metric (cm), Phase ID (phase), Fit (fit) and
            Number of Bands Matched (nmatch). There are some other
            metrics reported, but these are mostly for debugging
            purposes.
        bandData : numpy.ndarray
            Band identification data from the Radon transform.
        patstart : int
            Starting index of the indexed patterns.
        npats : int
            Number of patterns indexed. This and `patstart` are useful
            for the distributed indexing procedures.

        """
        tic = timer()

        if patsin is None:
            pats = self.fID.read_data(returnArrayOnly=True, patStartCount=[patstart, npats], convertToFloat=True)
        else:
            pshape = patsin.shape
            if len(pshape) == 2:
                pats = np.reshape(patsin, (1, pshape[0], pshape[1]))
            else:
                pats = patsin  # [patStart:patEnd, :,:]
            pshape = pats.shape

            if self.bandDetectPlan.patDim is None:
                self.bandDetectPlan.band_detect_setup(patDim=pshape[1:3])
            else:
                if np.all((np.array(pshape[1:3]) - self.bandDetectPlan.patDim) == 0):
                    self.bandDetectPlan.band_detect_setup(patDim=pshape[1:3])

        if self.bandDetectPlan.patDim is None:
            self.bandDetectPlan.band_detect_setup(patterns=pats)

        npoints = pats.shape[0]
        if npats == -1:
            npats = npoints

        # print(timer() - tic)
        tic = timer()
        bandData = self.bandDetectPlan.find_bands(pats, clparams=clparams, verbose=verbose, chunksize=chunksize)
        shpBandDat = bandData.shape
        if PC[0] is None:
            PC_0 = self.PC
        else:
            PC_0 = PC
        bandNorm = self.bandDetectPlan.radonPlan.radon2pole(bandData, PC=PC_0, vendor=self.vendor)
        # print('Find Band: ', timer() - tic)

        # return bandNorm,patStart,patEnd
        tic = timer()
        # bv = []
        # for tl in self.phaseLib:
        #  bv.append(band_vote.BandVote(tl))
        nPhases = len(self.phaseLib)
        q = np.zeros((nPhases, npoints, 4))
        indxData = np.zeros((nPhases + 1, npoints), dtype=self.dataTemplate)

        indxData['phase'] = -1
        indxData['fit'] = 180.0
        indxData['totvotes'] = 0
        earlyexit = max(7, shpBandDat[1])
        for i in range(npoints):
            bandNorm1 = bandNorm[i, :, :]
            bDat1 = bandData[i, :]
            whgood = np.nonzero(bDat1['max'] > -1.0e6)[0]
            if whgood.size >= 3:
                bDat1 = bDat1[whgood]
                bandNorm1 = bandNorm1[whgood, :]
                indxData['pq'][0:nPhases, i] = np.sum(bDat1['max'], axis=0)

                for j in range(len(self.phaseLib)):
                  (
                    avequat, fit, cm, bandmatch, nMatch, matchAttempts, totvotes
                  ) = self.phaseLib[j].tripvote(bandNorm1, band_intensity = bDat1['avemax'], goNumba=True, verbose=verbose)
                  # avequat,fit,cm,bandmatch,nMatch, matchAttempts = self.phaseLib[j].pairVoteOrientation(bandNorm1,goNumba=True)
                  if nMatch >= 3:
                      q[j, i, :] = avequat
                      indxData['fit'][j, i] = fit
                      indxData['cm'][j, i] = cm
                      indxData['phase'][j, i] = j
                      indxData['nmatch'][j, i] = nMatch
                      indxData['matchattempts'][j, i] = matchAttempts
                      indxData['totvotes'][j, i] = totvotes
                  if nMatch >= earlyexit:
                      break

        qref2detect = self.refframe2detector()
        q = q.reshape(nPhases * npoints, 4)
        q = rotlib.quat_multiply(q, qref2detect)
        q = rotlib.quatnorm(q)
        q = q.reshape(nPhases, npoints, 4)
        indxData['quat'][0:nPhases, :, :] = q
        if nPhases > 1:
            for j in range(nPhases - 1):
                indxData[-1, :] = np.where(
                    (indxData[j, :]['cm'] * indxData[j, :]['nmatch']) > (indxData[j + 1, :]['cm'] * indxData[j + 1, :]['nmatch']),
                    indxData[j, :],
                    indxData[j + 1, :]
                )
        else:
          indxData[-1, :] = indxData[0, :]

        if verbose > 0:
            print('Band Vote Time: ', timer() - tic)
        return indxData, bandData, patstart, npats

    def refframe2detector(self):
        ven = str.upper(self.vendor)
        if ven in ['EDAX', 'EMSOFT', 'KIKUCHIPY']:
            q0 = np.array([np.sqrt(2.0) * 0.5, 0.0, 0.0, -1.0 * np.sqrt(2.0) * 0.5])
            tiltang = -1.0 * (90.0 - self.sampleTilt + self.camElev) / RADEG
            q1 = np.array([np.cos(tiltang * 0.5), np.sin(tiltang * 0.5), 0.0, 0.0])
            quatref2detect = rotlib.quat_multiply(q1, q0)

        if ven in ['OXFORD', 'BRUKER']:
            tiltang = -1.0 * (90.0 - self.sampleTilt + self.camElev) / RADEG
            q1 = np.array([np.cos(tiltang * 0.5), np.sin(tiltang * 0.5), 0.0, 0.0])
            quatref2detect = q1

        return quatref2detect

    def pcCorrect(self, xy=[[0.0, 0.0]]):  # at somepoint we will put some methods here for correcting the PC
        # depending on the location within the scan.  Need to correct band_detect.radon2pole to accept a
        # PC for each point
        pass


def __main__(file=None,ncpu=-1):
  print('Hello')
  if file is None:
    file = '~/Desktop/SLMtest/scan2v3nlparl09sw7.up1'

  dat1,indxer = index_pats(filename=file,
                           patstart=0,npats=1,return_indexer_obj=True,
                           nTheta=180,nRho=90,
                           tSigma=1.0,rSigma=1.2,rhoMaskFrac=0.1,nBands=9, \
                           phaselist=['FCC'])

  dat = index_pats_distributed(filename=file,patstart=0,npats=-1,
                               chunksize=1008,ncpu=ncpu,ebsd_indexer_obj=indxer)
  imshape = (indxer.fID.nRows,indxer.fID.nCols)
  ipfim = IPFcolor.ipf_color_cubic(dat['quat']).reshape(imshape[0],imshape[1],3);
  plt.imshow(ipfim)
