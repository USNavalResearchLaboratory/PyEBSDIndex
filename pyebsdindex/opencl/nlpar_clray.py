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
# The US Naval Research Laboratory Date: 22 May 2024

# For more info see
# Patrick T. Brewick, Stuart I. Wright, David J. Rowenhorst. Ultramicroscopy, 200:50–61, May 2019.

"""Non-local pattern averaging and re-indexing (NLPAR)."""



import os, sys, platform
import logging
from timeit import default_timer as timer
import numpy as np
import pyopencl as cl

import ray


from pyebsdindex.opencl import openclparam, nlpar_cl
from time import time as timer

RAYIPADDRESS = '127.0.0.1'
OSPLATFORM  = platform.system()
#if OSPLATFORM  == 'Darwin':
#    RAYIPADDRESS = '0.0.0.0'  # the localhost address does not work on macOS when on a VPN


class NLPAR(nlpar_cl.NLPAR):
  def __init__( self,filename=None, **kwargs):
    nlpar_cl.NLPAR.__init__(self,filename=filename, **kwargs)
    self.useCPU = False

  def calcnlpar(self, **kwargs):
    return self.calcnlpar_clray(**kwargs)

  def calcsigma(self, nn=1, saturation_protect=True, automask=True, return_nndist=False, **kwargs):
    if self.sigmann > 7:
      print("Sigma optimization search limited to a search radius <= 7")
      print("The search radius has been clipped to 7")
      nn = 7
      self.sigmann = nn

    sig = self.calcsigma_clray(nn=nn,
                            saturation_protect=saturation_protect,
                            automask=automask, **kwargs)
    if return_nndist == True:
      return sig
    else:
      return sig[0]


  def calcnlpar_clsq(self, **kwargs):
    return nlpar_cl.NLPAR.calcnlpar_cl(self, **kwargs)

  def calcsigma_clsq(self, **kwargs):
    return nlpar_cl.NLPAR.calcsigma_cl(self, **kwargs)

  def calcsigma_clray(self, nn=1, saturation_protect=True, automask=True, normalize_d=False,
                      gpu_id = None, verbose=2, **kwargs):
    self.patternfile = self.getinfileobj()
    self.sigmann = nn

    if self.sigmann > 7:
      print("Sigma optimization search limited to a search radius <= 7")
      print("The search radius has been clipped to 7")
      nn = 7
      self.sigmann = nn

    if gpu_id is None:
      clparams = openclparam.OpenClParam()
      clparams.get_gpu()

      target_mem = 0
      gpu_id = 0
      count = 0

      for gpu in clparams.gpu:
        gmem = gpu.max_mem_alloc_size
        if target_mem < gmem:
          gpu_id = count
          target_mem = gmem
        count += 1
    else:
      clparams = openclparam.OpenClParam()
      clparams.get_gpu()
      gpu_id = min(len(clparams.gpu)-1, gpu_id)

    cudavis = ''
    for cdgpu in range(len(clparams.gpu)):
      cudavis += str(cdgpu) + ','

    #print(gpu_id)
    ngpuwrker = 6
    clparams.get_context(gpu_id=gpu_id, kfile = 'clnlpar.cl')
    clparams.get_queue()
    if clparams.gpu[gpu_id].host_unified_memory:
      return nlpar_cl.NLPAR.calcsigma_cl(self, nn=nn, saturation_protect=saturation_protect,
                                         automask=automask,
                                         normalize_d=normalize_d,
                                         gpu_id=gpu_id, **kwargs)

    target_mem = clparams.gpu[gpu_id].max_mem_alloc_size // 2
    max_mem = clparams.gpu[gpu_id].global_mem_size * 0.5
    if target_mem * ngpuwrker > max_mem:
      #print('revisemem:')
      target_mem = max_mem / ngpuwrker

    patternfile = self.getinfileobj()

    nrows = np.int64(self.nrows)  # np.uint64(patternfile.nRows)
    ncols = np.int64(self.ncols)  # np.uint64(patternfile.nCols)

    pwidth = np.uint64(patternfile.patternW)
    pheight = np.uint64(patternfile.patternH)
    npat_point = int(pwidth * pheight)

    if (automask is True) and (self.mask is None):
      self.mask = (self.automask(pheight, pwidth))
    if self.mask is None:
      self.mask = np.ones((pheight, pwidth), dtype=np.uint8)

    mask = self.mask.astype(np.float32)

    chunks = self._calcchunks([pwidth, pheight], ncols, nrows, target_bytes=target_mem,
                              col_overlap=nn, row_overlap=nn)


    jobqueue = []

    for rowchunk in range(chunks[1]):
      rstart = chunks[3][rowchunk, 0]
      rend = chunks[3][rowchunk, 1]
      nrowchunk = rend - rstart

      rstartcalc = nn if (rowchunk > 0) else 0
      rendcalc = nrowchunk - nn if (rowchunk < (chunks[1] - 1)) else nrowchunk
      nrowcalc = np.int64(rendcalc - rstartcalc)

      for colchunk in range(chunks[0]):
        cstart = chunks[2][colchunk, 0]
        cend = chunks[2][colchunk, 1]
        ncolchunk = cend - cstart

        cstartcalc = nn if (colchunk > 0) else 0
        cendcalc = ncolchunk - nn if (colchunk < (chunks[0] - 1)) else ncolchunk
        ncolcalc = np.int64(cendcalc - cstartcalc)

        jobqueue.append(NLPARGPUJob([colchunk, rowchunk], \
                                    [cstart, cend, rstart, rend], \
                                    [cstartcalc, cendcalc, rstartcalc, rendcalc]))


    # wrker = NLPARGPUWorker(actorid=1, gpu_id=gpu_id, cudavis=cudavis)
    # job = jobqueue[0]
    #
    # data = wrker.runsigma_chunk(job, nlparobj=self, saturation_protect=saturation_protect)
    #
    # return data

    ngpu_per_wrker = float(1.0 / ngpuwrker)

    # if verbose >=1:
    #   print("lambda:", self.lam, "search radius:", self.searchradius, "dthresh:", self.dthresh)

    ray.shutdown()
    rayclust = ray.init(
      num_cpus=int(ngpuwrker),
      num_gpus=1,
      # dashboard_host = RAYIPADDRESS,
      _node_ip_address=RAYIPADDRESS,  # "0.0.0.0",
      runtime_env={"env_vars":
                     {"PYTHONPATH": os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                      }},
      logging_level=logging.WARNING, )  # Supress INFO messages from ray.

    nlpar_remote = ray.put(self)

    nnn = int((2 * nn + 1) ** 2)
    sigma = np.zeros((nrows, ncols), dtype=np.float32) + 1e18
    dist = np.zeros((nrows, ncols, nnn), dtype=np.float32)
    countnn = np.zeros((nrows, ncols, nnn), dtype=np.float32)

    idlewrker = []
    busywrker = []
    tasks = []

    for w in range(ngpuwrker):
      idlewrker.append(NLPARGPUWorker.options(num_cpus=float(0.99), num_gpus=ngpu_per_wrker).remote(
        actorid=w, gpu_id=gpu_id, cudavis=cudavis))

    njobs = len(jobqueue)
    ndone = 0

    while ndone < njobs:
      if len(jobqueue) > 0:
        if len(idlewrker) > 0:
          wrker = idlewrker.pop()
          job = jobqueue.pop()

          tasks.append(wrker.runsigma_chunk.remote(job, nlparobj=nlpar_remote, saturation_protect=saturation_protect))
          busywrker.append(wrker)
      if len(tasks) > 0:
        donetasks, stillbusy = ray.wait(tasks, num_returns=len(busywrker), timeout=0.1)

        for tsk in donetasks:
          indx = tasks.index(tsk)
          message, job, newdata = ray.get(tsk)
          if message == 'Done':

            sigmachunk = newdata[0]
            distchunk = newdata[1]
            countchunk = newdata[2]
            cstart = job.cstart
            cend = job.cend
            ncolchunk = job.ncolchunk
            rstart = job.rstart
            rend = job.rend
            nrowchunk = job.nrowchunk

            rstartcalc = job.rstartcalc
            cstartcalc = job.cstartcalc
            nrowcalc = job.nrowcalc
            ncolcalc = job.ncolcalc

            countnn[rstart:rend, cstart:cend] = countchunk[0:int(ncolchunk * nrowchunk), :].reshape(nrowchunk,
                                                                                        ncolchunk, nnn)
            dist[rstart:rend, cstart:cend] = distchunk[0:int(ncolchunk * nrowchunk), :].reshape(nrowchunk, ncolchunk,
                                                                                                nnn)
            sigma[rstart:rend, cstart:cend] = np.minimum(sigma[rstart:rend, cstart:cend], sigmachunk)

            idlewrker.append(busywrker.pop(indx))
            tasks.remove(tsk)
            ndone += 1
          if verbose >= 2:
            print("tiles complete: ", ndone,"/", njobs,sep='', end='\r'  )
    if verbose >= 2:
      print('\n', end='')
    return sigma, dist, countnn

  def _sigmachunkcalc_cl(self, data, calclim, clparams=None, saturation_protect=True):
    nn = self.sigmann

    data = np.ascontiguousarray(data)
    ctx = clparams.ctx
    prg = clparams.prg
    clparams.get_queue()


    mf = clparams.memflags
    clvectlen = 16

    cstart = calclim.cstart
    cend = calclim.cend
    ncolchunk = calclim.ncolchunk
    rstart = calclim.rstart
    rend = calclim.rend
    nrowchunk = calclim.nrowchunk

    rstartcalc = calclim.rstartcalc
    cstartcalc = calclim.cstartcalc
    nrowcalc = calclim.nrowcalc
    ncolcalc = calclim.ncolcalc

    pheight = data.shape[1]
    pwidth = data.shape[2]
    npat_point = pheight*pwidth

    mask = self.mask

    npad = clvectlen * int(np.ceil(mask.size/clvectlen))
    maskpad = np.zeros((npad) , np.float32)
    maskpad[0:mask.size] = mask.reshape(-1).astype(np.float32)
    mask_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=maskpad)

    npatsteps = int(maskpad.size/clvectlen)

    #return chunks, chunksize
    mxchunk = data.shape[0]

    npadmx = clvectlen * int(np.ceil(mxchunk*npat_point/ clvectlen))

    datapad_gpu = cl.Buffer(ctx, mf.READ_WRITE, size=int(npadmx) * int(4))

    nnn = int((2 * nn + 1) ** 2)

    #dist_local = cl.LocalMemory(nnn*npadmx*4)
    dist_local = cl.Buffer(ctx, mf.READ_WRITE, size=int(mxchunk*nnn*4))
    distchunk = np.zeros((mxchunk, nnn), dtype=np.float32)
    #count_local = cl.LocalMemory(nnn*npadmx*4)
    count_local = cl.Buffer(ctx, mf.READ_WRITE, size=int(mxchunk * nnn * 4))
    countchunk = np.zeros((mxchunk, nnn), dtype=np.float32)

    mxval = data.max()
    if saturation_protect == False:
      mxval += 1.0
    else:
      mxval *= 0.9961

    evnt = cl.enqueue_fill_buffer(clparams.queue, datapad_gpu, np.float32(mxval+10), 0,int(4*npadmx))

    szdata = data.size
    npad = clvectlen * int(np.ceil(szdata / clvectlen))
    tic = timer()
    #datapad = np.zeros((npad), dtype=np.float32) + np.float32(mxval + 10)
    #datapad[0:szdata] = data.reshape(-1)
    data_gpu = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=data)

    if data.dtype.type is np.float32:
      prg.nlloadpat32flt(clparams.queue, (np.uint64(data.size),), None, data_gpu, datapad_gpu, wait_for=[evnt])
    if data.dtype.type is np.ubyte:
      prg.nlloadpat8bit(clparams.queue, (np.uint64(data.size),), None, data_gpu, datapad_gpu, wait_for=[evnt])
    if data.dtype.type is np.uint16:
      prg.nlloadpat16bit(clparams.queue, (np.uint64(data.size),), None, data_gpu, datapad_gpu, wait_for=[evnt])
    toc = timer()
    #print(toc - tic)

    sigmachunk = np.zeros((nrowchunk, ncolchunk ), dtype=np.float32)


    sigmachunk_gpu =  cl.Buffer(ctx, mf.WRITE_ONLY, size=sigmachunk.nbytes)
    cl.enqueue_barrier(clparams.queue)
    prg.calcsigma(clparams.queue, (np.uint32(ncolchunk), np.uint32(nrowchunk)), None,
                           datapad_gpu, mask_gpu,sigmachunk_gpu,
                           dist_local, count_local,
                           np.int64(nn), np.int64(npatsteps), np.int64(npat_point),
                           np.float32(mxval) )

    cl.enqueue_barrier(clparams.queue)
    prg.normd(clparams.queue, (np.uint32(ncolchunk), np.uint32(nrowchunk)), None,
                    sigmachunk_gpu,
                    count_local, dist_local,
                    np.int64(nn))
    clparams.queue.finish()

    cl.enqueue_copy(clparams.queue, distchunk, dist_local, is_blocking=False)
    cl.enqueue_copy(clparams.queue, countchunk, count_local,  is_blocking=False)
    cl.enqueue_copy(clparams.queue, sigmachunk, sigmachunk_gpu,  is_blocking=True)

    #sigmachunk_gpu.release()

    clparams.queue.finish()

    dist_local.release()
    count_local.release()
    datapad_gpu.release()
    clparams.queue.flush()
    clparams.queue = None
    #self.sigma = sigma
    return sigmachunk, distchunk, countchunk



  def calcnlpar_clray(self, searchradius=None, lam = None, dthresh = None, saturation_protect=True, automask=True,
                filename=None, fileout=None, reset_sigma=False, backsub = False, rescale = False,
                verbose = 2, gpu_id = None, **kwargs):

    if lam is not None:
      self.lam = lam

    self.saturation_protect = saturation_protect

    if dthresh is not None:
      self.dthresh = dthresh
    if self.dthresh is None:
      self.dthresh = 0.0

    if searchradius is not None:
      self.searchradius = searchradius

    if self.searchradius > 10:
      print("NLPAR on GPU is limited to a search radius <= 10")
      print("The search radius has been clipped to 10")
      searchradius = 10
      self.searchradius = searchradius

    lam = np.float32(self.lam)
    dthresh = np.float32(self.dthresh)
    sr = np.int64(self.searchradius)

    if filename is not None:
      self.setfile(filepath=filename)

    self.patternfile = self.getinfileobj()
    self.setoutfile(self.patternfile, filepath=fileout)
    self.patternfileout = self.getoutfileobj()

    nrows = np.int64(self.nrows)  # np.uint64(patternfile.nRows)
    ncols = np.int64(self.ncols)  # np.uint64(patternfile.nCols)

    pwidth = np.uint64(self.patternfile.patternW)
    pheight = np.uint64(self.patternfile.patternH)
    npat_point = int(pwidth * pheight)

    if reset_sigma:
      self.sigma = None

    if np.asarray(self.sigma).size == 1 and (self.sigma is not None):
      tmp = np.asarray(self.sigma)[0]
      self.sigma = np.zeros((nrows, ncols), dtype=np.float32) + tmp
      calcsigma = False

    if self.sigma is not None:
      shpsigma = np.asarray(self.sigma).shape
      if (shpsigma[0] != nrows) and (shpsigma[1] != ncols):
        self.sigma = None

    if self.sigma is None:
      self.sigma = self.calcsigma_cl(nn=1, saturation_protect=saturation_protect, automask=automask, gpu_id=gpu_id)[0]

    sigma = np.asarray(self.sigma).astype(np.float32)



    if (automask is True) and (self.mask is None):
      self.mask = (self.automask(pheight, pwidth))
    if self.mask is None:
      self.mask = np.ones((pheight, pwidth), dtype=np.uint8)

    scalemethod = 'clip'
    self.rescale = False
    if rescale == True:
      self.rescale = True
      if np.issubdtype(self.patternfileout.filedatatype, np.integer):
        mxval = np.iinfo(self.patternfileout.filedatatype).max
        scalemethod = 'fullscale'
      else:  # not int, so no rescale.
        self.rescale = False

    ngpuwrker = 6
    clparams = openclparam.OpenClParam()
    clparams.get_gpu()
    if gpu_id is None:
      target_mem = 0
      gpu_id = 0
      count = 0

      for gpu in clparams.gpu:
        gmem = gpu.max_mem_alloc_size
        if target_mem < gmem:
          gpu_id = count
          target_mem = gmem
        count += 1
    else:
      gpu_id = min(len(clparams.gpu)-1, gpu_id)
    cudavis = ''
    for cdgpu in range(len(clparams.gpu)):
        cudavis += str(cdgpu) + ','

    # print(gpu_id)
    clparams.get_context(gpu_id=gpu_id, kfile = 'clnlpar.cl')
    clparams.get_queue()
    if clparams.gpu[gpu_id].host_unified_memory:
      return nlpar_cl.NLPAR.calcnlpar_cl(self, saturation_protect=saturation_protect,
                                         automask=automask,
                                         filename=filename,
                                         fileout=fileout,
                                         reset_sigma=reset_sigma,
                                         backsub = backsub,
                                         rescale = rescale,
                                         gpu_id= gpu_id)

    target_mem = clparams.gpu[gpu_id].max_mem_alloc_size//6
    max_mem = clparams.gpu[gpu_id].global_mem_size*0.4
    if target_mem*ngpuwrker > max_mem:
      target_mem = max_mem/ngpuwrker
    #print(target_mem/1.0e9)

    chunks = self._calcchunks([pwidth, pheight], ncols, nrows, target_bytes=target_mem,
                              col_overlap=sr, row_overlap=sr)

    nnn = int((2 * sr + 1) ** 2)
    jobqueue = []


    for rowchunk in range(chunks[1]):
        rstart = chunks[3][rowchunk, 0]
        rend = chunks[3][rowchunk, 1]
        nrowchunk = rend - rstart

        rstartcalc = sr if (rowchunk > 0) else 0
        rendcalc = nrowchunk - sr if (rowchunk < (chunks[1] - 1)) else nrowchunk
        nrowcalc = np.int64(rendcalc - rstartcalc)

        for colchunk in range(chunks[0]):
            cstart = chunks[2][colchunk, 0]
            cend = chunks[2][colchunk, 1]
            ncolchunk = cend - cstart

            cstartcalc = sr if (colchunk > 0) else 0
            cendcalc = ncolchunk - sr if (colchunk < (chunks[0] - 1)) else ncolchunk
            ncolcalc = np.int64(cendcalc - cstartcalc)

            jobqueue.append( NLPARGPUJob([colchunk, rowchunk],\
                    [cstart,cend, rstart, rend],\
                  [cstartcalc,cendcalc, rstartcalc, rendcalc ]))


    if verbose >=1:
      print("lambda:", lam, "search radius:", sr, "dthresh:", dthresh)
    ngpu_per_wrker = float(1.0/ngpuwrker)
    ray.shutdown()

    rayclust = ray.init(
      num_cpus=int(ngpuwrker),
      num_gpus=1,
      # dashboard_host = RAYIPADDRESS,
      _node_ip_address=RAYIPADDRESS,  # "0.0.0.0",
      runtime_env={"env_vars":
                     {"PYTHONPATH": os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                      }},
      logging_level=logging.WARNING,)  # Supress INFO messages from ray.

    nlpar_remote = ray.put(self)

    idlewrker = []
    busywrker = []
    tasks = []

    for w in range(ngpuwrker):
        idlewrker.append(NLPARGPUWorker.options(num_cpus=float(0.99), num_gpus=ngpu_per_wrker).remote(
                actorid=w, gpu_id=gpu_id, cudavis=cudavis))

    njobs = len(jobqueue)
    ndone = 0
    while ndone < njobs:
        if len(jobqueue) > 0:
            if len(idlewrker) > 0:
                wrker = idlewrker.pop()
                job = jobqueue.pop(0)

                tasks.append(wrker.runnlpar_chunk.remote(job, nlparobj=nlpar_remote))
                busywrker.append(wrker)
        if len(tasks) > 0:
            donetasks, stillbusy = ray.wait(tasks, num_returns=len(busywrker), timeout=0.1)

            for tsk in donetasks:
                indx = tasks.index(tsk)
                message, job, newdata = ray.get(tsk)
                if message == 'Done':
                  idlewrker.append(busywrker.pop(indx))
                  tasks.remove(tsk)
                  ndone += 1
                  if verbose >= 2:
                    print("tiles complete: ", ndone, "/", njobs, sep='', end='\r')
                else: #An error has occured ... hopefully just need a re-process.
                  jobqueue.append(job)
                  print(message)

    if verbose >= 2:
      print('\n', end='')
    return str(self.patternfileout.filepath)

  def _nlparchunkcalc_cl(self, data, calclim, clparams=None):
    data = np.ascontiguousarray(data)
    ctx = clparams.ctx
    prg = clparams.prg
    clparams.get_queue()


    mf = clparams.memflags
    clvectlen = 16

    cstart = calclim.cstart
    cend = calclim.cend
    ncolchunk = calclim.ncolchunk
    rstart = calclim.rstart
    rend = calclim.rend
    nrowchunk = calclim.nrowchunk

    rstartcalc = calclim.rstartcalc
    cstartcalc = calclim.cstartcalc
    nrowcalc = calclim.nrowcalc
    ncolcalc = calclim.ncolcalc

    pheight = data.shape[1]
    pwidth = data.shape[2]



    lam = np.float32(self.lam)
    sr = np.int64(self.searchradius)
    nnn = int((2 * sr + 1) ** 2)
    dthresh = np.float32(self.dthresh)
    #print(chunks[2], chunks[3])
    #print(lam, sr, dthresh)


    # precalculate some needed arrays for the GPU
    mxval = data.max()
    if self.saturation_protect == False:
      mxval += 1.0
    else:
      mxval *= 0.9961

    shpdata = data.shape
    npat_point = int(shpdata[-1]*shpdata[-2])
    npat = shpdata[0]

    npadmx = clvectlen * int(np.ceil(float(npat)*npat_point/ clvectlen))
    data_gpu = cl.Buffer(ctx, mf.READ_ONLY| mf.COPY_HOST_PTR, hostbuf=data)
    datapad_gpu = cl.Buffer(ctx, mf.READ_WRITE, size=int(npadmx) * int(4))
    datapadout_gpu = cl.Buffer(ctx, mf.READ_WRITE, size=int(npadmx) * int(4))

    fill1 = cl.enqueue_fill_buffer(clparams.queue, datapad_gpu, np.float32(mxval + 10), 0, int(4 * npadmx))
    fill2 = cl.enqueue_fill_buffer(clparams.queue, datapadout_gpu, np.float32(0.0), 0, int(4 * npadmx))


    mask = self.mask.astype(np.float32)

    npad = clvectlen * int(np.ceil(mask.size / clvectlen))
    maskpad = np.zeros((npad), np.float32) - 1  # negative numbers will indicate a clvector overflow.
    maskpad[0:mask.size] = mask.reshape(-1).astype(np.float32)
    mask_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=maskpad)
    npatsteps = int(maskpad.size / clvectlen)

    sigmachunk = np.ascontiguousarray(self.sigma[rstart:rend, cstart:cend].astype(np.float32))
    sigmachunk_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=sigmachunk)

    szdata = data.size
    cl.enqueue_barrier(clparams.queue)
    data_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
    if data.dtype.type is np.float32:
      prg.nlloadpat32flt(clparams.queue, (np.uint64(data.size), 1), None, data_gpu, datapad_gpu)#, wait_for=[fill1,fill2])
    if data.dtype.type is np.ubyte:
      prg.nlloadpat8bit(clparams.queue, (np.uint64(data.size), 1), None, data_gpu, datapad_gpu)#, wait_for=[fill1,fill2])
    if data.dtype.type is np.uint16:
      prg.nlloadpat16bit(clparams.queue, (np.uint64(data.size), 1), None, data_gpu, datapad_gpu)#, wait_for=[fill1,fill2])

    calclim = np.array([cstartcalc, rstartcalc, ncolchunk, nrowchunk], dtype=np.int64)
    crlimits_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=calclim)
    cl.enqueue_barrier(clparams.queue)
    data_gpu.release()
    prg.calcnlpar(clparams.queue, (np.uint32(ncolcalc), np.uint32(nrowcalc)), None,
                  # prg.calcnlpar(clparams.queue, (1, 1), None,
                  datapad_gpu,
                  mask_gpu,
                  sigmachunk_gpu,
                  crlimits_gpu,
                  datapadout_gpu,
                  np.int64(sr),
                  np.int64(npatsteps),
                  np.int64(npat_point),
                  np.float32(mxval),
                  np.float32(1.0 / lam ** 2),
                  np.float32(dthresh))

    data = data.astype(np.float32)  # prepare to receive data back from GPU
    data.reshape(-1)[:] = 0.0
    data = data.reshape(nrowchunk, ncolchunk, pheight, pwidth)
    cl.enqueue_copy(clparams.queue, data, datapadout_gpu,  is_blocking=True)
    sigmachunk_gpu.release()
    clparams.queue.finish()
    if self.rescale == True:
      for i in range(data.shape[0]):
        temp = data[i, :, :]
        temp -= temp.min()
        temp *= np.float32(mxval) / temp.max()
        data[i, :, :] = temp
    data = data[rstartcalc: rstartcalc + nrowcalc, cstartcalc: cstartcalc + ncolcalc, :, :]
    data = data.reshape(nrowcalc * ncolcalc, pheight, pwidth)

    clparams.queue = None

    return data



@ray.remote
class NLPARGPUWorker:
  def __init__(self, actorid=0, gpu_id=None, cudavis = '0'):
      # sys.path.append(path.dirname(path.dirname(__file__)))  # do this to help Ray find the program files
      # import openclparam # do this to help Ray find the program files
      # device, context, queue, program, mf
      # self.dataout = None
      # self.indxstart = None
      # self.indxend = None
      # self.rate = None
      os.environ["CUDA_VISIBLE_DEVICES"] = cudavis
      self.actorID = actorid
      self.openCLParams = openclparam.OpenClParam()

      self.openCLParams.gpu_id = gpu_id
      self.openCLParams.get_context(gpu_id=gpu_id, kfile = 'clnlpar.cl')


          #elf.openCLParams = None

  def runsigma_chunk(self,gpujob, nlparobj=None, **kwargs):
    if gpujob is None:
        #time.sleep(0.001)
        return 'Bored', (None, None, None)
    try:
        # print(type(self.openCLParams.ctx))
        gpujob._starttime()
        #time.sleep(random.uniform(0, 1.0))

        #if self.openCLParams is not None:
        #    self.openCLParams.get_queue()
        data, xyloc = nlparobj.patternfile.read_data(patStartCount=[[gpujob.cstart, gpujob.rstart],
                                                                    [gpujob.ncolchunk, gpujob.nrowchunk]],
                                                                    convertToFloat=False, returnArrayOnly=True)

        newdata = nlparobj._sigmachunkcalc_cl(data, gpujob, clparams=self.openCLParams, **kwargs)

        if self.openCLParams.queue is not None:
            print("queue still here")
            self.openCLParams.queue.finish()
            self.openCLParams.queue = None

        gpujob._endtime()
        return 'Done', gpujob, newdata
    except Exception as e:
        print(e)
        gpujob.rate = None
        return "Error", gpujob, e


  def runnlpar_chunk(self, gpujob, nlparobj=None):

    if gpujob is None:
        #time.sleep(0.001)
        return 'Bored', (None, None, None)
    try:
        # print(type(self.openCLParams.ctx))
        gpujob._starttime()
        #time.sleep(random.uniform(0, 1.0))

        #if self.openCLParams is not None:
        #    self.openCLParams.get_queue()
        data, xyloc = nlparobj.patternfile.read_data(patStartCount=[[gpujob.cstart, gpujob.rstart],
                                                                    [gpujob.ncolchunk, gpujob.nrowchunk]],
                                                                    convertToFloat=False, returnArrayOnly=True)

        newdata = nlparobj._nlparchunkcalc_cl(data, gpujob, clparams=self.openCLParams)

        if self.openCLParams.queue is not None:
            print("queue still here")
            self.openCLParams.queue.finish()
            self.openCLParams.queue = None

        nlparobj.patternfileout.write_data(newpatterns=newdata,
                                       patStartCount=[[gpujob.cstart + gpujob.cstartcalc,
                                                       gpujob.rstart + gpujob.rstartcalc],
                                                      [gpujob.ncolcalc, gpujob.nrowcalc]],
                                       flt2int='clip', scalevalue=1.0)

        gpujob._endtime()
        return 'Done', gpujob, None
    except Exception as e:
        print(e)
        gpujob.rate = None
        return "Error", gpujob, e


class NLPARGPUJob:
  def __init__(self, jobid, chunk, calclim):
    self.jobid = jobid
    if self.jobid is not None:
      self.cstart = chunk[0]
      self.cend = chunk[1]
      self.ncolchunk = self.cend -  self.cstart
      self.rstart = chunk[2]
      self.rend = chunk[3]
      self.nrowchunk = self.rend - self.rstart
      self.cstartcalc = calclim[0]
      self.cendcalc = calclim[1]
      self.ncolcalc = self.cendcalc - self.cstartcalc
      self.rstartcalc = calclim[2]
      self.rendcalc = calclim[3]
      self.nrowcalc = self.rendcalc - self.rstartcalc
      self.starttime = 0.0
      self.endtime = 0.0
      self.extime = 0.0

  def _starttime(self):
    self.starttime = timer()

  def _endtime(self):
    self.endtime = timer()
    self.extime += self.endtime - self.starttime
    #self.rate = self.npat / (self.extime + 1e-12)



