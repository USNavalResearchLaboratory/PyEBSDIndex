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



import os
from timeit import default_timer as timer
import numpy as np
import pyopencl as cl



import scipy.optimize as sp_opt
from pyebsdindex import nlpar_cpu
from pyebsdindex.opencl import openclparam
from time import time as timer

class NLPAR(nlpar_cpu.NLPAR):
  def __init__( self, filename=None, **kwargs):
    nlpar_cpu.NLPAR.__init__(self, filename=filename, **kwargs)
    self.useCPU = False


  def opt_lambda(self,saturation_protect=True, automask=True, backsub=False,
                 target_weights=[0.5, 0.34, 0.25], dthresh=0.0, autoupdate=True, **kwargs):
    return self.opt_lambda_cl(saturation_protect=saturation_protect,
                              automask=automask,
                              backsub=backsub,
                              target_weights=target_weights,
                              dthresh=dthresh,
                              autoupdate=autoupdate, **kwargs)
  def calcnlpar(self, **kwargs):
    return self.calcnlpar_cl(**kwargs)


  def calcsigma(self,nn=1, saturation_protect=True,automask=True, return_nndist=False, **kwargs):
    self.sigmann = nn
    if self.sigmann > 7:
      print("Sigma optimization search limited to a search radius <= 7")
      print("The search radius has been clipped to 7")
      nn = 7
      self.sigmann = nn

    sig =  self.calcsigma_cl(nn=nn,
                            saturation_protect=saturation_protect,
                            automask=automask, **kwargs)
    if return_nndist == True:
      return sig
    else:
      return sig[0]
  def opt_lambda_cpu(self, **kwargs):
    return nlpar_cpu.NLPAR.opt_lambda(self, **kwargs)

  def calcnlpar_cpu(self, **kwargs):
    return nlpar_cpu.NLPAR.calcnlpar(self, **kwargs)

  def calcsigma_cpu(self,nn=1, saturation_protect=True,automask=True, **kwargs):
    return nlpar_cpu.NLPAR.calcsigma(self, nn=nn,
                                     saturation_protect=saturation_protect, automask=automask, **kwargs)

  def opt_lambda_cl(self, saturation_protect=True, automask=True, backsub=False,
                 target_weights=[0.5, 0.34, 0.25], dthresh=0.0, autoupdate=True, **kwargs):

    target_weights = np.asarray(target_weights)

    def loptfunc(lam, d2, tw, dthresh):
      temp = np.maximum(d2, dthresh)#(d2 > dthresh).choose(dthresh, d2)
      dw = np.exp(-(temp) / lam ** 2)
      w = np.sum(dw, axis=2) + 1e-12

      metric = np.mean(np.abs(tw - 1.0 / w))
      return metric

    patternfile = self.getinfileobj()
    patternfile.read_header()
    nrows = np.uint64(self.nrows)  # np.uint64(patternfile.nRows)
    ncols = np.uint64(self.ncols)  # np.uint64(patternfile.nCols)

    pwidth = np.uint64(patternfile.patternW)
    pheight = np.uint64(patternfile.patternH)
    phw = pheight * pwidth

    dthresh = np.float32(dthresh)
    lamopt_values = []

    sigma, d2, n2 = self.calcsigma(nn=1, saturation_protect=saturation_protect, automask=automask, normalize_d=True,
                                   return_nndist=True, **kwargs)

    #sigmapad = np.pad(sigma, 1, mode='reflect')
    #d2normcl(d2, n2, sigmapad)

   #print(d2.min(), d2.max(), d2.mean())

    lamopt_values_chnk = []
    for tw in target_weights:
      stride = 1 if sigma.size < 1e6 else 10

      lam = 1.0
      lambopt1 = sp_opt.minimize(loptfunc, lam, args=(d2[0::stride, :], tw, dthresh), method='Nelder-Mead',
                              bounds=[[0.001, 10.0]], options={'fatol': 0.0001})
      lamopt_values.append(lambopt1['x'])

    #lamopt_values.append(lamopt_values_chnk)
    lamopt_values = np.asarray(lamopt_values)
    print("Range of lambda values: ", lamopt_values.flatten())
    print("Optimal Choice: ", np.median(lamopt_values))
    if autoupdate == True:
      self.lam = np.median(np.mean(lamopt_values, axis=0))
    if self.sigma is None:
      self.sigma = sigma
    return lamopt_values.flatten()


  def calcsigma_cl(self,nn=1,saturation_protect=True,automask=True, normalize_d=False, gpu_id = None, verbose = 2, **kwargs):
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


    #print(gpu_id)
    clparams.get_context(gpu_id=gpu_id, kfile = 'clnlpar.cl')
    clparams.get_queue()
    target_mem = min(clparams.queue.device.max_mem_alloc_size//2, np.int64(4e9))
    ctx = clparams.ctx
    prg = clparams.prg
    queue = clparams.queue
    mf = clparams.memflags
    clvectlen = 16

    patternfile = self.getinfileobj()

    nrows = np.int64(self.nrows)  # np.uint64(patternfile.nRows)
    ncols = np.int64(self.ncols)  # np.uint64(patternfile.nCols)

    pwidth = np.uint64(patternfile.patternW)
    pheight = np.uint64(patternfile.patternH)
    npat_point = int(pwidth * pheight)
    #print(target_mem)
    chunks = self._calcchunks( [pwidth, pheight], ncols, nrows, target_bytes=target_mem,
                              col_overlap=1, row_overlap=1)

    if (automask is True) and (self.mask is None):
      self.mask = (self.automask(pheight, pwidth))
    if self.mask is None:
      self.mask = np.ones((pheight, pwidth), dtype=np.uint8)

    #indices = np.asarray((self.mask.flatten().nonzero())[0], np.uint64)
    #nindices = np.uint64(indices.size)
    #nindicespad =  np.uint64(clvectlen * int(np.ceil(nindices/clvectlen)))

    mask = self.mask.astype(np.float32)

    npad = clvectlen * int(np.ceil(mask.size/clvectlen))
    maskpad = np.zeros((npad) , np.float32)
    maskpad[0:mask.size] = mask.reshape(-1).astype(np.float32)
    mask_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=maskpad)

    npatsteps = int(maskpad.size/clvectlen)

    chunksize = (chunks[2][:,1] - chunks[2][:,0]).reshape(1,-1) * \
                     (chunks[3][:, 1] - chunks[3][:, 0]).reshape(-1, 1)

    #return chunks, chunksize
    mxchunk = chunksize.max()

    npadmx = clvectlen * int(np.ceil(mxchunk*npat_point/ clvectlen))

    datapad_gpu = cl.Buffer(ctx, mf.READ_WRITE, size=int(npadmx) * int(4))

    nnn = int((2 * nn + 1) ** 2)

    sigma = np.zeros((nrows, ncols), dtype=np.float32) + 1e18
    dist = np.zeros((nrows, ncols, nnn), dtype=np.float32)
    countnn = np.zeros((nrows, ncols, nnn), dtype=np.float32)

    #dist_local = cl.LocalMemory(nnn*npadmx*4)
    dist_local = cl.Buffer(ctx, mf.READ_WRITE, size=int(mxchunk*nnn*4))
    distchunk = np.zeros((mxchunk, nnn), dtype=np.float32)
    #count_local = cl.LocalMemory(nnn*npadmx*4)
    count_local = cl.Buffer(ctx, mf.READ_WRITE, size=int(mxchunk * nnn * 4))
    countchunk = np.zeros((mxchunk, nnn), dtype=np.float32)
    ndone = 0
    nchunks = int(chunks[1] * chunks[0])
    for rowchunk in range(chunks[1]):
      rstart = chunks[3][rowchunk, 0]
      rend = chunks[3][rowchunk, 1]
      nrowchunk = rend - rstart

      for colchunk in range(chunks[0]):
        cstart = chunks[2][colchunk, 0]
        cend = chunks[2][colchunk, 1]
        ncolchunk = cend - cstart

        data, xyloc = patternfile.read_data(patStartCount=[[cstart, rstart], [ncolchunk, nrowchunk]],
                                          convertToFloat=False, returnArrayOnly=True)

        mxval = data.max()
        if saturation_protect == False:
          mxval += 1.0
        else:
          mxval *= 0.9961

        evnt = cl.enqueue_fill_buffer(queue, datapad_gpu, np.float32(mxval+10), 0,int(4*npadmx))

        szdata = data.size
        npad = clvectlen * int(np.ceil(szdata / clvectlen))
        tic = timer()
        #datapad = np.zeros((npad), dtype=np.float32) + np.float32(mxval + 10)
        #datapad[0:szdata] = data.reshape(-1)
        data_gpu = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=data)

        if data.dtype.type is np.float32:
          prg.nlloadpat32flt(queue, (np.uint64(data.size),), None, data_gpu, datapad_gpu, wait_for=[evnt])
        if data.dtype.type is np.ubyte:
          prg.nlloadpat8bit(queue, (np.uint64(data.size),), None, data_gpu, datapad_gpu, wait_for=[evnt])
        if data.dtype.type is np.uint16:
          prg.nlloadpat16bit(queue, (np.uint64(data.size),), None, data_gpu, datapad_gpu, wait_for=[evnt])
        toc = timer()
        #print(toc - tic)

        sigmachunk = np.zeros((nrowchunk, ncolchunk ), dtype=np.float32)


        sigmachunk_gpu =  cl.Buffer(ctx, mf.WRITE_ONLY, size=sigmachunk.nbytes)

        cl.enqueue_barrier(queue)
        prg.calcsigma(queue, (np.uint32(ncolchunk), np.uint32(nrowchunk)), None,
                               datapad_gpu, mask_gpu,sigmachunk_gpu,
                               dist_local, count_local,
                               np.int64(nn), np.int64(npatsteps), np.int64(npat_point),
                               np.float32(mxval) )
        if normalize_d is True:
          cl.enqueue_barrier(queue)
          prg.normd(queue, (np.uint32(ncolchunk), np.uint32(nrowchunk)), None,
                          sigmachunk_gpu,
                          count_local, dist_local,
                          np.int64(nn))
        queue.finish()

        cl.enqueue_copy(queue, distchunk, dist_local, is_blocking=False)
        cl.enqueue_copy(queue, countchunk, count_local,  is_blocking=False)
        cl.enqueue_copy(queue, sigmachunk, sigmachunk_gpu,  is_blocking=True)

        #sigmachunk_gpu.release()

        queue.finish()
        countnn[rstart:rend, cstart:cend] = countchunk[0:int(ncolchunk*nrowchunk), :].reshape(nrowchunk, ncolchunk, nnn)
        dist[rstart:rend, cstart:cend] = distchunk[0:int(ncolchunk*nrowchunk), :].reshape(nrowchunk, ncolchunk, nnn)
        sigma[rstart:rend, cstart:cend] = np.minimum(sigma[rstart:rend, cstart:cend], sigmachunk)
        if verbose >= 2:
          print("tiles complete: ", ndone, "/", nchunks, sep='', end='\r')
        ndone +=1
    dist_local.release()
    count_local.release()
    datapad_gpu.release()
    queue.flush()
    queue = None
    self.sigma = sigma
    return sigma, dist, countnn



  def calcnlpar_cl(self, searchradius=None, lam = None, dthresh = None, saturation_protect=True, automask=True,
                   filename=None, fileout=None, reset_sigma=False, backsub = False, rescale = False,
                   gpu_id = None, verbose=2, **kwargs):

    class OpenCLClalcError(Exception):
      pass

    if lam is not None:
      self.lam = lam

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

    patternfile = self.getinfileobj()
    self.setoutfile(patternfile, filepath=fileout)
    patternfileout = self.getoutfileobj()

    nrows = np.int64(self.nrows)  # np.uint64(patternfile.nRows)
    ncols = np.int64(self.ncols)  # np.uint64(patternfile.nCols)

    pwidth = np.uint64(patternfile.patternW)
    pheight = np.uint64(patternfile.patternH)
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
      if np.issubdtype(patternfileout.filedatatype, np.integer):
        mxval = np.iinfo(patternfileout.filedatatype).max
        scalemethod = 'fullscale'
      else:  # not int, so no rescale.
        self.rescale = False
        rescale = False



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
      gpu_id = min(len(clparams.gpu) - 1, gpu_id)


    #print(gpu_id)
    clparams.get_context(gpu_id=gpu_id, kfile ='clnlpar.cl')
    clparams.get_queue()
    target_mem = min(clparams.queue.device.max_mem_alloc_size//4, np.int64(2e9))
    #target_mem = min(clparams.queue.device.max_mem_alloc_size*3, np.int64(18e9))
    ctx = clparams.ctx
    prg = clparams.prg
    queue = clparams.queue
    mf = clparams.memflags
    clvectlen = 16


    print("target mem:", target_mem)
    chunks = self._calcchunks( [pwidth, pheight], ncols, nrows, target_bytes=target_mem,
                              col_overlap=sr, row_overlap=sr)
    #print(chunks[2], chunks[3])
    if verbose >=1:
      print("lambda:", lam, "search radius:", sr, "dthresh:", dthresh)

    # precalculate some needed arrays for the GPU
    mask = self.mask.astype(np.float32)

    npad = clvectlen * np.int64(np.ceil(mask.size/clvectlen))
    maskpad = np.zeros((npad) , np.float32) -1 # negative numbers will indicate a clvector overflow.
    maskpad[0:mask.size] = mask.reshape(-1).astype(np.float32)
    mask_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=maskpad)

    npatsteps = np.int64(maskpad.size/clvectlen) # how many clvector chunks to move through a pattern.

    chunksize = (chunks[2][:,1] - chunks[2][:,0]).reshape(1,-1) * \
                     (chunks[3][:, 1] - chunks[3][:, 0]).reshape(-1, 1)
    nchunks = chunksize.size
    #return chunks, chunksize
    mxchunk = np.int64(chunksize.max())
    # print("max chunk:" , mxchunk)

    npadmx = clvectlen * np.int64(np.ceil(float(mxchunk)*npat_point/ clvectlen))

    datapad_gpu = cl.Buffer(ctx, mf.READ_WRITE, size=np.int64(npadmx) * np.int64(4))
    datapadout_gpu = cl.Buffer(ctx, mf.READ_WRITE, size=np.int64(npadmx) * np.int64(4))
    # print("data pad", datapad_gpu.size)
    # print("data out", datapadout_gpu.size)

    nnn = int((2 * sr + 1) ** 2)


    ndone = 0
    jqueue = []
    # if verbose >= 2:
    #   print('\n', end='')
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

        job = {"rstart": rstart,
               "rend": rend,
               "nrowchunk": nrowchunk,
               "rstartcalc": rstartcalc,
               "rendcalc": rendcalc,
               "nrowcalc": nrowcalc,
               "cstart": cstart,
               "cend": cend,
               "ncolchunk": ncolchunk,
               "cstartcalc": cstartcalc,
               "cendcalc": cendcalc,
               "ncolcalc": ncolcalc,
               "nattempts": -1}
        jqueue.append(job)


    while len(jqueue) > 0:
        j = jqueue.pop(0)
        j["nattempts"] += 1

        rstart = j["rstart"]
        cstart = j["cstart"]
        rend = j["rend"]
        cend = j["cend"]
        cstartcalc = j["cstartcalc"]
        rstartcalc = j["rstartcalc"]
        ncolchunk = j["ncolchunk"]
        nrowchunk = j["nrowchunk"]
        ncolcalc = j["ncolcalc"]
        nrowcalc = j["nrowcalc"]

        data, xyloc = patternfile.read_data(patStartCount=[[ cstart, rstart], [ncolchunk, nrowchunk]],
                                          convertToFloat=False, returnArrayOnly=True)


        mxval0 = data.max()
        mnval0 = data.min()
        mxval = mxval0
        if mnval0 < 0:
          data -= mnval0
          mxval = mxval - mnval0
       

        if saturation_protect == False:
          mxval += 1.0
        else:
          mxval *= 0.9961

        filldatain = cl.enqueue_fill_buffer(queue, datapad_gpu, np.float32(mxval+10), 0,np.int64(4*npadmx))
        cl.enqueue_fill_buffer(queue, datapadout_gpu, np.float32(0.0), 0, np.int64(4 * npadmx))

        sigmachunk = np.ascontiguousarray(sigma[rstart:rend, cstart:cend].astype(np.float32))
        sigmachunk_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=sigmachunk)
        # print("sigma", sigmachunk_gpu.size)
        szdata = data.size
        npad = clvectlen * np.int64(np.ceil(szdata / clvectlen))

        #datapad = np.zeros((npad), dtype=np.float32) + np.float32(mxval + 10)
        #datapad[0:szdata] = data.reshape(-1)

        data_gpu = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=data)
        # print("data", data_gpu.size)
        if data.dtype.type is np.float32:
          prg.nlloadpat32flt(queue, (np.uint64(data.size),1), None, data_gpu, datapad_gpu, wait_for=[filldatain])
        if data.dtype.type is np.ubyte:
          prg.nlloadpat8bit(queue, (np.uint64(data.size),1), None, data_gpu, datapad_gpu, wait_for=[filldatain])
        if data.dtype.type is np.uint16:
          prg.nlloadpat16bit(queue, (np.uint64(data.size),1), None, data_gpu, datapad_gpu, wait_for=[filldatain])



        calclim = np.array([cstartcalc, rstartcalc, ncolchunk, nrowchunk], dtype=np.int64)
        crlimits_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=calclim)
        cl.enqueue_barrier(queue)
        data_gpu.release()
        try:
          envt = prg.calcnlpar(queue, (np.uint32(ncolcalc), np.uint32(nrowcalc)), None,
          #prg.calcnlpar(queue, (1, 1), None,
                                 datapad_gpu,
                                 mask_gpu,
                                 sigmachunk_gpu,
                                 crlimits_gpu,
                                 datapadout_gpu,
                                 np.int64(sr),
                                 np.int64(npatsteps),
                                 np.int64(npat_point),
                                 np.float32(mxval),
                                 np.float32(1.0/lam**2),
                                 np.float32(dthresh) )

          data = data.astype(np.float32) # prepare to receive data back from GPU
          data.reshape(-1)[:] = 0.0
          data = data.reshape(nrowchunk, ncolchunk, pheight, pwidth)
          sigmachunk_gpu.release()
          cl.enqueue_copy(queue, data, datapadout_gpu,  is_blocking=True)
          #print(envt.command_execution_status)
          queue.finish()
          data = data[rstartcalc: rstartcalc + nrowcalc,
                 cstartcalc:cstartcalc + ncolcalc, :, :]
          mxout = data.max(axis=(-1,-2))
          mxtest = (np.float32(mxout < 1.e-8)).mean()
          # this check is because there is a rare, silent failure on apple-si chips, which
          # will just return zeros to the data array.  Not perfect, but this seems better than
          # nothing.  It will attempt to reprocess the data 3 times before just writing out
          # whatever it has.
          if (mxval0 < np.float32(1.e-8)) or ( mxtest < 0.1 ) or (j["nattempts"] >= 3):
            if mnval0 < 0:
              data += mnval0

            data = data.reshape(nrowcalc * ncolcalc, pheight, pwidth)
            if rescale == True:
              for i in range(data.shape[0]):
                temp = data[i, :, :]
                temp -= temp.min()
                temp *= np.float32(mxval) / temp.max()
                data[i, :, :] = temp


            patternfileout.write_data(newpatterns=data,
                                      patStartCount=[[np.int64(cstart + cstartcalc), np.int64(rstart + rstartcalc)],
                                                     [ncolcalc, nrowcalc]],
                                      flt2int='clip', scalevalue=1.0)
            ndone += 1
            if verbose >= 2:
              print("tiles complete: ", ndone, "/", nchunks, sep='', end='\r')




          else:
            if mxtest >= 0.1:
              raise OpenCLClalcError()

        except OpenCLClalcError:
          if j["nattempts"] < 3:
            print("Reattempting job: ", j['nattempts'])
            jqueue.append(j)
          else:
            print("Aborting job.")





    if verbose >= 2:
      print('', end='')
    queue.finish()
    queue = None
    return str(patternfileout.filepath)






