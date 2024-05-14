import os
from timeit import default_timer as timer
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

import numba
import scipy.optimize as sp_opt
from pyebsdindex import nlpar
from pyebsdindex.opencl import openclparam
from time import time as timer
import scipy
class NLPAR(nlpar.NLPAR):
  def __init__( self, filename=None, **kwargs):
    nlpar.NLPAR.__init__(self, filename=filename, **kwargs)
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
    sig =  self.calcsigma_cl(nn=nn,
                            saturation_protect=saturation_protect,
                            automask=automask, **kwargs)
    if return_nndist == True:
      return sig
    else:
      return sig[0]
  def opt_lambda_cpu(self, **kwargs):
    return nlpar.NLPAR.opt_lambda(self, **kwargs)

  def calcnlpar_cpu(self, **kwargs):
    return nlpar.NLPAR.calcnlpar(self, **kwargs)

  def calcsigma_cpu(self,nn=1, saturation_protect=True,automask=True, **kwargs):
    return nlpar.NLPAR.calcsigma(self, nn=nn,
                            saturation_protect=saturation_protect,automask=automask, **kwargs)

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
      lam = 1.0
      lambopt1 = sp_opt.minimize(loptfunc, lam, args=(d2, tw, dthresh), method='Nelder-Mead',
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


  def calcsigma_cl(self,nn=1,saturation_protect=True,automask=True, normalize_d=False, gpuid = None, **kwargs):

    if gpuid is None:
      clparams = openclparam.OpenClParam()
      clparams.get_gpu()
      target_mem = 0
      gpuid = 0
      count = 0

      for gpu in clparams.gpu:
        gmem = gpu.max_mem_alloc_size
        if target_mem < gmem:
          gpuid = count
          target_mem = gmem
        count += 1
    else:
      clparams = openclparam.OpenClParam()
      clparams.get_gpu()
      gpuid = min(len(clparams.gpu)-1, gpuid)


    #print(gpuid)
    clparams.get_context(gpu_id=gpuid, kfile = 'clnlpar.cl')
    clparams.get_queue()
    target_mem = clparams.queue.device.max_mem_alloc_size//2
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

    dist_local.release()
    count_local.release()
    datapad_gpu.release()
    queue.flush()
    queue = None
    self.sigma = sigma
    return sigma, dist, countnn

  def calcnlpar_cl(self,chunksize=0, searchradius=None, lam = None, dthresh = None, saturation_protect=True, automask=True,
                filename=None, fileout=None, reset_sigma=False, backsub = False, rescale = False, gpuid = None, **kwargs):

    if lam is not None:
      self.lam = lam

    if dthresh is not None:
      self.dthresh = dthresh
    if self.dthresh is None:
      self.dthresh = 0.0

    if searchradius is not None:
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
      self.sigma = self.calcsigma_cl(nn=1, saturation_protect=saturation_protect, automask=automask, gpuid=gpuid)[0]

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



    if gpuid is None:
      clparams = openclparam.OpenClParam()
      clparams.get_gpu()
      target_mem = 0
      gpuid = 0
      count = 0

      for gpu in clparams.gpu:
        gmem = gpu.max_mem_alloc_size
        if target_mem < gmem:
          gpuid = count
          target_mem = gmem
        count += 1
    else:
      clparams = openclparam.OpenClParam()
      clparams.get_gpu()
      gpuid = min(len(clparams.gpu)-1, gpuid)


    #print(gpuid)
    clparams.get_context(gpu_id=gpuid, kfile = 'clnlpar.cl')
    clparams.get_queue()
    target_mem = clparams.queue.device.max_mem_alloc_size//2
    ctx = clparams.ctx
    prg = clparams.prg
    queue = clparams.queue
    mf = clparams.memflags
    clvectlen = 16



    chunks = self._calcchunks( [pwidth, pheight], ncols, nrows, target_bytes=target_mem,
                              col_overlap=sr, row_overlap=sr)
    #print(chunks[2], chunks[3])
    print(lam, sr, dthresh)

    # precalculate some needed arrays for the GPU
    mask = self.mask.astype(np.float32)

    npad = clvectlen * int(np.ceil(mask.size/clvectlen))
    maskpad = np.zeros((npad) , np.float32) -1 # negative numbers will indicate a clvector overflow.
    maskpad[0:mask.size] = mask.reshape(-1).astype(np.float32)
    mask_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=maskpad)

    npatsteps = int(maskpad.size/clvectlen) # how many clvector chunks to move through a pattern.

    chunksize = (chunks[2][:,1] - chunks[2][:,0]).reshape(1,-1) * \
                     (chunks[3][:, 1] - chunks[3][:, 0]).reshape(-1, 1)

    #return chunks, chunksize
    mxchunk = int(chunksize.max())
    npadmx = clvectlen * int(np.ceil(float(mxchunk)*npat_point/ clvectlen))

    datapad_gpu = cl.Buffer(ctx, mf.READ_WRITE, size=int(npadmx) * int(4))
    datapadout_gpu = cl.Buffer(ctx, mf.READ_WRITE, size=int(npadmx) * int(4))

    nnn = int((2 * sr + 1) ** 2)



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

        data, xyloc = patternfile.read_data(patStartCount=[[cstart, rstart], [ncolchunk, nrowchunk]],
                                          convertToFloat=False, returnArrayOnly=True)

        mxval = data.max()
        if saturation_protect == False:
          mxval += 1.0
        else:
          mxval *= 0.9961

        filldatain = cl.enqueue_fill_buffer(queue, datapad_gpu, np.float32(mxval+10), 0,int(4*npadmx))
        cl.enqueue_fill_buffer(queue, datapadout_gpu, np.float32(0.0), 0, int(4 * npadmx))

        sigmachunk = np.ascontiguousarray(sigma[rstart:rend, cstart:cend].astype(np.float32))
        sigmachunk_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=sigmachunk)
        szdata = data.size
        npad = clvectlen * int(np.ceil(szdata / clvectlen))

        #datapad = np.zeros((npad), dtype=np.float32) + np.float32(mxval + 10)
        #datapad[0:szdata] = data.reshape(-1)

        data_gpu = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=data)

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
        prg.calcnlpar(queue, (np.uint32(ncolcalc), np.uint32(nrowcalc)), None,
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
        queue.finish()
        if rescale == True:
          for i in range(data.shape[0]):
            temp = data[i, :, :]
            temp -= temp.min()
            temp *= np.float32(mxval) / temp.max()
            data[i, :, :] = temp
        data = data[rstartcalc: rstartcalc+nrowcalc,cstartcalc: cstartcalc+ncolcalc, :,: ]
        data = data.reshape(nrowcalc*ncolcalc, pheight, pwidth)
        patternfileout.write_data(newpatterns=data, patStartCount=[[cstart+cstartcalc, rstart+rstartcalc],
                                                                   [ncolcalc, nrowcalc]],
                                  flt2int='clip', scalevalue=1.0)
    queue.finish()
    queue = None
    return str(patternfileout.filepath)






