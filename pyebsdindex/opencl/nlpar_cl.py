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
  def __init__( self, **kwargs):
    nlpar.NLPAR.__init__(self, **kwargs)
    self.useCPU = False

  def calcsigma(self,nn=1, saturation_protect=True,automask=True, **kwargs):
    return self.calcsigma_cl(nn=nn,
                            saturation_protect=saturation_protect,
                            automask=automask, **kwargs)[0]



  def calcsigma_cpu(self,nn=1, saturation_protect=True,automask=True, **kwargs):
    return nlpar.NLPAR.calcsigma(self, nn=nn,
                            saturation_protect=saturation_protect,automask=automask, **kwargs)

  def opt_lambda_cl(self, saturation_protect=True, automask=True, backsub=False,
                 target_weights=[0.5, 0.34, 0.25], dthresh=0.0, autoupdate=True):

    target_weights = np.asarray(target_weights)

    def loptfunc(lam, d2, tw, dthresh):
      temp = (d2 > dthresh).choose(dthresh, d2)
      dw = np.exp(-(temp) / lam ** 2)
      w = np.sum(dw, axis=2) + 1e-12

      metric = np.mean(np.abs(tw - 1.0 / w))
      return metric

    @numba.njit(fastmath=True, cache=True, parallel=True)
    def d2normcl(d2, n2, sigmapad):
      sftpat = np.array([[-1,-1], [-1, 0], [-1, 1],
                [0, -1], [0, 0], [0, 1],
                [1, -1], [1, 0], [1, 1]], dtype = np.int64)
      #sftpat = sftpat.reshape(-1)
      shp = d2.shape
      s2 = sigmapad ** 2
      for j in numba.prange(shp[0]):
        for i in range(shp[1]):
          s_ij = s2[j+1, i+1]
          for q in range(shp[2]):
            if n2[j, i, q] > 0:
              jj = np.int64(j+1+sftpat[q,0])
              ii = np.int64(i+1+sftpat[q, 1])
              s_q =s2[jj,ii ]
              s2_12 = (s_ij + s_q)
              d2[j, i, q] -= n2[j, i, q] * s2_12
              d2[j, i, q] /= s2_12 * np.sqrt(2.0 * n2[j, i, q])

    patternfile = self.getinfileobj()
    patternfile.read_header()
    nrows = np.uint64(self.nrows)  # np.uint64(patternfile.nRows)
    ncols = np.uint64(self.ncols)  # np.uint64(patternfile.nCols)

    pwidth = np.uint64(patternfile.patternW)
    pheight = np.uint64(patternfile.patternH)
    phw = pheight * pwidth

    dthresh = np.float32(dthresh)
    lamopt_values = []

    sigma, d2, n2 = self.calcsigma_cl(nn=1, saturation_protect=saturation_protect, automask=automask)

    sigmapad = np.pad(sigma, 1, mode='reflect')


    d2normcl(d2, n2, sigmapad)


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


  def calcsigma_cl(self,nn=1,saturation_protect=True,automask=True, gpuid = None, **kwargs):

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
    target_mem = clparams.queue.device.max_mem_alloc_size
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

    for colchunk in range(chunks[0]):
      cstart = chunks[2][colchunk, 0]
      cend = chunks[2][colchunk, 1]
      ncolchunk = cend - cstart
      for rowchunk in range(chunks[1]):
        rstart = chunks[3][rowchunk, 0]
        rend = chunks[3][rowchunk, 1]
        nrowchunk = rend - rstart
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

        prg.calcsigma(queue, (np.uint32(ncolchunk), np.uint32(nrowchunk)), None,
                               datapad_gpu, mask_gpu,sigmachunk_gpu,
                               dist_local, count_local,
                               np.int64(nn), np.int64(npatsteps), np.int64(npat_point),
                               np.float32(mxval) )
        queue.finish()

        cl.enqueue_copy(queue, distchunk, dist_local)
        cl.enqueue_copy(queue, countchunk, count_local)
        cl.enqueue_copy(queue, sigmachunk, sigmachunk_gpu).wait()

        sigmachunk_gpu.release()
        countnn[rstart:rend, cstart:cend] = countchunk[0:int(ncolchunk*nrowchunk), :].reshape(nrowchunk, ncolchunk, nnn)
        dist[rstart:rend, cstart:cend] = distchunk[0:int(ncolchunk*nrowchunk), :].reshape(nrowchunk, ncolchunk, nnn)
        sigma[rstart:rend, cstart:cend] = np.minimum(sigma[rstart:rend, cstart:cend], sigmachunk)

    dist_local.release()
    count_local.release()
    datapad_gpu.release()
    return sigma, dist, countnn

  def calcnlpar_cl(self,chunksize=0, searchradius=None, lam = None, dthresh = None, saturation_protect=True, automask=True,
                filename=None, fileout=None, reset_sigma=True, backsub = False, rescale = False, gpuid = None, **kwargs):

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
    if rescale == True:
      if np.issubdtype(patternfileout.filedatatype, np.integer):
        mxval = np.iinfo(patternfileout.filedatatype).max
        scalemethod = 'fullscale'
      else:  # not int, so no rescale.
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
    target_mem = clparams.queue.device.max_mem_alloc_size//4
    ctx = clparams.ctx
    prg = clparams.prg
    queue = clparams.queue
    mf = clparams.memflags
    clvectlen = 16



    chunks = self._calcchunks( [pwidth, pheight], ncols, nrows, target_bytes=target_mem,
                              col_overlap=sr, row_overlap=sr)

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

    for colchunk in range(chunks[0]):
      cstart = chunks[2][colchunk, 0]
      cend = chunks[2][colchunk, 1]
      ncolchunk = cend - cstart

      cstartcalc = sr if (colchunk > 0) else 0
      cendcalc = ncolchunk-sr if (colchunk < (chunks[0]-1)) else ncolchunk
      ncolcalc = np.int64(cendcalc - cstartcalc)

      for rowchunk in range(chunks[1]):
        rstart = chunks[3][rowchunk, 0]
        rend = chunks[3][rowchunk, 1]
        nrowchunk = rend - rstart

        rstartcalc = sr if (rowchunk > 0) else 0
        rendcalc = nrowchunk - sr if (rowchunk < (chunks[1] - 1)) else nrowchunk
        nrowcalc = np.int64(rendcalc - rstartcalc)

        data, xyloc = patternfile.read_data(patStartCount=[[cstart, rstart], [ncolchunk, nrowchunk]],
                                          convertToFloat=False, returnArrayOnly=True)

        mxval = data.max()
        if saturation_protect == False:
          mxval += 1.0
        else:
          mxval *= 0.9961

        #filldatain = cl.enqueue_fill_buffer(queue, datapad_gpu, np.float32(mxval+10), 0,int(4*npadmx))
        #cl.enqueue_fill_buffer(queue, datapadout_gpu, np.float32(0.0), 0, int(4 * npadmx))

        sigmachunk = np.ascontiguousarray(sigma[rstart:rend, cstart:cend].astype(np.float32))
        sigmachunk_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=sigmachunk)
        szdata = data.size
        npad = clvectlen * int(np.ceil(szdata / clvectlen))

        #datapad = np.zeros((npad), dtype=np.float32) + np.float32(mxval + 10)
        #datapad[0:szdata] = data.reshape(-1)

        data_gpu = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=data)

        if data.dtype.type is np.float32:
          prg.nlloadpat32flt(queue, (np.uint64(data.size),1), None, data_gpu, datapad_gpu)#, wait_for=[filldatain])
        if data.dtype.type is np.ubyte:
          prg.nlloadpat8bit(queue, (np.uint64(data.size),1), None, data_gpu, datapad_gpu)#, wait_for=[filldatain])
        if data.dtype.type is np.uint16:
          prg.nlloadpat16bit(queue, (np.uint64(data.size),1), None, data_gpu, datapad_gpu)#, wait_for=[filldatain])



        calclim = np.array([cstartcalc, rstartcalc, ncolchunk, nrowchunk], dtype=np.int64)
        crlimits_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=calclim)

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
        queue.finish()
        sigmachunk_gpu.release()
        cl.enqueue_copy(queue, data, datapadout_gpu).wait()

        if rescale == True:
          for i in range(data.shape[0]):
            temp = data[i, :, :]
            temp -= temp.min()
            temp *= np.float32(mxval) / temp.max()
            data[i, :, :] = temp

        patternfileout.write_data(newpatterns=data, patStartCount=[[cstart, rstart], [ncolchunk, nrowchunk]],
                                  flt2int='clip', scalevalue=1.0)

    return str(patternfileout.filepath)




  def _calcchunks(self, patdim, ncol, nrow, target_bytes=2e9, col_overlap=0, row_overlap=0, col_offset=0, row_offset=0):

    col_overlap = min(col_overlap, ncol - 1)
    row_overlap = min(row_overlap, nrow - 1)

    byteperpat = patdim[-1] * patdim[-2] * 4 * 2  # assume a 4 byte float input and output array
    byteperdataset = byteperpat * ncol * nrow
    nchunks = int(np.ceil(byteperdataset / target_bytes))



    ncolchunks = (max(np.round(np.sqrt(nchunks * float(ncol) / nrow)), 1))
    colstep = max((ncol / ncolchunks), 1)
    ncolchunks = max(ncol / colstep, 1)
    colstepov = min(colstep + 2 * col_overlap, ncol)
    ncolchunks = max(int(np.ceil(ncolchunks)), 1)
    colstep = max(int(np.round(colstep)), 1)
    colstepov = min(colstep + 2 * col_overlap, ncol)

    nrowchunks = max(np.ceil(nchunks / ncolchunks), 1)
    rowstep = max((nrow / nrowchunks), 1)
    nrowchunks = max(nrow / rowstep, 1)
    rowstepov = min(rowstep + 2 * row_overlap, nrow)
    nrowchunks = max(int(np.ceil(nrowchunks)), 1)
    rowstep = max(int(np.round(rowstep)), 1)
    rowstepov = min(rowstep + 2 * row_overlap, nrow)

    # colchunks = np.round(np.arange(ncolchunks+1)*ncol/ncolchunks).astype(int)
    colchunks = np.zeros((ncolchunks, 2), dtype=int)
    colchunks[:, 0] = (np.arange(ncolchunks) * colstep).astype(int)
    colchunks[:, 1] = colchunks[:, 0] + colstepov - int(col_overlap)
    colchunks[:, 0] -= col_overlap
    colchunks[0, 0] = 0;

    for i in range(ncolchunks - 1):
      if colchunks[i + 1, 0] >= ncol:
        colchunks = colchunks[0:i + 1, :]

    ncolchunks = colchunks.shape[0]
    colchunks[-1, 1] = ncol

    colchunks += col_offset

    # colproc = np.zeros((ncolchunks, 2), dtype=int)
    # if ncolchunks > 1:
    #   colproc[1:, 0] = col_overlap
    # if ncolchunks > 1:
    #   colproc[0:, 1] = colchunks[:, 1] - colchunks[:, 0] - col_overlap
    # colproc[-1, 1] = colchunks[-1, 1] - colchunks[-1, 0]

    # rowchunks = np.round(np.arange(nrowchunks + 1) * nrow / nrowchunks).astype(int)
    rowchunks = np.zeros((nrowchunks, 2), dtype=int)
    rowchunks[:, 0] = (np.arange(nrowchunks) * rowstep).astype(int)
    rowchunks[:, 1] = rowchunks[:, 0] + rowstepov - int(row_overlap)
    rowchunks[:, 0] -= row_overlap
    rowchunks[0, 0] = 0;

    for i in range(nrowchunks - 1):
      if rowchunks[i + 1, 0] >= nrow:
        rowchunks = rowchunks[0:i + 1, :]

    nrowchunks = rowchunks.shape[0]
    rowchunks[-1, 1] = nrow

    rowchunks += row_offset

    # rowproc = np.zeros((nrowchunks, 2), dtype=int)
    # if nrowchunks > 1:
    #   rowproc[1:, 0] = row_overlap
    # if nrowchunks > 1:
    #   rowproc[0:, 1] = rowchunks[:, 1] - rowchunks[:, 0] - row_overlap
    # rowproc[-1, 1] = rowchunks[-1, 1] - rowchunks[-1, 0]

    return ncolchunks, nrowchunks, colchunks, rowchunks


