import os, sys, platform
import logging
from timeit import default_timer as timer
import numpy as np
import pyopencl as cl

import ray

#from pyebsdindex import nlpar
from pyebsdindex.opencl import openclparam, nlpar_cl
from time import time as timer

RAYIPADDRESS = '127.0.0.1'
OSPLATFORM  = platform.system()
#if OSPLATFORM  == 'Darwin':
#    RAYIPADDRESS = '0.0.0.0'  # the localhost address does not work on macOS when on a VPN


class NLPAR(nlpar_cl.NLPAR):
  def __init__( self, **kwargs):
    nlpar_cl.NLPAR.__init__(self, **kwargs)
    self.useCPU = False

  def calcsigma(self,nn=1, saturation_protect=True,automask=True, **kwargs):
    return self.calcsigma_cl(nn=nn,
                            saturation_protect=saturation_protect,
                            automask=automask, **kwargs)[0]

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

    cudavis = str(int(gpuid))
    #print(gpuid)
    clparams.get_context(gpu_id=gpuid, kfile = 'clnlpar.cl')
    clparams.get_queue()

    target_mem = clparams.gpu[gpuid].max_mem_alloc_size
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
        queue.flush()
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
    queue.flush()
    queue = None
    return sigma, dist, countnn

  def calcnlpar_cl(self,chunksize=0, searchradius=None, lam = None, dthresh = None, saturation_protect=True, automask=True,
                filename=None, fileout=None, reset_sigma=True, backsub = False, rescale = False, gpuid = None, **kwargs):

    if lam is not None:
      self.lam = lam

    self.saturation_protect = saturation_protect

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
      if np.issubdtype(self.patternfileout.filedatatype, np.integer):
        mxval = np.iinfo(self.patternfileout.filedatatype).max
        scalemethod = 'fullscale'
      else:  # not int, so no rescale.
        self.rescale = False

    clparams = openclparam.OpenClParam()
    clparams.get_gpu()
    if gpuid is None:
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
      gpuid = min(len(clparams.gpu)-1, gpuid)
    cudavis = ''
    for cdgpu in range(len(clparams.gpu)):
        cudavis += str(cdgpu) + ','

    # print(gpuid)
    # clparams.get_context(gpu_id=gpuid, kfile = 'clnlpar.cl')
    # clparams.get_queue()

    target_mem = clparams.gpu[gpuid].max_mem_alloc_size//8

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

            jobqueue.append( GPUJob([colchunk, rowchunk],\
                    [cstart,cend, rstart, rend],\
                  [cstartcalc,cendcalc, rstartcalc, rendcalc ]))


    ngpuwrker = 4
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
      logging_level=logging.WARNING,
    )  # Supress INFO messages from ray.

    nlpar_remote = ray.put(self)

    idlewrker = []
    busywrker = []
    tasks = []
    for w in range(ngpuwrker):
        idlewrker.append(GPUWorker.options(num_cpus=float(0.99), num_gpus=ngpu_per_wrker).remote(
                actorid=w, gpu_id=gpuid, cudavis=cudavis))

    njobs = len(jobqueue)
    ndone = 0
    while ndone < njobs:
        if len(jobqueue) > 0:
            if len(idlewrker) > 0:
                wrker = idlewrker.pop()
                job = jobqueue.pop()

                tasks.append(wrker.runnlpar_chunk.remote(job, nlparobj=nlpar_remote))
                busywrker.append(wrker)
        if len(tasks) > 0:
            donetasks, stillbusy = ray.wait(tasks, num_returns=len(busywrker), timeout=0.1)

            for tsk in donetasks:
                indx = tasks.index(tsk)
                message, job, newdata = ray.get(tsk)
                if message == 'Done':
                  self.patternfileout.write_data(newpatterns=newdata,
                                                     patStartCount=[[job.cstart + job.cstartcalc,
                                                                     job.rstart + job.rstartcalc],
                                                                    [job.ncolcalc, job.nrowcalc]],
                                                     flt2int='clip', scalevalue=1.0)

                  idlewrker.append(busywrker.pop(indx))
                  tasks.remove(tsk)
                  ndone += 1
                print(message, ndone, njobs)

    return str(self.patternfileout.filepath)

  def _nlparchunkcalc_cl(self, data, calclim, clparams=None):
    data = np.ascontiguousarray(data)
    ctx = clparams.ctx
    prg = clparams.prg
    queue = clparams.queue
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

    fill1 = cl.enqueue_fill_buffer(queue, datapad_gpu, np.float32(mxval + 10), 0, int(4 * npadmx))
    fill2 = cl.enqueue_fill_buffer(queue, datapadout_gpu, np.float32(0.0), 0, int(4 * npadmx))


    mask = self.mask.astype(np.float32)

    npad = clvectlen * int(np.ceil(mask.size / clvectlen))
    maskpad = np.zeros((npad), np.float32) - 1  # negative numbers will indicate a clvector overflow.
    maskpad[0:mask.size] = mask.reshape(-1).astype(np.float32)
    mask_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=maskpad)
    npatsteps = int(maskpad.size / clvectlen)

    sigmachunk = np.ascontiguousarray(self.sigma[rstart:rend, cstart:cend].astype(np.float32))
    sigmachunk_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=sigmachunk)

    szdata = data.size
    cl.enqueue_barrier(queue)
    data_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
    if data.dtype.type is np.float32:
      prg.nlloadpat32flt(queue, (np.uint64(data.size), 1), None, data_gpu, datapad_gpu)#, wait_for=[fill1,fill2])
    if data.dtype.type is np.ubyte:
      prg.nlloadpat8bit(queue, (np.uint64(data.size), 1), None, data_gpu, datapad_gpu)#, wait_for=[fill1,fill2])
    if data.dtype.type is np.uint16:
      prg.nlloadpat16bit(queue, (np.uint64(data.size), 1), None, data_gpu, datapad_gpu)#, wait_for=[fill1,fill2])

    calclim = np.array([cstartcalc, rstartcalc, ncolchunk, nrowchunk], dtype=np.int64)
    crlimits_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=calclim)
    cl.enqueue_barrier(queue)
    data_gpu.release()
    prg.calcnlpar(queue, (np.uint32(ncolcalc), np.uint32(nrowcalc)), None,
                  # prg.calcnlpar(queue, (1, 1), None,
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
    cl.enqueue_copy(queue, data, datapadout_gpu).wait()
    sigmachunk_gpu.release()
    queue.finish()
    if self.rescale == True:
      for i in range(data.shape[0]):
        temp = data[i, :, :]
        temp -= temp.min()
        temp *= np.float32(mxval) / temp.max()
        data[i, :, :] = temp
    data = data[rstartcalc: rstartcalc + nrowcalc, cstartcalc: cstartcalc + ncolcalc, :, :]
    data = data.reshape(nrowcalc * ncolcalc, pheight, pwidth)

    queue = None

    return data









    #
    # filldatain = cl.enqueue_fill_buffer(queue, datapad_gpu, np.float32(mxval + 10), 0, int(4 * npadmx))
    # cl.enqueue_fill_buffer(queue, datapadout_gpu, np.float32(0.0), 0, int(4 * npadmx))
    #
    # sigmachunk = np.ascontiguousarray(sigma[rstart:rend, cstart:cend].astype(np.float32))
    # sigmachunk_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=sigmachunk)
    # szdata = data.size
    # npad = clvectlen * int(np.ceil(szdata / clvectlen))
    #
    # # datapad = np.zeros((npad), dtype=np.float32) + np.float32(mxval + 10)
    # # datapad[0:szdata] = data.reshape(-1)




@ray.remote
class GPUWorker:
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
        try:
            self.openCLParams.gpu_id = gpu_id
            self.openCLParams.get_context(gpu_id=gpu_id, kfile = 'clnlpar.cl')
        except:
            self.openCLParams = None

    def runnlpar_chunk(self, gpujob, nlparobj=None):

        if gpujob is None:
            #time.sleep(0.001)
            return 'Bored', (None, None, None)
        try:
            # print(type(self.openCLParams.ctx))
            gpujob._starttime()
            #time.sleep(random.uniform(0, 1.0))

            if self.openCLParams is not None:
                self.openCLParams.get_queue()



            data, xyloc = nlparobj.patternfile.read_data(patStartCount=[[gpujob.cstart, gpujob.rstart],
                                                                        [gpujob.ncolchunk, gpujob.nrowchunk]],
                                                                        convertToFloat=False, returnArrayOnly=True)


            newpats = nlparobj._nlparchunkcalc_cl(data, gpujob, clparams=self.openCLParams)
            if self.openCLParams is not None:
                self.openCLParams.queue.finish()
                self.openCLParams.queue = None


            gpujob._endtime()


            return 'Done', gpujob, newpats
        except Exception as e:
            print(e)
            gpujob.rate = None
            return "Error", gpujob, e


class GPUJob:
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



