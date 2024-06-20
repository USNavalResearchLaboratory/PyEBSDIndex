"""This software was developed by employees of the US Naval Research Laboratory (NRL), an
agency of the Federal Government. Pursuant to title 17 section 105 of the United States
Code, works of NRL employees are not subject to copyright protection, and this software
is in the public domain. PyEBSDIndex is an experimental system. NRL assumes no
responsibility whatsoever for its use by other parties, and makes no guarantees,
expressed or implied, about its quality, reliability, or any other characteristic. We
would appreciate acknowledgment if the software is used. To the extent that NRL may hold
copyright in countries other than the United States, you are hereby granted the
non-exclusive irrevocable and unconditional right to print, publish, prepare derivative
works and distribute this software, in any medium, or authorize others to do so on your
behalf, on a royalty-free basis throughout the world. You may improve, modify, and
create derivative works of the software or any portion of the software, and you may copy
and distribute such modifications or works. Modified works should carry a notice stating
that you changed the software and should note the date and nature of any such change.
Please explicitly acknowledge the US Naval Research Laboratory as the original source.
This software can be redistributed and/or modified freely provided that any derivative
works bear some notice that they are derived from it, and any modified versions bear
some notice that they have been modified.

Author: David Rowenhorst;
The US Naval Research Laboratory Date: 21 Aug 2020"""

from os import environ
from pathlib import PurePath
import platform
import tempfile
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import pyopencl as cl

from pyebsdindex import band_detect
from pyebsdindex.opencl import openclparam
import scipy

#from os import environ
#environ['PYOPENCL_COMPILER_OUTPUT'] = '1'


RADEG = 180.0/np.pi



class BandDetect(band_detect.BandDetect):
  def __init__( self, **kwargs):
    band_detect.BandDetect.__init__(self, **kwargs)
    self.useCPU = False


  def find_bands(self, patternsIn, verbose=0, clparams=None, chunksize=528, useCPU=None, **kwargs):
    if useCPU is None:
      useCPU = self.useCPU

    if useCPU == True:
      return band_detect.BandDetect.find_bands(self, patternsIn, verbose=verbose, chunksize=-1, **kwargs)
    #if clparams is None:
    #  print('noclparams')
    #else:
    #  print(type(clparams.queue))

    try:
      tic0 = timer()
      tic = timer()
      ndim = patternsIn.ndim
      if ndim == 2:
        patterns = (np.expand_dims(patternsIn, axis=0)).copy()
      else:
        patterns = (patternsIn).copy()

      pscale = np.array([0.0, 1.0])

      if patterns.dtype.kind =='f':
        mxp = patterns.max()
        mnp = patterns.min()
        patterns -= mnp
        patterns *= (2**16-2.0)/(mxp - mnp)
        pscale[:] = np.array([mnp,(mxp - mnp) ])
        patterns = patterns.astype(np.uint16)


      shape = patterns.shape
      nPats = shape[0]

      bandData = np.zeros((nPats,self.nBands),dtype=self.dataType)
      if chunksize < 0:
        nchunks = 1
        chunksize = nPats
      else:
        nchunks = (np.ceil(nPats / chunksize)).astype(np.long64)

      chunk_start_end = [[i * chunksize,(i + 1) * chunksize] for i in range(nchunks)]
      chunk_start_end[-1][1] = nPats
      # these are timers used to gauge performance
      rdntime = 0.0
      convtime = 0.0
      lmaxtime = 0.0
      blabeltime = 0.0

      for chnk in chunk_start_end:
        tic1 = timer()
        nPatsChunk = chnk[1] - chnk[0]
        #rdnNorm, clparams, rdnNorm_gpu = self.calc_rdn(patterns[chnk[0]:chnk[1],:,:], clparams, use_gpu=self.CLOps[0])
        rdnNorm, clparams = self.radon_fasterCL(patterns[chnk[0]:chnk[1],:,:], padding=self.padding,
                                                                       fixArtifacts=False, background=self.backgroundsub,
                                                                       returnBuff=True, clparams=clparams)

        #rdnNorm, clparams = self.rdn_mask(rdnNorm, clparams=clparams, returnBuff=False)

        #if (self.EDAXIQ == True): # I think EDAX actually uses the convolved radon for IQ
          #nTp = self.nTheta + 2 * self.padding[1]
          #nRp = self.nRho + 2 * self.padding[0]
          #nImCL = int(rdnNorm_gpu.size/(nTp*nRp*4))
          #rdnNorm_nocov = np.zeros((nRp,nTp,nImCL),dtype=np.float32)
          #cl.enqueue_copy(clparams.queue,rdnNorm_nocov,rdnNorm,is_blocking=True)

        rdntime += timer() - tic1
        tic1 = timer()
        rdnConv, clparams = self.rdn_convCL2(rdnNorm, clparams=clparams, returnBuff=True, separableKernel=True)
        rdnNorm.release()

        convtime += timer()-tic1
        tic1 = timer()
        lMaxRdn, clparams =  self.rdn_local_maxCL(rdnConv, clparams=clparams, returnBuff=True)
        lmaxtime +=  timer()-tic1
        tic1 = timer()

        bandDataChunk = self.band_labelCL(rdnConv, lMaxRdn, clparams=clparams)
        lMaxRdn.release()
        bandData['max'][chnk[0]:chnk[1]] = bandDataChunk[0][0:nPatsChunk, :]
        bandData['avemax'][chnk[0]:chnk[1]] = bandDataChunk[1][0:nPatsChunk, :]
        bandData['maxloc'][chnk[0]:chnk[1]] = bandDataChunk[2][0:nPatsChunk, :, :]
        bandData['aveloc'][chnk[0]:chnk[1]] = bandDataChunk[3][0:nPatsChunk, :, :]
        bandData['valid'][chnk[0]:chnk[1]] = bandDataChunk[4][0:nPatsChunk, :]
        bandData['width'][chnk[0]:chnk[1]] = bandDataChunk[5][0:nPatsChunk, :]
        bandData['maxloc'][chnk[0]:chnk[1]] -= self.padding.reshape(1, 1, 2) 
        bandData['aveloc'][chnk[0]:chnk[1]] -= self.padding.reshape(1, 1, 2)
        #bandDataChunk, rdnConvBuf = self.band_label(chnk[1]-chnk[0], rdnConv, rdnNorm, lMaxRdn,
        #                                    rdnConv_gpu,rdnConv_gpu,lMaxRdn_gpu,
        #                                    use_gpu = self.CLOps[3], clparams=clparams )

        if (verbose > 1) and (chnk[1] == nPats): # need to pull the radonconv off the gpu
          nTp = self.nTheta + 2 * self.padding[1]
          nRp = self.nRho + 2 * self.padding[0]
          nImCL = int(rdnConv.size / (nTp * nRp * 4))
          rdnConvarray = np.zeros((nRp,nTp,nImCL),dtype=np.float32)

          cl.enqueue_copy(clparams.queue,rdnConvarray,rdnConv,is_blocking=True)

          rdnConvarray = rdnConvarray[:,:,0:chnk[1]-chnk[0] ]
          #plt.imshow(rdnConvarray.squeeze())

        rdnConv.release()
        rdnConv = None

        blabeltime += timer() - tic1

      bandData['avemax'] *= pscale[1]
      bandData['avemax'] += pscale[0]
      bandData['max'] *= pscale[1]
      bandData['max'] += pscale[0]
      tottime = timer() - tic0
      # going to manually clear the clparams queue -- this should clear the memory of the queue off the GPU

      #if clparams is not None:
      #  clparams.queue.finish()
      #  clparams.queue = None

      if verbose > 0:
        print('Radon Time:',rdntime)
        print('Convolution Time:', convtime)
        print('Peak ID Time:', lmaxtime)
        print('Band Label Time:', blabeltime)
        print('Total Band Find Time:',tottime)
      if verbose > 1:
        self._display_radon_pattern(rdnConvarray, bandData, patterns)
        # if len(rdnConvarray.shape) == 3:
        #   im2show = rdnConvarray[self.padding[0]:-self.padding[0],self.padding[1]:-self.padding[1], -1]
        # else:
        #   im2show = rdnConvarray[self.padding[0]:-self.padding[0],self.padding[1]:-self.padding[1]]
        #
        # rhoMaskTrim = np.int32(im2show.shape[0] * self.rhoMaskFrac)
        # mean = np.mean(im2show[rhoMaskTrim:-rhoMaskTrim, 1:-2])
        # stdv = np.std(im2show[rhoMaskTrim:-rhoMaskTrim, 1:-2])
        # im2show -= mean
        # im2show /= stdv
        # im2show = im2show.clip(-4, None)
        # im2show += 6
        # im2show[0:rhoMaskTrim,:] = 0
        # im2show[-rhoMaskTrim:,:] = 0
        #
        # im2show = np.fliplr(im2show)
        # fig = plt.figure(figsize=(12, 4))
        # subrdn = fig.add_subplot(121, xlim=(0, 180), ylim=(-self.rhoMax, self.rhoMax))
        # subrdn.imshow(
        #     im2show,
        #     cmap='gray',
        #     extent=[0, 180, -self.rhoMax, self.rhoMax],
        #     interpolation='none',
        #     zorder=1,
        #     aspect='auto'
        # )
        # width = bandData['width'][-1, :]
        # width /= width.min()
        # width *= 2.0
        # xplt = np.squeeze(180.0 - np.interp(bandData['aveloc'][-1,:,1]+0.5, np.arange(self.radonPlan.nTheta), self.radonPlan.theta))
        # yplt = np.squeeze( -1.0 * np.interp(bandData['aveloc'][-1,:,0]-0.5, np.arange(self.radonPlan.nRho), self.radonPlan.rho))
        #
        # subrdn.scatter(y=yplt, x=xplt, c='r', s=width, zorder=2)
        #
        # for pt in range(self.nBands):
        #   subrdn.annotate(str(pt + 1), np.squeeze([xplt[pt] + 4, yplt[pt]]), color='yellow')
        # #subrdn.xlim(0,180)
        # #subrdn.ylim(-self.rhoMax, self.rhoMax)
        # subpat = fig.add_subplot(122)
        # subpat.imshow(patterns[-1, :, :], cmap='gray')

    except Exception as e: # something went wrong - try the CPU
      print(e)
      bandData = band_detect.BandDetect.find_bands(self, patternsIn, verbose=verbose, chunksize=-1, **kwargs)

    return bandData


  def radon_fasterCL(self,image,padding = np.array([0,0]), fixArtifacts = False, background = None, returnBuff = True, clparams=None ):
    # this function executes the radon sumations on the GPU
    tic = timer()
    image = np.asarray(image)

    # make sure we have an OpenCL environment
    if clparams is not None:
      if clparams.queue is None:
        clparams.get_queue()
      gpu = clparams.gpu
      gpu_id = clparams.gpu_id
      ctx = clparams.ctx
      prg = clparams.prg
      queue = clparams.queue
      mf = clparams.memflags
    else:
      clparams = openclparam.OpenClParam()
      clparams.get_queue()
      gpu = clparams.gpu
      gpu_id = clparams.gpu_id
      ctx = clparams.ctx
      prg = clparams.prg
      queue = clparams.queue
      mf = clparams.memflags

    shapeIm = np.shape(image)
    if image.ndim == 2:
      nIm = 1
      image = image.reshape(1, shapeIm[0], shapeIm[1])
      shapeIm = np.shape(image)
    else:
      nIm = shapeIm[0]
    #  reform = False

    clvtypesize = 16 # this is the vector size to be used in the openCL implementation.
    nImCL = np.uint64(clvtypesize * (np.int64(np.ceil(nIm/clvtypesize))))
    imtype = image.dtype

    tict = timer()
    #image_align = np.ones((shapeIm[1], shapeIm[2], nImCL), dtype = imtype)
    #image_align[:,:,0:nIm] = np.transpose(image, [1,2,0]).copy()
    toct = timer()



    #radon_gpu = cl.Buffer(ctx,mf.READ_WRITE,size=radon.nbytes)
    #radon_gpu = cl.Buffer(ctx,mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf=radon)
    image_gpu = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=image)
    imstep = np.uint64(np.product(shapeIm[-2:]))
    tic = timer()

    nImChunk = np.uint64(nImCL/clvtypesize)
    image_gpuflt = cl.Buffer(ctx, mf.READ_WRITE, size=int(int(shapeIm[1])*int(shapeIm[2])*int(nImCL) * int(4)))  # 32-bit float


    if image.dtype.type is np.float32:
      prg.loadfloat32(queue, (shapeIm[2], shapeIm[1], nIm), None, image_gpu, image_gpuflt, nImCL)
    if image.dtype.type is np.ubyte:
      prg.loadubyte8(queue, (shapeIm[2], shapeIm[1], nIm), None, image_gpu, image_gpuflt, nImCL)
    if image.dtype.type is np.uint16:
      prg.loaduint16(queue, (shapeIm[2], shapeIm[1], nIm), None, image_gpu, image_gpuflt, nImCL)
    queue.flush()
    image_gpu.release()
    image_gpu = None



    if background is not None:
      back_gpu = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=background.astype(np.float32))
      prg.backSub(queue,(imstep, 1, 1),None,image_gpuflt,back_gpu,nImChunk)
      #imBack = np.zeros((shapeIm[1], shapeIm[2], nImCL),dtype=np.float32)
      #cl.enqueue_copy(queue,imBack,image_gpu,is_blocking=True)

    indxstep = np.uint64(self.radonPlan.indexPlan.shape[-1])
    rdnstep = np.uint64(self.nRho * self.nTheta)
    padRho = np.uint64(padding[0])
    padTheta = np.uint64(padding[1])
    shpRdn = np.asarray(((self.nRho + 2 * padding[0]), (self.nTheta + 2 * padding[1]), nImCL), dtype=np.uint64)
    radon_gpu = cl.Buffer(ctx, mf.READ_WRITE,
                          size=int((self.nRho + 2 * padding[0]) * (self.nTheta + 2 * padding[1]) * nImCL * 4))

    rdnIndx_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.radonPlan.indexPlan)
    cl.enqueue_fill_buffer(queue, radon_gpu, np.float32(self.radonPlan.missingval), 0, radon_gpu.size)
    prg.radonSum(queue,(nImChunk,rdnstep),None,rdnIndx_gpu,image_gpuflt,radon_gpu,
                  imstep, indxstep,
                 shpRdn[0], shpRdn[1],
                 padRho, padTheta, np.uint64(self.nTheta))


    if (fixArtifacts == True):
       prg.radonFixArt(queue,(nImChunk,shpRdn[0]),None,radon_gpu,
                       shpRdn[0],shpRdn[1],padTheta)

    rdnIndx_gpu.release()
    rdnIndx_gpu = None


    if returnBuff == False:
      radon = np.zeros([self.nRho + 2 * padding[0],self.nTheta + 2 * padding[1],nImCL],dtype=np.float32)
      cl.enqueue_copy(queue,radon,radon_gpu,is_blocking=True)
      radon_gpu.release()
      radon = radon[:,:, 0:nIm]
      radon_gpu = None
      #clparams = None
      return radon, clparams

    return radon_gpu, clparams



  def rdn_convCL2(self, radonIn, clparams=None, separableKernel=True, returnBuff = True):
    # this will run (eventually sequential) convolutions and identify the peaks in the convolution
    tic = timer()
    #def preparray(array):
    #  return np.require(array,None,"C")

    clvtypesize = 16  # this is the vector size to be used in the openCL implementation.

    if clparams is not None:
      if clparams.queue is None:
        clparams.get_queue()
      gpu = clparams.gpu
      gpu_id = clparams.gpu_id
      ctx = clparams.ctx
      prg = clparams.prg
      queue = clparams.queue
      mf = clparams.memflags
    else:
      clparams = openclparam.OpenClParam()
      clparams.get_queue()
      gpu = clparams.gpu
      gpu_id = clparams.gpu_id
      ctx = clparams.ctx
      prg = clparams.prg
      queue = clparams.queue
      mf = clparams.memflags

    nT = self.nTheta
    nTp = nT + 2 * self.padding[1]
    nR = self.nRho
    nRp = nR + 2 * self.padding[0]

    if isinstance(radonIn, cl.Buffer):
      rdn_gpu = radonIn
      nImCL = np.int64(rdn_gpu.size/(nTp * nRp * 4))
      shp = (nRp, nTp, nImCL)
    else:
      shp = radonIn.shape
      if len(shp) == 2:
        radon = radonIn.reshape(shp[0],shp[1],1)
        nIm = 1
      else:
        radon = radonIn
      shp = radon.shape
      nIm = shp[2]
      nImCL = np.int32(clvtypesize * (np.int64(np.ceil(nIm / clvtypesize))))
      # there is something very strange that happens if the number of images
      # is an exact multiple of the max group size (typically 256)
      mxGroupSz = gpu[gpu_id].get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
      nImCL += np.int64(16 * (1 - np.int64(np.mod(nImCL,mxGroupSz) > 0)))
      radonCL = np.zeros( (nRp , nTp, nImCL), dtype = np.float32)
      radonCL[:,:,0:shp[2]] = radon
      rdn_gpu = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=radonCL)
      shp = (nRp, nTp, nImCL)

    nImChunk = np.uint64(nImCL / clvtypesize)
    resultConv = np.full(shp,0.0,dtype=np.float32)

    rdnConv_gpu = cl.Buffer(ctx,mf.WRITE_ONLY ,size=resultConv.nbytes)

    # maskrnd = np.zeros((self.nRho + 2 * self.padding[0], self.nTheta + 2 * self.padding[1]), dtype=np.ubyte)
    # maskrnd[self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]] = self.rhomask1
    #
    # maskrnd = maskrnd.astype(np.ubyte)
    # maskrnd_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=maskrnd)
    #
    # prg.maskrdn(queue, (np.uint32(nT), np.uint32(nR)), None, rdn_gpu, maskrnd_gpu,
    #             np.uint64(shp[1]), np.uint64(nImChunk),
    #             np.uint64(self.padding[1]), np.uint64(self.padding[0]))

    # # pad out the radon buffers
    # prg.radonPadTheta(queue,(shp[2],shp[0],1),None,rdn_gpu,
    #                 np.uint64(shp[0]),np.uint64(shp[1]),np.uint64(self.padding[1]))

    #prg.radonPadRho2(queue,(shp[2],shp[1],1),None,rdn_gpu,
    #                  np.uint64(shp[0]),np.uint64(shp[1]),np.uint64(self.padding[0]+1))

    prg.radonPadRho2(queue, (shp[2], shp[1], 1), None, rdn_gpu,
                 np.uint64(shp[0]),np.uint64(shp[1]),np.uint64(shp[0]//2-1))

    kern_gpu = None
    if separableKernel == False:
      # for now I will assume that the kernel(s) can fit in local memory on the GPU
      # also going to assume that there is only one kernel -- this will be something to fix at some point.
      k0 = np.array(self.kernel[0,:,:], dtype=np.float32)
      kshp = np.asarray(k0.shape, dtype=np.int32)
      pad = kshp/2
      kern_gpu = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=k0)
      prg.convolution3d2d(queue,(np.int32((shp[1]-2*pad[1])),np.int32((shp[0]-2*pad[0])), nImChunk),None,
                        rdn_gpu, kern_gpu,np.int32(shp[1]),np.int32(shp[0]),np.int32(shp[2]),
                        np.int32(kshp[1]), np.int32(kshp[0]), np.int32(pad[1]), np.int32(pad[0]), rdnConv_gpu)



      #tic = timer()
    else: # convolution is separable
      tempConvbuff = cl.Buffer(ctx,mf.HOST_NO_ACCESS,size=(shp[0]*shp[1]*shp[2]*4))

      kshp = np.asarray(self.kernel[0,:,:].shape,dtype=np.int32)
      pad = kshp
      k0x = np.require(self.kernel[0, np.int64(kshp[0] / 2), :], requirements=['C', 'A', 'W', 'O'], dtype=np.float32)
      k0x *= 1.0 / k0x.sum()
      k0x = (k0x[...,:]).reshape(1,kshp[1])



      kshp = np.asarray(k0x.shape,dtype=np.int32)

      kern_gpu_x = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=k0x)
      prg.convolution3d2d(queue,(np.int32((shp[1]-2*pad[1])),np.int32((shp[0]-2*pad[0])), nImChunk),None,
                          rdn_gpu,kern_gpu_x,np.int32(shp[1]),np.int32(shp[0]),np.int32(shp[2]),
                          np.int32(kshp[1]),np.int32(kshp[0]),np.int32(pad[1]),np.int32(pad[0]),tempConvbuff)

      kshp = np.asarray(self.kernel[0,:,:].shape,dtype=np.int32)
      k0y = np.require(self.kernel[0, :, np.int32(kshp[1] / 2)], requirements=['C', 'A', 'W', 'O'], dtype=np.float32)
      k0y *= 1.0 / k0y.sum()
      k0y = (k0y[...,:]).reshape(kshp[0],1)
      kshp = np.asarray(k0y.shape,dtype=np.int32)

      kern_gpu_y = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=k0y)
      prg.convolution3d2d(queue,(np.int32((shp[1]-2*pad[1])),np.int32((shp[0]-2*pad[0])), nImChunk),None,
                          tempConvbuff,kern_gpu_y,np.int32(shp[1]),np.int32(shp[0]),np.int32(shp[0]),
                          np.int32(kshp[1]),np.int32(kshp[0]),np.int32(pad[1]),np.int32(pad[0]),rdnConv_gpu)

    # for each radon, get the min value
    mns = cl.Buffer(ctx,mf.READ_WRITE,size=nImCL * 4)

    prg.imageMin(queue,(nImChunk,1,1),None,
                 rdnConv_gpu, mns,np.uint32(shp[1]),np.uint32(shp[0]),
                 np.uint32(self.padding[1]),np.uint32(self.padding[0]))
    # subtract the min value, clipping to 0.
    prg.imageSubMinWClip(queue,(np.int32(shp[1]), np.int32(shp[0]),nImChunk),None,
                     rdnConv_gpu,mns,np.uint32(shp[1]),np.uint32(shp[0]),
                     np.uint32(0),np.uint32(0))



    #rdn_gpu.release()
    mns.release()
    if kern_gpu is None:
      kern_gpu_y.release()
      kern_gpu_x.release()
      tempConvbuff.release()
    else:
      kern_gpu.release()

    #cl.enqueue_copy(queue,resultConv,rdnConv_gpu,is_blocking=True)

    if returnBuff == False:
      cl.enqueue_copy(queue, resultConv, rdnConv_gpu, is_blocking=True)
      rdnConv_gpu.release()
      rdnConv_gpu = None
      return resultConv, clparams
    else:
      return rdnConv_gpu, clparams



  def rdn_local_maxCL(self,radonIn, clparams=None,  returnBuff = True):
    # this will run a morphological max kernel over the convolved radon array
    # the local max identifies the location of the peak max within
    # the window size.

    tic = timer()
    clvtypesize = 16  # this is the vector size to be used in the openCL implementation.


    mf = cl.mem_flags
    if clparams is not None:
      if clparams.queue is None:
        clparams.get_queue()
      gpu = clparams.gpu
      gpu_id = clparams.gpu_id
      ctx = clparams.ctx
      prg = clparams.prg
      queue = clparams.queue
      mf = clparams.memflags
    else:
      clparams = openclparam.OpenClParam()
      clparams.get_queue()
      gpu = clparams.gpu
      gpu_id = clparams.gpu_id
      ctx = clparams.ctx
      prg = clparams.prg
      queue = clparams.queue
      mf = clparams.memflags



    nT = self.nTheta
    nTp = nT + 2 * self.padding[1]
    nR = self.nRho
    nRp = nR + 2 * self.padding[0]

    if isinstance(radonIn,cl.Buffer):
      rdn_gpu = radonIn
      nImCL = np.int64(rdn_gpu.size / (nTp * nRp * 4))
      shp = (nRp,nTp,nImCL)
    else:
      shp = radonIn.shape
      if len(shp) == 2:
        radon = radonIn.reshape(shp[0],shp[1],1)
      else:
        radon = radonIn
      shp = radon.shape
      nIm = shp[2]
      nImCL = np.int32(clvtypesize * (np.int64(np.ceil(nIm / clvtypesize))))
      # there is something very strange that happens if the number of images
      # is a exact multiple of the max group size (typically 256)
      mxGroupSz = gpu[gpu_id].get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
      nImCL += np.int64(16 * (1 - np.int64(np.mod(nImCL,mxGroupSz) > 0)))
      radonCL = np.zeros((nRp,nTp,nImCL),dtype=np.float32)
      radonCL[:,:,0:shp[2]] = radon
      rdn_gpu = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=radonCL)
      shp = (nRp,nTp,nImCL)


    nImChunk = np.uint64(nImCL / clvtypesize)
    #out = np.zeros((shp), dtype = np.int32)

    lmaxX = cl.Buffer(ctx, mf.READ_WRITE, size=rdn_gpu.size)
    lmaxXY = cl.Buffer(ctx, mf.READ_WRITE, size=rdn_gpu.size)


    nChunkT = np.uint64(np.ceil(nT / np.float64(self.peakPad[1])))
    nChunkR = np.uint64(np.ceil(nR / np.float64(self.peakPad[0])))
    winszX = np.uint64(self.peakPad[1])
    winszX2 = np.uint64((winszX-1) / 2)
    winszY = np.uint64(self.peakPad[0])
    winszY2 = np.uint64((winszY-1) / 2)


    # prg.morphDilateKernelBF(queue,(np.uint32(nT),np.uint32(nR),nImChunk),None,rdn_gpu,lmaxX,
    #                         np.int64(shp[1]),np.int64(shp[0]),
    #                         np.int64(self.padding[1]),np.int64(self.padding[0]),
    #                         np.int64(1),np.int64(self.peakPad[0]))
    #
    # prg.morphDilateKernelBF(queue,(np.uint32(nT),np.uint32(nR),nImChunk),None,lmaxX,lmaxXY,
    #                         np.int64(shp[1]),np.int64(shp[0]),
    #                         np.int64(self.padding[1]),np.int64(self.padding[0]),
    #                         np.int64(self.peakPad[1]),np.int64(1))
    # calculate the max in the x direction
    prg.morphDilateKernelBF(queue, (np.uint32(shp[1]), np.uint32(nR), nImChunk), None, rdn_gpu, lmaxX,
                            np.int64(shp[1]), np.int64(shp[0]),
                            np.int64(0), np.int64(self.padding[0]),
                            np.int64(self.peakPad[1]), np.int64(1))
    # take the max in the x output, use as input, and calculate in the y direction
    prg.morphDilateKernelBF(queue, (np.uint32(nT), np.uint32(nR), nImChunk), None, lmaxX, lmaxXY,
                            np.int64(shp[1]), np.int64(shp[0]),
                            np.int64(self.padding[1]), np.int64(self.padding[0]),
                            np.int64(1), np.int64(self.peakPad[0]))

    local_max = np.zeros((shp),dtype=np.ubyte)
    local_max_gpu = cl.Buffer(ctx,mf.WRITE_ONLY,size=local_max.nbytes)

    prg.im1EQim2(queue,(np.uint32(nT),np.uint32(nR),nImCL),None, lmaxXY, rdn_gpu, local_max_gpu,
                 np.uint64(shp[1]),np.uint64(shp[0]),
                 np.uint64(self.padding[1]),np.uint64(self.padding[0]))

    maskrnd = np.zeros((self.nRho + 2 * self.padding[0] , self.nTheta + 2 * self.padding[1]), dtype=np.ubyte)
    maskrnd[self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]] = self.rdnmask.astype(np.ubyte)

    maskrnd = maskrnd.astype(np.ubyte)
    maskrnd_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=maskrnd)

    prg.maxmask(queue, (np.uint32(nT), np.uint32(nR)), None, local_max_gpu, maskrnd_gpu,
                 np.uint64(shp[1]), np.uint64(nImChunk),
                 np.uint64(self.padding[1]), np.uint64(self.padding[0]))

    queue.flush()


    if returnBuff == False:
      local_maxX = np.zeros((shp), dtype=np.float32)
      local_maxXY = np.zeros((shp), dtype=np.float32)
      cl.enqueue_copy(queue,local_max,local_max_gpu,is_blocking=True)
      cl.enqueue_copy(queue, local_maxX, lmaxX, is_blocking=True)
      cl.enqueue_copy(queue, local_maxXY, lmaxXY, is_blocking=True)
      queue.flush()
      rdn_gpu.release()
      lmaxX.release()
      lmaxXY.release()
      local_max_gpu.release()
      local_max_gpu = None
      rhoMaskTrim = np.int32((shp[0] - 2 * self.padding[0]) * self.rhoMaskFrac + self.padding[0])
      local_max[0:rhoMaskTrim,:, :] = 0
      local_max[-rhoMaskTrim:,:,:] = 0
      local_max[:,0:self.padding[1],:] = 0
      local_max[:,-self.padding[1]:,:] = 0
      return (local_max,local_maxX, local_maxXY) , clparams
    else:
      lmaxX.release()
      lmaxXY.release()
      return local_max_gpu, clparams


  def band_labelCL(self,rdnConvIn, lMaxRdnIn,clparams=None):

    # an attempt to run the band label on the GPU

    tic = timer()


    if clparams is not None:
      if clparams.queue is None:
        clparams.get_queue()
      gpu = clparams.gpu
      ctx = clparams.ctx
      prg = clparams.prg
      queue = clparams.queue
      mf = clparams.memflags
    else:
      clparams = openclparam.OpenClParam()
      clparams.get_queue()
      gpu = clparams.gpu
      ctx = clparams.ctx
      prg = clparams.prg
      queue = clparams.queue
      mf = clparams.memflags


    nT = self.nTheta
    nTp = nT + 2 * self.padding[1]
    nR = self.nRho
    nRp = nR + 2 * self.padding[0]

    if isinstance(rdnConvIn,cl.Buffer):
      rdnConv_gpu = rdnConvIn
      nIm = np.int64(rdnConv_gpu.size / (nTp * nRp * 4))
      shp = (nRp,nTp,nIm)
    else:
      shp = rdnConvIn.shape
      if len(shp) == 2:
        radonConv = rdnConvIn.reshape(shp[0],shp[1],1)
      else:
        radonConv = rdnConvIn
      shp = radonConv.shape
      nIm = shp[2]

      radonConvCL = np.zeros((nRp,nTp,nIm),dtype=np.float32)
      radonConvCL[:,:,0:shp[2]] = radonConv
      rdnConv_gpu = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=radonConvCL)
      shp = (nRp,nTp,nIm)

    if isinstance(lMaxRdnIn,cl.Buffer):
      lMaxRdn_gpu = lMaxRdnIn
      nIm = np.int64(lMaxRdn_gpu.size / (nTp * nRp))
      shp = (nRp,nTp,nIm)
    else:
      shp = lMaxRdnIn.shape
      if len(shp) == 2:
        lMaxRdn = lMaxRdnIn.reshape(shp[0],shp[1],1)
      else:
        lMaxRdn = lMaxRdnIn
      shp = lMaxRdnIn.shape
      nIm = shp[2]

      lMaxRdnCL = np.zeros((nRp,nTp,nIm),dtype=np.float32)
      lMaxRdnCL[:,:,0:shp[2]] = lMaxRdn
      lMaxRdn_gpu = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=lMaxRdnCL)
      shp = (nRp,nTp,nIm)


    maxval = np.zeros((nIm, self.nBands),dtype=np.float32)-2.0e6
    maxval_gpu = cl.Buffer(ctx,mf.WRITE_ONLY,size=maxval.nbytes)
    maxloc = np.zeros((nIm, self.nBands),dtype=np.int64)
    maxloc_gpu = cl.Buffer(ctx,mf.WRITE_ONLY,size=maxloc.nbytes)
    aveval = np.zeros((nIm,self.nBands),dtype=np.float32) - 2.0e6
    aveval_gpu = cl.Buffer(ctx,mf.WRITE_ONLY,size=aveval.nbytes)
    width = np.zeros((nIm, self.nBands), dtype=np.float32)
    width_gpu = cl.Buffer(ctx, mf.WRITE_ONLY, size=width.nbytes)
    aveloc = np.zeros((nIm,self.nBands,2),dtype=np.float32)
    aveloc_gpu = cl.Buffer(ctx,mf.WRITE_ONLY,size=aveloc.nbytes)
    #rhoMaskTrim = np.int64((shp[0] - 2 * self.padding[0]) * self.rhoMaskFrac + self.padding[0])
    rhoMaskTrim = np.int64(self.padding[0])

    prg.maxlabel(queue,(nIm, 1,1),(1,1,1),
                 lMaxRdn_gpu,rdnConv_gpu,
                 maxloc_gpu, maxval_gpu,
                 aveloc_gpu,aveval_gpu,
                 width_gpu,
                 np.int64(shp[1]),np.int64(shp[0]),
                 np.int64(self.padding[1]),rhoMaskTrim,np.int64(self.nBands) )


    queue.finish()
    cl.enqueue_copy(queue,maxval,maxval_gpu,is_blocking=False)
    cl.enqueue_copy(queue,maxloc,maxloc_gpu,is_blocking=False)
    cl.enqueue_copy(queue,aveval,aveval_gpu,is_blocking=False)
    cl.enqueue_copy(queue,aveloc,aveloc_gpu,is_blocking=False)
    cl.enqueue_copy(queue, width, width_gpu, is_blocking=True)

    #rdnConv_out = np.zeros((nRp,nTp,nIm),dtype=np.float32)
    #cl.enqueue_copy(queue,rdnConv_out,rdnConv_gpu,is_blocking=True)
    #queue.finish()
    #rdnConv_gpu.release()
    maxval_gpu.release()
    maxloc_gpu.release()
    aveval_gpu.release()
    aveloc_gpu.release()
    width_gpu.release()
    queue.finish()

    maxlocxy = np.zeros((nIm, self.nBands, 2),dtype=np.float32)
    temp = np.asarray(np.unravel_index(maxloc, (shp[0], shp[1])), dtype = np.float32)
    maxlocxy[:,:,0] = temp[0,:,:]
    maxlocxy[:,:,1] = temp[1,:,:]
    valid = np.asarray(maxval > -1e6, dtype=np.int8)



    return (maxval,aveval,maxlocxy,aveloc,valid, width)

def getopenclparam(**kwargs):
  clparam = openclparam.OpenClParam(**kwargs)

  return clparam
