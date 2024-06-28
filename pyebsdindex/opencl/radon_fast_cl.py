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
from timeit import default_timer as timer

import numpy as np
import pyopencl as cl

from pyebsdindex import radon_fast
from pyebsdindex.opencl import openclparam

environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

RADDEG = 180.0/np.pi
DEGRAD = np.pi/180.0



class Radon(radon_fast.Radon):
  def __init__(self, clparams=None, **kwargs):
    radon_fast.Radon.__init__(self,**kwargs)
    #self.setcl(clparams)

  def setcl(self, clparams=None):
    if clparams is None:
      self.clparams = openclparam.OpenClParam()
      self.clparams.get_queue()
    else:
      self.clparams = clparams


  def radon_fasterCL(self,image,padding = np.array([0,0]), fixArtifacts = False,
                     background = None, background_method = 'SUBTRACT',
                     returnBuff = True, clparams=None ):

    tic = timer()
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
    nImCL = np.int32(clvtypesize * (np.int64(np.ceil(nIm/clvtypesize))))
    # there is something very strange that happens if the number of images
    # is a exact multiple of the max group size (typically 256)
    mxGroupSz = gpu[gpu_id].get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
    #nImCL += np.int64(16 * (1 - np.int64(np.mod(nImCL, mxGroupSz ) > 0)))
    image_align = np.ones((shapeIm[1], shapeIm[2], nImCL), dtype = np.float32)
    image_align[:,:,0:nIm] = np.transpose(image, [1,2,0]).copy()
    shpRdn = np.asarray( ((self.nRho+2*padding[0]), (self.nTheta+2*padding[1]), nImCL),dtype=np.uint64)
    radon_gpu = cl.Buffer(ctx,mf.READ_WRITE,size=int((self.nRho+2*padding[0])*(self.nTheta+2*padding[1])*nImCL*4))
    #radon_gpu = cl.Buffer(ctx,mf.READ_WRITE,size=radon.nbytes)
    #radon_gpu = cl.Buffer(ctx,mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf=radon)
    image_gpu = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=image_align)
    rdnIndx_gpu = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=self.indexPlan)

    imstep = np.uint64(np.prod(shapeIm[-2:], dtype=int))
    indxstep = np.uint64(self.indexPlan.shape[-1])
    rdnstep = np.uint64(self.nRho * self.nTheta)

    padRho = np.uint64(padding[0])
    padTheta = np.uint64(padding[1])
    tic = timer()

    nImChunk = np.uint64(nImCL/clvtypesize)

    if background is not None:
      back_gpu = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=background.astype(np.float32))
      if str.upper(background_method) == 'DIVIDE':
        prg.backDiv(queue,(imstep, 1, 1),None,image_gpu,back_gpu,nImChunk)
      else:
        prg.backSub(queue,(imstep, 1, 1),None,image_gpu,back_gpu,nImChunk)
        #imBack = np.zeros((shapeIm[1], shapeIm[2], nImCL),dtype=np.float32)
        #cl.enqueue_copy(queue,imBack,image_gpu,is_blocking=True)

    cl.enqueue_fill_buffer(queue, radon_gpu, np.float32(self.missingval), 0, radon_gpu.size)
    prg.radonSum(queue,(nImChunk,rdnstep),None,rdnIndx_gpu,image_gpu,radon_gpu,
                  imstep, indxstep,
                 shpRdn[0], shpRdn[1],
                 padRho, padTheta, np.uint64(self.nTheta))


    if (fixArtifacts == True):
       prg.radonFixArt(queue,(nImChunk,shpRdn[0]),None,radon_gpu,
                       shpRdn[0],shpRdn[1],padTheta)




    if returnBuff == False:
      radon = np.zeros([self.nRho + 2 * padding[0],self.nTheta + 2 * padding[1],nImCL],dtype=np.float32)
      cl.enqueue_copy(queue,radon,radon_gpu,is_blocking=True)
      radon_gpu.release()
      radon = radon[:,:, 0:nIm]
      radon_gpu = None
      #clparams = None
      return radon, clparams
    else:
      rdnIndx_gpu.release()
      image_gpu.release()

    return radon_gpu, clparams

    #if (fixArtifacts == True):
    #  radon[:,:,padding[1]] = radon[:,:,padding[1]+1]
    #  radon[:,:,-padding[1]-1] = radon[:,:,-2-padding[1]]




    #print(timer()-tic)





# if __name__ == "__main__":
#   import ebsd_pattern, ebsd_index
#   file = '~/Desktop/SLMtest/scan2v3nlparl09sw7.up1' ;f = ebsd_pattern.UPFile(file)
#
#   pat = f.read_data(patStartEnd=[0,1],convertToFloat=True,returnArrayOnly=True )
#   dat, indxer = ebsd_index.index_pats(filename = file, patStart = 0, patEnd = 1,return_indexer_obj = True)
#   dat = ebsd_index.index_pats_distributed(filename = file,patStart = 0, patEnd = -1, chunksize = 1000, ncpu = 34, ebsd_indexer_obj = indxer )
#