import numpy as np
from timeit import default_timer as timer
from os import path
from numba import jit, prange
import pyopencl as cl
from os import environ
environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

RADDEG = 180.0/np.pi
DEGRAD = np.pi/180.0

class Radon():
  def __init__(self, image=None, imageDim = None, nTheta = 180, nRho=90,rhoMax = None):
    self.nTheta = nTheta
    self.nRho = nRho
    self.rhoMax = rhoMax
    self.indexPlan = None
    if (image is None) and (imageDim is None):
      self.theta = None
      self.rho = None
      self.imDim = None
    else:
      if (image is not None):
        self.imDim = np.asarray(image.shape[-2:])
      else:
        self.imDim = np.asarray(imageDim[-2:])
      self.radon_plan_setup(imageDim=self.imDim, nTheta=self.nTheta, nRho=self.nRho, rhoMax=self.rhoMax)

  def radon_plan_setup(self, image=None, imageDim=None, nTheta=None, nRho=None, rhoMax=None):
    if (image is None) and (imageDim is not None):
      imDim = np.asarray(imageDim, dtype=np.int)
    elif (image is not None):
      imDim =  np.shape(image)[-2:] # this will catch if someone sends in a [1 x N x M] image
    else:
      return -1
    imDim = np.asarray(imDim)
    self.imDim = imDim
    if (nTheta is not None) : self.nTheta = nTheta
    if (nRho is not None): self.nRho = nRho
    self.rhoMax = rhoMax if (rhoMax is not None) else np.round(np.linalg.norm(imDim)*0.5)

    deltaRho = float(2 * self.rhoMax) / (self.nRho)
    self.theta = np.arange(self.nTheta, dtype = np.float32)*180.0/self.nTheta
    self.rho = np.arange(self.nRho, dtype = np.float32)*deltaRho - (self.rhoMax-deltaRho)

    xmin = -1.0*(self.imDim[0]-1)*0.5
    ymin = -1.0*(self.imDim[1]-1)*0.5

    #self.radon = np.zeros([self.nRho, self.nTheta])
    sTheta = np.sin(self.theta*DEGRAD)
    cTheta = np.cos(self.theta*DEGRAD)
    thetatest = np.abs(sTheta) > (np.sqrt(2.) * 0.5)

    m = np.arange(self.imDim[0], dtype = np.uint32)
    n = np.arange(self.imDim[1], dtype = np.uint32)

    a = -1.0*np.where(thetatest == 1, cTheta, sTheta)
    a /= np.where(thetatest == 1, sTheta, cTheta)
    b = xmin*cTheta + ymin*sTheta

    self.indexPlan = np.zeros([self.nRho, self.nTheta, self.imDim.max()], dtype=np.uint64)
    outofbounds = self.imDim[0]*self.imDim[1]
    for i in np.arange(self.nTheta):
      b1 = self.rho - b[i]
      if thetatest[i]:
        b1 /= sTheta[i]
        b1 = b1.reshape(self.nRho, 1)
        indx_y = np.floor(a[i]*m+b1).astype(np.int64)
        indx_y = np.where(indx_y < 0, outofbounds, indx_y)
        indx_y = np.where(indx_y >= self.imDim[1], outofbounds, indx_y)
        #indx_y = np.clip(indx_y, 0, self.imDim[1])
        indx1D = np.clip(m+self.imDim[0]*indx_y, 0, outofbounds)
        self.indexPlan[:,i, :] = indx1D
      else:
        b1 /= cTheta[i]
        b1 = b1.reshape(self.nRho, 1)
        indx_x = np.floor(a[i]*n + b1).astype(np.int64)
        indx_x = np.where(indx_x < 0, outofbounds, indx_x)
        indx_x = np.where(indx_x >= self.imDim[0], outofbounds, indx_x)
        indx1D = np.clip(indx_x+self.imDim[0]*n, 0, outofbounds)
        self.indexPlan[:, i, :] = indx1D
      self.indexPlan.sort(axis = -1)


  def radon_fast(self, image, padding = np.array([0,0]), fixArtifacts = False):
    tic = timer()
    shapeIm = np.shape(image)
    if image.ndim == 2:
      nIm = 1
      image = image[np.newaxis, : ,:]
      reform = True
    else:
      nIm = shapeIm[0]
      reform = False

    nPx = shapeIm[-1]*shapeIm[-2]
    im = np.zeros(nPx+1, dtype=np.float32)
    #radon = np.zeros([nIm, self.nRho, self.nTheta], dtype=np.float32)
    radon = np.zeros([nIm,self.nRho + 2 * padding[0],self.nTheta + 2 * padding[1]],dtype=np.float32)
    shpRdn = radon.shape
    norm = np.sum(self.indexPlan < nPx, axis = 2 ) + 1.0e-12
    for i in np.arange(nIm):
      im[:-1] = image[i,:,:].flatten()
      radon[i, padding[0]:shpRdn[1]-padding[0], padding[1]:shpRdn[2]-padding[1]] = np.sum(im.take(self.indexPlan.astype(np.int64)), axis=2) / norm

    if (fixArtifacts == True):
      radon[:,:,0] = radon[:,:,1]
      radon[:,:,-1] = radon[:,:,-2]

    if reform==True:
      image = image.reshape(shapeIm)

    #print(timer()-tic)
    return radon

  def radon_faster(self,image,padding = np.array([0,0]), fixArtifacts = False):
    tic = timer()
    shapeIm = np.shape(image)
    if image.ndim == 2:
      nIm = 1
      #image = image[np.newaxis, : ,:]
      #reform = True
    else:
      nIm = shapeIm[0]
    #  reform = False

    image = image.reshape(-1)

    nPx = shapeIm[-1]*shapeIm[-2]
    indxDim = np.asarray(self.indexPlan.shape)
    radon = np.zeros([nIm, self.nRho+2*padding[0], self.nTheta+2*padding[1]], dtype=np.float32)
    shp = radon.shape
    self.rdn_loops(image,self.indexPlan,nIm,nPx,indxDim,radon, np.asarray(padding))

    if (fixArtifacts == True):
      radon[:,:,padding[1]] = radon[:,:,padding[1]+1]
      radon[:,:,shp[2]-1-padding[1]] = radon[:,:,shp[2]-padding[1]-2]


    image = image.reshape(shapeIm)

    #print(timer()-tic)
    return radon

  @staticmethod
  @jit(nopython=True, fastmath=True, cache=True, parallel=False)
  def rdn_loops(images,index,nIm,nPx,indxdim,radon, padding):
    nRho = indxdim[0]
    nTheta = indxdim[1]
    nIndex = indxdim[2]
    count = 0.0
    sum = 0.0
    for q in prange(nIm):
      imstart = q*nPx
      for i in range(nRho):
        ip = i+padding[0]
        for j in range(nTheta):
          jp = j+padding[1]
          count = 0.0
          sum = 0.0
          for k in range(nIndex):
            indx1 = index[i,j,k]
            if (indx1 >= nPx):
              break
            #radon[q, i, j] += images[imstart+indx1]
            sum += images[imstart + indx1]
            count += 1.0
          radon[q,ip,jp] = sum/(count+1e-12)

  def radon_fasterCL(self,image,padding = np.array([0,0]),fixArtifacts = False, returnBuff = False):
    # while keeping this as a method works in multiprocessing on the Mac, it does not on Linux.  Thus I make it a separate function

    tic = timer()
    gpu = cl.get_platforms()[0].get_devices(device_type=cl.device_type.GPU)
    if len(gpu) == 0: # fall back to the numba implementation
      return self.radon_faster(image,padding=padding,fixArtifacts = fixArtifacts)
    # apparently it is very difficult to get a consistent ordering of multiple GPU systems.
    # my lazy way to do this is to assign them randomly, and figure it will even out in the long run
    gpuIdx = np.random.choice(len(gpu))
    ctx = cl.Context(devices = {gpu[gpuIdx]})
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    kernel_location = path.dirname(__file__)
    prg = cl.Program(ctx,open(path.join(kernel_location,'clkernels.cl')).read()).build()
    clparams = (gpu,ctx,queue,prg,mf)

    shapeIm = np.shape(image)
    if image.ndim == 2:
      nIm = 1
      #image = image[np.newaxis, : ,:]
      #reform = True
    else:
      nIm = shapeIm[0]
    #  reform = False

    image = image.reshape(-1).astype(np.float32)
    imstep = np.uint64(np.product(shapeIm[-2:]))
    indxstep = np.uint64(self.indexPlan.shape[-1])
    rdnstep = np.uint64(self.nRho * self.nTheta)

    #steps = np.asarray((imstep, indxstep, rdnstep), dtype = np.uint64)

    radon = np.zeros([nIm,self.nRho+2*padding[0],self.nTheta+2*padding[1]],dtype=np.float32)
    #image_gpu = cl_array.to_device(ctx, queue, image.astype(np.float32))
    #rdnIndx_gpu = cl_array.to_device(ctx, queue, self.indexPlan.astype(np.uint32).reshape[-1])
    #radon_gpu = cl_array.to_device(ctx, queue, radon.astype(np.float32).reshape[-1])
    radon_gpu = cl.Buffer(ctx,mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf=radon)
    image_gpu = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=image)
    rdnIndx_gpu = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=self.indexPlan)

    shpRdn = np.asarray(radon.shape, dtype = np.uint64)
    padRho = np.uint64(padding[0])
    padTheta = np.uint64(padding[1])
    prg.radonSum(queue,(nIm,self.nRho, self.nTheta),None,rdnIndx_gpu,image_gpu,radon_gpu,
                 imstep, indxstep, shpRdn[1], shpRdn[2], padRho, padTheta)
    if (fixArtifacts == True):
      prg.radonFixArt(queue,(nIm,self.nRho,1),None,radon_gpu,
                   shpRdn[1],shpRdn[2],padTheta)
    image = image.reshape(shapeIm)
    queue.finish()
    cl.enqueue_copy(queue,radon,radon_gpu,is_blocking=True).wait()
    if returnBuff == False:
      return radon, clparams, None
    else:
      return radon, clparams, radon_gpu

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