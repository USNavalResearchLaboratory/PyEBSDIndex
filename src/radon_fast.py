import numpy as np
from timeit import default_timer as timer
from numba import jit, prange

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

    self.indexPlan = np.zeros([self.nRho, self.nTheta, self.imDim.max()], dtype=np.int64)
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


  def radon_fast(self, image, fixArtifacts = False):
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
    radon = np.zeros([nIm, self.nRho, self.nTheta], dtype=np.float32)
    for i in np.arange(nIm):
      im[:-1] = image[i,:,:].flatten()
      radon[i, :, :] = np.sum(im.take(self.indexPlan), axis=2)

    if (fixArtifacts == True):
      radon[:,:,0] = radon[:,:,1]
      radon[:,:,-1] = radon[:,:,-2]

    if reform==True:
      image = image.reshape(shapeIm)

    #print(timer()-tic)
    return radon

  def radon_faster(self,image,fixArtifacts = False):
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
    radon = np.zeros([nIm, self.nRho, self.nTheta], dtype=np.float32)
    self.rdn_loops(image,self.indexPlan,nIm,nPx,indxDim,radon)

    if (fixArtifacts == True):
      radon[:,:,0] = radon[:,:,1]
      radon[:,:,-1] = radon[:,:,-2]


    image = image.reshape(shapeIm)

    #print(timer()-tic)
    return radon

  @staticmethod
  @jit(nopython=True, fastmath=True, cache=True, parallel=True)
  def rdn_loops(images,index,nIm,nPx,indxdim,radon):
    nRho = indxdim[0]
    nTheta = indxdim[1]
    nIndex = indxdim[2]
    for q in prange(nIm):
      imstart = q*nPx
      for i in range(nRho):
        for j in range(nTheta):
          for k in range(nIndex):
            indx1 = index[i,j,k]
            if (indx1 >= nPx):
              break
            radon[q, i, j] += images[imstart+indx1]






# if __name__ == "__main__":
#   file = '~/Desktop/SLMtest/scan2v3nlparl09sw7.up1'
#   f = EBSDPattern.upFile(file)
#   pat = f.ReadData(patStartEnd=[0,0],convertToFloat=True,returnArrayOnly=True )
#   RadonPlan(pat)