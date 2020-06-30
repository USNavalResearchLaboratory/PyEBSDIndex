import numpy as np
from scipy.ndimage import gaussian_filter
from radon_fast import Radon
from timeit import default_timer as timer
import matplotlib.pyplot as plt
#from numba import jit
RADEG = 180.0/np.pi


class BandDetect():
  def __init__(self, patterns=None, patDim = None, nTheta = 180, nRho=90,\
      tSigma= None, rSigma=None, rhoMaskFrac=0.1, nBands=9):
    self.patDim = None
    self.nTheta = nTheta
    self.nRho = nRho
    self.dTheta = None
    self.dRho = None
    self.rhoMax = None
    self.radonPlan = None
    self.rdnNorm = None
    self.tSigma = tSigma
    self.rSigma = rSigma
    self.kernel = None
    self.rhoMaskFrac = rhoMaskFrac
    self.rhoMask = None
    self.peakPad = None
    self.peakMask = None
    self.nBands = nBands
    if (patterns is None) and (patDim is None):
      pass
    else:
      if (patterns is not None):
        self.patDim = np.asarray(patterns.shape[-2:])
      else:
        self.patDim = np.asarray(patDim)
      self.band_detect_setup(patterns, self.patDim,self.nTheta,self.nRho,\
                self.tSigma, self.rSigma,self.rhoMaskFrac,self.nBands)

  def band_detect_setup(self, patterns=None,patDim=None,nTheta=None,nRho=None,\
                      tSigma=None, rSigma=None,rhoMaskFrac=None,nBands=None):
    p_dim = None
    recalc_radon = False
    recalc_masks = False
    if (patterns is None) and (patDim is not None):
      p_dim = np.asarray(patDim, dtype=np.int)
    if patterns is not None:
      p_dim = np.shape(patterns)[-2:]  # this will catch if someone sends in a [1 x N x M] image
    if p_dim is not None:
      if self.patDim is None:
        recalc_radon = True
        self.patDim = p_dim

      elif np.sum(np.abs(self.patDim[-2:]-p_dim[-2:]), dtype=np.int) != 0:
        recalc_radon = True
        self.patDim = p_dim

    if nTheta is not None:
      self.nTheta = nTheta
      recalc_radon = True
      recalc_masks = True

    if self.nTheta is not None:
      self.dTheta = 180.0/self.nTheta


    if nRho is not None:
      self.nRho = nRho
      self.dRho = 180. / self.nRho
      recalc_radon = True
      recalc_masks = True

    if self.dRho is None:
      recalc_radon = True

    if recalc_radon == True:
      self.rhoMax = 0.5 * np.float32(self.patDim.min())
      self.dRho = self.rhoMax/np.float32(self.nRho)
      self.radonPlan = Radon(imageDim=self.patDim, nTheta=self.nTheta, nRho=self.nRho, rhoMax=self.rhoMax)
      temp = np.ones(self.patDim[-2:], dtype=np.float32)
      back = self.radonPlan.radon_faster(temp,fixArtifacts=True)
      back = (back > 0).astype(np.float32) / (back + 1.0e-12)
      self.rdnNorm = back


    if tSigma is not None:
      self.tSigma = tSigma
      recalc_masks = True
    if rSigma is not None:
      self.rSigma = rSigma
      recalc_masks = True

    if rhoMaskFrac is not None:
      self.rhoMaskFrac = rhoMaskFrac
      recalc_masks = True
    if (self.rhoMask is None) and (self.rhoMaskFrac is not None):
      recalc_masks = True

    if (self.tSigma is None) and (self.dTheta is not None):
      self.tSigma = 1.0/self.dTheta
      recalc_masks = True

    if (self.rSigma is None) and (self.dRho is not None):
      self.rSigma = 0.25/np.float32(self.dRho)
      recalc_masks = True

    if recalc_masks == True:
      ksz = np.array([np.max([np.int(4*self.rSigma), 5]), np.max([np.int(4*self.tSigma), 5])])
      self.peakPad = np.array(np.around([ 4*ksz[0], 20.0/self.dTheta]), dtype=np.int)
      self.peakMask = np.ones(self.peakPad, dtype=np.int32)
      self.rhoMask = np.ones([1,self.nRho,self.nTheta+2*self.peakPad[1] ], dtype=np.float32)
      if self.rhoMaskFrac > 0:
        self.rhoMask[:,0:np.int(np.floor(self.nRho*self.rhoMaskFrac)), :] = 0.0
        self.rhoMask[:,-np.int(np.floor(self.nRho*self.rhoMaskFrac)):, :] = 0.0

    if nBands is not None:
      self.nBands = nBands

  def find_bands(self, patternsIn, faster=False, verbose=False):
    tic = timer()
    ndim = patternsIn.ndim
    if ndim == 2:
      patterns = np.expand_dims(patternsIn, axis=0)
    else:
      patterns = patternsIn

    shape = patterns.shape
    nPats = shape[0]
    peakmask_offset = np.array(-1 * np.floor(self.peakPad * 0.5), dtype=np.int)
    peakmask_offset = np.broadcast_to(peakmask_offset, (nPats, 2))

    peakloc = np.zeros((nPats,self.nRho+2*self.peakPad[0], self.nTheta+2*self.peakPad[1]), dtype=np.int)
    peaklocmask = np.ones((nPats,self.nRho+2*self.peakPad[0], self.nTheta+2*self.peakPad[1]), dtype=np.int)
    peaklocmask[:,:,0:self.peakPad[1]] = 0
    peaklocmask[:,:,-self.peakPad[1]:] = 0
    #plt.imshow(peaklocmask[0,:,:], origin='lower')

    bandDataType = np.dtype([('id', np.int32), ('max', np.float32), \
                             ('maxloc', np.float32, (2)), ('avemax', np.float32), ('aveloc', np.float32, (2)),\
                             ('pqmax', np.float32)])
    bandData = np.zeros((nPats, self.nBands), dtype=bandDataType)
    eps = 1.e-6

    rdn = self.radonPlan.radon_faster(patterns,fixArtifacts=True)

    rdnNorm = rdn*self.rdnNorm


    rdnNormP = np.zeros((nPats,self.nRho,self.nTheta+2*self.peakPad[1]), dtype=np.float32)
    rdnNormP[:,:,self.peakPad[1]:-self.peakPad[1]] = rdnNorm

    rdnNormP[:, :, 0:self.peakPad[1]] = np.flip(rdnNormP[:, :, -2*self.peakPad[1]:-self.peakPad[1]], axis=1)
    rdnNormP[:, :,-self.peakPad[1]:] = np.flip(rdnNormP[:, :, self.peakPad[1]:2*self.peakPad[1]], axis = 1)
    rdnConv = np.zeros_like(rdnNormP)
    mns = np.zeros(nPats, dtype=np.float32)

    for i in range(nPats):
      rdnConv[i,:,:] = -1.0*gaussian_filter(rdnNormP[i,:,:].reshape(self.nRho,self.nTheta+2*self.peakPad[1]), \
                                       [self.rSigma, self.tSigma], order=[2,0])
      #mns[i] = rdnConv[i,:,:].min()
      #rdnConv[i,:,:] -= mns[i]
    mns = rdnConv.min(axis=1).min(axis=1)
    rdnConv -= mns.reshape((nPats,1,1))

    rdnConv *= (rdnNormP > 0).astype(np.float32) / (rdnNormP+1e-12)
    rdnPad = np.array(rdnConv)
    rdnPad *= self.rhoMask
    rdnPad = np.pad(rdnPad, ((0,),(self.peakPad[1],),(0,)), mode ='constant',constant_values=0.0 )

    nnmask = np.array([-2,-1,0, 1,2, self.nTheta+2*self.peakPad[1],-1*(self.nTheta+2*self.peakPad[1]) ])
    nn = 7
    nlayer = rdnPad.shape[-2] * rdnPad.shape[-1]
    mskSz = self.peakMask.shape

    for i in range(self.nBands):
      mxloc = (rdnPad*(peakloc == 0)*peaklocmask).reshape((nPats, nlayer)).argmax(axis=1)
      #plt.imshow( (rdnPad*(peakloc == 0)*peaklocmask)[13,:,:])
      bandData['max'][ :, i] = (rdnPad.reshape((nPats, nlayer)))[np.arange(nPats), mxloc]
      nnindx = mxloc.reshape((nPats,1)) + nnmask.reshape((1,nn))
      rdnNNv = np.take(rdnPad.reshape((nPats, nlayer)), nnindx)
      bandData['avemax'][:,i] = np.mean(rdnNNv, axis =1)
      mxloc2 = np.array(np.unravel_index(mxloc, rdnPad.shape[-2:])).T
      bandData['maxloc'][:,i,:] = mxloc2
      nnloc = np.array(np.unravel_index(nnindx, rdnPad.shape[-2:]), dtype=np.float32).transpose([1,2,0])
      nnloc *= rdnNNv.reshape(nPats,nn,1)
      nnloc = np.sum(nnloc, axis = 1)
      nnloc /= np.sum(rdnNNv, axis=1).reshape(nPats,1)
      bandData['aveloc'][:, i, :] = nnloc.reshape(nPats, 2)
      mxloc2 += peakmask_offset
      tempmask = np.array(self.peakMask*(i+1))

      for j in range(nPats):
        #print(mxloc2[j,:])
        peakloc[j,mxloc2[j,0]:mxloc2[j,0]+mskSz[0], mxloc2[j,1]:mxloc2[j,1]+mskSz[1]] = np.array(tempmask)
      #plt.imshow(peakloc[-1,:,:])
      flipl = np.flip(peakloc[:,:,-2*self.peakPad[1]:], axis=1)
      rnflip = peakloc[:,:, 0: 2*self.peakPad[0]]
      rnflip = np.where(rnflip >= flipl,rnflip, flipl)
      peakloc[:, :, 0: 2 * self.peakPad[0]] = rnflip

      flipr = np.flip(peakloc[:, :,0:2 * self.peakPad[1]], axis=1)
      lnflip = peakloc[:, :, -2 * self.peakPad[0]:]
      lnflip = np.where(lnflip >= flipr, lnflip,flipr)
      peakloc[:, :, -2 * self.peakPad[0]:] = lnflip

    #rdnConv = rdnConv[:, :, self.peakPad[1]: -self.peakPad[1]]
    bandData['maxloc'] -= self.peakPad.reshape(1,1,2)
    bandData['aveloc'] -= self.peakPad.reshape(1, 1, 2)
    for j in range(nPats):
      bandData['pqmax'][j,:] = \
        rdnNorm[j, (bandData['aveloc'][j,:,0]).astype(int),(bandData['aveloc'][j,:,1]).astype(int)]
    #bandData['max'] += mns.reshape(nPats,1)


    if verbose == True:
      print(timer() - tic)
      plt.clf()
      plt.imshow(rdnConv[-1,:,:], origin='lower')
      plt.scatter(y = bandData['aveloc'][-1,:,0], x = bandData['aveloc'][-1,:,1]+self.peakPad[1], c ='r', s=5)
      plt.show()


    return bandData


  def radon2pole(self,bandData,PC=None,vendor='EDAX'):
    if PC is None:
      PC = np.array([0.471659,0.675044,0.630139])
    ven = str.upper(vendor)

    PC_PX = PC.copy()
    dimf = np.array(self.patDim, dtype=np.float32)
    if ven == 'EDAX':
      PC_PX *= np.array([dimf[0], dimf[1], -dimf[0]])


    nPats = bandData.shape[0]
    nBands = bandData.shape[1]

    theta = self.radonPlan.theta[np.array(bandData['aveloc'][:,:,1], dtype=np.int)]/RADEG
    rho = self.radonPlan.rho[np.array(bandData['aveloc'][:, :, 0], dtype=np.int)]

    stheta = np.sin(theta)
    ctheta = np.cos(theta)

    r = np.zeros((nPats, nBands, 3), dtype=np.float32)
    r[:,:,0] = -1*stheta
    r[:,:,1] = ctheta

    p = np.zeros((nPats, nBands, 3), dtype=np.float32)
    p[:,:,0] = rho*ctheta
    p[:,:,1] = rho*stheta
    p[:,:,0] += dimf[0] * 0.5
    p[:,:,1] += dimf[1] * 0.5

    n2 = p - PC_PX.reshape(1, 1, 3)
    n = np.cross(r.reshape(nPats*nBands, 3), n2.reshape(nPats*nBands, 3) )
    norm = np.linalg.norm(n, axis=1)
    n /= norm.reshape(nPats*nBands, 1)
    n = n.reshape(nPats, nBands, 3)
    return n
