import numpy as np
from os import path#, environ
import numba
from scipy.ndimage import gaussian_filter
from scipy.ndimage.morphology import grey_dilation as scipy_grey_dilation
import radon_fast
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import pyopencl as cl
#environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

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
    self.dataType = np.dtype([('id', np.int32), ('max', np.float32), \
                    ('maxloc', np.float32, (2)), ('avemax', np.float32), ('aveloc', np.float32, (2)),\
                    ('pqmax', np.float32)])
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
      self.radonPlan = radon_fast.Radon(imageDim=self.patDim, nTheta=self.nTheta, nRho=self.nRho, rhoMax=self.rhoMax)
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
      ksz = ksz + ((ksz % 2) == 0)
      kernel = np.zeros(ksz, dtype=np.float32)
      kernel[(ksz[0]/2).astype(int),(ksz[1]/2).astype(int) ] = 1
      kernel = -1.0*gaussian_filter(kernel, [self.rSigma, self.tSigma], order=[2,0])
      self.kernel = kernel.reshape((1,ksz[0], ksz[1]))
      #self.peakPad = np.array(np.around([ 4*ksz[0], 20.0/self.dTheta]), dtype=np.int)
      self.peakPad = np.array(np.around([4 * ksz[0], 4 * ksz[1]]), dtype=np.int)
      self.peakMask = np.ones(self.peakPad, dtype=np.int32)
      self.rhoMask = np.ones([1,self.nRho,self.nTheta+2*self.peakPad[1] ], dtype=np.float32)
      if self.rhoMaskFrac > 0:
        self.rhoMask[:,0:np.int(np.floor(self.nRho*self.rhoMaskFrac)), :] = 0.0
        self.rhoMask[:,-np.int(np.floor(self.nRho*self.rhoMaskFrac)):, :] = 0.0

    if nBands is not None:
      self.nBands = nBands

  def find_bands(self, patternsIn, faster=False, verbose=False):
    tic0 = timer()
    tic = timer()
    ndim = patternsIn.ndim
    if ndim == 2:
      patterns = np.expand_dims(patternsIn, axis=0)
    else:
      patterns = patternsIn

    shape = patterns.shape
    nPats = shape[0]

    eps = 1.e-6
    tic1 = timer()
    rdnNorm = radon_fast.radon_fasterCL(self.radonPlan, patterns,fixArtifacts=True)

    #rdnNorm = rdn*self.rdnNorm
    #print("Radon:",timer() - tic)
    tic = timer()
    rdnConv, lMaxRdn = self.band_conv(rdnNorm)
    tic = timer()
    #rdnConv,lMaxRdn = self.band_convCL(rdnNorm)
    #return rdnConv, lMaxRdn, rdnConv2, lMaxRdn2
    #print(rdnPad.shape,rdnConv2.shape)
    #print(lMaxRdn.shape,lMaxRdn2.shape)
    #
    # nnx = np.array([-2,-1,0,1,2, -2,-1,0,1,2, -2,-1,0,1,2], dtype = np.float32)
    # nny = np.array([-1,-1,-1,-1,-1, 0,0,0,0,0, 1,1,1,1,1], dtype = np.float32)
    # shp = rdnPad.shape
    # for q in range(nPats):
    #   rdnPad_q = rdnPad[q,:,:]
    #   lMaxRdn_q = lMaxRdn[q,:,:]
    #   peakLoc = np.nonzero((lMaxRdn_q == rdnPad_q) & (rdnPad_q > 0.001))
    #   #srt = np.argsort(rdnPad_q[peakLoc[0], peakLoc[1]])
    #   indx1D = peakLoc[1] + peakLoc[0] * shp[2]
    #   temp = (rdnPad_q.ravel())[indx1D]
    #   srt = np.argsort(temp)
    #   nBq = self.nBands if (len(srt) > self.nBands) else len(srt)
    #   for i in range(nBq):
    #     x = np.int32(peakLoc[0][srt[-1-i]])
    #     y = np.int32(peakLoc[1][srt[-1-i]])
    #     bandData['maxloc'][q,i,:] = np.array([x,y])
    #     bandData['max'][q,i] = rdnPad_q[x,y]
    #     nn = rdnPad_q[x-2:x+3,y-1:y+2].ravel()
    #     bandData['avemax'][q,i] = np.mean(nn)
    #     xnn = np.sum(nn*(np.float32(x)+nnx))/np.sum(nn)
    #     ynn = np.sum(nn * (np.float32(y) + nny)) / np.sum(nn)
    #     bandData['aveloc'][q,i,:] = np.array([xnn,ynn])


    #bdat = self.band_label(np.int(self.nBands),np.int(nPats),np.int(self.nRho),np.int(self.nTheta),rdnPad,self.peakPad,self.peakMask)
    bandData = np.zeros((nPats,self.nBands),dtype=self.dataType)
    bdat = self.band_label(np.int(self.nBands),np.int(nPats),np.int(self.nRho),np.int(self.nTheta),rdnConv,lMaxRdn)
    bandData['max'] = bdat[0]
    bandData['avemax'] = bdat[1]
    bandData['maxloc'] = bdat[2]
    bandData['aveloc'] = bdat[3]

    # peakloc = np.zeros((nPats,self.nRho + 2 * self.peakPad[0],self.nTheta + 2 * self.peakPad[1]),dtype=np.int)
    # peaklocmask = np.ones((nPats,self.nRho + 2 * self.peakPad[0],self.nTheta + 2 * self.peakPad[1]),dtype=np.int)
    # peaklocmask[:,:,0:self.peakPad[1]] = 0
    # peaklocmask[:,:,-self.peakPad[1]:] = 0
    # peakmask_offset = np.array(-1 * np.floor(self.peakPad * 0.5),dtype=np.int)
    # peakmask_offset = np.broadcast_to(peakmask_offset,(nPats,2))
    #
    # for i in range(self.nBands):
    #
    #   mxloc = (rdnPad*(peakloc == 0)*peaklocmask).reshape((nPats, nlayer)).argmax(axis=1)
    #   #plt.imshow( (rdnPad*(peakloc == 0)*peaklocmask)[13,:,:])
    #   bandData['max'][ :, i] = (rdnPad.reshape((nPats, nlayer)))[np.arange(nPats), mxloc]
    #   nnindx = mxloc.reshape((nPats,1)) + nnmask.reshape((1,nn))
    #   rdnNNv = np.take(rdnPad.reshape((nPats, nlayer)), nnindx)
    #   bandData['avemax'][:,i] = np.mean(rdnNNv, axis =1)
    #   mxloc2 = np.array(np.unravel_index(mxloc, rdnPad.shape[-2:])).T
    #   bandData['maxloc'][:,i,:] = mxloc2
    #   nnloc = np.array(np.unravel_index(nnindx, rdnPad.shape[-2:]), dtype=np.float32).transpose([1,2,0])
    #   nnloc *= rdnNNv.reshape(nPats,nn,1)
    #   nnloc = np.sum(nnloc, axis = 1)
    #   nnloc /= np.sum(rdnNNv, axis=1).reshape(nPats,1)
    #   bandData['aveloc'][:, i, :] = nnloc.reshape(nPats, 2)
    #   mxloc2 += peakmask_offset
    #   tempmask = np.array(self.peakMask*(i+1))
    #
    #   for j in range(nPats):
    #     #print(mxloc2[j,:])
    #     peakloc[j,mxloc2[j,0]:mxloc2[j,0]+mskSz[0], mxloc2[j,1]:mxloc2[j,1]+mskSz[1]] = np.array(tempmask)
    #
    #   #plt.imshow(peakloc[-1,:,:])
    #   flipl = np.flip(peakloc[:,:,-2*self.peakPad[1]:], axis=1)
    #   rnflip = peakloc[:,:, 0: 2*self.peakPad[1]]
    #   rnflip = np.where(rnflip >= flipl,rnflip, flipl)
    #   peakloc[:, :, 0: 2 * self.peakPad[1]] = rnflip
    #
    #   flipr = np.flip(peakloc[:, :,0:2 * self.peakPad[1]], axis=1)
    #   lnflip = peakloc[:, :, -2 * self.peakPad[1]:]
    #   lnflip = np.where(lnflip >= flipr, lnflip,flipr)
    #   peakloc[:, :, -2 * self.peakPad[1]:] = lnflip
    #print('loop: ',timer() - tic)
    #print(np.max(np.abs(bandData['max']-dave[0])), np.max(np.abs(bandData['avemax']-dave[1])),
    #      np.max(np.abs(bandData['maxloc']-dave[2])), np.max(np.abs(bandData['aveloc']-dave[3])) )
    #rdnConv = rdnConv[:, :, self.peakPad[1]: -self.peakPad[1]]

    bandData['maxloc'] -= self.peakPad.reshape(1,1,2)
    bandData['aveloc'] -= self.peakPad.reshape(1, 1, 2)
    for j in range(nPats):
      bandData['pqmax'][j,:] = \
        rdnNorm[j, (bandData['aveloc'][j,:,0]).astype(int),(bandData['aveloc'][j,:,1]).astype(int)]
    #bandData['max'] += mns.reshape(nPats,1)
    #print("BandLabel:",timer() - tic)

    if verbose == True:
      print('Total Band Find Time:',timer() - tic0)
      plt.clf()
      plt.imshow(rdnConv[-1,self.peakPad[0]:-self.peakPad[0],self.peakPad[1]:-self.peakPad[1]], origin='lower')
      #plt.scatter(y = bandData['aveloc'][-1,:,0], x = bandData['aveloc'][-1,:,1]+self.peakPad[1], c ='r', s=5)
      plt.scatter(y=bandData['aveloc'][-1,:,0],x=bandData['aveloc'][-1,:,1],c='r',s=5)
      plt.show()


    return bandData

  def radonPad(self,radon,rPad=0,tPad = 0, mirrorTheta = True):
    shp = radon.shape
    if (tPad==0)&(rPad == 0):
      return radon


    if (mirrorTheta == True)&(tPad > 0):
      rdnP = np.zeros((shp[0],shp[1],shp[2] + 2 * tPad),dtype=np.float32)
      rdnP[:,:,tPad:-tPad] = radon
        # now pad out the theta dimension with the flipped-wrapped radon -- accounting for radon periodicity
      rdnP[:,:,0:tPad] = np.flip(rdnP[:,:,-2 *tPad:-tPad],axis=1)
      rdnP[:,:,-tPad:] = np.flip(rdnP[:,:,tPad:2 * tPad],axis=1)
    else:
      if tPad > 0:
        rdnP = np.pad(radon,((0,),(0,),(tPad,)),mode='edge')
      elif tPad < 0:
        rdnP = radon[:,:,-tPad:tPad]
      else:
        rdnP = radon

    if rPad > 0:
      rdnP =  np.pad(rdnP,((0,),(rPad,),(0,)),mode='edge')
    elif rPad < 0:
      rdnP = rdnP[:,-rPad:rPad, :]

    return rdnP



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
    #theta = self.radonPlan.theta[np.array(bandData['maxloc'][:,:,1], dtype=np.int)]/RADEG
    #rho = self.radonPlan.rho[np.array(bandData['maxloc'][:, :, 0], dtype=np.int)]

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

  @staticmethod
  @numba.jit(nopython=True,fastmath=True,cache=True,parallel=False)
  def band_label(nBands, nPats, nRho, nTheta, rdnPad,  lMaxRdn ):
    nB = np.int(nBands)
    nP = np.int(nPats)
    nR = np.int(nRho)
    nT = np.int(nTheta)
    shp  = rdnPad.shape
    bandData_max = np.zeros((nP,nB), dtype = np.float32) # max of the convolved peak value
    bandData_avemax = np.zeros((nP,nB), dtype = np.float32) # mean of the nearest neighborhood values around the max
    bandData_maxloc = np.zeros((nP,nB,2), dtype = np.float32)-20.0 # location of the max within the radon transform
    bandData_aveloc = np.zeros((nP,nB,2), dtype = np.float32)-20.0 # location of the max based on the nearest neighbor interpolation

    nnx = np.array([-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2],dtype=np.float32)
    nny = np.array([-1,-1,-1,-1,-1,0,0,0,0,0,1,1,1,1,1],dtype=np.float32)

    for q in range(nPats):
      rdnPad_q = rdnPad[q,:,:]
      lMaxRdn_q = lMaxRdn[q,:,:]
      #peakLoc = np.nonzero((lMaxRdn_q == rdnPad_q) & (rdnPad_q > 1.0e-6))
      peakLoc = np.nonzero(lMaxRdn_q)
      indx1D = peakLoc[1] + peakLoc[0] * shp[2]
      temp = (rdnPad_q.ravel())[indx1D]
      srt = np.argsort(temp)
      nBq = nB if (len(srt) > nB) else len(srt)
      for i in range(nBq):
        x = np.int32(peakLoc[0][srt[-1 - i]])
        y = np.int32(peakLoc[1][srt[-1 - i]])
        bandData_maxloc[q,i,:] = np.array([x,y])
        bandData_max[q,i] = rdnPad_q[x,y]
        nn = rdnPad_q[x - 2:x + 3,y - 1:y + 2].ravel()
        bandData_avemax[q,i] = np.mean(nn)
        xnn = np.sum(nn * (np.float32(x) + nnx)) / (np.sum(nn) + 1.0e-12)
        ynn = np.sum(nn * (np.float32(y) + nny)) / (np.sum(nn) + 1.0e-12)
        bandData_aveloc[q,i,:] = np.array([xnn,ynn])

    return bandData_max,bandData_avemax,bandData_maxloc,bandData_aveloc

  def band_conv(self, radonIn):
    tic = timer()
    ## the code below was replaced by a gpu convolution routine.
    shp = radonIn.shape
    if len(shp) == 2:
      radon = radonIn.reshape(1,shp[0],shp[1])
      shp = radonIn.shape
    else:
      radon = radonIn

    rdnNormP = self.radonPad(radon,rPad=0,tPad=self.peakPad[1],mirrorTheta=True)
    # rdnNormP = np.zeros((nPats,self.nRho,self.nTheta+2*self.peakPad[1]), dtype=np.float32)
    # rdnNormP[:,:,self.peakPad[1]:-self.peakPad[1]] = rdnNorm
    # now pad out the theta dimension with the flipped-wrapped radon -- accounting for radon periodicity
    # rdnNormP[:, :, 0:self.peakPad[1]] = np.flip(rdnNormP[:, :, -2*self.peakPad[1]:-self.peakPad[1]], axis=1)
    # rdnNormP[:, :,-self.peakPad[1]:] = np.flip(rdnNormP[:, :, self.peakPad[1]:2*self.peakPad[1]], axis = 1)

    # pad again for doing the convolution -- this will be removed after the convolution
    rdnNormP = self.radonPad(rdnNormP,rPad=self.peakPad[0],tPad=self.peakPad[1],mirrorTheta=False)
    # rdnNormP = np.pad(rdnNormP, ((0,),(self.peakPad[0],),(self.peakPad[1],)), mode ='edge' )


    rdnConv = np.zeros_like(rdnNormP)

    for i in range(shp[0]):
      rdnConv[i,:,:] = -1.0 * gaussian_filter(np.squeeze(rdnNormP[i,:,:]),[self.rSigma,self.tSigma],order=[2,0])
    # rdnConv = gputools.convolve(rdnNormP, self.kernel)
    # remove the convolution padding.

    rdnConv = self.radonPad(rdnConv,rPad=-self.peakPad[0],tPad=-self.peakPad[1],mirrorTheta=False)
    rdnNormP = self.radonPad(rdnNormP,rPad=-self.peakPad[0],tPad=-self.peakPad[1],mirrorTheta=False)
    # rdnConv = rdnConv[:, self.peakPad[0]: -self.peakPad[0], self.peakPad[1]: -self.peakPad[1]]
    # rdnNormP = rdnNormP[:, self.peakPad[0]: -self.peakPad[0], self.peakPad[1]: -self.peakPad[1]]

    mns = rdnConv.min(axis=1).min(axis=1)
    rdnConv -= mns.reshape((shp[0],1,1))

    rdnConv *= (rdnNormP > 0).astype(np.float32) / (rdnNormP + 1e-12)
    rdnPad = np.array(rdnConv)
    rdnPad *= self.rhoMask

    rdnPad = np.pad(rdnPad,((0,),(self.peakPad[0],),(0,)),mode='constant',constant_values=0.0)

    nnmask = np.array([-2,-1,0,1,2,self.nTheta + 2 * self.peakPad[1],-1 * (self.nTheta + 2 * self.peakPad[1])])
    nn = nnmask.size
    nlayer = rdnPad.shape[-2] * rdnPad.shape[-1]
    mskSz = np.array(self.peakMask.shape)
    # print("Conv:",timer() - tic)
    # tic = timer()

    lMaxK = (1,self.peakMask.shape[0],self.peakMask.shape[1])

    lMaxRdn = scipy_grey_dilation(rdnPad,size=lMaxK)
    lMaxRdn[:,:,0:self.peakPad[1]] = 0
    lMaxRdn[:,:,-self.peakPad[1]:] = 0

    lMaxRdn = lMaxRdn == rdnPad

    rhoMaskTrim = np.int32((shp[1] - 2 * self.peakPad[0]) * self.rhoMaskFrac + self.peakPad[0])
    lMaxRdn[:,0:rhoMaskTrim,:] = 0
    lMaxRdn[:,-rhoMaskTrim:,:] = 0
    lMaxRdn[:,:,0:self.peakPad[1]] = 0
    lMaxRdn[:,:,-self.peakPad[1]:] = 0

    #print("Traditional:",timer() - tic)
    return rdnPad, lMaxRdn



  def band_convCL(self, radonIn):
    # this will run sequential convolutions and identify the peaks in the convolution
    def preparray(array):
      return np.require(array,None,"C")

    shp = radonIn.shape
    if len(shp) == 2:
      radonIn = radonIn.reshape(1,shp[0], shp[1])
      shp = radonIn.shape

    radon =  self.radonPad(radonIn,rPad=self.peakPad[0],tPad=self.peakPad[1],mirrorTheta=True)
    shp = radon.shape

    gpu = cl.get_platforms()[0].get_devices(device_type=cl.device_type.GPU)
    #if len(gpu) == 0:  # fall back to the numba implementation
    #  return self.radon_faster(radon,fixArtifacts=fixArtifacts)
    # apparently it is very difficult to get a consistent ordering of multiple GPU systems.
    # my lazy way to do this is to assign them randomly, and figure it will even out in the long run
    gpuIdx = np.random.choice(len(gpu))
    ctx = cl.Context(devices={gpu[gpuIdx]})
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags

    resultConv = np.full_like(radon,-1.0e12, dtype = np.float32)
    resultPeakLoc = np.full_like(radon,-1.0e12, dtype = np.float32)
    resultPeakLoc2 = np.full_like(radon,0,dtype=np.uint8)

    rdn_gpu = cl.Buffer(ctx,mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf=radon)
    rdnConv_gpu = cl.Buffer(ctx,mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf=resultConv)

    # for now I will assume that the kernel(s) can fit in local memory on the GPU
    # also going to assume that there is only one kernel -- this will be something to fix soon.
    k0 = self.kernel[0,:,:]
    kshp = np.asarray(k0.shape, dtype = np.int32)
    pad = kshp/2
    kern_gpu = cl.Buffer(ctx,mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf=k0)

    kernel_location = path.dirname(__file__)
    prg = cl.Program(ctx,open(path.join(kernel_location,'bandfindConvol.cl')).read()).build()


    prg.convolution3d2d(queue,(np.int32(shp[2]-2*pad[1]), np.int32(shp[1]-2*pad[0]), shp[0]),None,
                        rdn_gpu, kern_gpu,np.int32(shp[2]),np.int32(shp[1]),np.int32(shp[0]),
                        np.int32(kshp[1]), np.int32(kshp[0]), np.int32(pad[1]), np.int32(pad[0]), rdnConv_gpu)
    queue.finish()
    cl.enqueue_copy(queue,resultConv,rdnConv_gpu, is_blocking=True).wait()

    # going to reuse the original rdn buffer
    cl.enqueue_copy(queue,rdn_gpu,resultPeakLoc,is_blocking=True).wait()
    #rdn_gpu.release()
    #resultPeakLoc_gpu = cl.Buffer(ctx,mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf=resultPeakLoc)
    resultPeakLoc2_gpu = cl.Buffer(ctx,mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf=resultPeakLoc2)
    prg.morphDilateKernel(queue,(shp[2], shp[1], shp[0]),None,rdnConv_gpu,np.int32(shp[2]),np.int32(shp[1]),np.int32(shp[0]),
                          np.int32(self.peakMask.shape[1]), np.int32(self.peakMask.shape[0]), rdn_gpu)
    queue.finish()
    prg.im1NEim2(queue,(shp[2],shp[1],shp[0]),None,rdnConv_gpu,rdn_gpu,resultPeakLoc2_gpu,
                 np.int32(shp[2]),np.int32(shp[1]),np.int32(shp[0]))

    mns = (resultConv[:,self.peakPad[0]:-self.peakPad[0], self.peakPad[1]:-self.peakPad[1]]).min(axis=1).min(axis=1)
    resultConv -= mns.reshape((shp[0],1,1))
    resultConv = resultConv.clip(min = 0.0)

    queue.finish()
    #cl.enqueue_copy(queue,resultPeakLoc,rdn_gpu,is_blocking=True).wait()
    cl.enqueue_copy(queue,resultPeakLoc2,resultPeakLoc2_gpu,is_blocking=True).wait()

    #resultConv = self.radonPad(resultConv,rPad=-self.peakPad[0],tPad=0,mirrorTheta=False)
    #resultPeakLoc2 = self.radonPad(resultPeakLoc2,rPad=-self.peakPad[0],tPad=0,mirrorTheta=False)


    #resultPeakLoc2 *= np.asarray(self.rhoMask, dtype=np.uint8)
    rhoMaskTrim = np.int32((shp[1] - 2*self.peakPad[0])*self.rhoMaskFrac + self.peakPad[0])
    resultPeakLoc2[:,0:rhoMaskTrim, :] = 0
    resultPeakLoc2[:,-rhoMaskTrim:,:] = 0
    resultPeakLoc2[:,:,0:self.peakPad[1]] = 0
    resultPeakLoc2[:,:,-self.peakPad[1]:] = 0


    return resultConv, resultPeakLoc2






