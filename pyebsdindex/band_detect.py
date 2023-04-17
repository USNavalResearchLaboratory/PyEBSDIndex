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
import numba
import numpy as np
import scipy.ndimage as scipyndim #import gaussian_filter
#from scipy.ndimage #import grey_dilation as scipy_grey_dilation
#from scipy.ndimage #import median_filter
import scipy.optimize as scipyopt

from pyebsdindex import radon_fast


tempdir = PurePath("/tmp" if platform.system() == "Darwin" else tempfile.gettempdir())
tempdir = tempdir.joinpath('numba')
environ["NUMBA_CACHE_DIR"] = str(tempdir)

RADEG = 180.0/np.pi


class BandDetect():
  def __init__(
    self,
    patterns=None,
    patDim=None,
    nTheta=180,
    nRho=90,
    tSigma=None,
    rSigma=None,
    rhoMaskFrac=0.1,
    nBands=9,
    **kwargs
):
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
    self.peakPad = np.array([11, 11])
    self.padding = np.array([11, 11])
    self.rhoMaskFrac = rhoMaskFrac

    self.nBands = nBands
    self.EDAXIQ = False
    self.backgroundsub = None

    self.dataType = np.dtype([('id', np.int32), ('max', np.float32), \
                    ('maxloc', np.float32, (2)), ('avemax', np.float32), ('aveloc', np.float32, (2)),\
                    ('pqmax', np.float32), ('width', np.float32), ('theta', np.float32), ('rho', np.float32),
                    ('valid', np.int8)])


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
      p_dim = np.asarray(patDim, dtype=np.int64)
    if patterns is not None:
      p_dim = np.shape(patterns)[-2:]  # this will catch if someone sends in a [1 x N x M] image
    if p_dim is not None:
      if self.patDim is None:
        recalc_radon = True
        self.patDim = p_dim

      elif np.sum(np.abs(self.patDim[-2:]-p_dim[-2:]), dtype=np.int64) != 0:
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
      recalc_radon = True
      recalc_masks = True

    if self.dRho is None:
      recalc_radon = True

    if recalc_radon == True:
      #self.rhoMax = 0.5 * np.float32(np.sqrt(np.float32(self.patDim[0])**2  + np.float32(self.patDim[1])**2))
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
    if (self.rhoMaskFrac is not None):
      recalc_masks = True

    if (self.tSigma is None) and (self.dTheta is not None):
      self.tSigma = 1.0/self.dTheta
      recalc_masks = True

    if (self.rSigma is None) and (self.dRho is not None):
      self.rSigma = 0.5/np.float32(self.dRho)
      recalc_masks = True

    if recalc_masks == True:
      ksz = np.array([np.max([np.int64(4*self.rSigma), 5]), np.max([np.int64(4*self.tSigma), 5])])
      ksz = ksz + ((ksz % 2) == 0)
      kernel = np.zeros(ksz, dtype=np.float32)
      kernel[(ksz[0]/2).astype(int),(ksz[1]/2).astype(int) ] = 1
      kernel = -1.0*scipyndim.gaussian_filter(kernel, [self.rSigma, self.tSigma], order=[2,0])
      self.kernel = kernel.reshape((1,ksz[0], ksz[1]))
      #self.peakPad = np.array(np.around([ 4*ksz[0], 20.0/self.dTheta]), dtype=np.int64)
      self.peakPad = np.array(np.around([2 * ksz[0], 2 * ksz[1]]), dtype=np.int64)
      self.peakPad += 1 - np.mod(self.peakPad, 2)  # make sure we have it as odd.

    self.padding = np.array([np.max( [self.peakPad[0], self.padding[0]] ), np.max([self.peakPad[1], self.padding[1]])])

    if nBands is not None:
      self.nBands = nBands

  def collect_background(self, fileobj = None, patsIn = None, nsample = None, method = 'randomStride', sigma=None):

    back = None # default value
    # we got an array of patterns

    if patsIn is not None:
      ndim = patsIn.ndim
      if ndim == 2:
        patsIn = np.expand_dims(patsIn,axis=0)
      else:
        patsIn = patsIn
      npats = patsIn.shape[0]
      if nsample is None:
        nsample = npats
      #pshape = patsIn.shape
      if npats <= nsample:
        back = np.mean(patsIn, axis = 0)
        back = np.expand_dims(back,axis=0)
      else:
        if method.upper() == 'RANDOMSTRIDE':
          stride = np.random.choice(npats, size = nsample, replace = False )
          stride = np.sort(stride)
          back = np.mean(patsIn[stride,:,:],axis=0)
        elif method.upper() == 'EVENSTRIDE':
          stride = np.arange(0, npats, int(npats/nsample)) # not great, but maybe good enough.
          back = np.mean(patsIn[stride, :, :], axis=0)

    if (back is None) and (fileobj is not None):
      if fileobj.version is None:
        fileobj.read_header()
      npats = fileobj.nPatterns
      if nsample is None:
        nsample = npats
      if npats <= nsample:
        nsample = npats

      if method.upper() == 'RANDOMSTRIDE':
        stride = np.random.choice(npats, size = nsample, replace = False )
        stride = np.sort(stride)
      elif method.upper() == 'EVENSTRIDE':
        step = int(npats / nsample) # not great, but maybe good enough.
        stride = np.arange(0,npats, step, dypte = np.uint64)
      pat1 = fileobj.read_data(convertToFloat=True,patStartCount=[stride[0], 1],returnArrayOnly=True)[0]
      for i in stride[1:]:
        pat1 += fileobj.read_data(convertToFloat=True,patStartCount=[i, 1],returnArrayOnly=True)[0]
      back = pat1 / float(len(stride))
      #pshape = pat1.shape
    # a bit of image processing.
    if back is not None:
      back = np.squeeze(back)
      back = self.backsub_fit(back)
    self.backgroundsub = back

  def backsub_fit(self, back, mask = None):
    # This function will fit a 2D gaussian on top of a plane to the averaged set of patterns (data) that is provided.
    # It will automatically use whatever mask is defined for valid data.
    # If the gaussian fit fails to converge, it will fall back to just using the mean set of patterns for the background
    # with a warning.
    def gaussian_surf(x, y, a, x0, y0, sigx, sigy, c, d, e):
    # equation for 2D gaussian on top of a plane.
      return a * np.exp(- ((x - x0) ** 2) / (2.0 * sigx ** 2) - ((y - y0) ** 2) / (2.0 * sigy ** 2)) + c + d * x + e * y

    def fit_gauss(M, *args):
    # helper function
      x, y = M
      #arr = np.zeros(x.shape)
      return gaussian_surf(x, y, *args)

    #back = np.mean(data, axis=0) # start with the mean of all the data
    # now fit a 2D gaussian sitting on a plane.  See fuction def above.
    nx = back.shape[-1]
    ny = back.shape[-2]
    #plt.imshow(back)
    x = np.arange(nx, dtype=float)
    x = (np.broadcast_to(x.reshape(1,nx), (ny, nx))).ravel()
    y = np.arange(ny, dtype=float)
    y = (np.broadcast_to(y, (nx, ny)).T).ravel()
    if mask is None:
      # make a circular mask - even if not EDAX, this should work OK.
      cx = (np.arange(nx) - nx*0.5)**2
      cy = (np.arange(ny) - ny*0.5)**2
      cmask = np.sqrt(np.broadcast_to(cx.reshape(1,nx), (ny, nx)) + np.broadcast_to(cy, (nx, ny)).T) < (ny*0.49)
    else:
      cmask = mask
    # need to grab only the values that are in the mask.
    wh = np.nonzero(cmask.ravel())[0]
    xwh = x[wh]
    ywh = y[wh]
    xywh = np.vstack((xwh, ywh))
    zwh = (scipyndim.median_filter(np.squeeze(back),3).ravel())[wh]
    whmx = np.unravel_index(back.argmax(), back.shape)
    minz = zwh.min()
    # initialize a guess for the parameters.
    # [gauss amplitude, max loc x, max loc y, sigx, sigy, const offset, slope x, slope y]
    p0 = [(zwh.max() - zwh.min())*0.1, whmx[1], whmx[0], nx/2.355, ny/2.355, minz, 0, 0]
    try:
      popt, pcov = scipyopt.curve_fit(fit_gauss, xywh, zwh, p0)
      backfit = (gaussian_surf(x, y, *popt)).reshape(ny, nx)
      #print(p0, popt)
    except RuntimeError:
      print('Warning: no convergence on back subtract ... using mean of the patterns.')
      print('This may not be ideal for scans with few grains across the width of the scan.')
      backfit = back
    backfit -= np.mean(backfit)
    #f, axarr = plt.subplots(1, 3)
    #f.set_size_inches(10, 4)
    #axarr[0].imshow(data[0,:,:].squeeze(), cmap='gray')
    #axarr[1].imshow(data[0,:,:].squeeze() - backfit, cmap='gray')
    #axarr[2].imshow(backfit, cmap='gray')


    return backfit


  def find_bands(self, patternsIn, verbose=0, chunksize=-1,  **kwargs):
    tic0 = timer()
    tic = timer()
    ndim = patternsIn.ndim
    if ndim == 2:
      patterns = np.expand_dims(patternsIn, axis=0)
    else:
      patterns = patternsIn

    shape = patterns.shape
    nPats = shape[0]

    bandData = np.zeros((nPats,self.nBands),dtype=self.dataType)
    if chunksize < 0:
      nchunks = 1
      chunksize = nPats
      chunk_start_end = [[0,nPats]]
    else:
      nchunks = (np.ceil(nPats / chunksize)).astype(np.compat.long)
      chunk_start_end = [[i * chunksize, (i + 1) * chunksize] for i in range(nchunks)]
      chunk_start_end[-1][1] = nPats

    # these are timers used to gauge performance
    rdntime = 0.0
    convtime = 0.0
    lmaxtime = 0.0
    blabeltime = 0.0

    for chnk in chunk_start_end:
      tic1 = timer()
      rdnNorm = self.radonPlan.radon_faster(patterns[chnk[0]:chnk[1],:,:], self.padding, fixArtifacts=False, background=self.backgroundsub)
      rdntime += timer() - tic1
      tic1 = timer()
      rdnConv = self.rdn_conv(rdnNorm)
      convtime += timer()-tic1
      tic1 = timer()
      lMaxRdn= self.rdn_local_max(rdnConv)
      lmaxtime +=  timer()-tic1
      tic1 = timer()
      bandDataChunk= self.band_label(chnk[1]-chnk[0], rdnConv, rdnNorm, lMaxRdn)
      bandData[chnk[0]:chnk[1]] = bandDataChunk
      if (verbose > 1) and (chnk[1] == nPats): # need to pull the radonconv off the gpu
        rdnConv = rdnConv[:,:,0:chnk[1]-chnk[0] ]

      blabeltime += timer() - tic1

    tottime = timer() - tic0

    if verbose > 0:
      print('Radon Time:',rdntime)
      print('Convolution Time:', convtime)
      print('Peak ID Time:', lmaxtime)
      print('Band Label Time:', blabeltime)
      print('Total Band Find Time:',tottime)
    if verbose > 1:
      plt.clf()

      if len(rdnConv.shape) == 3:
        im2show = rdnConv[self.padding[0]:-self.padding[0],self.padding[1]:-self.padding[1], -1]
      else:
        im2show = rdnConv[self.padding[0]:-self.padding[0],self.padding[1]:-self.padding[1]]

      rhoMaskTrim = np.int32(im2show.shape[0] * self.rhoMaskFrac)
      mean = np.mean(im2show[rhoMaskTrim:-rhoMaskTrim, 1:-2])
      stdv = np.std(im2show[rhoMaskTrim:-rhoMaskTrim, 1:-2])

      im2show -= mean
      im2show /= stdv
      im2show = im2show.clip(-4, None)
      im2show += 6
      im2show[0:rhoMaskTrim,:] = 0
      im2show[-rhoMaskTrim:,:] = 0
      im2show = np.fliplr(im2show)

      plt.figure()
      plt.imshow(im2show, cmap='gray', extent=[self.radonPlan.theta.min(), self.radonPlan.theta.max(),
                                               self.radonPlan.rho.min(), self.radonPlan.rho.max()],
                 interpolation='none', zorder=1, aspect='auto')
      width = bandData['width'][-1, :]
      width /= width.min()
      width *= 2
      xplt = np.squeeze(
        180.0 - np.interp(bandData['aveloc'][-1, :, 1]+0.5, np.arange(self.radonPlan.nTheta), self.radonPlan.theta))
      yplt = np.squeeze(
        -1.0 * np.interp(bandData['aveloc'][-1, :, 0]-0.5, np.arange(self.radonPlan.nRho), self.radonPlan.rho))

      plt.scatter(y=yplt, x=xplt, c='r', s=width, zorder=2)

      for pt in range(self.nBands):
        plt.annotate(str(pt + 1), np.squeeze([xplt[pt]+4, yplt[pt]]), color='yellow')
      plt.xlim(0,180)
      plt.ylim(-self.rhoMax, self.rhoMax)


    return bandData

  def radonPad(self,radon,rPad=0,tPad = 0, mirrorTheta = True):
    # function for padding the radon transform
    # theta padding (tPad) will use the symmetry of the radon and will vertical flip the transform and place it on
    # the other side.
    # rho padding simply repeats the top/bottom rows into the padded region
    # negative padding values will result in a crop (remove the padding).
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



  def rdn_conv(self, radonIn):
    tic = timer()
    shp = radonIn.shape
    if len(shp) == 2:
      radon = radonIn.reshape(shp[0],shp[1],1)
      shp = radon.shape
    else:
      radon = radonIn
    shprdn = radon.shape
    #rdnNormP = self.radonPad(radon,rPad=0,tPad=self.peakPad[1],mirrorTheta=True)
    if self.padding[1] > 0:
      radon[:,0:self.padding[1],:] = np.flip(radon[:,-2 * self.padding[1]:-self.padding[1],:],axis=0)
      radon[:,-self.padding[1]:,:] = np.flip(radon[:,self.padding[1]:2 * self.padding[1],:],axis=0)
    # pad again for doing the convolution
    #rdnNormP = self.radonPad(rdnNormP,rPad=self.peakPad[0],tPad=self.peakPad[1],mirrorTheta=False)
    if self.padding[0] > 0:
      radon[0:self.padding[0], :,:] = radon[self.padding[0],:,:].reshape(1,shp[1], shp[2])
      radon[-self.padding[0]:, :,:] = radon[-self.padding[0]-1, :,:].reshape(1, shp[1],shp[2])


    rdnConv = np.zeros_like(radon)

    for i in range(shp[2]):
      rdnConv[:,:,i] = -1.0 * scipyndim.gaussian_filter(np.squeeze(radon[:,:,i]),[self.rSigma,self.tSigma],order=[2,0])

    #print(rdnConv.min(),rdnConv.max())
    mns = (rdnConv[self.padding[0]:shprdn[1]-self.padding[0],self.padding[1]:shprdn[1]-self.padding[1],:]).min(axis=0).min(axis=0)

    rdnConv -= mns.reshape((1,1, shp[2]))
    rdnConv = rdnConv.clip(min=0.0)

    return rdnConv

  def rdn_local_max(self, rdn, clparams=None, rdn_gpu=None, use_gpu=False):

    shp = rdn.shape
    # find the local max
    lMaxK = (self.peakPad[0],self.peakPad[1],1)

    lMaxRdn = scipyndim.grey_dilation(rdn,size=lMaxK)
    #lMaxRdn[:,:,0:self.peakPad[1]] = 0
    #lMaxRdn[:,:,-self.peakPad[1]:] = 0
    #location of the max is where the local max is equal to the original.
    lMaxRdn = lMaxRdn == rdn

    rhoMaskTrim = np.int32((shp[0] - 2 * self.padding[0]) * self.rhoMaskFrac + self.padding[0])
    lMaxRdn[0:rhoMaskTrim,:,:] = 0
    lMaxRdn[-rhoMaskTrim:,:,:] = 0
    lMaxRdn[:,0:self.padding[1],:] = 0
    lMaxRdn[:,-self.padding[1]:,:] = 0
    #print("Traditional:",timer() - tic)
    return lMaxRdn


  def band_label(self,nPats,rdnConvIn,rdnNormIn,lMaxRdnIn):
    bandData = np.zeros((nPats,self.nBands),dtype=self.dataType)


    bdat = self.band_label_numba(
      np.int64(self.nBands),
      np.int64(nPats),
      np.int64(self.nRho),
      np.int64(self.nTheta),
      rdnConvIn,
      rdnConvIn,
      lMaxRdnIn
    )

    bandData['max']    = bdat[0][0:nPats, :]
    bandData['avemax'] = bdat[1][0:nPats, :]
    bandData['maxloc'] = bdat[2][0:nPats, :, :]
    bandData['aveloc'] = bdat[3][0:nPats, :, :]
    bandData['valid']  = bdat[4][0:nPats, :]
    bandData['width']  = bdat[5][0:nPats, :]
    bandData['maxloc'] -= self.padding.reshape(1, 1, 2)
    bandData['aveloc'] -= self.padding.reshape(1, 1, 2)

    return bandData

  @staticmethod
  @numba.jit(nopython=True,fastmath=True,cache=True,parallel=False)
  def band_label_numba(nBands,nPats,nRho,nTheta,rdnConv,rdnPad,lMaxRdn):
    nB = np.int64(nBands)
    nP = np.int64(nPats)

    shp = rdnPad.shape

    bandData_max = np.zeros((nP,nB),dtype=np.float32) - 2.0e6  # max of the convolved peak value
    bandData_avemax = np.zeros((nP,nB),
                               dtype=np.float32) - 2.0e6  # mean of the nearest neighborhood values around the max
    bandData_valid = np.zeros((nP, nB), dtype=np.int8)
    bandData_maxloc = np.zeros((nP,nB,2),dtype=np.float32)  # location of the max within the radon transform
    bandData_aveloc = np.zeros((nP,nB,2),
                               dtype=np.float32)  # location of the max based on the nearest neighbor interpolation
    bandData_width = np.zeros((nP,nB),dtype=np.float32) # a metric of the band width

    #nnc = np.array([-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2],dtype=np.float32)
    #nnr = np.array([-1,-1,-1,-1,-1,0,0,0,0,0,1,1,1,1,1],dtype=np.float32)
    #nnN = numba.float32(15)
    nnc = np.array([-1,0,1,-1,0,1,-1,0,1],dtype=np.float32)
    nnr = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1], dtype=np.float32)
    nnN = numba.float32(9)
    for q in range(nPats):
      # rdnConv_q = np.copy(rdnConv[:,:,q])
      # rdnPad_q = np.copy(rdnPad[:,:,q])
      # lMaxRdn_q = np.copy(lMaxRdn[:,:,q])
      # peakLoc = np.nonzero((lMaxRdn_q == rdnPad_q) & (rdnPad_q > 1.0e-6))
      peakLoc = lMaxRdn[:,:,q].nonzero()
      indx1D = peakLoc[1] + peakLoc[0] * shp[1]
      temp = (rdnConv[:,:,q].ravel())[indx1D]
      srt = np.argsort(temp)
      nBq = nB if (len(srt) > nB) else len(srt)
      for i in numba.prange(nBq):
        r = np.int32(peakLoc[0][srt[-1 - i]])
        c = np.int32(peakLoc[1][srt[-1 - i]])
        bandData_maxloc[q,i,:] = np.array([r,c])
        bandData_max[q,i] = rdnPad[r,c,q]
        bandData_width[q, i] = 1.0 / (bandData_max[q,i] - 0.5* (rdnPad[r+1, c, q] + rdnPad[r-1, c, q]) + 1.0e-12)

        #center of mass peak localization
        #nn = rdnConv[r - 1:r + 2,c - 2:c + 3,q].ravel()
        #sumnn = (np.sum(nn) + 1.e-12)
        #nn /= sumnn
        #bandData_avemax[q,i] = sumnn / nnN
        #rnn = np.sum(nn * (np.float32(r) + nnr))
        #cnn = np.sum(nn * (np.float32(c) + nnc))

        # taylor expansion quadratic
        nn = rdnConv[r - 1:r + 2,c - 1:c + 2,q].copy()
        sumnn = (np.sum(nn) + 1.e-12)
        nn /= sumnn
        bandData_avemax[q,i] = sumnn / nnN
        # rnn = np.sum(nn * (np.float32(r) + nnr))
        # cnn = np.sum(nn * (np.float32(c) + nnc))
        dx  = 0.5*(nn[1,2] - nn[1,0])
        dy  = 0.5*(nn[2,1] - nn[0,1])
        dxx = nn[1,2] + nn[1,0] - 2 * nn[1,1]
        dyy = nn[2,1] + nn[0,1] - 2 * nn[1,1]
        dxy = 0.25*(nn[2,2] - nn [0,2] - nn[2,0] + nn[0,0])
        #det = 1.0 / (dxx * dyy - dxy * dxy)
        det = (dxx * dyy - dxy * dxy)
        det = det if np.fabs(det) > 1e-12 else 1.0e-12
        det = 1.0/det
        dc =  (dyy * dx - dxy * dy) * det
        rc = (dxx * dy - dxy * dx) * det
        dc = max(-1.0, dc) ; rc = max(-1.0, rc)
        dc = min(1.0, dc) ;  rc = min(1.0, rc)
        cnn = c - dc
        rnn = r - rc
        bandData_aveloc[q,i,:] = np.array([rnn,cnn])
        bandData_valid[q,i] = 1

    return bandData_max,bandData_avemax,bandData_maxloc,bandData_aveloc, bandData_valid, bandData_width

def getopenclparam(**kwargs): # dummy function to maintain compatability with openCL version
  return None


