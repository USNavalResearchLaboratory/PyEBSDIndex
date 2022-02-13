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
import pyopencl as cl
from scipy.ndimage import gaussian_filter
from scipy.ndimage import grey_dilation as scipy_grey_dilation

from pyebsdindex import openclparam, radon_fast


#from os import environ
#environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
tempdir = PurePath("/tmp" if platform.system() == "Darwin" else tempfile.gettempdir())
tempdir = tempdir.joinpath('numba')
environ["NUMBA_CACHE_DIR"] = str(tempdir)

RADEG = 180.0/np.pi



class BandDetect:
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
    clOps=[True, True, True, True]
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
                    ('pqmax', np.float32), ('valid', np.int8)])

    self.CLOps = [False, False, False, False]
    if clOps is not None:
      self.CLOps =  clOps

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
    if (self.rhoMaskFrac is not None):
      recalc_masks = True

    if (self.tSigma is None) and (self.dTheta is not None):
      self.tSigma = 1.0/self.dTheta
      recalc_masks = True

    if (self.rSigma is None) and (self.dRho is not None):
      self.rSigma = 0.25/np.float32(self.dRho)
      recalc_masks = True

    if recalc_masks == True:
      ksz = np.array([np.max([np.int64(4*self.rSigma), 5]), np.max([np.int64(4*self.tSigma), 5])])
      ksz = ksz + ((ksz % 2) == 0)
      kernel = np.zeros(ksz, dtype=np.float32)
      kernel[(ksz[0]/2).astype(int),(ksz[1]/2).astype(int) ] = 1
      kernel = -1.0*gaussian_filter(kernel, [self.rSigma, self.tSigma], order=[2,0])
      self.kernel = kernel.reshape((1,ksz[0], ksz[1]))
      #self.peakPad = np.array(np.around([ 4*ksz[0], 20.0/self.dTheta]), dtype=np.int64)
      self.peakPad = np.array(np.around([3 * ksz[0], 4 * ksz[1]]), dtype=np.int64)
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
      pat1 = fileobj.read_data(convertToFloat=True,patStartCount=[stride[0], 1],returnArrayOnly=True)
      for i in stride[1:]:
        pat1 += fileobj.read_data(convertToFloat=True,patStartCount=[i, 1],returnArrayOnly=True)
      back = pat1 / float(len(stride))
      #pshape = pat1.shape
    # a bit of image processing.
    if back is not None:
      #if sigma is None:
       #sigma = 2.0 * float(pshape[-1]) / 80.0
      #back[0,:,:] = gaussian_filter(back[0,:,:], sigma = sigma )
      back -= np.mean(back)
    self.backgroundsub = back

  def find_bands(self, patternsIn, verbose=0, clparams=None, chunksize=528):
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
    else:
      nchunks = (np.ceil(nPats / chunksize)).astype(np.compat.long)

    chunk_start_end = [[i * chunksize,(i + 1) * chunksize] for i in range(nchunks)]
    chunk_start_end[-1][1] = nPats
    # these are timers used to gauge performance
    rdntime = 0.0
    convtime = 0.0
    lmaxtime = 0.0
    blabeltime = 0.0

    for chnk in chunk_start_end:
      tic1 = timer()
      rdnNorm, clparams, rdnNorm_gpu = self.calc_rdn(patterns[chnk[0]:chnk[1],:,:], clparams, use_gpu=self.CLOps[0])
      if (self.EDAXIQ == True):
        if rdnNorm is None:
          nTp = self.nTheta + 2 * self.padding[1]
          nRp = self.nRho + 2 * self.padding[0]
          nImCL = int(rdnNorm_gpu.size/(nTp*nRp*4))
          rdnNorm = np.zeros((nRp,nTp,nImCL),dtype=np.float32)
          cl.enqueue_copy(clparams.queue,rdnNorm,rdnNorm_gpu,is_blocking=True)

      if self.CLOps[1] == False:
        rdnNorm_gpu  = None
        clparams = None
      rdntime += timer() - tic1
      tic1 = timer()
      rdnConv, clparams, rdnConv_gpu = self.rdn_conv(rdnNorm, clparams, rdnNorm_gpu, use_gpu=self.CLOps[1])
      if self.CLOps[2] == False:
        rdnConv_gpu = None
        clparams = None
      convtime += timer()-tic1
      tic1 = timer()
      lMaxRdn, lMaxRdn_gpu = self.rdn_local_max(rdnConv, clparams, rdnConv_gpu, use_gpu=self.CLOps[2])
      lmaxtime +=  timer()-tic1
      tic1 = timer()


      bandDataChunk, rdnConvBuf = self.band_label(chnk[1]-chnk[0], rdnConv, rdnNorm, lMaxRdn,
                                          rdnConv_gpu,rdnConv_gpu,lMaxRdn_gpu,
                                          use_gpu = self.CLOps[3], clparams=clparams )
      bandData[chnk[0]:chnk[1]] = bandDataChunk
      if (verbose > 1) and (chnk[1] == nPats): # need to pull the radonconv off the gpu
        if isinstance(rdnConvBuf , cl.Buffer):
          nTp = self.nTheta + 2 * self.padding[1]
          nRp = self.nRho + 2 * self.padding[0]
          nImCL = int(rdnConvBuf.size / (nTp * nRp * 4))
          rdnConv = np.zeros((nRp,nTp,nImCL),dtype=np.float32)
          cl.enqueue_copy(clparams.queue,rdnConv,rdnConvBuf,is_blocking=True)
        else:
          rdnConv = rdnConvBuf
        rdnConv = rdnConv[:,:,0:chnk[1]-chnk[0] ]




      # for j in range(nPats):
      #   bandData['pqmax'][j,:] = \
      #     rdnNorm[(bandData['aveloc'][j,:,0]).astype(int),(bandData['aveloc'][j,:,1]).astype(int), j]

      blabeltime += timer() - tic1

    tottime = timer() - tic0
    # going to manually clear the clparams queue -- this should clear the memory of the queue off the GPU
    if clparams is not None:
      clparams.queue.finish()
      clparams.queue = None

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

      #im2show -= mean
      #im2show /= stdv
      #im2show += 7
      im2show[0:rhoMaskTrim,:] = 0
      im2show[-rhoMaskTrim:,:] = 0

      plt.imshow(im2show, origin='lower', cmap='gray')
      #plt.scatter(y = bandData['aveloc'][-1,:,0], x = bandData['aveloc'][-1,:,1]+self.peakPad[1], c ='r', s=5)
      plt.scatter(y=bandData['aveloc'][-1,:,0],x=bandData['aveloc'][-1,:,1],c='r',s=5)
      plt.xlim(0,self.nTheta)
      plt.ylim(0,self.nRho)
      plt.show()


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



  def radon2pole(self,bandData,PC=None,vendor='EDAX'):
    # Following Krieger-Lassen1994 eq 3.1.6 //figure 3.1.1
    if PC is None:
      PC = np.array([0.471659,0.675044,0.630139])
    ven = str.upper(vendor)

    nPats = bandData.shape[0]
    nBands = bandData.shape[1]

    # This translation from the Radon to theta and rho assumes that the first pixel read
    # in off the detector is in the bottom left corner. -- No longer the assumption --- see below.  
    # theta = self.radonPlan.theta[np.array(bandData['aveloc'][:,:,1], dtype=np.int)]/RADEG
    # rho = self.radonPlan.rho[np.array(bandData['aveloc'][:, :, 0], dtype=np.int)]

    # This translation from the Radon to theta and rho assumes that the first pixel read
    # in off the detector is in the top left corner.
    theta = np.pi - self.radonPlan.theta[np.array(bandData['aveloc'][:,:,1],dtype=np.int64)] / RADEG
    rho = -1.0*self.radonPlan.rho[np.array(bandData['aveloc'][:,:,0],dtype=np.int64)]

    # from this point on, we will assume the image origin and t-vector (aka pattern center) is described
    # at the bottom left of the pattern
    stheta = np.sin(theta)
    ctheta = np.cos(theta)

    t = np.asfarray(PC).copy()
    shapet = t.shape
    if len(shapet) < 2:
      t = np.tile(t, nPats).reshape(nPats,3)
    else:
      if shapet[0] != nPats:
        t = np.tile(t[0,:], nPats).reshape(nPats,3)

    dimf = np.array(self.patDim, dtype=np.float32)
    if ven in ['EDAX', 'OXFORD']:
      t *= np.array([dimf[1], dimf[1], -dimf[1]])
    if ven == 'EMSOFT':
      t += np.array([dimf[1] / 2.0, dimf[0] / 2.0, 0.0])
      t[:, 2] *= -1.0
    if ven in ['KIKUCHIPY', 'BRUKER']:
      t *=  np.array([dimf[1], dimf[0], -dimf[0]])
      t[:, 1] = dimf[0] - t[:, 1]
    # describes the translation from the bottom left corner of the pattern image to the point on the detector
    # perpendicular to where the beam contacts the sample.

    t = np.tile(t.reshape(nPats,1, 3), (1, nBands,1))

    r = np.zeros((nPats, nBands, 3), dtype=np.float32)
    r[:,:,0] = -1*stheta
    r[:,:,1] = ctheta # now defined as r_v

    p = np.zeros((nPats, nBands, 3), dtype=np.float32)
    p[:,:,0] = rho*ctheta # get a point within the band -- here it is the point perpendicular to the image center.
    p[:,:,1] = rho*stheta
    p[:,:,0] += dimf[1] * 0.5 # now convert this with reference to the image origin.
    p[:,:,1] += dimf[0] * 0.5 # this is now [O_vP]_v in Eq 3.1.6

    #n2 = p - t.reshape(1,1,3)
    n2 = p - t
    n = np.cross(r.reshape(nPats*nBands, 3), n2.reshape(nPats*nBands, 3) )
    norm = np.linalg.norm(n, axis=1)
    n /= norm.reshape(nPats*nBands, 1)
    n = n.reshape(nPats, nBands, 3)
    return n



  def calc_rdn(self, patterns, clparams = None, use_gpu=False):
    rdnNorm_gpu = None
    if use_gpu == False:
      rdnNorm = self.radonPlan.radon_faster(patterns,self.padding,fixArtifacts=True, background = self.backgroundsub)
    else:
      try:
        rdnNorm, clparams, rdnNorm_gpu = self.radonPlan.radon_fasterCL(patterns, self.padding,
                                                                     fixArtifacts=True,background = self.backgroundsub,
                                                                     returnBuff = self.CLOps[1], clparams = clparams)
      except Exception as e:
        print(e)
        clparams.queue = None
        rdnNorm = self.radonPlan.radon_faster(patterns,self.padding,fixArtifacts=True, background = self.backgroundsub)

    return rdnNorm, clparams, rdnNorm_gpu

  def rdn_conv(self, radonIn, clparams = None, radonIn_gpu = None, use_gpu=False):
    tic = timer()

    if use_gpu == True: # perform this operation on the GPU
      try:
        return self.rdn_convCL2(radonIn, clparams, radonIn_gpu = radonIn_gpu, returnBuff = self.CLOps[2])
      except Exception as e:
        print(e)
        if isinstance(radonIn_gpu, cl.Buffer):
          nT = self.nTheta
          nTp = nT + 2 * self.padding[1]
          nR = self.nRho
          nRp = nR + 2 * self.padding[0]
          rdn_gpu = radonIn_gpu
          nImCL = np.int64(rdn_gpu.size / (nTp * nRp * 4))
          shp = (nRp,nTp,nImCL)
          radonTry = np.zeros(shp, dtype=np.float32)
          cl.enqueue_copy(clparams.queue,radonTry,radonIn_gpu,is_blocking=True)
        else:
          clparams.queue = None
          radonTry = radonIn
        return self.rdn_conv(radonTry, use_gpu=False)
    else: # perform on the CPU

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
        rdnConv[:,:,i] = -1.0 * gaussian_filter(np.squeeze(radon[:,:,i]),[self.rSigma,self.tSigma],order=[2,0])

      #print(rdnConv.min(),rdnConv.max())
      mns = (rdnConv[self.padding[0]:shprdn[1]-self.padding[0],self.padding[1]:shprdn[1]-self.padding[1],:]).min(axis=0).min(axis=0)

      rdnConv -= mns.reshape((1,1, shp[2]))
      rdnConv = rdnConv.clip(min=0.0)
      rdnConv_gpu = None
    return rdnConv, clparams, rdnConv_gpu

  def rdn_local_max(self, rdn, clparams=None, rdn_gpu=None, use_gpu=False):

    if use_gpu == True: # perform this operation on the GPU
      try:
        return self.rdn_local_maxCL(rdn, clparams, radonIn_gpu = rdn_gpu, returnBuff=self.CLOps[3])
      except Exception as e:
        print(e)
        if isinstance(rdn_gpu, cl.Buffer):
          nT = self.nTheta
          nTp = nT + 2 * self.padding[1]
          nR = self.nRho
          nRp = nR + 2 * self.padding[0]
          nImCL = np.int64(rdn_gpu.size / (nTp * nRp * 4))
          shp = (nRp,nTp,nImCL)
          radonTry = np.zeros(shp, dtype=np.float32)
          cl.enqueue_copy(clparams.queue,radonTry,rdn_gpu,is_blocking=True)
        else:
          clparams.queue = None
          radonTry = rdn
        return self.rdn_local_max(radonTry, use_gpu=False)
    else: # perform on the CPU

      shp = rdn.shape
      # find the local max
      lMaxK = (self.peakPad[0],self.peakPad[1],1)

      lMaxRdn = scipy_grey_dilation(rdn,size=lMaxK)
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
      return lMaxRdn, None

  def rdn_convCL2(self, radonIn, clparams=None, radonIn_gpu = None, separableKernel=True, returnBuff = False):
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
      try:
        clparams = openclparam.OpenClParam()
        clparams.get_queue()
        gpu = clparams.gpu
        gpu_id = clparams.gpu_id
        ctx = clparams.ctx
        prg = clparams.prg
        queue = clparams.queue
        mf = clparams.memflags
      except:
        clparams = None
        return self.rdn_conv(radonIn, clparams=None, radonIn_gpu = None)


    nT = self.nTheta
    nTp = nT + 2 * self.padding[1]
    nR = self.nRho
    nRp = nR + 2 * self.padding[0]

    if isinstance(radonIn_gpu, cl.Buffer):
      rdn_gpu = radonIn_gpu
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
      # is a exact multiple of the max group size (typically 256)
      mxGroupSz = gpu[gpu_id].get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
      #nImCL += np.int64(16 * (1 - np.int64(np.mod(nImCL,mxGroupSz) > 0)))
      radonCL = np.zeros( (nRp , nTp, nImCL), dtype = np.float32)
      radonCL[:,:,0:shp[2]] = radon
      rdn_gpu = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=radonCL)
      shp = (nRp, nTp, nImCL)

    nImChunk = np.uint64(nImCL / clvtypesize)
    resultConv = np.full(shp,0.0,dtype=np.float32)
    #resultPeakLoc = np.full_like(radon,-1.0e12, dtype = np.float32)
    #resultPeakLoc2 = np.full_like(radon,0,dtype=np.uint8)
    #rdnConv_gpu = cl.Buffer(ctx,mf.WRITE_ONLY | mf.COPY_HOST_PTR,hostbuf=resultConv)
    rdnConv_gpu = cl.Buffer(ctx,mf.WRITE_ONLY ,size=resultConv.nbytes)

    prg.radonPadTheta(queue,(shp[2],shp[0],1),None,rdn_gpu,
                    np.uint64(shp[0]),np.uint64(shp[1]),np.uint64(self.padding[1]))
    prg.radonPadRho(queue,(shp[2],shp[1],1),None,rdn_gpu,
                      np.uint64(shp[0]),np.uint64(shp[1]),np.uint64(self.padding[0]))
    kern_gpu = None
    if separableKernel == False:
      # for now I will assume that the kernel(s) can fit in local memory on the GPU
      # also going to assume that there is only one kernel -- this will be something to fix soon.
      k0 = self.kernel[0,:,:]
      kshp = np.asarray(k0.shape, dtype=np.int32)
      pad = kshp/2
      kern_gpu = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=k0)
      prg.convolution3d2d(queue,(np.int32((shp[1]-2*pad[1])),np.int32((shp[0]-2*pad[0])), nImChunk),None,
                        rdn_gpu, kern_gpu,np.int32(shp[1]),np.int32(shp[0]),np.int32(shp[2]),
                        np.int32(kshp[1]), np.int32(kshp[0]), np.int32(pad[1]), np.int32(pad[0]), rdnConv_gpu)



      #tic = timer()
    else:
      tempConvbuff = cl.Buffer(ctx,mf.HOST_NO_ACCESS,size=(shp[0]*shp[1]*shp[2]*4))

      kshp = np.asarray(self.kernel[0,:,:].shape,dtype=np.int32)
      pad = kshp
      k0x = np.require(self.kernel[0, np.int64(kshp[0] / 2), :], requirements=['C', 'A', 'W', 'O'])
      k0x *= 1.0 / k0x.sum()
      k0x = (k0x[...,:]).reshape(1,kshp[1])



      kshp = np.asarray(k0x.shape,dtype=np.int32)

      kern_gpu_x = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=k0x)
      prg.convolution3d2d(queue,(np.int32((shp[1]-2*pad[1])),np.int32((shp[0]-2*pad[0])), nImChunk),None,
                          rdn_gpu,kern_gpu_x,np.int32(shp[1]),np.int32(shp[0]),np.int32(shp[2]),
                          np.int32(kshp[1]),np.int32(kshp[0]),np.int32(pad[1]),np.int32(pad[0]),tempConvbuff)

      kshp = np.asarray(self.kernel[0,:,:].shape,dtype=np.int32)
      k0y = np.require(self.kernel[0, :, np.int32(kshp[1] / 2)], requirements=['C', 'A', 'W', 'O'])
      k0y *= 1.0 / k0y.sum()
      k0y = (k0y[...,:]).reshape(kshp[0],1)
      kshp = np.asarray(k0y.shape,dtype=np.int32)

      kern_gpu_y = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=k0y)
      prg.convolution3d2d(queue,(np.int32((shp[1]-2*pad[1])),np.int32((shp[0]-2*pad[0])), nImChunk),None,
                          tempConvbuff,kern_gpu_y,np.int32(shp[1]),np.int32(shp[0]),np.int32(shp[0]),
                          np.int32(kshp[1]),np.int32(kshp[0]),np.int32(pad[1]),np.int32(pad[0]),rdnConv_gpu)

    mns = cl.Buffer(ctx,mf.READ_WRITE,size=nImCL * 4)

    prg.imageMin(queue,(nImChunk,1,1),None,
                 rdnConv_gpu, mns,np.uint32(shp[1]),np.uint32(shp[0]),
                 np.uint32(self.padding[1]),np.uint32(self.padding[0]))

    prg.imageSubMinWClip(queue,(np.int32(shp[1]), np.int32(shp[0]),nImChunk),None,
                     rdnConv_gpu,mns,np.uint32(shp[1]),np.uint32(shp[0]),
                     np.uint32(0),np.uint32(0))



    rdn_gpu.release()
    mns.release()
    if kern_gpu is None:
      kern_gpu_y.release()
      kern_gpu_x.release()
      tempConvbuff.release()
    else:
      kern_gpu.release()

    cl.enqueue_copy(queue,resultConv,rdnConv_gpu,is_blocking=True)

    if returnBuff == False:
      rdnConv_gpu.release()
      rdnConv_gpu = None

    return resultConv, clparams, rdnConv_gpu

  def rdn_local_maxCL(self,radonIn,clparams=None,radonIn_gpu=None,  returnBuff = False):
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
      try:
        clparams = openclparam.OpenClParam()
        clparams.get_queue()
        gpu = clparams.gpu
        gpu_id = clparams.gpu_id
        ctx = clparams.ctx
        prg = clparams.prg
        queue = clparams.queue
        mf = clparams.memflags
      except Exception as e: # fall back to CPU
        print(e)
        return self.rdn_local_max(radonIn, clparams=None, rdn_gpu=None)


    nT = self.nTheta
    nTp = nT + 2 * self.padding[1]
    nR = self.nRho
    nRp = nR + 2 * self.padding[0]

    if isinstance(radonIn_gpu,cl.Buffer):
      rdn_gpu = radonIn_gpu
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
      #nImCL += np.int(16 * (1 - np.int(np.mod(nImCL,mxGroupSz) > 0)))
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

    # wrkGrpsize = np.int(winszX * 4 * clvtypesize)
    # wrkGrpsize2 = np.int((winszX*2-1) * 4 * clvtypesize)
    # prg.morphDilateKernelX(queue,(nChunkT,nR,nImChunk),None,rdn_gpu,lmaxX,
    #                        winszX, winszX2, np.uint64(shp[1]),np.uint64(shp[0]),
    #                        np.uint64(self.padding[1]),np.uint64(self.padding[0]),
    #                        cl.LocalMemory(wrkGrpsize),cl.LocalMemory(wrkGrpsize),
    #                        cl.LocalMemory(wrkGrpsize2) )
    #
    # wrkGrpsize = np.int(winszY * 4 * clvtypesize)
    # wrkGrpsize2 = np.int((winszY * 2 - 1) * 4 * clvtypesize)
    # prg.morphDilateKernelY(queue,(nT, nChunkR,nImChunk),None,lmaxX, lmaxXY,
    #                        winszY,winszY2,np.uint64(shp[1]),np.uint64(shp[0]),
    #                        np.uint64(self.padding[1]),np.uint64(self.padding[0]),
    #                        cl.LocalMemory(wrkGrpsize),cl.LocalMemory(wrkGrpsize),
    #                        cl.LocalMemory(wrkGrpsize2 ))

    # prg.morphDilateKernelBF(queue,(np.uint32(nT),np.uint32(nR),nImChunk),None,rdn_gpu,lmaxXY,
    #                       np.int64(shp[1]),np.int64(shp[0]),
    #                       np.int64(self.padding[1]),np.int64(self.padding[0]),
    #                       np.int64(self.peakPad[1]), np.int64(self.peakPad[0]))

    prg.morphDilateKernelBF(queue,(np.uint32(nT),np.uint32(nR),nImChunk),None,rdn_gpu,lmaxX,
                            np.int64(shp[1]),np.int64(shp[0]),
                            np.int64(self.padding[1]),np.int64(self.padding[0]),
                            np.int64(1),np.int64(self.peakPad[0]))

    prg.morphDilateKernelBF(queue,(np.uint32(nT),np.uint32(nR),nImChunk),None,lmaxX,lmaxXY,
                            np.int64(shp[1]),np.int64(shp[0]),
                            np.int64(self.padding[1]),np.int64(self.padding[0]),
                            np.int64(self.peakPad[1]),np.int64(1))

    local_max = np.zeros((shp),dtype=np.ubyte)
    local_max_gpu = cl.Buffer(ctx,mf.WRITE_ONLY,size=local_max.nbytes)

    prg.im1EQim2(queue,(np.uint32(nT),np.uint32(nR),nImCL),None, lmaxXY, rdn_gpu, local_max_gpu,
                 np.uint64(shp[1]),np.uint64(shp[0]),
                 np.uint64(self.padding[1]),np.uint64(self.padding[0]))

    queue.flush()


    if returnBuff == False:
      cl.enqueue_copy(queue,local_max,local_max_gpu,is_blocking=True)
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
    else:
      local_max = None
    return local_max, local_max_gpu

  def band_label(self,nPats,rdnConvIn,rdnNormIn,lMaxRdnIn,
                 rdnConvIn_gpu,rdnNormIn_gpu,lMaxRdnIn_gpu,
                 use_gpu = False, clparams=None ):
    bandData = np.zeros((nPats,self.nBands),dtype=self.dataType)

    if use_gpu == True:
      try:
        if isinstance(rdnConvIn_gpu, cl.Buffer):
          rdnConv = rdnConvIn_gpu
        else:
          rdnConv = rdnConvIn

        if isinstance(lMaxRdnIn_gpu, cl.Buffer):
          lMaxRdn = lMaxRdnIn_gpu
        else:
          lMaxRdn = lMaxRdnIn

        bdat, rdnConv_out = self.band_labelCL(rdnConv, rdnConv, lMaxRdn, clparams=clparams )
      except Exception as e:
        print(e)
        if isinstance(rdnConvIn_gpu, cl.Buffer):
          nT = self.nTheta
          nTp = nT + 2 * self.padding[1]
          nR = self.nRho
          nRp = nR + 2 * self.padding[0]
          nImCL = np.int64(rdnConvIn_gpu.size / (nTp * nRp * 4))
          shp = (nRp,nTp,nImCL)
          rdnConv = np.zeros(shp,dtype=np.float32)
          cl.enqueue_copy(clparams.queue,rdnConv,rdnConvIn_gpu,is_blocking=True)
        else:
          rdnConv = rdnConvIn

        if isinstance(lMaxRdnIn_gpu, cl.Buffer):
          nT = self.nTheta
          nTp = nT + 2 * self.padding[1]
          nR = self.nRho
          nRp = nR + 2 * self.padding[0]

          nImCL = np.int64(lMaxRdnIn_gpu.size / (nTp * nRp))
          shp = (nRp,nTp,nImCL)
          lMaxRdn = np.zeros(shp,dtype=np.ubyte)
          cl.enqueue_copy(clparams.queue,lMaxRdn,lMaxRdnIn_gpu,is_blocking=True)
        else:
          lMaxRdn = lMaxRdnIn

        bdat = self.band_label_numba(
          np.int64(self.nBands),
          np.int64(nPats),
          np.int64(self.nRho),
          np.int64(self.nTheta),
          rdnConv,
          rdnConv,
          lMaxRdn
        )
        rdnConv_out = rdnConv
    else:
      if self.EDAXIQ:
        bdat = self.band_label_numba(
          np.int64(self.nBands),
          np.int64(nPats),
          np.int64(self.nRho),
          np.int64(self.nTheta),
          rdnConvIn,
          rdnNormIn,
          lMaxRdnIn
        )
      else:
        bdat = self.band_label_numba(
          np.int64(self.nBands),
          np.int64(nPats),
          np.int64(self.nRho),
          np.int64(self.nTheta),
          rdnConvIn,
          rdnConvIn,
          lMaxRdnIn
        )
      rdnConv_out = rdnConvIn

    bandData['max']    = bdat[0][0:nPats, :]
    bandData['avemax'] = bdat[1][0:nPats, :]
    bandData['maxloc'] = bdat[2][0:nPats, :, :]
    bandData['aveloc'] = bdat[3][0:nPats, :, :]
    bandData['valid']  = bdat[4][0:nPats, :]
    bandData['maxloc'] -= self.padding.reshape(1, 1, 2)
    bandData['aveloc'] -= self.padding.reshape(1, 1, 2)

    return bandData, rdnConv_out

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

    nnc = np.array([-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2],dtype=np.float32)
    nnr = np.array([-1,-1,-1,-1,-1,0,0,0,0,0,1,1,1,1,1],dtype=np.float32)
    nnN = numba.float32(15)

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
        # nn = rdnPad_q[r - 1:r + 2,c - 2:c + 3].ravel()
        nn = rdnConv[r - 1:r + 2,c - 2:c + 3,q].ravel()
        sumnn = (np.sum(nn) + 1.e-12)
        nn /= sumnn
        bandData_avemax[q,i] = sumnn / nnN
        rnn = np.sum(nn * (np.float32(r) + nnr))
        cnn = np.sum(nn * (np.float32(c) + nnc))
        bandData_aveloc[q,i,:] = np.array([rnn,cnn])
        bandData_valid[q,i] = 1
    return bandData_max,bandData_avemax,bandData_maxloc,bandData_aveloc, bandData_valid

  def band_labelCL(self,rdnConvIn, rdnPadIn, lMaxRdnIn,clparams=None):

    # an attempt to to run the band label on the GPU

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
      try:
        clparams = openclparam.OpenClParam()
        clparams.get_queue()
        gpu = clparams.gpu
        ctx = clparams.ctx
        prg = clparams.prg
        queue = clparams.queue
        mf = clparams.memflags
      except:
        # fall back to CPU
        return self.band_label(rdnConvIn, rdnPadIn, lMaxRdnIn, clparams=None, use_gpu=False)

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
    aveloc = np.zeros((nIm,self.nBands,2),dtype=np.float32)
    aveloc_gpu = cl.Buffer(ctx,mf.WRITE_ONLY,size=aveloc.nbytes)
    rhoMaskTrim = np.int64((shp[0] - 2 * self.padding[0]) * self.rhoMaskFrac + self.padding[0])


    prg.maxlabel(queue,(nIm, 1,1),(1,1,1),
                 lMaxRdn_gpu,rdnConv_gpu,
                 maxloc_gpu, maxval_gpu,
                 aveloc_gpu,aveval_gpu,
                 np.int64(shp[1]),np.int64(shp[0]),
                 np.int64(self.padding[1]),rhoMaskTrim,np.int64(self.nBands) )

    queue.finish()
    cl.enqueue_copy(queue,maxval,maxval_gpu,is_blocking=False)
    cl.enqueue_copy(queue,maxloc,maxloc_gpu,is_blocking=False)
    cl.enqueue_copy(queue,aveval,aveval_gpu,is_blocking=False)
    cl.enqueue_copy(queue,aveloc,aveloc_gpu,is_blocking=True)

    #rdnConv_out = np.zeros((nRp,nTp,nIm),dtype=np.float32)
    #cl.enqueue_copy(queue,rdnConv_out,rdnConv_gpu,is_blocking=True)
    #queue.finish()
    #rdnConv_gpu.release()
    maxval_gpu.release()
    maxloc_gpu.release()
    aveval_gpu.release()
    aveloc_gpu.release()
    queue.finish()

    maxlocxy = np.zeros((nIm, self.nBands, 2),dtype=np.float32)
    temp = np.asarray(np.unravel_index(maxloc, (shp[0], shp[1])), dtype = np.float32)
    maxlocxy[:,:,0] = temp[0,:,:]
    maxlocxy[:,:,1] = temp[1,:,:]
    valid = np.asarray(maxval > -1e6, dtype=np.int8)
    return (maxval,aveval,maxlocxy,aveloc,valid), rdnConv_gpu
