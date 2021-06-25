import numpy as np
from os import path#, environ
import numba
from scipy.ndimage import gaussian_filter
from scipy.ndimage.morphology import grey_dilation as scipy_grey_dilation
import radon_fast
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import pyopencl as cl
#from os import environ
#environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

RADEG = 180.0/np.pi



class BandDetect():
  def __init__(self, patterns=None, patDim = None, nTheta = 180, nRho=90,\
      tSigma= None, rSigma=None, rhoMaskFrac=0.1, nBands=9, clOps = [True, True, True]):
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
    self.peakPad = np.array([11,11])
    self.padding = np.array([11,11])
    self.rhoMaskFrac = rhoMaskFrac

    self.nBands = nBands
    self.EDAXIQ = False
    self.backgroundsub = None

    self.dataType = np.dtype([('id', np.int32), ('max', np.float32), \
                    ('maxloc', np.float32, (2)), ('avemax', np.float32), ('aveloc', np.float32, (2)),\
                    ('pqmax', np.float32)])

    self.CLOps = [False, False, False]
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
    if (self.rhoMaskFrac is not None):
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
      self.peakPad = np.array(np.around([3 * ksz[0], 4 * ksz[1]]), dtype=np.int)
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
      pshape = patsIn.shape
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
      pat1 = fileobj.read_data(convertToFloat=True,patStartEnd=[stride[0],stride[0] + 1],returnArrayOnly=True)
      for i in stride[1:]:
        pat1 += fileobj.read_data(convertToFloat=True,patStartEnd=[i,i + 1],returnArrayOnly=True)
      back = pat1 / float(len(stride))
      pshape = pat1.shape
    # a bit of image processing.
    if back is not None:
      if sigma is None:
       sigma = 2.0 * float(pshape[-1]) / 80.0
      #back[0,:,:] = gaussian_filter(back[0,:,:], sigma = sigma )
      back -= np.mean(back)
    self.backgroundsub = back

  def find_bands(self, patternsIn, faster=False, verbose=False, clparams = [None, None, None, None, None]):
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
    rdnNorm, clparams, rdnNorm_gpu = self.calc_rdn(patterns, clparams, use_gpu=self.CLOps[0])
    if self.EDAXIQ == True:
      if rdnNorm is None:
        nTp = self.nTheta + 2 * self.padding[1]
        nRp = self.nRho + 2 * self.padding[0]
        nImCL = int(rdnNorm_gpu.size/(nTp*nRp*4))
        rdnNorm = np.zeros((nRp,nTp,nImCL),dtype=np.float32)
        cl.enqueue_copy(clparams[2],rdnNorm,rdnNorm_gpu,is_blocking=True)

    if self.CLOps[1] == False:
      rdnNorm_gpu  = None
      clparams = [None,None,None,None,None]
    rdntime = timer() - tic1
    tic1 = timer()
    rdnConv, clparams, rdnConv_gpu = self.rdn_conv(rdnNorm, clparams, rdnNorm_gpu, use_gpu=self.CLOps[1])
    if self.CLOps[2] == False:
      rdnConv_gpu = None
      clparams = [None,None,None,None,None]
    convtime = timer()-tic1
    tic1 = timer()
    lMaxRdn = self.rdn_local_max(rdnConv, clparams, rdnConv_gpu, use_gpu=self.CLOps[2])
    lmaxtime =  timer()-tic1
    tic1 = timer()
    # going to manually clear the clparams queue -- this should clear the memory of the queue off the GPU
    clparams[2] = None
    bandData = np.zeros((nPats,self.nBands),dtype=self.dataType)

    if self.EDAXIQ == True:
      bdat = self.band_label(np.int(self.nBands),np.int(nPats),np.int(self.nRho),np.int(self.nTheta),rdnConv,rdnNorm,lMaxRdn)
    else:
      bdat = self.band_label(np.int(self.nBands),np.int(nPats),np.int(self.nRho),np.int(self.nTheta),rdnConv,rdnConv,lMaxRdn)
    bandData['max'] = bdat[0]
    bandData['avemax'] = bdat[1]
    bandData['maxloc'] = bdat[2]
    bandData['aveloc'] = bdat[3]
    bandData['maxloc'] -= self.padding.reshape(1,1,2)
    bandData['aveloc'] -= self.padding.reshape(1, 1, 2)
    # for j in range(nPats):
    #   bandData['pqmax'][j,:] = \
    #     rdnNorm[(bandData['aveloc'][j,:,0]).astype(int),(bandData['aveloc'][j,:,1]).astype(int), j]

    blabeltime = timer() - tic1


    if verbose == True:
      print('Radon Time:',rdntime)
      print('Convolution Time:', convtime)
      print('Peak ID Time:', lmaxtime)
      print('Band Label Time:', blabeltime)
      print('Total Band Find Time:',timer() - tic0)
      plt.clf()
      im2show = rdnConv[self.padding[0]:-self.padding[0],self.padding[1]:-self.padding[1], nPats-1]

      rhoMaskTrim = np.int32(im2show.shape[0] * self.rhoMaskFrac)
      mean = np.mean(im2show[rhoMaskTrim:-rhoMaskTrim, 1:-2])
      stdv = np.std(im2show[rhoMaskTrim:-rhoMaskTrim, 1:-2])

      im2show -= mean
      im2show /= stdv
      im2show += 4
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
    # in off the detector is in the bottom left corner.
    # theta = self.radonPlan.theta[np.array(bandData['aveloc'][:,:,1], dtype=np.int)]/RADEG
    # rho = self.radonPlan.rho[np.array(bandData['aveloc'][:, :, 0], dtype=np.int)]

    # This translation from the Radon to theta and rho assumes that the first pixel read
    # in off the detector is in the top left corner.
    theta = np.pi - self.radonPlan.theta[np.array(bandData['aveloc'][:,:,1],dtype=np.int64)] / RADEG
    # rho = self.radonPlan.rho[np.array(self.nRho -
    #                                   bandData['aveloc'][:,:,0],dtype=np.int64).clip(0,self.radonPlan.nRho-1)]
    rho = -1.0*self.radonPlan.rho[np.array(bandData['aveloc'][:,:,0],dtype=np.int64)]
    # from this point on, we will assume the image origin and t-vector (aka pattern center) is described
    # at the bottom left of the pattern
    stheta = np.sin(theta)
    ctheta = np.cos(theta)

    t = PC.copy()
    dimf = np.array(self.patDim, dtype=np.float32)
    if ven == 'EDAX':
      t *= np.array([dimf[0],dimf[1],-dimf[0]])
    # describes the translation from the bottom left corner of the pattern image to the point on the detector
    # perpendicular to where the beam contacts the sample.

    r = np.zeros((nPats, nBands, 3), dtype=np.float32)
    r[:,:,0] = -1*stheta
    r[:,:,1] = ctheta # now defined as r_v

    p = np.zeros((nPats, nBands, 3), dtype=np.float32)
    p[:,:,0] = rho*ctheta # get a point within the band -- here it is the point perpendicular to the image center.
    p[:,:,1] = rho*stheta
    p[:,:,0] += dimf[0] * 0.5 # now convert this with reference to the image origin.
    p[:,:,1] += dimf[1] * 0.5 # this is now [O_vP]_v in Eq 3.1.6

    n2 = p - t.reshape(1,1,3)
    n = np.cross(r.reshape(nPats*nBands, 3), n2.reshape(nPats*nBands, 3) )
    norm = np.linalg.norm(n, axis=1)
    n /= norm.reshape(nPats*nBands, 1)
    n = n.reshape(nPats, nBands, 3)
    return n

  @staticmethod
  @numba.jit(nopython=True,fastmath=True,cache=True,parallel=False)
  def band_label(nBands, nPats, nRho, nTheta, rdnConv, rdnPad,  lMaxRdn ):
    nB = np.int(nBands)
    nP = np.int(nPats)
    nR = np.int(nRho)
    nT = np.int(nTheta)
    shp  = rdnPad.shape
    #print(shp)
    bandData_max = np.zeros((nP,nB), dtype = np.float32) - 2.0e6 # max of the convolved peak value
    bandData_avemax = np.zeros((nP,nB), dtype = np.float32) - 2.0e6 # mean of the nearest neighborhood values around the max
    bandData_maxloc = np.zeros((nP,nB,2), dtype = np.float32) # location of the max within the radon transform
    bandData_aveloc = np.zeros((nP,nB,2), dtype = np.float32) # location of the max based on the nearest neighbor interpolation

    nnx = np.array([-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2],dtype=np.float32)
    nny = np.array([-1,-1,-1,-1,-1,0,0,0,0,0,1,1,1,1,1],dtype=np.float32)

    for q in range(nPats):
      rdnConv_q = rdnConv[:,:,q]
      rdnPad_q = rdnPad[:,:,q]
      lMaxRdn_q = lMaxRdn[:,:,q]
      #peakLoc = np.nonzero((lMaxRdn_q == rdnPad_q) & (rdnPad_q > 1.0e-6))
      peakLoc = np.nonzero(lMaxRdn_q)
      indx1D = peakLoc[1] + peakLoc[0] * shp[1]
      temp = (rdnConv_q.ravel())[indx1D]
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

  def calc_rdn(self, patterns, clparams = [None, None, None, None], use_gpu=False):
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
        rdnNorm = self.radonPlan.radon_faster(patterns,self.padding,fixArtifacts=True, background = self.backgroundsub)

    return rdnNorm, clparams, rdnNorm_gpu

  def rdn_conv(self, radonIn, clparams=[None, None, None, None, None], radonIn_gpu = None, use_gpu=False):
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
          nImCL = np.int(rdn_gpu.size / (nTp * nRp * 4))
          shp = (nRp,nTp,nImCL)
          radonTry = np.zeros(shp, dtype=np.float32)
          cl.enqueue_copy(clparams[2],radonTry,radonIn_gpu,is_blocking=True)
        else:
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

  def rdn_local_max(self, rdn, clparams=[None, None, None, None, None], rdn_gpu=None, use_gpu=False):

    if use_gpu == True: # perform this operation on the GPU
      try:
        return self.rdn_local_maxCL(rdn, clparams, radonIn_gpu = rdn_gpu)
      except Exception as e:
        print(e)
        if isinstance(rdn_gpu, cl.Buffer):
          nT = self.nTheta
          nTp = nT + 2 * self.padding[1]
          nR = self.nRho
          nRp = nR + 2 * self.padding[0]
          nImCL = np.int(rdn_gpu.size / (nTp * nRp * 4))
          shp = (nRp,nTp,nImCL)
          radonTry = np.zeros(shp, dtype=np.float32)
          cl.enqueue_copy(clparams[2],radonTry,rdn_gpu,is_blocking=True)
        else:
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
      return lMaxRdn

  def rdn_convCL2(self, radonIn, clparams=[None, None, None, None, None], radonIn_gpu = None, separableKernel=True, returnBuff = False):
    # this will run (eventually sequential) convolutions and identify the peaks in the convolution
    tic = timer()
    #def preparray(array):
    #  return np.require(array,None,"C")

    clvtypesize = 16  # this is the vector size to be used in the openCL implementation.

    mf = cl.mem_flags
    if isinstance(clparams[1],cl.Context):
      gpu = clparams[0]
      ctx = clparams[1]
      prg = clparams[3]
      if isinstance(clparams[2], cl.CommandQueue):
        queue = clparams[2]
      else:
        queue = cl.CommandQueue(ctx)
      mf = clparams[4]
    else:
      try:
        gpu = cl.get_platforms()[0].get_devices(device_type=cl.device_type.GPU)
        ctx = cl.Context(devices={gpu[0]})
        queue = cl.CommandQueue(ctx)
        kernel_location = path.dirname(__file__)
        prg = cl.Program(ctx,open(path.join(kernel_location,'clkernels.cl')).read()).build()
        mf = cl.mem_flags
      except:
        return self.rdn_conv(radonIn, clparams=[None, None, None, None, None], radonIn_gpu = None)
    clparams = [gpu,ctx,queue,prg,mf]

    nT = self.nTheta
    nTp = nT + 2 * self.padding[1]
    nR = self.nRho
    nRp = nR + 2 * self.padding[0]

    if isinstance(radonIn_gpu, cl.Buffer):
      rdn_gpu = radonIn_gpu
      nImCL = np.int(rdn_gpu.size/(nTp * nRp * 4))
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
      nImCL = np.int32(clvtypesize * (np.int(np.ceil(nIm / clvtypesize))))
      # there is something very strange that happens if the number of images
      # is a exact multiple of the max group size (typically 256)
      mxGroupSz = gpu[0].get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
      nImCL += np.int(16 * (1 - np.int(np.mod(nImCL,mxGroupSz) > 0)))
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
      kshp = np.asarray(k0.shape, dtype = np.int32)
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
      k0x = np.require(self.kernel[0,np.int(kshp[0]/2),:],requirements=['C','A', 'W', 'O'])
      k0x *= 1.0 / k0x.sum()
      k0x = (k0x[...,:]).reshape(1,kshp[1])



      kshp = np.asarray(k0x.shape,dtype=np.int32)

      kern_gpu_x = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=k0x)
      prg.convolution3d2d(queue,(np.int32((shp[1]-2*pad[1])),np.int32((shp[0]-2*pad[0])), nImChunk),None,
                          rdn_gpu,kern_gpu_x,np.int32(shp[1]),np.int32(shp[0]),np.int32(shp[2]),
                          np.int32(kshp[1]),np.int32(kshp[0]),np.int32(pad[1]),np.int32(pad[0]),tempConvbuff)

      kshp = np.asarray(self.kernel[0,:,:].shape,dtype=np.int32)
      k0y = np.require(self.kernel[0,:,np.int(kshp[1] / 2)],requirements=['C','A', 'W', 'O'])
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

  def rdn_local_maxCL(self,radonIn,clparams=[None,None,None,None,None],radonIn_gpu=None,):
    # this will run a morphological max kernel over the convolved radon array
    # the local max identifies the location of the peak max within
    # the window size.

    tic = timer()
    clvtypesize = 16  # this is the vector size to be used in the openCL implementation.


    mf = cl.mem_flags
    if isinstance(clparams[1],cl.Context):
      gpu = clparams[0]
      ctx = clparams[1]
      prg = clparams[3]
      if isinstance(clparams[2],cl.CommandQueue):
        queue = clparams[2]
      else:
        queue = cl.CommandQueue(ctx)
      mf = clparams[4]
    else:
      try:
        gpu = cl.get_platforms()[0].get_devices(device_type=cl.device_type.GPU)
        ctx = cl.Context(devices={gpu[0]})
        queue = cl.CommandQueue(ctx)
        kernel_location = path.dirname(__file__)
        prg = cl.Program(ctx,open(path.join(kernel_location,'clkernels.cl')).read()).build()
        mf = cl.mem_flags
      except: # fall back to CPU
        return self.rdn_local_max(radonIn, clparams=[None, None, None, None, None], rdn_gpu=None)

    clparams = [gpu,ctx,queue,prg,mf]
    nT = self.nTheta
    nTp = nT + 2 * self.padding[1]
    nR = self.nRho
    nRp = nR + 2 * self.padding[0]

    if isinstance(radonIn_gpu,cl.Buffer):
      rdn_gpu = radonIn_gpu
      nImCL = np.int(rdn_gpu.size / (nTp * nRp * 4))
      shp = (nRp,nTp,nImCL)
    else:
      shp = radonIn.shape
      if len(shp) == 2:
        radon = radonIn.reshape(shp[0],shp[1],1)
      else:
        radon = radonIn
      shp = radon.shape
      nIm = shp[2]
      nImCL = np.int32(clvtypesize * (np.int(np.ceil(nIm / clvtypesize))))
      # there is something very strange that happens if the number of images
      # is a exact multiple of the max group size (typically 256)
      mxGroupSz = gpu[0].get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
      nImCL += np.int(16 * (1 - np.int(np.mod(nImCL,mxGroupSz) > 0)))
      radonCL = np.zeros((nRp,nTp,nImCL),dtype=np.float32)
      radonCL[:,:,0:shp[2]] = radon
      rdn_gpu = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=radonCL)
      shp = (nRp,nTp,nImCL)

    nImChunk = np.uint64(nImCL / clvtypesize)
    #out = np.zeros((shp), dtype = np.int32)

    lmaxX = cl.Buffer(ctx,mf.READ_WRITE ,size=rdn_gpu.size)
    lmaxXY = cl.Buffer(ctx,mf.READ_WRITE ,size=rdn_gpu.size)


    nChunkT = np.uint64(np.ceil(nT / np.float(self.peakPad[1])))
    nChunkR = np.uint64(np.ceil(nR / np.float(self.peakPad[0])))
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

    out = np.zeros((shp),dtype=np.ubyte)
    out_gpu = cl.Buffer(ctx,mf.WRITE_ONLY,size=out.nbytes)

    prg.im1EQim2(queue,(np.uint32(nT),np.uint32(nR),nImCL),None, lmaxXY, rdn_gpu, out_gpu,
                 np.uint64(shp[1]),np.uint64(shp[0]),
                 np.uint64(self.padding[1]),np.uint64(self.padding[0]))


    cl.enqueue_copy(queue,out,out_gpu,is_blocking=True)
    rdn_gpu.release()
    lmaxX.release()
    lmaxXY.release()
    out_gpu.release()
    rhoMaskTrim = np.int32((shp[0] - 2 * self.padding[0]) * self.rhoMaskFrac + self.padding[0])
    out[0:rhoMaskTrim,:, :] = 0
    out[-rhoMaskTrim:,:,:] = 0
    out[:,0:self.padding[1],:] = 0
    out[:,-self.padding[1]:,:] = 0

    return out



