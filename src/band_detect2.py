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
      tSigma= None, rSigma=None, rhoMaskFrac=0.1, nBands=9, clOps = [True, False, False]):
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
      self.peakPad = np.array(np.around([3 * ksz[0], 3 * ksz[1]]), dtype=np.int)
      self.peakPad += 1 - np.mod(self.peakPad, 2)  # make sure we have it as odd.

    self.padding = np.array([np.max( [self.peakPad[0], self.padding[0]] ), np.max([self.peakPad[1], self.padding[1]])])

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
    #rdnNorm = self.radonPlan.radon_faster(patterns,self.padding, fixArtifacts=True)
    rdnNorm, clparams, rdnNorm_gpu = self.calc_rdn(patterns)
    #print("Radon", timer()-tic1)
    tic1 = timer()
    if self.CLOps[1] == False:
      rdnNorm_gpu  = None
      clparams = [None,None,None,None,None]
    #rdnNorm = self.radonPlan.radon_fasterCL(patterns,fixArtifacts=True)
    rdnConv, clparams, rdnConv_gpu = self.rdn_conv(rdnNorm, clparams, rdnNorm_gpu)
    if self.CLOps[2] == False:
      rdnConv_gpu = None
      clparams = [None,None,None,None,None]
    #print('Conv: ', timer()-tic1)
    tic1 = timer()
    #plt.imshow(rdnConv[-1,:,:])
    lMaxRdn = self.rdn_local_max(rdnConv, clparams, rdnConv_gpu)
    #print("lMax: ", timer()-tic1)
    tic = timer()
    #rdnConv, lMaxRdn = self.band_conv(rdnNorm)
    #print("Conv:",timer() - tic)
    tic = timer()
    #rdnConv,lMaxRdn = self.band_convCL(rdnNorm)


    #bdat = self.band_label(np.int(self.nBands),np.int(nPats),np.int(self.nRho),np.int(self.nTheta),rdnPad,self.peakPad,self.peakMask)
    bandData = np.zeros((nPats,self.nBands),dtype=self.dataType)
    bdat = self.band_label(np.int(self.nBands),np.int(nPats),np.int(self.nRho),np.int(self.nTheta),rdnConv,lMaxRdn)
    bandData['max'] = bdat[0]
    bandData['avemax'] = bdat[1]
    bandData['maxloc'] = bdat[2]
    bandData['aveloc'] = bdat[3]
    bandData['maxloc'] -= self.padding.reshape(1,1,2)
    bandData['aveloc'] -= self.padding.reshape(1, 1, 2)
    for j in range(nPats):
      bandData['pqmax'][j,:] = \
        rdnNorm[j, (bandData['aveloc'][j,:,0]).astype(int),(bandData['aveloc'][j,:,1]).astype(int)]
    #bandData['max'] += mns.reshape(nPats,1)
    #print("BandLabel:",timer() - tic)


    if verbose == True:
      print('Total Band Find Time:',timer() - tic0)
      plt.clf()
      plt.imshow(rdnConv[-1,self.padding[0]:-self.padding[0],self.padding[1]:-self.padding[1]], origin='lower')
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

  def calc_rdn(self, patterns):
    clparams = [None,None,None,None,None]
    rdnNorm_gpu = None
    if self.CLOps[0] == False:
      rdnNorm = self.radonPlan.radon_faster(patterns,self.padding,fixArtifacts=True)
    else:
      rdnNorm, clparams, rdnNorm_gpu = self.radonPlan.radon_fasterCL(patterns, self.padding, fixArtifacts=True, returnBuff = self.CLOps[1])

    return rdnNorm, clparams, rdnNorm_gpu

  def rdn_conv(self, radonIn, clparams=[None, None, None, None, None], radonIn_gpu = None):
    tic = timer()
    ## the code below was replaced by a gpu convolution routine.

    if self.CLOps[1] == True: # perform this operation on the GPU
      return self.rdn_convCL2(radonIn, clparams, radonIn_gpu = radonIn_gpu, returnBuff = self.CLOps[2])

    else: # perform on the CPU

      shp = radonIn.shape
      if len(shp) == 2:
        radon = radonIn.reshape(1,shp[0],shp[1])
        shp = radonIn.shape
      else:
        radon = radonIn
      shprdn = radon.shape
      #rdnNormP = self.radonPad(radon,rPad=0,tPad=self.peakPad[1],mirrorTheta=True)
      if self.padding[1] > 0:
        radon[:,:,0:self.padding[1]] = np.flip(radon[:,:,-2 * self.padding[1]:-self.padding[1]],axis=1)
        radon[:,:,-self.padding[1]:] = np.flip(radon[:,:,self.padding[1]:2 * self.padding[1]],axis=1)
      # pad again for doing the convolution -- this will be removed after the convolution
      #rdnNormP = self.radonPad(rdnNormP,rPad=self.peakPad[0],tPad=self.peakPad[1],mirrorTheta=False)
      if self.padding[0] > 0:
        radon[:,0:self.padding[0], :] = radon[:,self.padding[0],:].reshape(shp[0],1,shp[2])
        radon[:,-self.padding[0]:, :] = radon[:,-self.padding[0]-1, :].reshape(shp[0],1,shp[2])


      rdnConv = np.zeros_like(radon)

      for i in range(shp[0]):
        rdnConv[i,:,:] = -1.0 * gaussian_filter(np.squeeze(radon[i,:,:]),[self.rSigma,self.tSigma],order=[2,0])

      #print(rdnConv.min(),rdnConv.max())
      mns = (rdnConv[:,self.padding[0]:shprdn[1]-self.padding[0],self.padding[1]:shprdn[1]-self.padding[1]]).min(axis=1).min(axis=1)

      rdnConv -= mns.reshape((shp[0],1,1))
      rdnConv.clip(min=0.0)
      rdnConv_gpu = None
    return rdnConv, clparams, rdnConv_gpu

  def rdn_local_max(self, rdn, clparams=[None, None, None, None, None], rdn_gpu=None):

    if self.CLOps[2] == True: # perform this operation on the GPU
      return self.rdn_local_maxCL(rdn, clparams, radonIn_gpu = rdn_gpu)

    else: # perform on the CPU

      shp = rdn.shape
      # find the local max
      lMaxK = (1,self.peakPad[0],self.peakPad[1])

      lMaxRdn = scipy_grey_dilation(rdn,size=lMaxK)
      #lMaxRdn[:,:,0:self.peakPad[1]] = 0
      #lMaxRdn[:,:,-self.peakPad[1]:] = 0
      #location of the max is where the local max is equal to the original.
      lMaxRdn = lMaxRdn == rdn

      rhoMaskTrim = np.int32((shp[1] - 2 * self.padding[0]) * self.rhoMaskFrac + self.padding[0])
      lMaxRdn[:,0:rhoMaskTrim,:] = 0
      lMaxRdn[:,-rhoMaskTrim:,:] = 0
      lMaxRdn[:,:,0:self.padding[1]] = 0
      lMaxRdn[:,:,-self.padding[1]:] = 0


      #print("Traditional:",timer() - tic)
      return lMaxRdn

  def rdn_convCL(self, radonIn, clinfo):
    # this will run sequential convolutions and identify the peaks in the convolution
    tic = timer()
    #def preparray(array):
    #  return np.require(array,None,"C")

    mf = cl.mem_flags
    if isinstance(clinfo[2], cl.CommandQueue):
      gpu = cl.get_platforms()[0].get_devices(device_type=cl.device_type.GPU)
      #if len(gpu) == 0:  # fall back to the numba implementation
      #  return self.radon_faster(radon,fixArtifacts=fixArtifacts)
      # apparently it is very difficult to get a consistent ordering of multiple GPU systems.
      # my lazy way to do this is to assign them randomly, and figure it will even out in the long run
      gpuIdx = np.random.choice(len(gpu))
      ctx = cl.Context(devices={gpu[gpuIdx]})
      queue = cl.CommandQueue(ctx)
      kernel_location = path.dirname(__file__)
      prg = cl.Program(ctx,open(path.join(kernel_location,'clkernels.cl')).read()).build()
    else:
      queue = clinfo[2]
      prg = clinfo[3]

    shp = radonIn.shape
    if len(shp) == 2:
      radonIn = radonIn.reshape(1,shp[0],shp[1])
      shp = radonIn.shape

    radon = self.radonPad(radonIn,rPad=self.peakPad[0],tPad=self.peakPad[1],mirrorTheta=True)
    shp = radon.shape

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
    #print('inital buffers', timer()-tic)
    #tic = timer()

    #print('build program', timer()-tic)
    #tic = timer()
    prg.convolution3d2d(queue,(np.int32(shp[2]-2*pad[1]), np.int32(shp[1]-2*pad[0]), shp[0]),None,
                        rdn_gpu, kern_gpu,np.int32(shp[2]),np.int32(shp[1]),np.int32(shp[0]),
                        np.int32(kshp[1]), np.int32(kshp[0]), np.int32(pad[1]), np.int32(pad[0]), rdnConv_gpu)
    queue.finish()
    #print('convolution', timer()-tic)
    tic = timer()
    cl.enqueue_copy(queue,resultConv,rdnConv_gpu, is_blocking=True).wait()

    # going to reuse the original rdn buffer
    cl.enqueue_copy(queue,rdn_gpu,resultPeakLoc,is_blocking=True).wait()
    #print('second buffers',timer() - tic)
    tic = timer()
    #rdn_gpu.release()
    #resultPeakLoc_gpu = cl.Buffer(ctx,mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf=resultPeakLoc)
    resultPeakLoc2_gpu = cl.Buffer(ctx,mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf=resultPeakLoc2)
    prg.morphDilateKernel(queue,(shp[2], shp[1], shp[0]),None,rdnConv_gpu,np.int32(shp[2]),np.int32(shp[1]),np.int32(shp[0]),
                          np.int32(self.peakMask.shape[1]), np.int32(self.peakMask.shape[0]), rdn_gpu)
    queue.finish()
    prg.im1NEim2(queue,(shp[2],shp[1],shp[0]),None,rdnConv_gpu,rdn_gpu,resultPeakLoc2_gpu,
                 np.int32(shp[2]),np.int32(shp[1]),np.int32(shp[0]))

    mns = (resultConv[:,self.peakPad[0]:-self.peakPad[0],self.peakPad[1]:-self.peakPad[1]]).min(axis=1).min(axis=1)
    resultConv -= mns.reshape((shp[0],1,1))
    resultConv = resultConv.clip(min=0.0)
    #print('find min',timer() - tic)
    #tic = timer()

    queue.finish()
    #print('find max',timer() - tic)
    tic = timer()



    #cl.enqueue_copy(queue,resultPeakLoc,rdn_gpu,is_blocking=True).wait()
    cl.enqueue_copy(queue,resultPeakLoc2,resultPeakLoc2_gpu,is_blocking=True).wait()
    #print('copyback',timer() - tic)
    tic = timer()

    rhoMaskTrim = np.int32((shp[1] - 2*self.peakPad[0])*self.rhoMaskFrac + self.peakPad[0])
    resultPeakLoc2[:,0:rhoMaskTrim, :] = 0
    resultPeakLoc2[:,-rhoMaskTrim:,:] = 0
    resultPeakLoc2[:,:,0:self.peakPad[1]] = 0
    resultPeakLoc2[:,:,-self.peakPad[1]:] = 0
    #print('trim',timer() - tic)
    #tic = timer()

    return resultConv, resultPeakLoc2

  def rdn_convCL2(self, radonIn, clparams=[None, None, None, None, None], radonIn_gpu = None, separableKernel=True, returnBuff = False):
    # this will run (eventually sequential) convolutions and identify the peaks in the convolution
    tic = timer()
    #def preparray(array):
    #  return np.require(array,None,"C")
    shp = radonIn.shape
    if len(shp) == 2:
      radon = radonIn.reshape(1,shp[0],shp[1])
    else:
      radon = radonIn
    shp = radon.shape
    mf = cl.mem_flags
    if isinstance(clparams[2],cl.CommandQueue):
      ctx = clparams[1]
      queue = clparams[2]
      prg = clparams[3]
    else:
      gpu = cl.get_platforms()[0].get_devices(device_type=cl.device_type.GPU)
      # if len(gpu) == 0:  # fall back to the numba implementation
      #  return self.radon_faster(radon,fixArtifacts=fixArtifacts)
      # apparently it is very difficult to get a consistent ordering of multiple GPU systems.
      # my lazy way to do this is to assign them randomly, and figure it will even out in the long run
      gpuIdx = np.random.choice(len(gpu))
      ctx = cl.Context(devices={gpu[gpuIdx]})
      queue = cl.CommandQueue(ctx)
      kernel_location = path.dirname(__file__)
      prg = cl.Program(ctx,open(path.join(kernel_location,'clkernels.cl')).read()).build()


    nT = self.nTheta
    nTp = nT + 2 * self.padding[1]
    nR = self.nRho
    nRp = nR + 2 * self.padding[0]

    if isinstance(radonIn_gpu, cl.Buffer):
      rdn_gpu = radonIn_gpu
      nIm = np.int(rdn_gpu.size/(nTp * nRp * 4))
      shp = (nIm, nRp, nTp)
    else:
      rdn_gpu = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=radon)

    #print('gpu platform setup', timer()-tic)
    resultConv = np.full(shp,0.0, dtype = np.float32)
    #resultPeakLoc = np.full_like(radon,-1.0e12, dtype = np.float32)
    #resultPeakLoc2 = np.full_like(radon,0,dtype=np.uint8)
    rdnConv_gpu = cl.Buffer(ctx,mf.WRITE_ONLY | mf.COPY_HOST_PTR,hostbuf=resultConv)
    mns = cl.Buffer(ctx, mf.READ_WRITE|mf.COPY_HOST_PTR, hostbuf=np.full((shp[0]), -1.0e12, dtype=np.float32))
    prg.radonPadTheta(queue,(shp[0],shp[1],1),None,rdn_gpu,
                    np.uint64(shp[1]),np.uint64(shp[2]),np.uint64(self.padding[1]))
    prg.radonPadRho(queue,(shp[0],shp[2],1),None,rdn_gpu,
                      np.uint64(shp[1]),np.uint64(shp[2]),np.uint64(self.padding[0]))

    if separableKernel == False:
      # for now I will assume that the kernel(s) can fit in local memory on the GPU
      # also going to assume that there is only one kernel -- this will be something to fix soon.
      k0 = self.kernel[0,:,:]
      kshp = np.asarray(k0.shape, dtype = np.int32)
      pad = kshp/2
      kern_gpu = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=k0)
      prg.convolution3d2d(queue,(np.int32(shp[2]-2*pad[1]), np.int32(shp[1]-2*pad[0]), shp[0]),None,
                        rdn_gpu, kern_gpu,np.int32(shp[2]),np.int32(shp[1]),np.int32(shp[0]),
                        np.int32(kshp[1]), np.int32(kshp[0]), np.int32(pad[1]), np.int32(pad[0]), rdnConv_gpu)

    #print('convolution', timer()-tic)
      tic = timer()
    else:
      tempConvbuff = cl.Buffer(ctx,mf.HOST_NO_ACCESS,size=(shp[0]*shp[1]*shp[2]*4))

      kshp = np.asarray(self.kernel[0,:,:].shape,dtype=np.int32)
      k0x = np.require(self.kernel[0,np.int(kshp[0]/2),:],requirements=['C','A'])
      k0x = (k0x[...,:]).reshape(1,kshp[1])
      k0y = np.require(self.kernel[0,:,np.int(kshp[1]/2)],requirements=['C','A'])
      k0y = (k0y[...,:]).reshape(kshp[0],1)

      kshp = np.asarray(k0x.shape,dtype=np.int32)
      pad = kshp / 2
      kern_gpu_x = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=k0x)
      prg.convolution3d2d(queue,(np.int32(shp[2] - 2 * pad[1]),np.int32(shp[1] - 2 * pad[0]),shp[0]),None,
                          rdn_gpu,kern_gpu_x,np.int32(shp[2]),np.int32(shp[1]),np.int32(shp[0]),
                          np.int32(kshp[1]),np.int32(kshp[0]),np.int32(pad[1]),np.int32(pad[0]),tempConvbuff)

      kshp = np.asarray(k0y.shape,dtype=np.int32)
      pad = kshp / 2

      kern_gpu_y = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=k0y)
      prg.convolution3d2d(queue,(np.int32(shp[2] - 2 * pad[1]),np.int32(shp[1] - 2 * pad[0]),shp[0]),None,
                          tempConvbuff,kern_gpu_y,np.int32(shp[2]),np.int32(shp[1]),np.int32(shp[0]),
                          np.int32(kshp[1]),np.int32(kshp[0]),np.int32(pad[1]),np.int32(pad[0]),rdnConv_gpu)
      kern_gpu_x.release()
      kern_gpu_y.release()
      tempConvbuff.release()

    prg.imageMin(queue,(shp[0],1,1),None,
                 rdnConv_gpu, mns,np.int32(shp[2]),np.int32(shp[1]),
                 np.int32(self.padding[1]),np.int32(self.padding[0]))

    # prg.imageSubMin(queue,(np.int32(shp[2] - 2 * pad[1]),np.int32(shp[1] - 2 * pad[0]),shp[0]),None,
    #              rdnConv_gpu,mns,np.int32(shp[2]),np.int32(shp[1]),
    #              np.int32(self.padding[1]),np.int32(self.padding[0]))

    prg.imageSubMinWClip(queue,(np.int32(shp[2]),np.int32(shp[1]),shp[0]),None,
                     rdnConv_gpu,mns,np.int32(shp[2]),np.int32(shp[1]),
                     np.int32(0),np.int32(0))

    queue.finish()
    cl.enqueue_copy(queue,resultConv,rdnConv_gpu,is_blocking=True).wait()
    if returnBuff == False:
      return resultConv, clparams, None
    else:
      return resultConv, clparams, rdnConv_gpu

  def rdn_local_maxCL(self,radonIn,clparams=[None,None,None,None,None],radonIn_gpu=None,):
    # this will run a morphological max kernel over the convolved radon array
    # the local max identifies the location of the peak max within
    # the window size.

    tic = timer()
    shp = radonIn.shape
    if len(shp) == 2:
      radon = radonIn.reshape(1,shp[0],shp[1])
    else:
      radon = radonIn
    shp = radon.shape

    mf = cl.mem_flags
    if isinstance(clparams[2],cl.CommandQueue):
      ctx = clparams[1]
      queue = clparams[2]
      prg = clparams[3]
    else:
      gpu = cl.get_platforms()[0].get_devices(device_type=cl.device_type.GPU)
      # if len(gpu) == 0:  # fall back to the numba implementation
      #  return self.radon_faster(radon,fixArtifacts=fixArtifacts)
      # apparently it is very difficult to get a consistent ordering of multiple GPU systems.
      # my lazy way to do this is to assign them randomly, and figure it will even out in the long run
      gpuIdx = np.random.choice(len(gpu))
      ctx = cl.Context(devices={gpu[gpuIdx]})
      queue = cl.CommandQueue(ctx)
      kernel_location = path.dirname(__file__)
      prg = cl.Program(ctx,open(path.join(kernel_location,'clkernels.cl')).read()).build()

    nT = self.nTheta
    nTp = nT + 2 * self.padding[1]
    nR = self.nRho
    nRp = nR + 2 * self.padding[0]

    if isinstance(radonIn_gpu,cl.Buffer):
      rdn_gpu = radonIn_gpu
      nIm = np.int(rdn_gpu.size / (nTp * nRp * 4))
      shp = (nIm,nRp,nTp)
    else:
      rdn_gpu = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=radon)

    out = np.zeros((shp), dtype = np.ubyte)
    #out = np.zeros((shp),dtype=np.float32)
    out_gpu = cl.Buffer(ctx,mf.WRITE_ONLY | mf.COPY_HOST_PTR,hostbuf=out)
    lmaxX = cl.Buffer(ctx,mf.READ_WRITE | mf.HOST_NO_ACCESS,size=radon.size * 4)
    lmaxXY = cl.Buffer(ctx,mf.READ_WRITE | mf.HOST_NO_ACCESS,size=radon.size * 4)


    nChunkT = np.uint32(np.ceil(nT / np.float(self.peakPad[1])))
    nChunkR = np.uint32(np.ceil(nR / np.float(self.peakPad[0])))
    winszX = np.uint32(self.peakPad[1])
    winszX2 = np.uint32((winszX-1) / 2)
    winszY = np.uint32(self.peakPad[0])
    winszY2 = np.uint32((winszY-1) / 2)

    prg.morphDilateKernelX(queue,(nChunkT,nR,shp[0]),(1,1,1),rdn_gpu,lmaxX,
                           winszX, winszX2, np.uint32(shp[2]),np.uint32(shp[1]),
                           np.uint32(self.padding[1]),np.uint32(self.padding[0]),
                           cl.LocalMemory(winszX * 4),cl.LocalMemory(winszX * 4), cl.LocalMemory((winszX*2-1) * 4) )


    prg.morphDilateKernelY(queue,(nT, nChunkR,shp[0]),(1,1,1),lmaxX, lmaxXY,
                           winszY,winszY2,np.uint32(shp[2]),np.uint32(shp[1]),
                           np.uint32(self.padding[1]),np.uint32(self.padding[0]),
                           cl.LocalMemory(winszX * 4),cl.LocalMemory(winszX * 4), cl.LocalMemory((winszX*2-1) * 4) )

    # prg.morphDilateKernelBF(queue,(nT,nR,shp[0]),None,rdn_gpu,lmaxXY,
    #                       np.uint32(shp[2]),np.uint32(shp[1]),
    #                       np.uint32(self.padding[1]),np.uint32(self.padding[0]),
    #                       np.uint32(self.peakPad[1]), np.uint32(self.peakPad[0]))

    prg.im1EQim2(queue,(nT, nR,shp[0]),None, lmaxXY, rdn_gpu, out_gpu,
                 np.uint32(shp[2]),np.uint32(shp[1]),
                 np.uint32(self.padding[1]),np.uint32(self.padding[0]))
    queue.finish()
    cl.enqueue_copy(queue,out,out_gpu,is_blocking=True).wait()

    rhoMaskTrim = np.int32((shp[1] - 2 * self.padding[0]) * self.rhoMaskFrac + self.padding[0])
    out[:,0:rhoMaskTrim,:] = 0
    out[:,-rhoMaskTrim:,:] = 0
    out[:,:,0:self.padding[1]] = 0
    out[:,:,-self.padding[1]:] = 0

    return out
    # going to reuse the original rdn buffer
    #cl.enqueue_copy(queue,rdn_gpu,resultPeakLoc,is_blocking=True).wait()
    #print('second buffers',timer() - tic)
    #tic = timer()
    #rdn_gpu.release()
    #resultPeakLoc_gpu = cl.Buffer(ctx,mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf=resultPeakLoc)
    #resultPeakLoc2_gpu = cl.Buffer(ctx,mf.READ_WRITE | mf.COPY_HOST_PTR,hostbuf=resultPeakLoc2)
    #prg.morphDilateKernel(queue,(shp[2], shp[1], shp[0]),None,rdnConv_gpu,np.int32(shp[2]),np.int32(shp[1]),np.int32(shp[0]),
    #                      np.int32(self.peakMask.shape[1]), np.int32(self.peakMask.shape[0]), rdn_gpu)
    #queue.finish()
    #prg.im1NEim2(queue,(shp[2],shp[1],shp[0]),None,rdnConv_gpu,rdn_gpu,resultPeakLoc2_gpu,
    #             np.int32(shp[2]),np.int32(shp[1]),np.int32(shp[0]))


