import numpy as np
import numba
import rotlib
from timeit import default_timer as timer
RADEG = 180.0/np.pi


class BandVotePair():
  def __init__(self, pairLib, angTol = 3.0):
    self.pairLib = pairLib
    self.angTol = angTol
    # same sorting used in building the triplet library
    #LUTA = np.array([[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]], dtype=np.int64)
    #LUTB = np.array([[0,1,2],[1,0,2],[0,2,1],[2,0,1],[1,2,0],[2,1,0]], dtype=np.int64)

    #LUT = np.zeros((3,3,3,3),dtype=np.int64)
    #for i in range(6):
    #  LUT[:,LUTA[i,0],LUTA[i,1],LUTA[i,2]] = LUTB[i,:]
    #self.LUT = np.asarray(LUT).copy()


  def pairvote(self, bandnormsIN, goNumba = False):
    tic0 = timer()
    nfam = self.pairLib.family.shape[0]
    nPairs = self.pairLib.angLib.shape[0]
    tolRad = self.angTol/RADEG


    shp = bandnormsIN.shape
    if bandnormsIN.ndim == 2:
      bandnorms = bandnormsIN.reshape((1,shp[0], shp[1]))
    else:
      bandnorms = bandnormsIN

    shp = bandnorms.shape

    nPats = shp[0]
    nBands = shp[1]
    accumulator = np.zeros((nPats,nfam,nBands), dtype = np.int32)

    angTable = np.arccos(np.abs(np.einsum('ijk,ilk->ijl', bandnorms, bandnorms).clip(-1.0,1.0)))

    for dtest in range(nPairs):
      wh = np.nonzero( np.abs( angTable - self.pairLib.angLib['ang'][dtest])  <= tolRad)
      if len(wh) > 0:
        f1 = self.pairLib.angLib['f1'][dtest]
        f2 = self.pairLib.angLib['f2'][dtest]
        np.add.at(accumulator, (wh[0], f1, wh[1]),1)
        np.add.at(accumulator, (wh[0], f2, wh[1]),1)


    mxvote = np.amax(accumulator, axis = 1)
    tvotes = np.sum(accumulator, axis = 1)
    band_cm = np.zeros((nPats, nBands)) #confidence metric


    # for i in range(nBands):
    #   if tvotes[i] < 1:
    #     band_cm[i] = 0.0
    #   else:
    #     srt = np.argsort(accumulator[:,i])
    #     band_cm[i] = (accumulator[srt[-1], i] - accumulator[srt[-2], i])/tvotes[i]

    srt = np.argsort(accumulator, axis = 1)
    srtA = np.take_along_axis(accumulator, srt, axis = 1)
    band_cm = (srtA[:,-1,:] - srtA[:,-2,:])/(tvotes.clip(min=1))

    bandFam = np.argmax(accumulator, axis=1)
    bandRank = (nBands - np.arange(nBands))/nBands * band_cm * mxvote

    #print(accumulator)
    #print(tvotes, band_cm, mxvote)
    #print('vote loops: ', timer() - tic0)
    tic = timer()
    avequatN = np.zeros((nPats, 4), dtype = np.float32)
    fitN = np.zeros((nPats,1), dtype = np.float32)
    bandmatchN = np.zeros((nPats,nBands), dtype = np.int32)
    nMatchN = np.zeros((nPats),dtype=np.int32)
    print(accumulator)
    for q in range(nPats):

      avequat,fit,bandmatch,nMatch = self.band_vote_refine(bandnorms[q,:,:],bandRank[q,:],bandFam[q,:], angTol = self.angTol)
      if nMatch == 0:
        srt = np.argsort(bandRank[q,:])
        for i in range(np.min([5, nBands])):
          bandRank2 = bandRank[q,:]
          bandRank2[srt[i]] = -1.0
          #avequat, fit, bandmatch, nMatch = self.band_vote_refine(bandnorms,bandRank2,bandFam,self.tripLib.completelib,self.angTol)
          avequat,fit,bandmatch,nMatch = self.band_vote_refine(bandnorms[q,:,:],bandRank2,bandFam[q,:])
          if nMatch > 2:
            break
      avequatN[q,:] = avequat
      fitN[q,:] = fit
      bandmatchN[q,:] = bandmatch
      nMatchN[q] = nMatch
    print('refinement: ', timer() - tic)
    #print('tripvote: ',timer() - tic0)
    return avequatN, fitN, np.mean(band_cm, axis = 1), bandmatchN, nMatchN

  def band_vote_refine(self,bandnorms,bandRank,familyLabel,angTol=3.0):
    tic0 = timer()
    tic = timer()
    nBands = np.int(bandnorms.size/3)
    angTable = self.pairLib.completelib['angTable']
    sztable = angTable.shape

    famIndx = self.pairLib.completelib['famIndex']
    nFam = self.pairLib.completelib['nFamily']
    poles = self.pairLib.completelib['polesCart']
    nPoles = np.int(poles.size / 3)
    srt = np.flip(np.argsort(bandRank))
    v0indx = -1
    v1indx = 0
    ang01 = 0.0
    while ang01 < angTol:
      v0indx += 1
      v1indx += 1
      v0 = bandnorms[srt[v0indx], :]
      f0 = familyLabel[srt[v0indx]]
      v1 = bandnorms[srt[v1indx], :]
      f1 = familyLabel[srt[v1indx]]
      ang01 = np.clip(np.dot(v0, v1), -1.0, 1.0)
      ang01 = np.arccos(ang01)*RADEG


    v0v1c = np.cross(v0, v1)

    v0v1c /= np.linalg.norm(v0v1c)

    wh01 = np.nonzero(np.abs(angTable[famIndx[f0], famIndx[f1]:np.int(famIndx[f1]+nFam[f1])] - ang01) < angTol)[0]

    n01 = wh01.size
    if  n01 == 0:
      return (np.array([1.0, 0, 0, 0]), 360.0, nBands-1, 0)

    wh01 += famIndx[f1]
    p0 = poles[famIndx[f0], :]

    # attempt to see which solution gives the best match to all the poles
    # Use the TRIAD method for finding the rotation matrix

    Rtry = np.zeros((n01,3,3))
    score = np.zeros(n01)
    A = np.zeros((3,3))
    #B = np.zeros((3,3))
    b2 = np.cross(v0,v0v1c)
    B = np.squeeze([[v0],[v0v1c],[b2]])
    A[:,0] = p0
    score = -1

    for i in range(n01):
      p1 = poles[wh01[i], :]
      ntemp = np.linalg.norm(p1)+1.0e-35
      p1 = p1/ntemp
      p0p1c = np.cross(p0, p1)
      ntemp = np.linalg.norm(p0p1c)+1.0e-35
      p0p1c = p0p1c/ntemp
      A[:,1] = p0p1c
      A[:,2] = np.cross(p0, p0p1c)
      Rtry[i, :,:] = A.dot(B)

      testp = (Rtry[i, :, :].dot(bandnorms.T))
      test = poles.dot(testp)

      angfitTry = np.clip(np.amax(test, axis=0), -1.0, 1.0)

      angfitTry = np.arccos(angfitTry)*RADEG
      whMatch = np.nonzero(angfitTry < angTol)[0]

      nmatch = whMatch.size
      scoreTry = nmatch * np.mean(np.abs(angTol - angfitTry[whMatch]))

      if scoreTry > score:
        score = scoreTry
        angFit = angfitTry
        polematch = np.argmax(test, axis=0)
        R = Rtry[i, :,:]


    whGood = np.nonzero(angFit < angTol)[0]
    nGood = np.int64(whGood.size)
    if nGood < 3:
      return (np.array([1.0,0,0,0]),360.0,nBands - 1,0)
    whNoGood = np.nonzero(angFit >= angTol)[0]
    nNoGood = whNoGood.size

    if nNoGood > 0:
      polematch[whNoGood] = -1
    #print('Refine -- first find: ', timer() - tic)
    tic = timer()
    # do a n choose 2 of the rest of the poles
    # n choose k combinations --> C = n! / (k!(n-k)! == product(lindgen(k)+(n-k+1)) / factorial(k)
    # N Choose K with N = good band poles and K = 2
    #print('first loop: ', timer() - tic)
    tic = timer()
    n2Fit = np.int64(np.product(np.arange(2)+(nGood-2+1))/np.int(2))
    whGood = np.asarray(whGood,dtype=np.int64)
    #qweight = np.zeros(n2Fit)
    AB = band_vote_refine_loops(nGood,whGood,poles,bandnorms,polematch,n2Fit)

    #quats = np.zeros((n2Fit, 4), dtype = np.float32)
    # AB = np.zeros((n2Fit, 3,3), dtype = np.float32)
    # counter = 0
    # for i in range(nGood):
    #   v0 = bandnorms[whGood[i], :]
    #   p0 = poles[polematch[whGood[i]], :]
    #   A[:,0] = p0
    #   B[0,:] = v0
    #   for j in range(i+1, nGood):
    #     v1 = bandnorms[whGood[j], :]
    #     p1 = poles[polematch[whGood[j]], :]
    #     v0v1c = np.cross(v0, v1)
    #     #v0v1c /= np.linalg.norm(v0v1c)+1.0e-35
    #     v0v1c = vectnorm(v0v1c)
    #     p0p1c = np.cross(p0, p1)
    #     #p0p1c /= (np.linalg.norm(p0p1c))+1.0e-35
    #     p0p1c = vectnorm(p0p1c)
    #     A[:,1] = p0p1c
    #     B[1,:] = v0v1c
    #     A[:,2] = np.cross(p0, p0p1c)
    #     B[2,:] = np.cross(v0, v0v1c)
    #     AB[counter, :,:] = A.dot(B)
    #     counter += 1


    #print('Refine -- refine: ',timer() - tic)
    tic = timer()
    quats = rotlib.om2qu(AB)

    sign0 = np.sum(quats[0,:] * quats, axis = 1)
    sign = (sign0 >= 0).astype(np.float32) - (sign0 < 0).astype(np.float32)
    quats *= sign.reshape(n2Fit,1)
    avequat = np.mean(quats, axis = 0)
    avequat = rotlib.quatnorm(avequat)
    #if avequat[0] < 0:
    #  avequat *= -1.0

    test = rotlib.quat_vector(avequat,bandnorms[whGood,:])
    test = np.sum(test * poles[polematch[whGood], :], axis = 1)
    test = np.arccos(np.clip(test, -1.0, 1.0))*RADEG
    fit = np.mean(test)
    #print('Refine -- ave fit: ',timer() - tic)
    #print('Refine -- total: ',timer() - tic0)
    return (avequat, fit, polematch, nGood )


@numba.jit(nopython=True, cache=True,fastmath=True,parallel=False)
def tripvote_numba( bandangs, LUT, angTol, tripAngles, tripID, nfam, n_bands):
    LUTTemp = np.asarray(LUT).copy()
    accumulator = np.zeros((nfam, n_bands), dtype=np.int32)
    tshape = np.shape(tripAngles)
    ntrip = int(tshape[0])

    angTest2 = np.zeros(ntrip, dtype=numba.boolean)
    for i in range(n_bands):
      for j in range(i + 1,n_bands):
        for k in range(j + 1,n_bands):
          angtri = np.array([bandangs[i,j],bandangs[i,k],bandangs[j,k]], dtype=numba.float32)
          srt = angtri.argsort(kind='quicksort') #np.array(np.argsort(angtri), dtype=numba.int64)
          srt2 = np.asarray(LUTTemp[:,srt[0],srt[1],srt[2]], dtype=numba.int64).copy()
          unsrtFID = np.argsort(srt2,kind='quicksort').astype(np.int64)
          angtriSRT = np.asarray(angtri[srt])
          angTest = (np.abs(tripAngles - angtriSRT)) <= angTol
          for q in range(ntrip):
            #angTest2[q] = np.all(angTest[q,:])
            angTest2[q] = (angTest[q,0] + angTest[q,1] + angTest[q,2]) == 3

          wh = angTest2.nonzero()[0]

          for q in wh:
            f = tripID[q,:]
            f = f[unsrtFID]
            accumulator[f,i] += 1
            accumulator[f,j] += 1
            accumulator[f,k] += 1
    mxvote = np.zeros(n_bands, dtype = np.int32)
    tvotes = np.zeros(n_bands, dtype = np.int32)
    band_cm = np.zeros(n_bands,dtype=np.float32)
    for q in range(n_bands):
      mxvote[q] = np.amax(accumulator[:,q])
      tvotes[q] = np.sum(accumulator[:,q])



    for i in range(n_bands):
      if tvotes[i] < 1:
        band_cm[i] = 0.0
      else:
        srt = np.argsort(accumulator[:,i])
        band_cm[i] = (accumulator[srt[-1],i] - accumulator[srt[-2],i]) / tvotes[i]

    bandRank = np.zeros(n_bands, dtype=np.float32)
    bandFam = np.zeros(n_bands, dtype=np.int32)
    for q in range(n_bands):
      bandFam[q] = np.argmax(accumulator[:,q])
    bandRank = (n_bands - np.arange(n_bands)) / n_bands * band_cm * mxvote

    return accumulator, bandFam, bandRank, band_cm

@numba.jit(nopython=True, cache=True,fastmath=True,parallel=False)
def vectnorm(v):
  sum = numba.float32(0.0)
  for i in range(3):
    sum += v[i]*v[i]
  sum = np.sqrt(sum)+1.0e-35
  for i in range(3):
    v[i] = v[i]/sum
  return v

@numba.jit(nopython=True, cache=True, fastmath=True,parallel=False)
def band_vote_refine_loops(nGood, whGood, poles, bandnorms, polematch, n2Fit):
  quats = np.zeros((n2Fit,4),dtype=np.float32)
  counter = 0
  A = np.zeros((3,3), dtype = np.float32)
  B = np.zeros((3,3), dtype = np.float32)
  AB = np.zeros((n2Fit, 3,3), dtype= np.float32)
  for i in range(nGood):
    v0 = bandnorms[whGood[i],:]
    p0 = poles[polematch[whGood[i]],:]
    A[:,0] = p0
    B[0,:] = v0
    for j in range(i + 1,nGood):
      v1 = bandnorms[whGood[j],:]
      p1 = poles[polematch[whGood[j]],:]
      v0v1c = np.cross(v0,v1)
      # v0v1c /= np.linalg.norm(v0v1c)+1.0e-35
      # v0v1c = vectnorm(v0v1c) # faster to inline these functions
      norm = numba.float32(0.0)
      for ii in range(3):
        norm += v0v1c[ii] * v0v1c[ii]
      norm = np.sqrt(norm) + 1.0e-35
      for ii in range(3):
        v0v1c[ii] = v0v1c[ii] / norm

      p0p1c = np.cross(p0,p1)
      # p0p1c /= (np.linalg.norm(p0p1c))+1.0e-35
      #p0p1c = vectnorm(p0p1c) # faster to inline these functions
      norm = numba.float32(0.0)
      for ii in range(3):
        norm += p0p1c[ii] * p0p1c[ii]
      norm = np.sqrt(norm) + 1.0e-35
      for ii in range(3):
        p0p1c[ii] = p0p1c[ii] / norm

      A[:,1] = p0p1c
      B[1,:] = v0v1c
      A[:,2] = np.cross(p0,p0p1c)
      B[2,:] = np.cross(v0,v0v1c)
      AB[counter, :,:] = A.dot(B)
      #AB = np.reshape(AB, (1,3,3))
      #quats[counter,:] = rotlib.om2quL(AB)
      counter += 1

  return AB