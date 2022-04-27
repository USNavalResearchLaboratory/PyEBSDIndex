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

import numba
import numpy as np

from pyebsdindex import rotlib


RADEG = 180.0/np.pi

tempdir = PurePath("/tmp" if platform.system() == "Darwin" else tempfile.gettempdir())
tempdir = tempdir.joinpath('numba')
environ["NUMBA_CACHE_DIR"] = str(tempdir)


class BandVote:
  def __init__(self, tripLib, angTol=3.0, high_fidelity=True):
    self.tripLib = tripLib
    self.phase_name = self.tripLib.phaseName
    self.phase_sym = self.tripLib.symmetry
    self.lattice_param = self.tripLib.latticeParameter
    self.angTol = angTol
    self.n_band_early_exit = 8
    self.high_fidelity = high_fidelity
    # these lookup tables are used to order the index for the pole-family when
    # sorting triplet angles from low to high.
    LUTA = np.array([[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]],dtype=np.int64)
    LUTB = np.array([[0,1,2],[1,0,2],[0,2,1],[2,0,1],[1,2,0],[2,1,0]],dtype=np.int64)

    LUT = np.zeros((3,3,3,3),dtype=np.int64)
    for i in range(6):
      LUT[:,LUTA[i,0],LUTA[i,1],LUTA[i,2]] = LUTB[i,:]
    self.LUT = np.asarray(LUT).copy()

  def tripvote(self, band_norms, band_intensity = None, goNumba = True, verbose=0):
    tic0 = timer()
    nfam = self.tripLib.family.shape[0]
    bandnorms = np.squeeze(band_norms)
    n_bands = np.int64(bandnorms.size/3)
    if band_intensity is None:
      band_intensity = np.ones((n_bands))
    tic = timer()
    bandangs = np.abs(bandnorms.dot(bandnorms.T))
    bandangs = np.clip(bandangs, -1.0, 1.0)
    bandangs  = np.arccos(bandangs)*RADEG


    if goNumba == True:
      accumulator, bandFam, bandRank, band_cm = self.tripvote_numba(bandangs, self.LUT, self.angTol, self.tripLib.tripAngles, self.tripLib.tripID, nfam, n_bands)
    else:
      count = 0
      accumulator = np.zeros((nfam,n_bands),dtype=np.int32)
      for i in range(n_bands):
        for j in range(i+1,n_bands):
          for k in range(j+1, n_bands):
            angtri = np.array([bandangs[i,j], bandangs[i,k], bandangs[j,k]])
            srt = np.argsort(angtri)
            srt2 = np.array(self.LUT[:, srt[0], srt[1], srt[2]])
            unsrtFID = np.argsort(srt2)
            angtriSRT = np.array(angtri[srt])
            angTest = (np.abs(self.tripLib.tripAngles - angtriSRT)) <= self.angTol
            angTest = np.all(angTest, axis = 1)
            wh = angTest.nonzero()[0]

            for q in wh:
              f = self.tripLib.tripID[q,:]
              f = f[unsrtFID]
              accumulator[f, [i,j,k]] += 1


      mxvote = np.amax(accumulator, axis = 0)
      tvotes = np.sum(accumulator, axis = 0)
      band_cm = np.zeros(n_bands)



      for i in range(n_bands):
        if tvotes[i] < 1:
          band_cm[i] = 0.0
        else:
          srt = np.argsort(accumulator[:,i])
          band_cm[i] = (accumulator[srt[-1], i] - accumulator[srt[-2], i])/tvotes[i]

      bandFam = np.argmax(accumulator, axis=0)
      bandRank = (n_bands - np.arange(n_bands))/n_bands * band_cm * mxvote
    #print(np.sum(accumulator))
    #print(accumulator)
    #print(bandRank)
    #print(tvotes, band_cm, mxvote)
    #print('vote loops: ', timer() - tic)
    if verbose > 2:
      print('band Vote time:',timer() - tic)
    tic = timer()

    # avequat,fit,bandmatch,nMatch = self.band_vote_refine(bandnorms,bandRank,bandFam, angTol = self.angTol)
    # if nMatch == 0:
    #   srt = np.argsort(bandRank)
    #   for i in range(np.min([5, n_bands])):
    #     bandRank2 = bandRank
    #     bandRank2[srt[i]] = -1.0
    #     #avequat, fit, bandmatch, nMatch = self.band_vote_refine(bandnorms,bandRank2,bandFam,self.tripLib.completelib,self.angTol)
    #     avequat,fit,bandmatch,nMatch = self.band_vote_refine(bandnorms,bandRank2,bandFam)
    #     if nMatch > 2:
    #       break
    #print('refinement: ', timer() - tic)
    #print('tripvote: ',timer() - tic0)
    sumaccum = np.sum(accumulator)
    bandRank_arg = np.argsort(bandRank).astype(np.int64)
    test  = 0
    fit = 1000.0
    nMatch = -1
    avequat = np.zeros(4, dtype=np.float32)
    polematch = np.array([-1])
    whGood = -1

    angTable = self.tripLib.completelib['angTable']
    sztable = angTable.shape
    famIndx = self.tripLib.completelib['famIndex']
    nFam = self.tripLib.completelib['nFamily']
    poles = self.tripLib.completelib['polesCart']
    angTol = self.angTol
    n_band_early = np.int64(self.n_band_early_exit)

    # this will check the vote, and return the exact band matching to specific poles of the best fitting solution.
    fit, polematch, nMatch, whGood, ij, R = \
      self.band_index_nb(poles, bandRank_arg, bandFam,  famIndx, nFam, angTable, bandnorms, angTol, n_band_early)

    if verbose > 2:
      print('band index: ',timer() - tic)
    tic = timer()

    cm2 = 0.0
    if nMatch >=2:
      if self.high_fidelity == True:

        avequat, fit = self.refine_orientation_quest(bandnorms, whGood, polematch, weights=band_intensity)
        fit = np.arccos(np.clip(fit, -1.0, 1.0))*RADEG
        #avequat, fit = self.refine_orientation(bandnorms,whGood,polematch)
      else:
        avequat = rotlib.om2qu(R)
      whmatch = np.nonzero(polematch >= 0)[0]
      cm = np.mean(band_cm[whmatch])
      whfam = self.tripLib.completelib['poleFamID'][polematch[whmatch]]
      cm2 = np.sum(accumulator[[whfam], [whmatch]]).astype(np.float32)
      cm2 /= np.sum(accumulator.clip(1))

    if verbose > 2:
      print('refinement: ', timer() - tic)
      print('all: ',timer() - tic0)
    return avequat, fit, cm2, polematch, nMatch, ij, sumaccum



  def band_index(self,bandnorms,bnd1,bnd2,familyLabel,angTol=3.0, verbose = 0):

    #nBands = np.int32(bandnorms.size/3)
    angTable = self.tripLib.completelib['angTable']
    sztable = angTable.shape
    #whGood = -1
    famIndx = self.tripLib.completelib['famIndex']
    nFam = self.tripLib.completelib['nFamily']
    poles = self.tripLib.completelib['polesCart']
    #ang01 = 0.0
    # need to check that the two selected bands are not parallel.
    #v0 = bandnorms[bnd1, :]
    #f0 = familyLabel[bnd1]
    #v1 = bandnorms[bnd2, :]
    #f1 = familyLabel[bnd2]
    #ang01 = np.clip(np.dot(v0, v1), -1.0, 1.0)
    #ang01 = np.arccos(ang01)*RADEG
    #if ang01 < angTol: # the two poles are parallel, send in another two poles if available.
    #  return 360.0, 0, whGood, -1

    #wh01 = np.nonzero(np.abs(angTable[famIndx[f0], famIndx[f1]:np.int(famIndx[f1]+nFam[f1])] - ang01) < angTol)[0]

    #n01 = wh01.size
    #if  n01 == 0:
    #  return 360.0, 0, whGood, -1

    #wh01 += famIndx[f1]
    #p0 = poles[famIndx[f0], :]
    #print('pre first loop: ',timer() - tic)
    #tic = timer()
    # place numba code here ...



    #fit, polematch, R, nGood, whGood = self.band_vote_refine_loops1(poles,v0,v1, p0, wh01, bandnorms, angTol)
    fit,polematch,R,nGood,whGood = self.band_vote_refine_loops1(poles, bnd1, bnd2, familyLabel,  famIndx, nFam, angTable, bandnorms, angTol)

    #print('numba first loops',timer() - tic)
    #whGood = np.nonzero(angFit < angTol)[0]
    #nGood = np.int64(whGood.size)
    #if nGood < 3:
    #  return 360.0, -1, -1, -1

    #fit = np.mean(angFit[whGood])
    #print('all bindexed time', timer()-tic0)
    return fit, nGood, whGood, polematch

  def refine_orientation(self,bandnorms, whGood, polematch):
    tic = timer()
    poles = self.tripLib.completelib['polesCart']
    nGood = whGood.size
    n2Fit = np.int64(np.product(np.arange(2)+(nGood-2+1))/np.int64(2))
    whGood = np.asarray(whGood,dtype=np.int64)
    #AB, ABgood = self.orientation_refine_loops_am(nGood,whGood,poles,bandnorms,polematch,n2Fit)
    # tic = timer()
    # quats = rotlib.om2quL(AB[ABgood.nonzero()[0], :, :])
    # print("om2qu", timer() - tic)
    # tic = timer()
    # avequat = rotlib.quatave(quats)

    AB, weights = self.orientation_refine_loops_am(nGood, whGood, poles, bandnorms, polematch, n2Fit)

    wh_weight = np.nonzero(weights < 359.0)[0]
    quats = rotlib.om2quL(AB[wh_weight, :, :])

    expw  = weights[wh_weight]

    rng = expw.max()-expw.min()
    #print(rng, len(wh_weight))
    if rng > 1e-6:
      expw -= expw.min()
      #print(np.arccos(1.0 - expw)*RADEG)
      expw = np.exp(-expw/(0.5*(rng)))
      expw /= np.sum(expw)
      #print(quats)
      #print(expw)
      #print(expw*len(wh_weight))
      avequat = rotlib.quatave(quats * np.expand_dims(expw, axis=-1))
      #print(avequat)
    else:
      avequat = rotlib.quatave(quats)


    test = rotlib.quat_vectorL(avequat,bandnorms[whGood,:])

    tic = timer()
    test = np.sum(test * poles[polematch[whGood], :], axis = 1)
    test = np.arccos(np.clip(test, -1.0, 1.0))*RADEG


    fit = np.mean(test)

    #print('fitting: ',timer() - tic)
    return avequat, fit

  def refine_orientation_quest(self,bandnorms, whGood, polematch, weights = None):
    tic = timer()
    poles = self.tripLib.completelib['polesCart']
    whGood = np.asarray(whGood,dtype=np.int64)

    if weights is None:
      weights = np.ones((bandnorms.shape[0]), dtype=np.float64)
    weightsn = weights[whGood]
    weightsn /= np.sum(weightsn)
    #print(weightsn)
    pflt = (np.asarray(poles[polematch[whGood], :], dtype=np.float64))
    bndnorm = (np.asarray(bandnorms[whGood, :], dtype=np.float64))

    avequat, fit = self.orientation_quest(pflt, bndnorm, weightsn)

    return avequat, fit


  @staticmethod
  @numba.jit(nopython=True, cache=True,fastmath=True,parallel=False)
  def tripvote_numba( bandangs, LUT, angTol, tripAngles, tripID, nfam, n_bands):
    LUTTemp = np.asarray(LUT).copy()
    accumulator = np.zeros((nfam, n_bands), dtype=np.int32)
    tshape = np.shape(tripAngles)
    ntrip = int(tshape[0])
    count  = 0.0
    #angTest2 = np.zeros(ntrip, dtype=numba.boolean)
    #angTest2 = np.empty(ntrip,dtype=numba.boolean)
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
            angTest2 = (angTest[q,0] + angTest[q,1] + angTest[q,2]) == 3
            if angTest2:
              f = tripID[q,:]
              f = f[unsrtFID]
              accumulator[f[0],i] += 1
              accumulator[f[1],j] += 1
              accumulator[f[2],k] += 1
              t1 = False
              t2 = False
              t3 = False
              if np.abs(angtriSRT[0] - angtriSRT[1]) < angTol:
                accumulator[f[0],i] += 1
                accumulator[f[1],k] += 1
                accumulator[f[2],j] += 1
                t1 = True
              if np.abs(angtriSRT[1] - angtriSRT[2]) < angTol:
                accumulator[f[0],j] += 1
                accumulator[f[1],i] += 1
                accumulator[f[2],k] += 1
                t2 = True
              if np.abs(angtriSRT[2] - angtriSRT[0]) < angTol:
                accumulator[f[0],k] += 1
                accumulator[f[1],j] += 1
                accumulator[f[2],i] += 1
                t3 = True
              if (t1 and t2 and t3):
                accumulator[f[0],k] += 1
                accumulator[f[1],i] += 1
                accumulator[f[2],j] += 1

    mxvote = np.zeros(n_bands, dtype=np.int32)
    tvotes = np.zeros(n_bands, dtype=np.int32)
    band_cm = np.zeros(n_bands, dtype=np.float32)
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

  @staticmethod
  @numba.jit(nopython=True, cache=True, fastmath=True,parallel=False)
  def band_index_nb(poles, bandRank_arg, familyLabel,  famIndx, nFam, angTable, bandnorms, angTol, n_band_early):
    eps = np.float32(1.0e-12)
    nBnds = bandnorms.shape[0]

    whGood_out = np.zeros(nBnds, dtype=np.int64)-1


    nMatch = np.int64(-1)
    Rout = np.zeros((1,3,3), dtype=np.float32)
    #Rout[0,0,0] = 1.0; Rout[0,1,1] = 1.0; Rout[0,2,2] = 1.0
    polematch_out = np.zeros((nBnds),dtype=np.int64) - 1
    pflt = np.asarray(poles, dtype=np.float32)
    bndnorm = np.transpose(np.asarray(bandnorms, dtype=np.float32))

    fit = np.float32(360.0)
    fitout = np.float32(360.0)
    R = np.zeros((1, 3, 3), dtype=np.float32)
    #fit = np.float32(360.0)
    #whGood = np.zeros(nBnds, dtype=np.int64) - 1
    nGood = np.int64(-1)

    for ii in range(nBnds-1):
      for jj in range(ii+1,nBnds):

        polematch = np.zeros((nBnds),dtype=np.int64) - 1

        bnd1 = bandRank_arg[-1 - ii]
        bnd2 = bandRank_arg[-1 - jj]

        v0 = bandnorms[bnd1,:]
        f0 = familyLabel[bnd1]
        v1 = bandnorms[bnd2,:]
        f1 = familyLabel[bnd2]
        ang01 = np.dot(v0,v1)
        if ang01 > np.float32(1.0):
          ang01 = np.float32(1.0-eps)
        if ang01 < np.float32(-11.0):
          ang01 = np.float32(-1.0+eps)

        ang01 = np.arccos(ang01) * RADEG
        if ang01 < angTol:  # the two poles are parallel, send in another two poles if available.
          continue

        wh01 = np.nonzero(np.abs(angTable[famIndx[f0],famIndx[f1]:np.int64(famIndx[f1] + nFam[f1])] - ang01) < angTol)[0]

        n01 = wh01.size
        if n01 == 0:
          continue

        wh01 += famIndx[f1]
        p0 = poles[famIndx[f0],:]

        n01 = wh01.size
        v0v1c = np.cross(v0,v1)
        v0v1c /= np.linalg.norm(v0v1c)
        # attempt to see which solution gives the best match to all the poles
        # best is measured as the number of poles that are within tolerance,
        # divided by the angular deviation.
        # Use the TRIAD method for finding the rotation matrix

        Rtry = np.zeros((n01,3,3), dtype = np.float32)

        #score = np.zeros((n01), dtype = np.float32)
        A = np.zeros((3,3), dtype = np.float32)
        B = np.zeros((3,3), dtype = np.float32)
        #AB = np.zeros((3,3),dtype=np.float32)
        b2 = np.cross(v0,v0v1c)
        B[0,:] = v0
        B[1,:] = v0v1c
        B[2,:] = b2
        A[:,0] = p0
        score = -1.0

        for i in range(n01):
          p1 = poles[wh01[i],:]
          ntemp = np.linalg.norm(p1) + 1.0e-35
          p1 = p1 / ntemp
          p0p1c = np.cross(p0,p1)
          ntemp = np.linalg.norm(p0p1c) + 1.0e-35
          p0p1c = p0p1c / ntemp
          A[:,1] = p0p1c
          A[:,2] = np.cross(p0,p0p1c)
          AB = A.dot(B)
          Rtry[i,:,:] = AB

          testp = (AB.dot(bndnorm))
          test = pflt.dot(testp)
          #print(test.shape)
          angfitTry = np.zeros((nBnds), dtype = np.float32)
          #angfitTry = np.max(test,axis=0)
          for j in range(nBnds):
            angfitTry[j] = np.max(test[:,j])
            angfitTry[j] = -1.0 if angfitTry[j] < -1.0 else angfitTry[j]
            angfitTry[j] =  1.0 if angfitTry[j] >  1.0 else angfitTry[j]

          #angfitTry = np.clip(np.amax(test,axis=0),-1.0,1.0)

          angfitTry = np.arccos(angfitTry) * RADEG
          whMatch = np.nonzero(angfitTry < angTol)[0]
          nmatch = whMatch.size
          #scoreTry = np.float32(nmatch) * np.mean(np.abs(angTol - angfitTry[whMatch]))
          scoreTry = np.float32(nmatch) /( np.mean(angfitTry[whMatch]) + 1e-6)
          if scoreTry > score:
            score = scoreTry
            angFit = angfitTry
            for j in range(nBnds):
              polematch[j] = np.argmax(test[:,j]) * ( 2*np.int32(angfitTry[j] < angTol)-1)
            R[0, :,:] = Rtry[i,:,:]


        whGood = (np.nonzero(angFit < angTol)[0]).astype(np.int64)
        nGood = max(np.int64(whGood.size), np.int64(0))

        if nGood < 3:
          continue
          #return 360.0,-1,-1,-1
          #whGood = -1*np.ones((1), dtype=np.int64)
          #fit = np.float32(360.0)
          #polematch[:] = -1
          #nGood = np.int64(-1)
        else:
          fit = np.float32(0.0)
          for q in range(nGood):
            fit += np.float32(angFit[whGood[q]])
          fit /= np.float32(nGood)


        if nGood >= (n_band_early):
          fitout = fit
          nMatch = nGood
          whGood_out = whGood
          polematch_out = polematch
          Rout = R
          break
        else:
          if nMatch < nGood:
            fitout = np.float32(fit)
            nMatch = nGood
            whGood_out = whGood
            polematch_out = polematch
            Rout = R
          elif nMatch == nGood:
            if fitout > fit:
              fitout = np.float32(fit)
              nMatch = nGood
              whGood_out = whGood
              polematch_out = polematch
              Rout = R
      if nMatch >= (n_band_early):
        break
    #quatout = rotlib.om2quL(Rout)
    return fitout, polematch_out,nMatch, whGood_out, (ii,jj), Rout

  @staticmethod
  @numba.jit(nopython=True, cache=True, fastmath=True,parallel=False)
  def orientation_refine_loops_triad(nGood, whGood, poles, bandnorms, polematch, n2Fit):
    #uses the TRIAD method for getting rotation matrix from imperfect poles.
    quats = np.zeros((n2Fit, 4), dtype=np.float32)
    counter = 0
    A = np.zeros((3, 3), dtype=np.float32)
    B = np.zeros((3, 3), dtype=np.float32)
    AB = np.zeros((n2Fit, 3, 3), dtype=np.float32)
    whgood2 = np.zeros(n2Fit, dtype=np.int32)
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
        #print(norm)
        if norm > (0.087): # these vectors are not parallel (greater than 5 degrees)
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
          whgood2[counter] = 1
          #AB = np.reshape(AB, (1,3,3))
          #quats[counter,:] = rotlib.om2quL(AB)
          counter += 1
        else: # the two are parallel - throwout the result.
          whgood2[counter] = 0
          counter +=1
    return AB, whgood2

  @staticmethod
  @numba.jit(nopython=True,cache=True,fastmath=True,parallel=False)
  def orientation_refine_loops_am(nGood,whGood,poles,bandnorms,polematch,n2Fit):
    # this uses the method laid out by A. Morawiec 2020 Eq.4 for getting rotation matrix
    # from imperfect poles
    counter = 0

    pflt = np.transpose(np.asarray(poles[polematch[whGood], :], dtype=np.float32))
    bndnorm = np.transpose(np.asarray(bandnorms[whGood,:], dtype=np.float32))

    A = np.zeros((3, 3), dtype=np.float32)
    B = np.zeros((3, 3), dtype=np.float32)
    AB = np.zeros((n2Fit, 3, 3),dtype=np.float32)
    #whgood2 = np.zeros(n2Fit, dtype=np.int32)
    whgood2 = np.zeros(n2Fit, dtype=np.float32)

    for i in range(nGood):
      v0 = bandnorms[whGood[i],:]
      p0 = poles[polematch[whGood[i]],:]

      for j in range(i + 1,nGood):
        v1 = bandnorms[whGood[j],:]
        p1 = poles[polematch[whGood[j]],:]
        v0v1c = np.cross(v0,v1)
        p0p1c = np.cross(p0,p1)
        p0p1add = p0 + p1
        v0v1add = v0 + v1
        p0p1sub = p0 - p1
        v0v1sub = v0 - v1

        normPCross = numba.float32(0.0)
        normVCross = numba.float32(0.0)
        normPAdd = numba.float32(0.0)
        normVAdd = numba.float32(0.0)
        normPSub = numba.float32(0.0)
        normVSub = numba.float32(0.0)

        for ii in range(3):
          normPCross += p0p1c[ii] * p0p1c[ii]
          normVCross += v0v1c[ii] * v0v1c[ii]
          normPAdd += p0p1add[ii] * p0p1add[ii]
          normVAdd += v0v1add[ii] * v0v1add[ii]
          normPSub += p0p1sub[ii] * p0p1sub[ii]
          normVSub += v0v1sub[ii] * v0v1sub[ii]

        normPCross = np.sqrt(normPCross) + 1.0e-35
        normVCross = np.sqrt(normVCross) + 1.0e-35
        normPAdd = np.sqrt(normPAdd) + 1.0e-35
        normVAdd = np.sqrt(normVAdd) + 1.0e-35
        normPSub = np.sqrt(normPSub) + 1.0e-35
        normVSub = np.sqrt(normVSub) + 1.0e-35

        # print(norm)
        if normVCross > (0.087):  # these vectors are not parallel (greater than 5 degrees)
          for ii in range(3):
            v0v1c[ii] /= normVCross
            p0p1c[ii] /= normPCross
            v0v1add[ii] /= normVAdd
            p0p1add[ii] /= normPAdd
            v0v1sub[ii] /= normVSub
            p0p1sub[ii] /= normPSub

          A[:,0] = p0p1c
          B[0,:] = v0v1c

          A[:,1] = p0p1add
          B[1,:] = v0v1add

          A[:,2] = p0p1sub
          B[2,:] = v0v1sub
          R = A.dot(B)
          AB[counter,:,:] = A.dot(B)

          # test the fit of each candidate
          testp = (R.dot(bndnorm))
          #test = pflt.dot(testp)
          test = np.sum(pflt*testp, axis=0)
          #print(test.shape)
          #angfitTry = np.zeros((nGood), dtype=np.float32)
          # angfitTry = np.max(test,axis=0)
          #for qq in range(nGood):
          #  angfitTry[qq] = np.max(test[:, qq])
          #  angfitTry[qq] = -1.0 if angfitTry[qq] < -1.0 else angfitTry[qq]
          #  angfitTry[qq] = 1.0 if angfitTry[qq] > 1.0 else angfitTry[qq]
          #angfitTry = np.mean(np.arccos(angfitTry) * RADEG)
          #print(test)


          #whgood2[counter] = 1
          whgood2[counter] = 1.0 - np.mean(test)
          #print(1.0 - np.mean(test))
          counter += 1
        else:  # the two are parallel - throwout the result.
          whgood2[counter] = np.float32(360.0)
          counter += 1
    return AB,whgood2

  @staticmethod
  @numba.jit(nopython=True, cache=True, fastmath=True, parallel=False)
  def orientation_quest(poles, bandnorms, weights):
    # this uses the Quaternion Estimator AKA quest algorithm.

    #pflt = (np.asarray(poles[polematch[whGood], :], dtype=np.float32))
    #bndnorm = (np.asarray(bandnorms[whGood, :], dtype=np.float32))
    pflt = np.asarray(poles, dtype=np.float64)
    bndnorm = np.asarray(bandnorms, dtype=np.float64)
    npoles = pflt.shape[0]
    wn = (np.asarray(weights, dtype=np.float64)).reshape(npoles, 1)
    #wn = np.ones((nGood,1), dtype=np.float32)/np.float32(nGood)  #(weights[whGood]).reshape(nGood,1)
    wn /= np.sum(wn)

    I = np.zeros((3,3), dtype=np.float64)
    I[0,0] = 1.0 ; I[1,1] = 1.0 ; I[2,2] = 1.0
    q = np.zeros((4), dtype=np.float64)

    B = (wn * bndnorm).T @ pflt
    S = B + B.T
    z = np.asarray(np.sum(wn * np.cross(bndnorm, pflt), axis = 0), dtype=np.float64)
    S2 = S @ S
    det = np.linalg.det(S)
    k = (S[1,1]*S[2,2] - S[1,2]*S[2,1]) + (S[0,0]*S[2,2] - S[0,2]*S[2,0]) + (S[0,0]*S[1,1] - S[1,0]*S[0,1])
    sig = 0.5 * (S[0,0] + S[1,1] + S[2,2])
    sig2 = sig * sig
    d = z.T @ S2 @ z
    c = det + (z.T @ S @ z)
    b = sig2 + (z.T @ z)
    a = sig2 -k

    lam = 1.0
    tol = 1.0e-6
    iter = 0
    dlam = 1e6
    #for i in range(10):
    while (dlam > tol) and (iter < 10):
      lam0 = lam
      lam = lam - (lam**4 - (a + b) * lam**2 - c * lam + (a * b + c * sig - d))/ (4 * lam**3 - 2 * (a + b) * lam - c)
      dlam = np.fabs(lam0-lam)
      iter += 1

    beta = lam - sig
    alpha = lam ** 2 - sig2 + k
    gamma = (lam + sig) * alpha - det
    X = np.asarray( (alpha * I + beta * S + S2), dtype=np.float64)  @ z
    qn = np.float64(0.0)
    qn += gamma ** 2 + X[0] **2 + X[1] **2 + X[2] **2
    qn = np.sqrt(qn)
    q[0] = gamma
    q[1:4] = X[0:3]
    q /= qn
    if (np.sign(gamma) < 0):
      q *= -1.0

    return q, lam

  def pairVoteOrientation(self,bandnormsIN,goNumba=True):
    tic0 = timer()
    nfam = self.tripLib.family.shape[0]
    bandnorms = np.squeeze(bandnormsIN)
    n_bands = np.int64(bandnorms.size / 3)

    bandangs = np.abs(bandnorms.dot(bandnorms.T))
    bandangs = np.clip(bandangs,-1.0,1.0)
    bandangs = np.arccos(bandangs) * RADEG

    angTable = self.tripLib.completelib['angTable']
    sztable = angTable.shape
    whGood = -1
    famIndx = self.tripLib.completelib['famIndex']
    nFam = self.tripLib.completelib['nFamily']
    poles = self.tripLib.completelib['polesCart']
    angTableReduce = angTable[famIndx,:]
    polesReduce = poles[famIndx,:]
    qsym = self.tripLib.qsymops
    tic = timer()


    if goNumba == True:
      solutions, nsolutions, solutionVotes, solSrt = self.pairVoteOrientationNumba(bandnorms,bandangs,qsym,angTableReduce,poles,polesReduce, self.angTol)
    else:
      solutions = np.empty((500, 24, 4), dtype=np.float32)
      solutions[0,:,:] = rotlib.quat_multiply(qsym, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
      solutionVotes = np.zeros(500, dtype=np.int32)
      nsolutions = 1
      soltol = np.cos(5.0/RADEG/2.0)

      A = np.zeros((3,3), dtype=np.float32)
      B = np.zeros((3,3), dtype=np.float32) # used for TRIAD calculations

      for i in range(n_bands):
        for j in range(i + 1,n_bands):

            angPair = bandangs[i,j]
            if angPair > 10.0:
              angTest = (np.abs(angTableReduce - angPair)) <= self.angTol
              wh = angTest.nonzero()
              if len(wh[0] > 0):

                v0 = bandnorms[i,:]
                v1 = bandnorms[j,:]
                v0v1c = np.cross(v0,v1)
                v0v1c /= np.linalg.norm(v0v1c)
                B[0,:] = v0
                B[1,:] = v0v1c
                B[2,:] = np.cross(v0,v0v1c)
                for k in range(len(wh[0])):

                  p0 = polesReduce[wh[0][k],:]
                  p1 = poles[wh[1][k],:]
                  p0p1c = np.cross(p0,p1)
                  p0p1c /= np.linalg.norm(v0v1c)
                  A[:,0] = p0
                  A[:,1] = p0p1c
                  A[:,2] = np.cross(p0,p0p1c)
                  AB = A.dot(B)

                  qAB = rotlib.om2qu(AB)
                  qABinv = rotlib.quatconj(qAB)
                  #qABsym = rotlib.quat_multiply(qsym, qAB)

                  solutionFound = False
                  for q in range(nsolutions):
                    #rotlib.quat_multiplyLNN(q1,q2,n,intype,p=P)

                    soltest = np.max( np.abs( (rotlib.quat_multiply((solutions[q,:,:]), qABinv))[:,0] ))

                    if soltest >= soltol:
                      solutionVotes[q] += 1
                      solutionFound = True
                  if solutionFound == False:
                    solutions[nsolutions, :, :] = rotlib.quat_multiply(qsym, qAB)
                    solutionVotes[nsolutions] += 1
                    nsolutions += 1

      solSrt = np.argsort(solutionVotes)
    #print(nsolutions, solutionVotes[solSrt[-10:]])
    mxvote = np.max(solutionVotes)
    #print(timer()-tic)
    tic = timer()
    if mxvote > 0:
      whmxvotes = np.nonzero(solutionVotes == mxvote)
      nmxvote = len(whmxvotes[0])
      fit = np.zeros(nmxvote, dtype=np.float32)
      avequat = np.zeros((nmxvote,4),dtype=np.float32)
      nMatch = np.zeros((nmxvote),dtype=np.float32)
      poleMatch = np.zeros((nmxvote, n_bands), dtype=np.int32) - 1
      #cm = (mxvote-solutionVotes[solSrt[-nmxvote-1]])/np.sum(solutionVotes)
      cm = mxvote/np.sum(solutionVotes)
      for q in range(nmxvote):
        testbands = rotlib.quat_vector(solutions[whmxvotes[0][q], 0, : ], bandnorms)
        fittest = (testbands.dot(poles.T)).clip(-1.0, 1.0)
        poleMatch1 = np.argmax(fittest, axis = 1)
        fittemp = np.arccos(np.max(fittest,axis=1)) * RADEG

        whGood = np.nonzero(fittemp < self.angTol)
        nMatch[q] = len(whGood[0])
        poleMatch[q,whGood[0]] = poleMatch1[whGood[0]]

        avequat1, fit1 = self.refine_orientation(bandnorms,whGood[0],poleMatch1)

        fit[q] = fit1
        avequat[q,:] = avequat1

      keep = np.argmax(nMatch)
      #print(timer() - tic)
      return avequat[keep,:],fit[keep],cm,poleMatch[keep,:],nMatch[keep],(0,0)

    else: # no solutions
      fit = 1000.0
      nMatch = -1
      avequat = np.zeros(4,dtype=np.float32)
      polematch = np.array([-1])
      whGood = -1
      print(timer() - tic)
      return avequat,fit,-1,polematch,nMatch,(0,0)


  @staticmethod
  @numba.jit(nopython=True,cache=True,fastmath=True,parallel=False)
  def pairVoteOrientationNumba(bandnorms,bandangs, qsym, angTableReduce, poles, polesReduce, angTol):
    n_bands = bandnorms.shape[0]
    nsym = qsym.shape[0]
    solutions = np.empty((500,24,4),dtype=np.float32)
    solutions[0,:,:] = rotlib.quat_multiplyL(qsym,np.array([1.0,0.0,0.0,0.0],dtype=np.float32))
    solutionVotes = np.zeros(500, dtype=np.int32)
    nsolutions = 1
    soltol = np.cos(5.0 / RADEG / 2.0)

    A = np.empty((3,3),dtype=np.float32)
    B = np.empty((3,3),dtype=np.float32)  # used for TRIAD calculations
    AB = np.empty((1,3,3), dtype=np.float32)
    qAB = np.empty((1,4), dtype=np.float32)
    qABinv = np.empty((1,4),dtype=np.float32)
    soltemp = np.empty((nsym,4), dtype=np.float32)

    for i in range(n_bands):
      for j in range(i + 1,n_bands):

        angPair = bandangs[i,j]
        if angPair > 10.0:
          angTest = (np.abs(angTableReduce - angPair)) <= angTol
          wh = angTest.nonzero()
          if len(wh[0] > 0):

            v0 = bandnorms[i,:]
            v1 = bandnorms[j,:]
            v0v1c = np.cross(v0,v1)
            v0v1c /= np.linalg.norm(v0v1c)
            B[0,:] = v0
            B[1,:] = v0v1c
            B[2,:] = np.cross(v0,v0v1c)
            for k in range(len(wh[0])):

              p0 = polesReduce[wh[0][k],:]
              p1 = poles[wh[1][k],:]
              p0p1c = np.cross(p0,p1)
              p0p1c /= np.linalg.norm(v0v1c)
              A[:,0] = p0
              A[:,1] = p0p1c
              A[:,2] = np.cross(p0,p0p1c)
              AB[0,:,:] = A.dot(B)

              qAB = rotlib.om2quL(AB)
              qABinv = rotlib.quatconjL(qAB)
              #print(qABinv.shape)
              # qABsym = rotlib.quat_multiply(qsym, qAB)

              solutionFound = False
              for q in range(nsolutions):
                # rotlib.quat_multiplyLNN(q1,q2,n,intype,p=P)

                soltemp = np.copy(solutions[q,:,:])
                #print(soltemp.shape)
                soltemp =  rotlib.quat_multiplyL(soltemp,qABinv)

                soltest = -1.0
                for qq in range(nsym):
                  if soltemp[qq,0] > soltest:
                    soltest = soltemp[qq,0]

                if soltest >= soltol:
                  solutionVotes[q] += 1
                  solutionFound = True
              if solutionFound == False:
                solutions[nsolutions,:,:] = rotlib.quat_multiplyL(qsym,qAB)
                solutionVotes[nsolutions] += 1
                nsolutions += 1

    solSrt = np.argsort(solutionVotes)

    return solutions, nsolutions, solutionVotes, solSrt
