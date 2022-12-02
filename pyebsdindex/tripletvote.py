'''This software was developed by employees of the US Naval Research Laboratory (NRL), an
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
The US Naval Research Laboratory Date: 21 Aug 2020'''

from os import environ
from pathlib import PurePath
import platform
import tempfile
from timeit import default_timer as timer

import numpy as np
import numba

from pyebsdindex import crystal_sym, rotlib, crystallometry


RADEG = 180.0/np.pi

tempdir = PurePath("/tmp" if platform.system() == "Darwin" else tempfile.gettempdir())
tempdir = tempdir.joinpath('numba')
environ["NUMBA_CACHE_DIR"] = str(tempdir)

def addphase(libtype=None, phasename=None,
             spacegroup=None,
             latticeparameter=None,
             polefamilies=None):

  if libtype is not None:

    #set up generic FCC
    if str(libtype).upper() == 'FCC':
      if phasename is None:
        phasename = 'FCC'
      if spacegroup is None:
        spacegroup = 225
      if latticeparameter is None:
        latticeparameter = np.array([1.0, 1.0, 1.0, 90.0, 90.0, 90.0])
      else:
        latticeparameter = np.array(latticeparameter)
      if polefamilies is None:
        polefamilies = np.array([[0, 0, 2], [1, 1, 1], [0, 2, 2], [1, 1, 3]])
      else:
        polefamilies = np.array(polefamilies)

    # Set up a generic HCP
    if str(libtype).upper() == 'BCC':
      if phasename is None:
        phasename = 'BCC'
      if spacegroup is None:
        spacegroup = 229
      if latticeparameter is None:
        latticeparameter = np.array([1.0, 1.0, 1.0, 90.0, 90.0, 90.0])
      else:
        latticeparameter = np.array(latticeparameter)
      if polefamilies is None:
        polefamilies = np.array([[0, 1, 1], [0, 0, 2], [1, 1, 2], [0, 1, 3]])
      else:
        polefamilies = np.array(polefamilies)

    # Set up a generic HCP
    if str(libtype).upper() == 'HCP':
      if phasename is None:
        phasename = 'HCP'
      if spacegroup is None:
        spacegroup = 229
      if latticeparameter is None:
        latticeparameter = np.array([1.0, 1.0, 1.63, 90.0, 90.0, 120.0])
      else:
        latticeparameter = np.array(latticeparameter)
      if polefamilies is None:
        polefamilies = np.array([[1, 0, -1, 0], [1, 0, -1, 1], [0, 0, 0, 2], [1, 0, -1, 3], [1, 1, -2, 0], [1, 0, -1, 2]])
      else:
        polefamilies = np.array(polefamilies)

  else:
    if spacegroup is None:
      return addphase(libtype='FCC', latticeparameter=latticeparameter, polefamilies=polefamilies, phasename = phasename)
    if latticeparameter is None:
      latticeparameter = np.array([1.0, 1.0, 1.0, 90.0, 90.0, 90.0])
    if polefamilies is None:
      polefamilies = np.array([[0, 0, 2], [1, 1, 1], [0, 2, 2], [1, 1, 3]])

  triplib = BandIndexer(phasename=phasename, spacegroup=spacegroup,
                        latticeparameter=latticeparameter, polefamilies=polefamilies)

  triplib.build_trip_lib()
  return triplib

class BandIndexer():
  #def __init__(self, libType='FCC', phaseName=None, laticeParameter = None):
  def __init__(self,
               phasename=None,
               spacegroup = None,
               latticeparameter=None,
               polefamilies = None,
               angTol=3.0,
               nband_earlyexit = 8):
    self.phaseName = None  # User provided name of the phase.
    self.spacegroup = None  # space group id 1-230
    self.latticeParameter = None  # 6 element array for the lattice parameter.
    self.polefamilies = None  # array of integer pole normals that should have reflections
    self.npolefamilies = None  # number of unique reflector families
    self.crystalmats = None  # store the four crystal matrices useful for angle/cartisian conversions.

    self.lauecode = None  # Laue code for the space group (following DREAM.3D notation.
    self.qsymops = None  # array of quaternions that represent proper symmetry operations for the laue group

    self.pointgroup = ' '  # point group nomenclature
    self.pointgroupid = None

    self.angTol = angTol
    self.nband_earlyexit = nband_earlyexit
    self.high_fidelity = True

    # many objects to hold the information about the reflecting poles, angles between them ...
    self.angpairs = None # dictionary that will store the possible unique angles between all pole families.
    self.angtriplets = None # dictionary that will store all possible angle triplets within the pole family.
    self.completelib = None # dictionary that will hold all possible angles (non-unique) between the families and
    # all possible poles

    # these Look Up Tables are used in the sorting/unsorting of angle triplets.
    luta = np.array([[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]])
    lutb = np.array([[0, 1, 2], [1, 0, 2], [0, 2, 1], [2, 0, 1], [1, 2, 0], [2, 1, 0]])
    lut = np.zeros((3, 3, 3, 3), dtype=np.int64)
    for i in range(6):
      lut[:, luta[i, 0], luta[i, 1], luta[i, 2]] = lutb[i, :]
    self.lut = np.asarray(lut).copy()

    if phasename is None:
      self.phasename = ' '
    else:
      self.phasename = str(phasename)

    if latticeparameter is not None:
      self.setlatticeparameter(latticeparameter)


    if spacegroup is not None:
      self.setspacegroup(spacegroup)

    if polefamilies is not None:
      self.setpolefamilies(polefamilies)


  def setlatticeparameter(self, latticeparameter):
    self.latticeparameter = np.array(latticeparameter)
    self.crystalmats = crystallometry.Crystal(self.phaseName,
                                              self.latticeparameter[0],
                                              self.latticeparameter[1],
                                              self.latticeparameter[2],
                                              self.latticeparameter[3],
                                              self.latticeparameter[4],
                                              self.latticeparameter[5])

  def setspacegroup(self, spacegroup = 225):
    self.spacegroup = spacegroup
    self.lauecode = crystal_sym.spacegroup2lauenumber(self.spacegroup)
    self.qsymops = crystal_sym.laueid2symops(self.lauecode)

  def setpolefamilies(self, reflectors):
    self.polefamilies = np.array(reflectors)

  # def build_fcc(self):
  #   if self.phaseName is None:
  #     self.phaseName = 'FCC'
  #   self.pointgroup = "Cubic m3m"
  #   self.pointgroupid = 131
  #   self.spacegroup = 225
  #   self.lauecode = crystal_sym.spacegroup2lauenumber(self.spacegroup)
  #   self.qsymops = crystal_sym.laueid2symops(self.lauecode)
  #   poles = np.array([[0,0,2], [1,1,1], [0,2,2], [1,1,3]])
  #   self.build_trip_lib(poles)
  #
  # def build_dc(self):
  #   if self.phaseName is None:
  #     self.phaseName = 'Diamond Cubic'
  #   self.pointgroup = "Cubic m3m"
  #   self.pointgroupid = 131
  #   self.spacegroup = 227
  #   self.lauecode = crystal_sym.spacegroup2lauenumber(self.spacegroup)
  #   self.qsymops = crystal_sym.laueid2symops(self.lauecode)
  #   poles = np.array([[1, 1, 1], [0, 2, 2], [0, 0, 4], [1, 1, 3], [2, 2, 4], [1, 3, 3]])
  #   self.build_trip_lib(poles)
  #
  # def build_bcc(self):
  #   if self.phaseName is None:
  #     self.phaseName = 'BCC'
  #   self.pointgroup = "Cubic m3m"
  #   self.pointgroupid = 131
  #   self.spacegroup = 229
  #   self.lauecode = crystal_sym.spacegroup2lauenumber(self.spacegroup)
  #   self.qsymops = crystal_sym.laueid2symops(self.lauecode)
  #   poles = np.array([[0,1,1],[0,0,2],[1,1,2],[0,1,3]])
  #   self.build_trip_lib(poles)



  # def build_hcp(self):
  #   if self.phaseName is None:
  #     self.phaseName = 'HCP'
  #   self.pointgroup = "Hexagonal 6/mmm"
  #   self.spacegroup = 194
  #   self.lauecode = crystal_sym.spacegroup2lauenumber(self.spacegroup)
  #   self.qsymops = crystal_sym.laueid2symops(self.lauecode)
  #   poles4 = np.array([[1,0, -1, 0], [1, 0, -1, 1], [0,0, 0, 2], [1, 0, -1, 3], [1,1,-2,0], [1,0,-1,2]])
  #   self.build_hex_trip_lib(poles4)
  #
  # def build_hex_trip_lib(self, poles4):
  #   poles3 = crystal_sym.hex4poles2hex3poles(poles4)
  #   self.build_trip_lib(poles3)
  #   p3temp = self.polefamilies
  #   p4temp = crystal_sym.hex3poles2hex4poles(p3temp)
  #   self.polefamilies = p4temp

  def build_trip_lib(self):

    if self.spacegroup is None:
      print('No Space Group ID is set')
      return
    if self.latticeparameter is None:
      print('No lattice parameter is set')
      return
    if self.polefamilies is None:
      print('No pole familes are set')
      return

    crystalmats = self.crystalmats

    poles = np.array(self.polefamilies)
    if (self.lauecode == 62) or (self.lauecode == 6):
      if self.polefamilies.shape[-1] == 4:
        poles = crystal_sym.hex4poles2hex3poles(np.array(self.poles))
    poles = np.reshape(poles, (-1,3) )

    npoles = poles.shape[0]
    sympoles = [] # list of all HKL variants which does not count the invariant pole as unique.

    sympolesComplete = [] # list of all HKL variants with no duplicates
    nFamComplete = np.zeros(npoles, dtype = np.int32) # number of
    nFamily = np.zeros(npoles, dtype = np.int32)
    polesFlt = np.array(poles, dtype=np.float32) # convert the input poles to floating point (but still HKL int values)

    for i in range(npoles):
      family = self._symrotpoles(polesFlt[i, :], crystalmats) #rotlib.quat_vector(symmetry,polesFlt[i,:])
      uniqHKL = self._hkl_unique(family, reduceInversion=False)
      uniqHKL = np.flip(uniqHKL, axis=0)
      sympolesComplete.append(uniqHKL)
      nFamComplete[i] = np.reshape(sympolesComplete[-1],(-1,3)).shape[0] #np.int32((sympolesComplete[-1]).size/3)

      uniqHKL2 = self._hkl_unique(family, reduceInversion=True, rMT = crystalmats.reciprocalMetricTensor)
      nFamily[i] = np.reshape(uniqHKL2,(-1,3)).shape[0] #np.int32(uniqHKL2.size/3)
      sign = np.squeeze(self._calc_pole_dot_int(uniqHKL2, polesFlt[i, :], rMetricTensor=crystalmats.reciprocalMetricTensor))
      sign = np.atleast_1d(sign)
      whmx = (np.abs(sign)).argmax()

      sign = np.round(sign[whmx])
      uniqHKL2 *= sign

      sympoles.append(np.round(uniqHKL2))
      #sympolesN.append(self.xstalPlane2cart(family))

    sympolesComplete = np.concatenate(sympolesComplete)
    nsyms = np.sum(nFamily).astype(np.int32)
    famindx = np.concatenate( ([0],np.cumsum(nFamComplete)) )
    angs = []
    familyID = []
    polePairs = []
    for i in range(npoles):
      for j in range(i, npoles):
        fampoles = sympolesComplete[famindx[j]:famindx[j+1], :].astype(np.float32)
        #print('______', i,j)
        #print(np.round(fampoles).astype(int))

        ang = np.squeeze(self._calc_pole_dot_int(polesFlt[i, :], fampoles,
                                                 rMetricTensor=crystalmats.reciprocalMetricTensor)) # for each input pole, calculate

        ang = np.clip(ang, -1.0, 1.0)
        #sign = (ang >= 0).astype(np.float32) - (ang < 0).astype(np.float32)
        #sign = np.atleast_1d(sign)
        ang = np.round(np.arccos(np.abs(ang))*RADEG*100).astype(np.int32) # get the unique angles between the input
        ang = np.atleast_1d(ang)
        # pole, and the family poles. Angles within 0.01 deg are taken as the same.
        unqang, argunq = np.unique(ang, return_index=True)
        unqang = unqang/100.0 # revert back to the actual angle in degrees.


        wh = np.nonzero(unqang > 1.0)[0]
        nwh = wh.size
        #sign = sign[wh]
        #sign = sign.reshape(nwh,1)
        temp = np.zeros((nwh, 2, 3))
        temp[:,0,:] = np.broadcast_to(poles[i,:], (nwh, 3))
        temp[:,1,:] = np.broadcast_to(fampoles[argunq[wh],:], (nwh, 3))
        for k in range(nwh):
          angs.append(unqang[wh[k]])
          familyID.append([i,j])
          polePairs.append(temp[k,:,:])

    angs = np.squeeze(np.array(angs))
    nangs = angs.size
    familyID = np.array(familyID)
    polePairs = np.array(polePairs)

    stuff, nFamilyID = np.unique(familyID[:,0], return_counts=True)
    indx0FID = (np.concatenate( ([0],np.cumsum(nFamilyID)) ))[0:npoles]
    #print(indx0FID)
    #This completely over previsions the arrays, this is essentially 
    #N Choose K with N = number of angles and K = 3
    nlib = npoles*np.prod(np.arange(3, dtype=np.int64)+(nangs-2+1))/np.compat.long(np.math.factorial(3))
    nlib = nlib.astype(int)

    libANG = np.zeros((nlib, 3))
    libID = np.zeros((nlib, 3), dtype=int)
    counter = 0
    # now actually catalog all the triplet angles.
    for i in range(npoles):
      id0 = familyID[indx0FID[i], 0]
      for j in range(0,nFamilyID[i]):

        ang0 = angs[j + indx0FID[i]]
        id1 = familyID[j + indx0FID[i], 1]
        for k in range(j, nFamilyID[i]):
          ang1 = angs[k + indx0FID[i]]
          id2 = familyID[k + indx0FID[i], 1]

          whjk = np.nonzero( np.logical_and( familyID[:,0] == id1, familyID[:,1] == id2 ))[0]
          for q in range(whjk.size):
            ang2 = angs[whjk[q]]
            libANG[counter, :] = np.array([ang0, ang1, ang2])
            libID[counter, :] =  np.array([id0, id1, id2])
            counter += 1

    libANG = libANG[0:counter, :]
    libID = libID[0:counter, :]

    libANG, libID = self._sortlib_id(libANG, libID, findDups = True) # sorts each row of the library to make sure
    # the triplets are in increasing order.

    #print(libANG)
    #print(libANG.shape)
    # now make a table of the angle between all the poles (allowing inversino)
    angTable = self._calc_pole_dot_int(sympolesComplete, sympolesComplete, rMetricTensor=crystalmats.reciprocalMetricTensor)
    angTable = np.arccos(angTable)*RADEG
    famindx0 = ((np.concatenate( ([0],np.cumsum(nFamComplete)) ))[0:-1]).astype(dtype=np.int64)
    cartPoles = self._xstalplane2cart(sympolesComplete, rStructMatrix=crystalmats.reciprocalStructureMatrix)
    cartPoles /= np.linalg.norm(cartPoles, axis = 1).reshape(np.int64(cartPoles.size/3),1)
    completePoleFamId = np.zeros(sympolesComplete.shape[0], dtype=np.int32)
    for i in range(npoles):
      for j in range(nFamComplete[i]):
        completePoleFamId[j+famindx0[i]] = i
    self.completelib = {
                   'poles' : sympolesComplete,
                   'polesCart': cartPoles,
                   'familyid': completePoleFamId,
                   'angTable' : angTable,
                   'nFamily'  : nFamComplete,
                   'famIndex' : famindx0
                  }

    self.angpairs = {
      'familyid': familyID,
      'polepairs':polePairs,
      'angles':angs
    }
    self.angtriplets = {
      'angles': libANG,
      'familyid': libID
    }
    if (self.lauecode == 62) or (self.lauecode == 6):
      poles = crystal_sym.hex3poles2hex4poles(poles)
    self.polefamilies = poles
    self.npolefamilies = npoles

    #self.angles = angs
    #self.polePairs = polePairs
    #self.angleFamilyID = familyID
    #self.tripAngles = libANG
    #self.tripID = libID


  def bandindex(self, band_norms, band_intensity = None, band_widths=None, verbose=0):
    tic0 = timer()
    nfam = self.polefamilies.shape[0]
    bandnorms = np.squeeze(band_norms)
    n_bands = np.reshape(bandnorms, (-1,3)).shape[0] #np.int64(bandnorms.size/3)
    if band_intensity is None:
      band_intensity = np.ones((n_bands))
    tic = timer()
    bandangs = np.abs(bandnorms.dot(bandnorms.T))
    bandangs = np.clip(bandangs, -1.0, 1.0)
    bandangs  = np.arccos(bandangs)*RADEG

    tripangs = self.angtriplets['angles']
    tripid = self.angtriplets['familyid']

    accumulator, bandFam, bandRank, band_cm = self._tripvote_numba(bandangs, self.lut, self.angTol, tripangs, tripid, nfam, n_bands)


    if verbose > 2:
      print('band Vote time:',timer() - tic)
    tic = timer()

    sumaccum = np.sum(accumulator)
    bandRank_arg = np.argsort(bandRank).astype(np.int64)
    test  = 0
    fit = 1000.0
    nMatch = -1
    avequat = np.zeros(4, dtype=np.float32)
    polematch = np.array([-1])
    whGood = -1

    angTable = self.completelib['angTable']
    sztable = angTable.shape
    famIndx = self.completelib['famIndex']
    nFam = self.completelib['nFamily']
    polesCart = self.completelib['polesCart']
    angTol = self.angTol
    n_band_early = np.int64(self.nband_earlyexit)

    # this will check the vote, and return the exact band matching to specific poles of the best fitting solution.
    fit, polematch, nMatch, whGood, ij, R, fitb = \
      self._assign_bands_nb(polesCart, bandRank_arg, bandFam, famIndx, nFam, angTable, bandnorms, angTol, n_band_early)

    if verbose > 2:
      print('band index: ',timer() - tic)
    tic = timer()

    cm2 = 0.0
    if nMatch >=2:
      if self.high_fidelity == True:

        srt = np.argsort(fitb[whGood])
        whgood6 = whGood[srt[0:np.min([8, whGood.shape[0]])]]

        weights6 = band_intensity[whgood6]
        pflt6 = (np.asarray(polesCart[polematch[whgood6], :], dtype=np.float64))
        bndnorm6 = (np.asarray(bandnorms[whgood6, :], dtype=np.float64))

        avequat, fit = self._refine_orientation_quest(bndnorm6, pflt6, weights=weights6)
        fit = np.arccos(np.clip(fit, -1.0, 1.0))*RADEG
        #avequat, fit = self.refine_orientation(bandnorms,whGood,polematch)
      else:
        avequat = rotlib.om2qu(R)
      whmatch = np.nonzero(polematch >= 0)[0]
      cm = np.mean(band_cm[whmatch])
      whfam = self.completelib['familyid'][polematch[whmatch]]
      cm2 = np.sum(accumulator[[whfam], [whmatch]]).astype(np.float32)
      cm2 /= np.sum(accumulator.clip(1))

    if verbose > 2:
      print('refinement: ', timer() - tic)
      print('all: ',timer() - tic0)
    return avequat, fit, cm2, polematch, nMatch, ij, sumaccum


  def _symrotpoles(self, pole, crystalmats):

    polecart = np.matmul(crystalmats.reciprocalStructureMatrix, np.array(pole).T)
    sympolescart = rotlib.quat_vector(self.qsymops, polecart)
    return np.transpose(np.matmul(crystalmats.invReciprocalStructureMatrix, sympolescart.T))

  def _symrotdir(self, pole, crystalmats):

    polecart = np.matmul(crystalmats.directStructureMatrix, np.array(pole).T)
    sympolescart = rotlib.quat_vector(self.qsymops, polecart)
    return np.transpose(np.matmul(crystalmats.invDirectStructureMatrix, sympolescart.T))

  def _hkl_unique(self, poles, reduceInversion=True, rMT = np.identity(3)):
    """
    When given a list of integer HKL poles (plane normals), will return only the unique HKL variants

    Parameters
    ----------
    poles: numpy.ndarray (n,3) in HKL integer form.
    reduceInversion: True/False.  If True, then the any inverted crystal pole
    will also be removed from the uniquelist.  The angle between poles
    rMT: reciprocol metric tensor -- needed to calculated

    Returns
    -------
    numpy.ndarray (n,3) in HKL integer form of the unique poles.
    """

    npoles = poles.shape[0]
    intPoles =np.array(poles.round().astype(np.int32))
    mn = intPoles.min()
    intPoles -= mn
    basis = intPoles.max()+1
    basis3 = np.array([1,basis, basis**2])
    test = intPoles.dot(basis3)

    un, unq = np.unique(test, return_index=True)

    polesout = poles[unq, :]

    if reduceInversion == True:
      family = polesout
      nf = family.shape[0]
      test = self._calc_pole_dot_int(family, family, rMetricTensor = rMT)

      testSum = np.sum( (test < -0.99999).astype(np.int32)*np.arange(nf).reshape(1,nf), axis = 1)
      whpos = np.nonzero( np.logical_or(testSum < np.arange(nf), (testSum == 0)))[0]
      polesout = polesout[whpos, :]
    return polesout

  def _calc_pole_dot_int(self, poles1, poles2, rMetricTensor = np.identity(3)):

    p1 = poles1.reshape(np.int64(poles1.size / 3), 3)
    p2 = poles2.reshape(np.int64(poles2.size / 3), 3)

    n1 = p1.shape[0]
    n2 = p2.shape[0]

    t1 = p1.dot(rMetricTensor)
    t2 = rMetricTensor.dot(p2.T)
    dot = t1.dot(p2.T)
    dotnum = np.sqrt(np.diag(t1.dot(p1.T)))
    dotnum = dotnum.reshape(n1,1)
    dotnum2 = np.sqrt(np.diag(p2.dot(t2)))
    dotnum2 = dotnum2.reshape(1,n2)
    dotnum = dotnum.dot(dotnum2)

    dot /= dotnum
    dot = np.clip(dot, -1.0, 1.0)
    return dot

  def _xstalplane2cart(self, poles, rStructMatrix = np.identity(3)):
    polesout = rStructMatrix.dot(poles.T)
    return np.transpose(polesout)

  def _sortlib_id(self, libANG, libID, findDups = False):
    # will make sure that triplets are ordered from lowest to highest
    # and maintain the pole family id
    # optionally will locate any duplicates in the triplet list.

    # LUTA = np.array([[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]])
    # LUTB = np.array([[0,1,2],[1,0,2],[0,2,1],[2,0,1],[1,2,0],[2,1,0]])
    #
    # LUT = np.zeros((3,3,3,3), dtype=np.int64)
    # for i in range(6):
    #   LUT[:, LUTA[i,0], LUTA[i,1], LUTA[i,2]] = LUTB[i,:]
    lut = self.lut

    ntrips = np.int64(libANG.size / 3)
    for i in range(ntrips):
      temp = np.squeeze(libANG[i,:])
      srt = np.argsort(temp)
      libANG[i,:] = temp[srt]
      srt2 = lut[:,srt[0], srt[1], srt[2]]
      temp2 = libID[i,:]
      temp2 = temp2[srt2]
      libID[i,:] = temp2

    if findDups == True:
      angID = np.sum(np.round(libANG*100), axis = 1).astype(np.longlong)
      basis = np.longlong(libID.max() + 1)
      libID_ID = libID.dot(np.array([1,basis, basis**2]))
      UID = np.ceil(np.log10(libID_ID.max()))
      UID = np.where(UID > 2, UID, 2)
      UID = (angID * 10**UID) + libID_ID

      stuff, unq = np.unique(UID, return_index=True)
      libANG = libANG[unq, :]
      libID = libID[unq,:]
      libID_ID = libID_ID[unq]
      srt = np.argsort(libID_ID)
      libANG  = libANG[srt, :]
      libID = libID[srt, :]

    return (libANG, libID)






  # def band_index(self,bandnorms,bnd1,bnd2,familyLabel,angTol=3.0, verbose = 0):
  #
  #   #nBands = np.int32(bandnorms.size/3)
  #   angTable = self.tripLib.completelib['angTable']
  #   sztable = angTable.shape
  #   #whGood = -1
  #   famIndx = self.tripLib.completelib['famIndex']
  #   nFam = self.tripLib.completelib['nFamily']
  #   poles = self.tripLib.completelib['polesCart']
  #   #ang01 = 0.0
  #   # need to check that the two selected bands are not parallel.
  #   #v0 = bandnorms[bnd1, :]
  #   #f0 = familyLabel[bnd1]
  #   #v1 = bandnorms[bnd2, :]
  #   #f1 = familyLabel[bnd2]
  #   #ang01 = np.clip(np.dot(v0, v1), -1.0, 1.0)
  #   #ang01 = np.arccos(ang01)*RADEG
  #   #if ang01 < angTol: # the two poles are parallel, send in another two poles if available.
  #   #  return 360.0, 0, whGood, -1
  #
  #   #wh01 = np.nonzero(np.abs(angTable[famIndx[f0], famIndx[f1]:np.int(famIndx[f1]+nFam[f1])] - ang01) < angTol)[0]
  #
  #   #n01 = wh01.size
  #   #if  n01 == 0:
  #   #  return 360.0, 0, whGood, -1
  #
  #   #wh01 += famIndx[f1]
  #   #p0 = poles[famIndx[f0], :]
  #   #print('pre first loop: ',timer() - tic)
  #   #tic = timer()
  #   # place numba code here ...
  #
  #
  #
  #   #fit, polematch, R, nGood, whGood = self.band_vote_refine_loops1(poles,v0,v1, p0, wh01, bandnorms, angTol)
  #   fit,polematch,R,nGood,whGood = self.band_vote_refine_loops1(poles, bnd1, bnd2, familyLabel,  famIndx, nFam, angTable, bandnorms, angTol)
  #
  #   #print('numba first loops',timer() - tic)
  #   #whGood = np.nonzero(angFit < angTol)[0]
  #   #nGood = np.int64(whGood.size)
  #   #if nGood < 3:
  #   #  return 360.0, -1, -1, -1
  #
  #   #fit = np.mean(angFit[whGood])
  #   #print('all bindexed time', timer()-tic0)
  #   return fit, nGood, whGood, polematch

  def _refine_orientation(self, bandnorms, whGood, polematch):
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

    AB, weights = self._orientation_refine_loops_am(nGood, whGood, poles, bandnorms, polematch, n2Fit)

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

  def _refine_orientation_quest(self, bandnorms, polecartmatch, weights = None):
    tic = timer()


    if weights is None:
      weights = np.ones((bandnorms.shape[0]), dtype=np.float64)
    weightsn = np.asarray(weights, dtype=np.float64)
    weightsn /= np.sum(weightsn)
    #print(weightsn)
    pflt = np.asarray(polecartmatch, dtype=np.float64)
    bndnorm = np.asarray(bandnorms, dtype=np.float64)

    avequat, fit = self._orientation_quest_nb(pflt, bndnorm, weightsn)

    return avequat, fit

  @staticmethod
  @numba.jit(nopython=True, cache=True, fastmath=True, parallel=False)
  def _orientation_quest_nb(polescart, bandnorms, weights):
    # this uses the Quaternion Estimator AKA quest algorithm.

    pflt = np.asarray(polescart, dtype=np.float64)
    bndnorm = np.asarray(bandnorms, dtype=np.float64)
    npoles = pflt.shape[0]
    wn = (np.asarray(weights, dtype=np.float64)).reshape(npoles, 1)

    # wn = np.ones((nGood,1), dtype=np.float32)/np.float32(nGood)  #(weights[whGood]).reshape(nGood,1)
    wn /= np.sum(wn)

    I = np.zeros((3, 3), dtype=np.float64)
    I[0, 0] = 1.0;
    I[1, 1] = 1.0;
    I[2, 2] = 1.0
    q = np.zeros((4), dtype=np.float64)

    B = (wn * bndnorm).T @ pflt
    S = B + B.T
    z = np.asarray(np.sum(wn * np.cross(bndnorm, pflt), axis=0), dtype=np.float64)
    S2 = S @ S
    det = np.linalg.det(S)
    k = (S[1, 1] * S[2, 2] - S[1, 2] * S[2, 1]) + (S[0, 0] * S[2, 2] - S[0, 2] * S[2, 0]) + (
          S[0, 0] * S[1, 1] - S[1, 0] * S[0, 1])
    sig = 0.5 * (S[0, 0] + S[1, 1] + S[2, 2])
    sig2 = sig * sig
    d = z.T @ S2 @ z
    c = det + (z.T @ S @ z)
    b = sig2 + (z.T @ z)
    a = sig2 - k

    lam = 1.0
    tol = 1.0e-6
    iter = 0
    dlam = 1e6
    # for i in range(10):
    while (dlam > tol) and (iter < 10):
      lam0 = lam
      lam = lam - (lam ** 4 - (a + b) * lam ** 2 - c * lam + (a * b + c * sig - d)) / (
            4 * lam ** 3 - 2 * (a + b) * lam - c)
      dlam = np.fabs(lam0 - lam)
      iter += 1

    beta = lam - sig
    alpha = lam ** 2 - sig2 + k
    gamma = (lam + sig) * alpha - det
    X = np.asarray((alpha * I + beta * S + S2), dtype=np.float64) @ z
    qn = np.float64(0.0)
    qn += gamma ** 2 + X[0] ** 2 + X[1] ** 2 + X[2] ** 2
    qn = np.sqrt(qn)
    q[0] = gamma
    q[1:4] = X[0:3]
    q /= qn
    if (np.sign(gamma) < 0):
      q *= -1.0

    # polesrot = rotlib.quat_vectorL1N(q, pflt, npoles, np.float64, p=1)
    # pdot = np.sum(polesrot*bndnorm, axis = 1, dtype=np.float64)
    return q, lam  # , pdot


  @staticmethod
  @numba.jit(nopython=True, cache=True,fastmath=True,parallel=False)
  def _tripvote_numba(bandangs, LUT, angTol, tripAngles, tripID, nfam, n_bands):
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
  def _assign_bands_nb(polesCart, bandRank_arg, familyLabel, famIndx, nFam, angTable, bandnorms, angTol, n_band_early):
    eps = np.float32(1.0e-12)
    nBnds = bandnorms.shape[0]

    whGood_out = np.zeros(nBnds, dtype=np.int64)-1


    nMatch = np.int64(-1)
    Rout = np.zeros((1,3,3), dtype=np.float32)
    #Rout[0,0,0] = 1.0; Rout[0,1,1] = 1.0; Rout[0,2,2] = 1.0
    polematch_out = np.zeros((nBnds),dtype=np.int64) - 1
    pflt = np.asarray(polesCart, dtype=np.float32)
    bndnorm = np.transpose(np.asarray(bandnorms, dtype=np.float32))

    fit = np.float32(360.0)
    fitout = np.float32(360.0)
    fitbout = np.zeros((nBnds))
    R = np.zeros((1, 3, 3), dtype=np.float32)
    #fit = np.float32(360.0)
    #whGood = np.zeros(nBnds, dtype=np.int64) - 1
    nGood = np.int64(-1)
    ij = (-1,-1)
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
        if ang01 < np.float32(-1.0):
          ang01 = np.float32(-1.0+eps)

        paralleltest = np.arccos(np.fabs(ang01)) * RADEG
        if paralleltest < angTol:  # the two poles are parallel, send in another two poles if available.
          continue

        ang01 = np.arccos(ang01) * RADEG

        wh01 = np.nonzero(np.abs(angTable[famIndx[f0],famIndx[f1]:np.int64(famIndx[f1] + nFam[f1])] - ang01) < angTol)[0]

        n01 = wh01.size
        if n01 == 0:
          continue

        wh01 += famIndx[f1]
        p0 = polesCart[famIndx[f0], :]

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
          p1 = polesCart[wh01[i], :]
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
          fitb = angFit
          #fit = np.mean(fitb[whGood])
          fit = np.float32(0.0)
          for q in range(nGood):
            fit += np.float32(fitb[whGood[q]])
          fit /= np.float32(nGood)


        if nGood >= (n_band_early):
          fitout = fit
          fitbout = fitb
          nMatch = nGood
          whGood_out = whGood
          polematch_out = polematch
          Rout = R
          ij  = (bnd1,bnd2)
          break
        else:
          if nMatch < nGood:
            fitout = np.float32(fit)
            fitbout = fitb
            nMatch = nGood
            whGood_out = whGood
            polematch_out = polematch
            Rout = R
            ij = (bnd1, bnd2)
          elif nMatch == nGood:
            if fitout > fit:
              fitout = np.float32(fit)
              fitbout = fitb
              nMatch = nGood
              whGood_out = whGood
              polematch_out = polematch
              Rout = R
              ij = (bnd1, bnd2)
      if nMatch >= (n_band_early):
        break
    #quatout = rotlib.om2quL(Rout)
    return fitout, polematch_out,nMatch, whGood_out, ij, Rout, fitbout

  @staticmethod
  @numba.jit(nopython=True, cache=True, fastmath=True,parallel=False)
  def _orientation_refine_loops_triad(nGood, whGood, poles, bandnorms, polematch, n2Fit):
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
  def _orientation_refine_loops_am(nGood, whGood, poles, bandnorms, polematch, n2Fit):
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



  def pairVoteOrientation(self,bandnormsIN,goNumba=True):
    tic0 = timer()
    nfam = self.tripLib.polefamilies.shape[0]
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
      solutions, nsolutions, solutionVotes, solSrt = self._pairvote_nb(bandnorms, bandangs, qsym, angTableReduce, poles, polesReduce, self.angTol)
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

        avequat1, fit1 = self._refine_orientation(bandnorms, whGood[0], poleMatch1)

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
  def _pairvote_nb(bandnorms, bandangs, qsym, angTableReduce, poles, polesReduce, angTol):
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