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


import numpy as np
from pyebsdindex import crystal_sym, rotlib


RADEG = 180.0/np.pi


class triplib():
  def __init__(self, libType='FCC', phaseName=None, laticeParameter = None):
    self.family = None
    self.nfamily = None
    self.angles = None
    self.polePairs = None
    self.angleFamilyID = None
    self.tripAngles = None
    self.tripID = None
    self.completelib = None
    self.symmetry = None
    self.qsymops = None
    self.phaseName = None
    self.latticeParameter = np.array([1.0, 1.0, 1.0, 90.0, 90.0, 90.0])

    if libType is None:
      return

    if phaseName is None:
      self.phaseName = libType
    else:
      self.phaseName = phaseName

    if libType.upper() == 'FCC':
      self.build_fcc()
      self.symmetry = 43
      self.qsymops = crystal_sym.cubicsym_q()
      if phaseName is None:
        self.phaseName = 'FCC'

      if laticeParameter is None:
        self.latticeParameter = np.array([1.0,1.0,1.0,90.0,90.0,90.0])
      else:
        self.latticeParameter = laticeParameter

    if libType.upper() == 'BCC':
      self.build_bcc()

      if phaseName is None:
        self.phaseName = 'BCC'
      if laticeParameter is None:
        self.latticeParameter = np.array([1.0,1.0,1.0,90.0,90.0,90.0])
      else:
        self.latticeParameter = laticeParameter

  def build_fcc(self):
    if self.phaseName is None:
      self.phaseName = 'FCC'
    self.symmetry = 43
    self.qsymops = crystal_sym.cubicsym_q()
    poles = np.array([[0,0,2], [1,1,1], [0,2,2], [1,1,3]])
    self.build_trip_lib(poles,crystal_sym.cubicsym_q())

  def build_bcc(self):
    if self.phaseName is None:
      self.phaseName = 'BCC'
    self.symmetry = 43
    self.qsymops = crystal_sym.cubicsym_q()
    poles = np.array([[0,1,1],[0,0,2],[1,1,2],[0,1,3]])
    self.build_trip_lib(poles,crystal_sym.cubicsym_q())

  def build_trip_lib(self,poles,symmetry):
    nsym = symmetry.shape[0]
    npoles = poles.shape[0]
    sympoles = []
    sympolesN = []
    sympolesComplete = []
    nFamComplete = np.zeros(npoles, dtype = np.int32)
    nFamily = np.zeros(npoles, dtype = np.int32)
    polesFlt = np.array(poles, dtype=np.float32)

    for i in range(npoles):
      family = rotlib.quat_vector(symmetry,polesFlt[i,:])
      uniqHKL = self.hkl_unique(family,reduceInversion=False)
      uniqHKL = np.flip(uniqHKL, axis=0)
      sympolesComplete.append(uniqHKL)
      nFamComplete[i] = np.int32((sympolesComplete[-1]).size/3)

      uniqHKL2 = self.hkl_unique(family,reduceInversion=True)
      nFamily[i] = np.int32(uniqHKL2.size/3)
      sign = np.squeeze(self.calc_pole_dot(uniqHKL2,polesFlt[i,:]))
      whmx = (np.abs(sign)).argmax()
      sign = np.round(sign[whmx])
      uniqHKL2 *= sign

      sympoles.append(np.round(uniqHKL2))
      #sympolesN.append(self.xstalPlane2cart(family))

    sympolesComplete = np.concatenate(sympolesComplete)
    nsyms = np.sum(nFamily).astype(np.int32)

    angs = []
    familyID = []
    polePairs = []
    for i in range(npoles):
      for j in range(i, npoles):
        ang = np.squeeze(self.calc_pole_dot(polesFlt[i,:],sympoles[j]))
        ang = np.clip(ang, -1.0, 1.0)
        sign = (ang >= 0).astype(np.float32) - (ang < 0).astype(np.float32)
        ang = np.round(np.arccos(sign * ang)*RADEG*100).astype(np.int32)
        unqang, argunq = np.unique(ang, return_index=True)
        unqang = unqang/100.0
        sign = sign[argunq]

        wh = np.nonzero(unqang > 1.0)[0]
        nwh = wh.size
        sign = sign[wh]
        sign = sign.reshape(nwh,1)
        temp = np.zeros((nwh, 2, 3))
        temp[:,0,:] = np.broadcast_to(poles[i,:], (nwh, 3))
        temp[:,1,:] = np.broadcast_to(sympoles[j][argunq[wh],:]*sign, (nwh, 3))
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
    nlib = npoles*np.prod(np.arange(3, dtype=np.int64)+(nangs-2+1))/np.long(np.math.factorial(3))
    nlib = nlib.astype(np.int)

    libANG = np.zeros((nlib, 3))
    libID = np.zeros((nlib, 3), dtype = np.int)
    counter = 0

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

    libANG, libID = self.sortlib_id(libANG,libID,findDups = True)
    #print(libANG)
    #print(libANG.shape)
    angTable = self.calc_pole_dot(sympolesComplete,sympolesComplete)
    angTable = np.arccos(angTable)*RADEG
    famindx0 = ((np.concatenate( ([0],np.cumsum(nFamComplete)) ))[0:-1]).astype(dtype=np.int)
    cartPoles = self.xstalplane2cart(sympolesComplete)
    cartPoles /= np.linalg.norm(cartPoles, axis = 1).reshape(np.int(cartPoles.size/3),1)
    completePoleFamId = np.zeros(sympolesComplete.shape[0], dtype=np.int32)
    for i in range(npoles):
      for j in range(nFamComplete[i]):
        completePoleFamId[j+famindx0[i]] = i
    self.completelib = {
                   'poles' : sympolesComplete,
                   'polesCart': cartPoles,
                   'poleFamID': completePoleFamId,
                   'angTable' : angTable,
                   'nFamily'  : nFamComplete,
                   'famIndex' : famindx0
                  }

    self.family = poles
    self.nfamily = npoles
    self.angles = angs
    self.polePairs = polePairs
    self.angleFamilyID = familyID
    self.tripAngles = libANG
    self.tripID = libID



  def hkl_unique(self,poles, reduceInversion=True, rMT = np.identity(3)):
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
      test = self.calc_pole_dot(family,family,rMetricTensor = rMT)

      testSum = np.sum( (test < -0.99999).astype(np.int32)*np.arange(nf).reshape(1,nf), axis = 1)
      whpos = np.nonzero( np.logical_or(testSum < np.arange(nf), (testSum == 0)))[0]
      polesout = polesout[whpos, :]
    return polesout

  def calc_pole_dot(self,poles1,poles2,rMetricTensor = np.identity(3)):

    p1 = poles1.reshape(np.int(poles1.size / 3), 3)
    p2 = poles2.reshape(np.int(poles2.size / 3), 3)

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

  def xstalplane2cart(self,poles,rStructMatrix = np.identity(3)):
    polesout = rStructMatrix.dot(poles.T)
    return np.transpose(polesout)

  def sortlib_id(self,libANG,libID,findDups = False):
    LUTA = np.array([[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]])
    LUTB = np.array([[0,1,2],[1,0,2],[0,2,1],[2,0,1],[1,2,0],[2,1,0]])

    LUT = np.zeros((3,3,3,3), dtype=np.int64)
    for i in range(6):
      LUT[:, LUTA[i,0], LUTA[i,1], LUTA[i,2]] = LUTB[i,:]

    ntrips = np.int(libANG.size / 3)
    for i in range(ntrips):
      temp = np.squeeze(libANG[i,:])
      srt = np.argsort(temp)
      libANG[i,:] = temp[srt]
      srt2 = LUT[:,srt[0], srt[1], srt[2]]
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
