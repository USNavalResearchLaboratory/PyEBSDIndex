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
The US Naval Research Laboratory Date: 23 Nov 2022

This is heavily inspired/borrowed from the DREAM.3D EBSDLib LaueOps library
https://github.com/BlueQuartzSoftware/EbsdLib
'''


import numpy as np

from pyebsdindex import rotlib


PI = np.pi

def cubicsym_q(quatin=None, low = False):
  """Provide quaternion proper rotation symmetry operators for the cubic crystal systems.
   If provided with input quaternions, the symmetrically equivalent quaternions will be returned.

      Parameters
      ----------
      quatin : numpy.ndarray
          quaternions, of shape``(4)``
          If not provided [default] then
          the quaternions operators are returned.
      low : Default is ``False`` and the full m-3m symmetry operations are provided.
      Set to ``True`` to return the m-3 symmetry operators

      Returns
      -------
      numpy.ndarray
          array of ``(24,4)`` (or if ``low=True``, ``(12,4)``) rotation symmetry operators for cubic system.

      Notes
      -----
      """

  symops = np.zeros((24,4), dtype=np.float32)
  #identity
  symops [0,:] = np.array([1.0,0.0,0.0,0.0])
  #four fold
  cang1 = np.cos(0.25 * PI)
  sang1 = np.sin(0.25 * PI)
  cang2 = np.cos(0.75 * PI)
  sang2 = np.sin(0.75 * PI)
  #[001]
  symops[1,:] = np.array([cang1, 0.0,0.0,sang1])
  symops[2,:] = np.array([cang2,0.0,0.0,sang2])
  #[010]
  symops[3,:] = np.array([cang1,0.0,sang1,0.0])
  symops[4,:] = np.array([cang2,0.0,sang2,0.0])
  #[100]
  symops[5,:] = np.array([cang1,sang1,0.0,0.0])
  symops[6,:] = np.array([cang2,sang2,0.0,0.0])

  #two fold
  cang1 = np.cos(PI*0.5)
  sang1 = np.sin(PI*0.5)
  #[001]
  symops[7,:] = np.array([cang1,0.0,0.0,sang1])
  #[010]
  symops[8,:] = np.array([cang1,0.0,sang1,0.0])
  #[100]
  symops[9,:] = np.array([cang1,sang1,0.0,0.0])

  t = 1.0/np.sqrt(2.0)
  sang1 *= t
  #[-101]
  symops[10,:] = np.array([cang1,-sang1,0.0,sang1])
  # [0-11]
  symops[11,:] = np.array([cang1,0.0,-sang1,sang1])
  # [101]
  symops[12,:] = np.array([cang1,sang1,0.0,sang1])
  #[011]
  symops[13,:] = np.array([cang1,0.0,sang1,sang1])
  #[-110]
  symops[14,:] = np.array([cang1,-1*sang1,sang1,0.0])
  #[110]
  symops[15,:] = np.array([cang1,sang1,sang1,0.0])

  #three fold axis
  t = 1.0 / np.sqrt(3.0)
  cang1 = np.cos(1.0/3.0 * PI)
  sang1 = np.sin(1.0/3.0 * PI) * t
  cang2 = np.cos(2.0/3.0 * PI)
  sang2 = np.sin(2.0/3.0 * PI) * t
  #[111]
  symops[16,:] = np.array([cang1,sang1,sang1,sang1])
  symops[17,:] = np.array([cang2,sang2,sang2,sang2])
  #[-111]
  symops[18,:] = np.array([cang1,-sang1,sang1,sang1])
  symops[19,:] = np.array([cang2,-sang2,sang2,sang2])
  #[-1-11]
  symops[20,:] = np.array([cang1,-sang1,-sang1,sang1])
  symops[21,:] = np.array([cang2,-sang2,-sang2,sang2])
  #[1-11]
  symops[22,:] = np.array([cang1,sang1,-sang1,sang1])
  symops[23,:] = np.array([cang2,sang2,-sang2,sang2])

  if low:
    symops = symops[[0,7,8,9,16,17,18,19,20,21,22,23],:]

  if quatin is None:
    return symops
  else:
    qsym = rotlib.quat_multiply(symops,quatin)
    return qsym


def hexsym_q(quatin=None, low = False):
  """Provide quaternion proper rotation symmetry operators for the hexagonal crystal systems.
     If provided with input quaternions, the symmetrically equivalent quaternions will be returned.

        Parameters
        ----------
        quatin : numpy.ndarray
            quaternions, of shape``(4)``
            If not provided [default] then
            the quaternions operators are returned.
        low : Default is ``False`` and the full 6/mmm symmetry operations are provided.
        Set to ``True`` to return the 6/m symmetry operators

        Returns
        -------
        numpy.ndarray
            array of ``(12,4)`` (or if ``low=True``, ``(6,4)``) rotation symmetry
            operators for hexagonal system.

        Notes
        -----
        """

  symops = np.zeros((12,4), dtype=np.float32)
  #identity
  symops [0,:] = np.array([1.0,0.0,0.0,0.0])
  #60 deg about c [0001]; remember that quat rotation is cos(ang/2)
  symops[1, :] = np.array([np.cos(PI/6.0), 0.0,0.0,np.sin(PI/6.0)]) # rotation of 60
  symops[2, :] = np.array([np.cos(PI/3.0), 0.0,0.0,np.sin(PI/3.0)]) # rotation of 120
  symops[3, :] = np.array([0.0, 0.0,0.0,1.0]) # rotation of 180
  symops[4, :] = np.array([np.cos(2.0*PI/3.0), 0.0, 0.0, np.sin(2.0*PI/3.0)])  # rotation of 240
  symops[5, :] = np.array([np.cos(5.0*PI/6.0 ), 0.0, 0.0, np.sin(5.0*PI/6.0 )])  # rotation of 300

  # 180 deg around the a axes
  symops[6, :] =  np.array([0.0000000000, 1.000000000, 0.000000000, 0.000000000])
  symops[7, :] =  np.array([0.0000000000, 0.866025400, 0.500000000, 0.000000000])
  symops[8, :] =  np.array([0.000000000, 0.500000000, 0.866025400, 0.000000000])
  symops[9, :] =  np.array([0.000000000, 0.000000000, 1.000000000, 0.000000000])
  symops[10, :] = np.array([0.000000000, -0.50000000, 0.866025400, 0.000000000])
  symops[11, :] = np.array([0.000000000, -0.86602540, 0.500000000, 0.000000000])

  if low:
    symops = symops[0:6,:]

  if quatin is None:
    return symops
  else:
    qsym = rotlib.quat_multiply(symops,quatin)
    return qsym

def trigonal_q(quatin=None, low = False):
  """Provide quaternion proper rotation symmetry operators for the trigonal crystal systems.
       If provided with input quaternions, the symmetrically equivalent quaternions will be returned.

          Parameters
          ----------
          quatin : numpy.ndarray
              quaternions, of shape``(4)``
              If not provided [default] then
              the quaternions operators are returned.
          low : Default is ``False`` and the full -3m symmetry operations are returned.
          Set to ``True`` to return the -3 symmetry operators are returned

          Returns
          -------
          numpy.ndarray
              array of ``(6,4)`` (or if ``low=True``, ``(3,4)``) rotation symmetry
              operators for trigonal system.

          Notes
          -----
          """
  symops = np.zeros((6, 4), dtype=np.float32)
  # identity
  symops[0, :] = np.array([1.0, 0.0, 0.0, 0.0])
  # 120 deg rotations about c axis: remember that quat rotation is cos(ang/2)
  symops[1, :] = np.array([np.cos(PI / 3.0), 0.0, 0.0, np.sin(PI / 3.0)])  # rotation of 120
  symops[2, :] = np.array([np.cos(2.0 * PI / 3.0), 0.0, 0.0, np.sin(2.0 * PI / 3.0)])  # rotation of 240

  # 180 deg around the a axes
  symops[3, :] = np.array([0.0000000000, 1.000000000, 0.000000000, 0.000000000])
  symops[4, :] = np.array([0.000000000, -0.50000000, 0.866025400, 0.000000000])
  symops[5, :] = np.array([0.000000000, -0.50000000, -0.866025400, 0.000000000])

  if low:
    symops = symops[0:3, :]

  if quatin is None:
    return symops
  else:
    qsym = rotlib.quat_multiply(symops, quatin)
    return qsym

def tetragonal_q(quatin=None, low = False):
  """Provide quaternion proper rotation symmetry operators for the tetragonal crystal systems.
         If provided with input quaternions, the symmetrically equivalent quaternions will be returned.

            Parameters
            ----------
            quatin : numpy.ndarray
                quaternions, of shape``(4)``
                If not provided [default] then
                the quaternions operators are returned.
            low : Default is ``False`` and the full 4/mmm symmetry operations are returned.
            Set to ``True`` to return the 4/m symmetry operators are returned

            Returns
            -------
            numpy.ndarray
                array of ``(8,4)`` (or if ``low=True``, ``(4,4)``) rotation symmetry
                operators for tetragonal system.

            Notes
            -----
            """

  symops = np.zeros((8, 4), dtype=np.float32)
  # identity
  symops[0, :] = np.array([1.0, 0.0, 0.0, 0.0])
  # 4-fold around c axis [001]; remember that quat rotation is cos(ang/2)
  symops[1, :] = np.array([np.cos(PI /4.0), 0.0, 0.0, np.sin(PI / 4.0)])  # rotation of 90
  symops[2, :] = np.array([0.0, 0.0, 0.0, 1.0])  # rotation of 180
  symops[3, :] = np.array([np.cos(0.75*PI ), 0.0, 0.0, np.sin(0.75 * PI )])  # rotation of 270

  # 2-fold, 180 deg around the [110] axes
  symops[4, :] = np.array([0.0000000000, 0.5*np.sqrt(2.0), 0.5*np.sqrt(2.0), 0.000000000])
  symops[5, :] = np.array([0.0000000000, -0.5 * np.sqrt(2.0), 0.5 * np.sqrt(2.0), 0.000000000])
  #180 deg around [100], [010]
  symops[6, :] = np.array([0.0000000000, 1.0, 0.0, 0.0])
  symops[7, :] = np.array([0.0000000000, 0.0, 1.0, 0.0])

  if low:
    symops = symops[0:4, :]

  if quatin is None:
    return symops
  else:
    qsym = rotlib.quat_multiply(symops, quatin)
    return qsym

def orthorhombic_q(quatin=None, low = False):
  """Provide quaternion proper rotation symmetry operators for the orthorhombic crystal systems.
           If provided with input quaternions, the symmetrically equivalent quaternions will be returned.

              Parameters
              ----------
              quatin : numpy.ndarray
                  quaternions, of shape``(4)``
                  If not provided [default] then
                  the quaternions operators are returned.
              low : This parameter does nothing, but is maintained to provide
              function calling similarity with the other functions in this module.
              The mmm rotation operators will always be returned.

              Returns
              -------
              numpy.ndarray
                  array of ``(4,4)`` rotation symmetry
                  operators for orthorhombic system.

              Notes
              -----
              """
  symops = np.zeros((4, 4), dtype=np.float32)
  # identity
  symops[0, :] = np.array([1.0, 0.0, 0.0, 0.0])
  #
  symops[1, :] = np.array([0.0, 1.0, 0.0, 0.0])  # rotation of 180 a
  symops[2, :] = np.array([0.0, 0.0, 1.0, 0.0])  # rotation of 180 b
  symops[3, :] = np.array([0.0, 0.0, 0.0, 1.0])  # rotation of 180 c


  if quatin is None:
    return symops
  else:
    qsym = rotlib.quat_multiply(symops, quatin)
    return qsym

def monoclinic_q(quatin=None, low = False):
  """Provide quaternion proper rotation symmetry operators for the monoclinic crystal systems.
             If provided with input quaternions, the symmetrically equivalent quaternions will be returned.

                Parameters
                ----------
                quatin : numpy.ndarray
                    quaternions, of shape``(4)``
                    If not provided [default] then
                    the quaternions operators are returned.
                low : This parameter does nothing, but is maintained to provide
                function calling similarity with the other functions in this module.
                The 2/m rotation operators will always be returned.

                Returns
                -------
                numpy.ndarray
                    array of ``(2,4)`` rotation symmetry
                    operators for monoclinic system.

                Notes
                -----
                Convention is that the rotational axis is along b-axis.
                """
  symops = np.zeros((2, 4), dtype=np.float32)
  # identity
  symops[0, :] = np.array([1.0, 0.0, 0.0, 0.0])
  # rotation of 180 around b-axis
  symops[1, :] = np.array([0.0, 0.0, 1.0, 0.0])

  if quatin is None:
    return symops
  else:
    qsym = rotlib.quat_multiply(symops, quatin)
    return qsym

def triclinic_q(quatin=None, low=False):
  """Provide quaternion proper rotation symmetry operators for the triclinic crystal systems.
              If provided with input quaternions, the symmetrically equivalent quaternions will be returned.

                 Parameters
                 ----------
                 quatin : numpy.ndarray
                     quaternions, of shape``(4)``
                     If not provided [default] then
                     the quaternions operators are returned.
                 low : This parameter does nothing, but is maintained to provide
                 function calling similarity with the other functions in this module.
                 The -1 rotation operators will always be returned.

                 Returns
                 -------
                 numpy.ndarray
                     array of ``(1,4)``  rotation symmetry
                     operators for triclinic system.

                 Notes
                 -----
                 This will be no different than the identity quaternion.
                 """
  symops = np.zeros((1,4),dtype=np.float32)
  # identity
  symops[0,:] = np.array([1.0,0.0,0.0,0.0])
  if low:
    pass

  if quatin is None:
    return symops
  else:
    qsym = rotlib.quat_multiply(symops,quatin)
    return qsym

def spacegroup2lauenumber(spacegroupid):
    """Given a space group number, this will provide a laue group id (EDAX laue group convention)

       Parameters
       ----------
       spacegroupid : int
           number between 1 and 230

       Returns
       -------
       int
          numeric code that is associated with a Laue group (EDAX convention)

       Notes
       -----
       """

    sgpg = [1, 2, 3, 6, 10, 16, 25, 47, 75, 81, 83, 89, 99, 111, 123, 143, 147, 149, 156, 162, 168, 174, 175, 177, 183,
            187, 191, 195, 200, 207, 215, 221]
    pgLaue = [1, 1, 2, 2, 2, 22, 22, 22, 4, 4, 4, 42, 42, 42, 42, 3, 3, 32, 32, 32, 6, 6, 6, 62, 62, 62, 62, 23, 23, 43,
              43, 43]

    for pgNum in range(len(sgpg)):
      if (sgpg[pgNum] > spacegroupid):
        break

    lauenumber = pgLaue[pgNum-1]
    return lauenumber

def laueid2symops(lauenumber):
  """Given a laue group number (EDAX integer convention), this will
  return the proper rotation quaternions for that crystal system.

         Parameters
         ----------
         lauenumber : int
             number between 1 and 43

         Returns
         -------
         numpy.ndarray
            array of ``(n,4)`` quaternions that represent rotational symetry operations.

         Notes
         -----
         """
  if lauenumber == 43:
    return cubicsym_q()
  if lauenumber == 23:
    return cubicsym_q(low = True)
  if lauenumber == 62:
    return hexsym_q()
  if lauenumber == 6:
    return hexsym_q(low = True)
  if lauenumber == 32:
    return trigonal_q()
  if lauenumber == 3:
    return trigonal_q(low = True)
  if lauenumber == 42:
    return tetragonal_q()
  if lauenumber == 4:
    return tetragonal_q(low = True)
  if lauenumber == 22:
    return orthorhombic_q()
  if lauenumber == 2:
    return monoclinic_q()
  if lauenumber == 1:
    return triclinic_q()
  return None

def spacegroupnum2symops(spacegroupid):
  """Given a space group id, this will
   return the proper rotation quaternions for that crystal system.

          Parameters
          ----------
          spacegroupid : int
              number between 1 and 230

          Returns
          -------
          numpy.ndarray
             array of ``(n,4)`` quaternions that represent rotational symetry operations.

          Notes
          -----
          """

  return laueid2symops(spacegroup2lauenumber(spacegroupid))

def hex4poles2hex3poles(poles):
  npoles = int(np.array(poles).size/4)
  poles4 = np.reshape(np.array(poles), (npoles, 4))
  poles3 = poles4[:, [0,1,3]]
  return poles3

def hex3poles2hex4poles(poles):
  npoles = int(poles.size/3)
  poles3 = np.reshape(poles, (npoles, 3))
  poles4 = np.zeros((npoles, 4))
  poles4[:,0] = poles3[:,0]
  poles4[:, 1] = poles3[:, 1]
  poles4[:, 3] = poles3[:, 2]
  poles4[:, 2] = - (poles3[:,0] + poles3[:,1])
  return poles4