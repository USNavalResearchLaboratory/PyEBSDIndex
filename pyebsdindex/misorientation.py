# This software was developed by employees of the US Naval Research Laboratory (NRL), an
# agency of the Federal Government. Pursuant to title 17 section 105 of the United States
# Code, works of NRL employees are not subject to copyright protection, and this software
# is in the public domain. PyEBSDIndex is an experimental system. NRL assumes no
# responsibility whatsoever for its use by other parties, and makes no guarantees,
# expressed or implied, about its quality, reliability, or any other characteristic. We
# would appreciate acknowledgment if the software is used. To the extent that NRL may hold
# copyright in countries other than the United States, you are hereby granted the
# non-exclusive irrevocable and unconditional right to print, publish, prepare derivative
# works and distribute this software, in any medium, or authorize others to do so on your
# behalf, on a royalty-free basis throughout the world. You may improve, modify, and
# create derivative works of the software or any portion of the software, and you may copy
# and distribute such modifications or works. Modified works should carry a notice stating
# that you changed the software and should note the date and nature of any such change.
# Please explicitly acknowledge the US Naval Research Laboratory as the original source.
# This software can be redistributed and/or modified freely provided that any derivative
# works bear some notice that they are derived from it, and any modified versions bear
# some notice that they have been modified.
#
#
# Author: David Rowenhorst;
# The US Naval Research Laboratory Date: 22 May 2024
#
# For further information see:
# David J. Rowenhorst, Patrick G. Callahan, Håkon W. Ånes. Fast Radon transforms for
# high-precision EBSD orientation determination using PyEBSDIndex.
# Journal of Applied Crystallography, 57(1):3–19, 2024.
# DOI: 10.1107/S1600576723010221
#
#

import numpy as np
import numba
import os
import tempfile
from pathlib import PurePath, Path
import platform
tempdir = PurePath(Path.home())
#tempdir = PurePath("/tmp" if platform.system() == "Darwin" else tempfile.gettempdir())
#tempdir = tempdir.joinpath('numbacache')
tempdir = tempdir.joinpath('.pyebsdindex').joinpath('numbacache')
Path(tempdir).mkdir(parents=True, exist_ok=True)
os.environ["NUMBA_CACHE_DIR"] = str(tempdir)+str(os.sep)

import rotlib

def misorientcubic_quick(q1,q2):
  '''Misorientation between two cubic crystals using the quaternion trick from Sutton and Buluffi'''

  type = np.dtype('float32')
  if (q1.dtype is np.dtype('float64')) or (q2.dtype is np.dtype('float64')):
    type = np.dtype('float64')

  shape1 = q1.shape
  shape2 = q2.shape

  m1 = shape1[-1]
  n1 = numba.int64(q1.size / m1)

  m2 = shape2[-1]
  n2 = numba.int64(q2.size / m2)

  n12 = np.array((n1,n2))
  q1in = np.require(q1.reshape(n1,m1).astype(type),requirements=['C','A'])
  q2in = np.require(q2.reshape(n2, m2).astype(type), requirements=['C', 'A'])
  q12 = misorientcubic_quicknb(q1in, q2in)

  q12 = np.squeeze(q12)
  return q12


@numba.jit(nopython=True,fastmath=True,cache=True)
def misorientcubic_quicknb(q1In,q2In):
  n1 = q1In.shape[0]
  n2 = q2In.shape[0]
  n = max(n1,n2)
  q12 = np.zeros((n,4), dtype = q1In.dtype)

  sqrt2 = np.float32(np.sqrt(2))#.astype(q1In.dtype)
  qAB = np.zeros((4), dtype=q1In.dtype)
  qAB_t1 = np.zeros((4), dtype = q1In.dtype)
  qAB_t2 = np.zeros((4), dtype=q1In.dtype)

  order1 = np.array([3,0,1,2], dtype=np.uint64)
  order2 = np.array([2,1,0], dtype=np.uint64)
  for i in numba.prange(n):
    i1 = i % n1
    i2 = i % n2

    q1i = q1In[i1, :].copy().reshape(4)
    q2i = q2In[i2,:].copy()
    q2i = q2i.reshape(4)
    q2i[1:4] *= -1.0 # take the conjugate/inverse of q2

    qAB = np.abs(rotlib.quat_multiply1(q1i, q2i))

    qAB = np.sort(qAB)

    qAB = qAB[order1]

    qAB_t1[0] = qAB[0] + qAB[3]
    qAB_t1[1] =  qAB[1] - qAB[2]
    qAB_t1[2] = qAB[2] + qAB[1]
    qAB_t1[3] = qAB[3] - qAB[0]

    if (qAB_t1[0] / sqrt2) > qAB[0]:
       qAB[:] = qAB_t1 / sqrt2

    qAB_t2[0] = qAB_t1[0] + qAB_t1[2]
    qAB_t2[1] = qAB_t1[1] + qAB_t1[3]
    qAB_t2[2] = qAB_t1[2] - qAB_t1[0]
    qAB_t2[3] = qAB_t1[3] - qAB_t1[1]
    qAB_t2 *= 0.5

    if (qAB_t2[0] > qAB[0]):
       qAB[:] = qAB_t2

    vect = np.abs(qAB[1:4])
    vect = np.sort(vect)
    qAB[1:] = vect[order2]

    q12[i, :]  = qAB


  return q12



