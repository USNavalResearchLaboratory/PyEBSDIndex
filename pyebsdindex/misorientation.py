'''
2022, David Rowenhorst/The US Naval Research Laboratory, Washington DC
# Pursuant to title 17 section 105 of the United States Code, works of US Governement employees
# are not not subject to copyright protection.
#
# Copyright (c) 2013-2014, Marc De Graef/Carnegie Mellon University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are
# permitted provided that the following conditions are met:
#
#     - Redistributions of source code must retain the above copyright notice, this list
#        of conditions and the following disclaimer.
#     - Redistributions in binary form must reproduce the above copyright notice, this
#        list of conditions and the following disclaimer in the documentation and/or
#        other materials provided with the distribution.
#     - Neither the names of Marc De Graef, Carnegie Mellon University nor the names
#        of its contributors may be used to endorse or promote products derived from
#        this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
####################################################################################################


#Author David Rowenhorst/The US Naval Research Laboratory, Washington DC
#
#
'''

import numpy as np
import numba
from os import environ
import tempfile
from pathlib import PurePath
import platform
tempdir = PurePath("/tmp" if platform.system() == "Darwin" else tempfile.gettempdir())
tempdir = tempdir.joinpath('numba')
environ["NUMBA_CACHE_DIR"] = str(tempdir)

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

    q2i = q2In[i2,:].copy()
    q2i = q2i.reshape(4)
    q2i[1:4] *= -1.0

    q1i = q1In[i1,:].copy().reshape(4)

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



