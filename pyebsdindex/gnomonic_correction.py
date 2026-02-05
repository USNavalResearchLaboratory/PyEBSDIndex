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
# The US Naval Research Laboratory Date: 28 Jan 2026
#
# For further information see:
# David J. Rowenhorst, Patrick G. Callahan, Håkon W. Ånes. Fast Radon transforms for
# high-precision EBSD orientation determination using PyEBSDIndex.
# Journal of Applied Crystallography, 57(1):3–19, 2024.
# DOI: 10.1107/S1600576723010221
#
#

import os
from pathlib import PurePath, Path
import platform
# import tempfile
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numba
import numpy as np

import scipy.ndimage as scipyndim #import gaussian_filter, dilation ...
#from scipy.ndimage #import grey_dilation as scipy_grey_dilation
#from scipy.ndimage #import median_filter
import scipy.optimize as scipyopt

from pyebsdindex import radon_fast

tempdir = PurePath(Path.home())
#tempdir = PurePath("/tmp" if platform.system() == "Darwin" else tempfile.gettempdir())
#tempdir = tempdir.joinpath('numbacache')
tempdir = tempdir.joinpath('.pyebsdindex').joinpath('numbacache')
Path(tempdir).mkdir(parents=True, exist_ok=True)
os.environ["NUMBA_CACHE_DIR"] = str(tempdir)+str(os.sep)

RADEG = 180.0/np.pi

class GnomoicCorrection():
  def __init__(
    self,
    radonPlan=None,
    PC = np.array([0.5, 0.5, 0.5]),
    vendor='EDAX',
    **kwargs
    ):
    self.PC = PC
    self.vendor = vendor
    self.setradonPlan(radonPlan)



  def setradonPlan(
    self,
    radonPlan=None
    ):

    if radonPlan is not None:
      if not isinstance(radonPlan, radon_fast.Radon):
        print('Set to radonplan object')
        return
    else:
      return

    self.radonPlan = radonPlan
    #self.dx2rnd = np.zeros([self.radonPlan.nRho, self.radonPlan.nTheta], dtype=np.float32)
    #self.dy2rnd = np.zeros([self.radonPlan.nRho, self.radonPlan.nTheta], dtype=np.float32)
    self.patdim = self.radonPlan.imDim

  def calccorrection(
    self,
    PC = None,
    **kwargs
  ):
    if PC is not None:
      self.PC = PC

    pctemp = np.asarray(self.PC, dtype=np.float32).copy()
    shapet = pctemp.shape
    ven = self.vendor
    if ven != 'EMSOFT':
      t = pctemp
    else:  # EMSOFT pc to ebsdindex needs four numbers for PC
      t = pctemp[0:3]
      t[2] /= pctemp[3]  # normalize by pixel size

    dimf = np.array(self.patdim, dtype=np.float32)
    if ven in ['EDAX']:
      t *= np.array([dimf[1], dimf[0], np.min(dimf[0:2])])
      t[ 1] = dimf[0] - t[1]
    if ven in ['OXFORD']:
      t *= np.array([dimf[1], dimf[1], dimf[1]])
      t[ 1] = dimf[0] - t[1]
    if ven == 'EMSOFT':
      t[0] *= -1.0
      t += np.array([dimf[1] / 2.0, dimf[0] / 2.0, 0.0])
      t[1] = dimf[0] - t[1]
    if ven in ['KIKUCHIPY', 'BRUKER']:
      t *= np.array([dimf[1], dimf[0], dimf[0]])


    print(t)
    nx = self.patdim[1]
    ny = self.patdim[0]
    x = np.arange(nx, dtype=float) - t[0]
    x = (np.broadcast_to(x.reshape(1, nx), (ny, nx)))
    y = np.arange(ny, dtype=float) - t[1]
    y = (np.broadcast_to(y, (nx, ny)).T)

    x2 = x*x
    y2 = y*y



    rdnx2 = np.squeeze(self.radonPlan.radon_faster(x2, fixArtifacts = True))

    rdncos = np.broadcast_to(
       np.abs(np.cos(self.radonPlan.theta*np.pi/180.)),
     (self.radonPlan.nRho, self.radonPlan.nTheta))
    rdnx2 *= rdncos

    rdny2 = np.squeeze(self.radonPlan.radon_faster(y2, fixArtifacts = True))
    rdnsin = np.broadcast_to(
      (np.sin(self.radonPlan.theta * np.pi / 180.)),
      (self.radonPlan.nRho, self.radonPlan.nTheta))
    rdny2 *= rdnsin

    rdncorrect = np.sqrt(rdnx2 + rdny2)
    self.rdncorrect = rdncorrect

    #return rdncorrect, rdnx2, rdny2, rdncos, rdnsin

  def applycorrection(
          self,
          bnddata,
          rsigma,
          **kwargs
    ):

    for bnd in bnddata:
      fwhm = bnd['width']
      #sigma12 = sqrt(sigma1^2 + sigma2^2)
      # FWHM = 2 * sqrt(2*ln(2)) * sigma
      pass