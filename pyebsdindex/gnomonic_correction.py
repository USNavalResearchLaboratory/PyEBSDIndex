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

class gnomoic_correction():
  def __init__(
    self,
    radonPlan=None,
    PC = np.array([0.5, 0.5, 0.5]),
    **kwargs
    ):
    self.PC = PC
    self.setradonPlan(radonPlan)

  def setradonPlan(
    self,
    radonPlan=None
    ):
    if radonPlan is not None:
      if radonPlan is not isinstance(radonPlan, radon_fast.Radon):
        print('Set to radonplan object')
        return
    else:
      return

    self.radonPlan = radonPlan
    self.dx2rnd = np.zeros([self.radonPlan.nRho, self.radonPlan.nTheta], dtype=np.float32)
    self.dy2rnd = np.zeros([self.radonPlan.nRho, self.radonPlan.nTheta], dtype=np.float32)
    self.patdim = self.radonPlan.imDim

  def calccorrection(
    self,
    PC = None,
    **kwargs
  ):
    if PC is not None:
      self.PC = PC


    nx = self.imDim[1]
    ny = self.imDim[0]
    x = np.arange(nx, dtype=float)
    x = (np.broadcast_to(x.reshape(1, nx), (ny, nx))).ravel()
    y = np.arange(ny, dtype=float)
    y = (np.broadcast_to(y, (nx, ny)).T).ravel()

