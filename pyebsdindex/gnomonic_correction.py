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

class GnomonicCorrection():
  def __init__(
    self,
    radonPlan=None,
    PC = np.array([0.5, 0.5, 0.5]),
    vendor='EDAX',
    **kwargs
    ):
    self.PC = PC
    self.PCpx = None
    self.vendor = vendor
    self.setradonPlan(radonPlan)
    if self.radonPlan is not None:
      if self.radonPlan.imDim is not None:
        self.calccorrection()




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
      self.PC = np.array(PC)


    pctemp = np.asarray(self.PC, dtype=np.float32).copy()
    shapet = pctemp.shape
    if len(shapet) == 2:
      pctemp = np.mean(pctemp, axis=0)
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

    t[1] = dimf[0] - t[1]
    self.PCpx = t


    # nx = self.patdim[1]
    # ny = self.patdim[0]
    # x = np.arange(nx, dtype=float) - t[0]
    # x = (np.broadcast_to(x.reshape(1, nx), (ny, nx)))
    # y = np.arange(ny, dtype=float) - (self.patdim[0] - t[1])
    # y = (np.broadcast_to(y, (nx, ny)).T)
    #
    # x2 = x*x
    # y2 = y*y
    #
    # #x2 *= np.abs(x)/np.sqrt(x**2 + y**2).clip(1e-8)
    # #y2 *= np.abs(y) / np.sqrt(x ** 2 + y ** 2).clip(1e-8)
    #
    # rdnx2 = np.squeeze(self.radonPlan.radon_faster(x2, fixArtifacts = True)).clip(0)
    #
    # rdncos = np.broadcast_to(
    #    np.abs(np.cos(self.radonPlan.theta*np.pi/180.)),
    #  (self.radonPlan.nRho, self.radonPlan.nTheta))
    # rdnx2 *= rdncos
    #
    #
    #
    # rdny2 = np.squeeze(self.radonPlan.radon_faster(y2, fixArtifacts = True)).clip(0)
    # rdnsin = np.broadcast_to(
    #   np.abs(np.sin(self.radonPlan.theta * np.pi / 180.)),
    #   (self.radonPlan.nRho, self.radonPlan.nTheta))
    # rdny2 *= rdnsin
    #
    #
    # rdncorrect = np.sqrt(rdnx2 + rdny2)
    # self.rdncorrect = rdncorrect

    #return rdncorrect, x2, y2, #rdncos, rdnsin
    #return rdncorrect,rdnx2, rdny2, rdncos, rdnsin

  def applycorrection(
          self,
          bnddata,
          rsigma,
          convolfactor = 1.0537092,
          PC = None,
          **kwargs
    ):

    if PC is not None:
      self.calccorrection(PC=np.array(PC))

    PCpx = self.PCpx
    valid = bnddata['valid']
    npat = bnddata.shape[0]
    nband = bnddata.shape[1]
    width = bnddata['width']
    maxloc = bnddata['maxloc']
    theta = bnddata['theta']
    rho = bnddata['rho']
    patdim = self.patdim
    #rdncorrect = self.rdncorrect
    #print(PCpx)
    bdndata_out = bnddata.copy()

    rho_new = self.__correction_loops_nb( npat, nband,
                            valid, width, maxloc, theta, rho, PCpx, patdim,
                            convolfactor, rsigma)

    bdndata_out['rho'] = rho_new

    #print('PCpx: ', PCpx)
    # for j in range(bnddata.shape[0]):
    #   bnddataj = bnddata[j].copy()
    #   for indx in range(bnddata.shape[1]):
    #     bnd = bnddataj[indx]
    #     if bnd['valid'] > 0:
    #       fwhm = bnd['width']
    #       # FWHM_measured = sqrt((c*rsigma)^2 + (c*bndsigma)^2) ; c = 1.0537
    #       # FWHM_measured = sqrt((c*rsigma)^2 + (FWHM_band)^2)
    #       bdnwith_2 = np.sqrt( np.clip(fwhm**2 - (convolfactor * rsigma)**2,0, None) )
    #       #print(bdnwith_2)
    #
    #
    #       theta = bnd['maxloc'].astype(int)[1]
    #       rho = bnd['maxloc'].astype(int)[0]
    #
    #       d = self.rdncorrect[rho,theta]
    #
    #       phi1 = np.arctan((d+bdnwith_2) / PCpx[2])
    #       phi2 = np.arctan((d-bdnwith_2)/ PCpx[2])
    #       phi = (phi1 + phi2)*0.5
    #       shft = np.abs(self.PCpx[2] * np.tan(phi) - d)
    #       #print(shft)
    #       #print(bdnwith_2, d, phi1, phi2, phi, shft)
    #       rho_0 =  bnd['rho']
    #       theta = bnd['theta']
    #       # this is the adjusted rho that is centered on the pattern center, not the detector center.
    #       dx = PCpx[0] - self.patdim[1] * 0.5
    #       dy = PCpx[1] - self.patdim[0] * 0.5
    #       rho_prime = rho_0 - (dx * np.cos(theta) + dy * np.sin(theta))
    #
    #       sign = 1.0 if rho_prime >= 0 else -1.0 # this then gives the correct direction.
    #       rho_1 = rho_0 + sign*shft
    #
    #       bnd['rho'] = rho_1
    #       bnddataj[indx] = bnd
    #       #print('______')
    #     bnddata[j] = bnddataj
    return bdndata_out

  @staticmethod
  @numba.jit(nopython=True, cache=True, fastmath=True, parallel=True)
  def __correction_loops_nb( npat, nband,
                            valid, width, maxloc, theta, rho, PCpx, patdim,
                            convolfactor, rsigma):

      for j in numba.prange(npat):
        #bnddataj = bnddata[j].copy()
        for i in range(nband):
          if valid[j,i] > 0:
            fwhm = width[j,i]
            # FWHM_measured = sqrt((c*rsigma)^2 + (c*bndsigma)^2) ; c = 1.0537
            # FWHM_measured = sqrt((c*rsigma)^2 + (FWHM_band)^2)
            a = fwhm ** 2 - (convolfactor * rsigma) ** 2
            bdnwith_2 = np.sqrt(a) if a > 0 else 0.0

            rho_indx = int(maxloc[j,i,0])
            theta_indx = int(maxloc[j, i,1])

            rho_ji = rho[j, i]


            theta_ji = theta[j, i]
            # this is the adjusted rho that is centered on the pattern center, not the detector center.
            dx =  (PCpx[0] - patdim[1] * 0.5)
            dy =  (PCpx[1] - patdim[0] * 0.5)
            rho_prime = rho_ji - (dx * np.cos(theta_ji) + dy * np.sin(theta_ji))



            d = np.abs(rho_prime) #rdncorrect[rho_indx, theta_indx]

            phi1 = np.arctan((d + bdnwith_2) / PCpx[2])
            phi2 = np.arctan((d - bdnwith_2) / PCpx[2])
            phi = (phi1 + phi2) * 0.5
            shft = np.abs(PCpx[2] * np.tan(phi) - d)

            # print(shft)
            # print(bdnwith_2, d, phi1, phi2, phi, shft)
            if shft < 1.0:
              sign = 1.0 if rho_prime >= 0 else -1.0  # this then gives the correct direction.
              rho_1 = rho_ji - sign * shft
            else:
              rho_1 = rho_ji
            rho[j,i] = rho_1


      return rho