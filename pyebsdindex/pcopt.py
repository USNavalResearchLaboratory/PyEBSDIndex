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
# Author: David Rowenhorst;
# The US Naval Research Laboratory Date: 21 Aug 2020

"""Optimization of the pattern center (PC) of EBSD patterns."""


import copy

import numpy as np
import pyswarms as pso
import scipy.optimize as opt


RADEG = 180.0 / np.pi


def optfunction(PC_i, indexer, banddat):
    band_norm = indexer.bandDetectPlan.radon2pole(banddat, PC=PC_i, vendor=indexer.vendor)
    n_points = banddat.shape[0]
    n_averages = 0
    average_fit = 0
    phase = indexer.phaseLib[0]

    for i in range(n_points):
        band_norm1 = band_norm[i, :, :]
        band_data1 = banddat[i, :]
        whgood = np.nonzero(band_data1['max'] > -1e6)[0]
        if whgood.size >= 3:
            band_norm1 = band_norm1[whgood, :]
            fit = phase.tripvote(band_norm1, goNumba=True)[1]
            if fit < 90:
                average_fit += fit
                n_averages += 1

    if n_averages < 0.9:
        average_fit = 100
    else:
        average_fit /= n_averages
    return average_fit


def optimize(pats, indexer, PC0=None, batch=False):
    """Optimize pattern center (PC) (PCx, PCy, PCz) in the convention
    of the `indexer.vendor` (default is EDAX) with Nelder-Mead.

    Parameters
    ----------
    pats : numpy.ndarray
      EBSD patterns.
    indexer : pyebsdindex.ebsd_index.EBSDIndexer
      EBSD indexer instance storing all relevant parameters for band
      detection.
    PC0 : list, optional
      Initial guess of PC. If not given, `indexer.PC` is used.
    batch : bool, optional
      Default is False.

    Returns
    -------
    PCoutRet : numpy.ndarray
      Optimized PC.

    Notes
    -----
    SciPy's Nelder-Mead minimization function is used with a tolerance
    `fatol` of 0.00001 between each iteration.
    """
    indxerCPU = copy.deepcopy(indexer)
    indxerCPU.bandDetectPlan.CLOps = [False, False, False, False]
    banddat = indxerCPU.bandDetectPlan.find_bands(pats)
    npoints = banddat.shape[0]
    if PC0 is None:
        PC0 = indexer.PC

    if not batch:
        PCopt = opt.minimize(optfunction, PC0, args=(indexer, banddat), method='Nelder-Mead', options={'fatol': 0.00001})
        PCoutRet = PCopt['x']
    else:
        PCoutRet = np.zeros((npoints, 3))
        for i in range(npoints):
            PCopt = opt.minimize(optfunction, PC0, args=(indexer, banddat[i, :, :]), method='Nelder-Mead')
            PCoutRet[i, :] = PCopt['x']
    return PCoutRet


def optimize_pso(pats, indexer, PC0=None, batch=False):
    """Optimize pattern center (PC) (PCx, PCy, PCz) in the convention
    of the `indexer.vendor` (default is EDAX) with particle swarms.

    Parameters
    ----------
    pats : numpy.ndarray
      EBSD patterns.
    indexer : pyebsdindex.ebsd_index.EBSDIndexer
      EBSD indexer instance storing all relevant parameters for band
      detection.
    PC0 : list, optional
      Initial guess of PC. If not given, `indexer.PC` is used.
    batch : bool, optional
      Default is False.

    Returns
    -------
    PCoutRet : numpy.ndarray
      Optimized PC.

    Notes
    -----
    `pyswarms` particle swarm algorithm is used with 50 particles,
    bounds of +/- 0.05 on the PC values, and parameters c1 = 2.05, c2 =
    2.05 and w = 0.8.
    """
    banddat = indexer.bandDetectPlan.find_bands(pats)
    npoints = banddat.shape[0]

    if PC0 is None:
        PC0 = indexer.PC
    else:
        PC0 = np.asarray(PC0)

    pso_options = {'c1': 2.05, 'c2': 2.05, 'w': 0.8}
    bounds = (PC0 - 0.05, PC0 + 0.05)
    optimizer = pso.single.GlobalBestPSO(n_particles=50, dimensions=3, options=pso_options, bounds=bounds)

    if not batch:
        cost, PCopt = optimizer.optimize(optfunction, 1000, indexer=indexer, banddat=banddat)
        PCoutRet = PCopt
        print(cost)
    else:
        PCoutRet = np.zeros((npoints, 3))
        for i in range(npoints):
            cost, PCopt = optimizer.optimize(optfunction, 100, indexer=indexer, banddat=banddat[i, :, :])
            PCoutRet[i, :] = PCopt
    return PCoutRet


def file_opt(fobj, indexer):
    nCols = fobj.nCols
    nRows = fobj.nRows
    stride = 20
    pcopt = np.zeros((int(nRows / stride), int(nCols / stride), 3), dtype=np.float32)

    for i in range(int(nRows / stride)):
        ii = i * stride
        print(ii)
        for j in range(int(nCols / stride)):
            jj = j * stride

            pats = fobj.read_data(
                returnArrayOnly=True, convertToFloat=True, patStartEnd=[ii*nCols + jj, ii*nCols + jj + 1]
            )

            pc = optimize(pats, indexer)
            #print(pc, pc.shape)
            pcopt[i, j, :] = pc

    return pcopt
