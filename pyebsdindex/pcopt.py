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

import numpy as np
import pyswarms as pso
import scipy.optimize as opt


__all__ = [
    "optimize",
    "optimize_pso",
]

RADEG = 180.0 / np.pi


def _optfunction(PC_i, indexer, banddat):
    bandnorm = indexer.bandDetectPlan.radonPlan.radon2pole(
        banddat, PC=PC_i, vendor=indexer.vendor
    )
    npoints = banddat.shape[0]
    #n_averages = 0
    #average_fit = 0
    #nbands_fit = 0
    #phase = indexer.phaseLib[0]
    nbands = indexer.bandDetectPlan.nBands
    indexdata = indexer._indexbandsphase( banddat, bandnorm)



    fit = indexdata[-1]['fit']
    nmatch = indexdata[-1]['nmatch']
    average_fit = fit*(nbands+1 - nmatch)
    #average_fit = -1.0*(3.0-fit)*nmatch
    whgood = np.nonzero(fit < 90.0)

    n_averages = len(whgood[0])


    if n_averages < 0.9:
        average_fit = 1000
    else:
        average_fit = np.sum(average_fit[whgood[0]]) + 4.0*(nbands+1)*(npoints - n_averages)
        average_fit /= npoints
        #average_fit /= n_averages
        #average_fit *=  (n_averages*(nbands+1) - nbands_fit)/(n_averages*nbands)
    return average_fit


def optimize(pats, indexer, PC0=None, batch=False):
    """Optimize pattern center (PC) (PCx, PCy, PCz) in the convention
    of the :attr:`indexer.vendor` with Nelder-Mead.

    Parameters
    ----------
    pats : numpy.ndarray
        EBSD pattern(s), of shape
        ``(n detector rows, n detector columns)``,
        or ``(n patterns, n detector rows, n detector columns)``.
    indexer : pyebsdindex.ebsd_index.EBSDIndexer
        EBSD indexer instance storing all relevant parameters for band
        detection.
    PC0 : list, optional
        Initial guess of PC. If not given, :attr:`indexer.PC` is used.
        If :attr:`indexer.vendor` is ``"EMSOFT"``, the PC must be four
        numbers, the final number being the pixel size.
    batch : bool, optional
        Default is ``False`` which indicates the fit for a set of
        patterns should be optimized using the cumulative fit for all
        the patterns, and one PC will be returned. If ``True``, then an
        optimization is run for each individual pattern, and an array of
        PC values is returned.

    Returns
    -------
    numpy.ndarray
        Optimized PC.

    Notes
    -----
    SciPy's Nelder-Mead minimization function is used with a tolerance
    ``fatol=0.00001`` between each iteration, ending the optimization
    when the improvement is below this value.
    """
    banddat = indexer.bandDetectPlan.find_bands(pats)
    npoints, nbands = banddat.shape[:2]

    if PC0 is None:
        PC0 = indexer.PC
    emsoftflag = False
    if indexer.vendor == "EMSOFT":  # Convert to EDAX for optimization
        emsoftflag = True
        indexer.vendor = "EDAX"
        delta = indexer.PC
        PCtemp = PC0[0:3]
        PCtemp[0] *= -1.0
        PCtemp[0] += 0.5 * indexer.bandDetectPlan.patDim[1]
        PCtemp[1] += 0.5 * indexer.bandDetectPlan.patDim[0]
        PCtemp /= indexer.bandDetectPlan.patDim[1]
        PCtemp[2] /= delta[3]
        PC0 = PCtemp

    if not batch:
        PCopt = opt.minimize(
            _optfunction,
            PC0,
            args=(indexer, banddat),
            method="Nelder-Mead",
            options={"fatol": 0.00001}
        )
        PCoutRet = PCopt['x']
    else:
        PCoutRet = np.zeros((npoints, 3))
        for i in range(npoints):
            PCopt = opt.minimize(
                _optfunction,
                PC0,
                args=(indexer, banddat[i, :].reshape(1, nbands)),
                method="Nelder-Mead"
            )
            PCoutRet[i, :] = PCopt['x']

    if emsoftflag:  # Return original state for indexer
        indexer.vendor = "EMSOFT"
        indexer.PC = delta
        if PCoutRet.ndim == 2:
            newout = np.zeros((npoints, 4))
            PCoutRet[:, 0] -= 0.5
            PCoutRet[:, :3] *= indexer.bandDetectPlan.patDim[1]
            PCoutRet[:, 1] -= 0.5 * indexer.bandDetectPlan.patDim[0]
            PCoutRet[:, 0] *= -1.0
            PCoutRet[:, 2] *= delta[3]
            newout[:, :3] = PCoutRet
            newout[:, 3] = delta[3]
            PCoutRet = newout
        else:
            newout = np.zeros(4)
            PCoutRet[0] -= 0.5
            PCoutRet[:3] *= indexer.bandDetectPlan.patDim[1]
            PCoutRet[1] -= 0.5 * indexer.bandDetectPlan.patDim[0]
            PCoutRet[0] *= -1.0
            PCoutRet[2] *= delta[3]
            newout[:3] = PCoutRet
            newout[3] = delta[3]
            PCoutRet = newout

    return PCoutRet


def optimize_pso(pats, indexer, PC0=None, batch=False, search_limit = 0.05,
                 nswarmpoints=None, pswarmpar=None, ninter=500):
    """Optimize pattern center (PC) (PCx, PCy, PCz) in the convention
    of the :attr:`indexer.vendor` with particle swarms.

    Parameters
    ----------
    pats : numpy.ndarray
        EBSD pattern(s), of shape
        ``(n detector rows, n detector columns)``,
        or ``(n patterns, n detector rows, n detector columns)``.
    indexer : pyebsdindex.ebsd_index.EBSDIndexer
        EBSD indexer instance storing all relevant parameters for band
        detection.
    PC0 : list, optional
        Initial guess of PC. If not given, :attr:`indexer.PC` is used.
        If :attr:`indexer.vendor` is ``"EMSOFT"``, the PC must be four
        numbers, the final number being the pixel size.
    batch : bool, optional
        Default is ``False`` which indicates the fit for a set of
        patterns should be optimized using the cumulative fit for all
        the patterns, and one PC will be returned. If ``True``, then an
        optimization is run for each individual pattern, and an array of
        PC values is returned.
    search_limit : float, optional
        Default is 0.05 for all PC values, and sets the +/- limit for the optimization search.

    Returns
    -------
    numpy.ndarray
        Optimized PC.

    Notes
    -----
    :mod:`pyswarms` particle swarm algorithm is used with 50 particles,
    and parameters c1 = 2.05, c2 = 2.05 and w = 0.8.
    """
    banddat = indexer.bandDetectPlan.find_bands(pats)
    npoints = banddat.shape[0]
    if pswarmpar is None:
        pswarmpar = {"c1": 2.05, "c2": 2.05, "w": 0.8}#, 'k': 2, 'p': 2}

    if nswarmpoints is None:
        nswarmpoints = int(np.array(search_limit).max() * (100.0/0.2))
        nswarmpoints = max(50, nswarmpoints)


    if PC0 is None:
        PC0 = np.asarray(indexer.PC)
    else:
        PC0 = np.asarray(PC0)

    emsoftflag = False
    if indexer.vendor == "EMSOFT":  # Convert to EDAX for optimization
        emsoftflag = True
        indexer.vendor = "EDAX"
        delta = indexer.PC
        PCtemp = PC0[0:3]
        PCtemp[0] *= -1.0
        PCtemp[0] += 0.5 * indexer.bandDetectPlan.patDim[1]
        PCtemp[1] += 0.5 * indexer.bandDetectPlan.patDim[0]
        PCtemp /= indexer.bandDetectPlan.patDim[1]
        PCtemp[2] /= delta[3]
        PC0 = np.array(PCtemp)


    #optimizer = pso.single.GlobalBestPSO(
    optimizer = pso.single.GlobalBestPSO(
        n_particles=nswarmpoints,
        dimensions=3,
        options=pswarmpar,
        bounds=(PC0 - np.array(search_limit), PC0 + np.array(search_limit)),
    )

    if not batch:
        cost, PCoutRet = optimizer.optimize(
            _optfunction, ninter, indexer=indexer, banddat=banddat
        )
        #print(cost)
    else:
        PCoutRet = np.zeros((npoints, 3))
        for i in range(npoints):
            cost, PCoutRet[i, :] = optimizer.optimize(
                _optfunction, ninter, indexer=indexer, banddat=banddat[i, :, :]
            )

    if emsoftflag:  # Return original state for indexer
        indexer.vendor = "EMSOFT"
        indexer.PC = delta
        if PCoutRet.ndim == 2:
            newout = np.zeros((npoints, 4))
            PCoutRet[:, 0] -= 0.5
            PCoutRet[:, :3] *= indexer.bandDetectPlan.patDim[1]
            PCoutRet[:, 1] -= 0.5 * indexer.bandDetectPlan.patDim[0]
            PCoutRet[:, 0] *= -1.0
            PCoutRet[:, 2] *= delta[3]
            newout[:, :3] = PCoutRet
            newout[:, 3] = delta[3]
            PCoutRet = newout
        else:
            newout = np.zeros(4)
            PCoutRet[0] -= 0.5
            PCoutRet[:3] *= indexer.bandDetectPlan.patDim[1]
            PCoutRet[1] -= 0.5 * indexer.bandDetectPlan.patDim[0]
            PCoutRet[0] *= -1.0
            PCoutRet[2] *= delta[3]
            newout[:3] = PCoutRet
            newout[3] = delta[3]
            PCoutRet = newout

    return PCoutRet


def _file_opt(fobj, indexer, stride=200, groupsz = 3):
    nCols = fobj.nCols
    nRows = fobj.nRows
    pcopt = np.zeros((int(nRows / stride), int(nCols / stride), 3), dtype=np.float32)

    for i in range(int(nRows / stride)):
        ii = i * stride
        print(ii)
        for j in range(int(nCols / stride)):
            jj = j * stride

            pats = fobj.read_data(
                returnArrayOnly=True,
                convertToFloat=True,
                patStartCount=[ii*nCols + jj, groupsz]
            )

            pc = optimize(pats, indexer)
            pcopt[i, j, :] = pc

    return pcopt
