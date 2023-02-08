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
import multiprocessing
import pyswarms as pso
import scipy.optimize as opt
from functools import partial
from timeit import default_timer as timer



__all__ = [
    "optimize",
    "optimize_pso",
]

RADEG = 180.0 / np.pi


def _optfunction(PC_i, indexer, banddat):
    tic = timer()
    PC = np.atleast_2d(PC_i)
    result = np.zeros(PC.shape[0])
    # this loop is here because pyswarms expects a vectorized function
    #print(PC.shape)
    for q in range(PC.shape[0]):

        bandnorm = indexer.bandDetectPlan.radonPlan.radon2pole(
            banddat, PC=PC[q,:], vendor=indexer.vendor
        )
        #print(timer() - tic)
        npoints = banddat.shape[0]
        #n_averages = 0
        #average_fit = 0
        #nbands_fit = 0
        #phase = indexer.phaseLib[0]
        nbands = indexer.bandDetectPlan.nBands
        indexdata = indexer._indexbandsphase( banddat, bandnorm)



        fit = indexdata[-1]['fit']
        nmatch = indexdata[-1]['nmatch']
        average_fit = fit + 1.0*(nbands - nmatch)
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
        result[q] = average_fit
    #print(timer()-tic)
    return result


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
        patDim = np.array(indexer.bandDetectPlan.patDim)
        delta = indexer.PC
        PCtemp = PC0[0:3]
        patdimnorm = (np.array([patDim[1], patDim[0], np.max(patDim[0:2])]))
        PCtemp[0] *= -1.0
        PCtemp[0] += 0.5 * indexer.bandDetectPlan.patDim[1]
        PCtemp[1] += 0.5 * indexer.bandDetectPlan.patDim[0]
        #PCtemp /= indexer.bandDetectPlan.patDim[1]
        PCtemp /= patdimnorm
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
        patDim = np.array(indexer.bandDetectPlan.patDim)
        patdimnorm = (np.array([patDim[1], patDim[0], np.max(patDim[0:2])]))
        if PCoutRet.ndim == 2:
            newout = np.zeros((npoints, 4))
            PCoutRet[:, 0] -= 0.5
            #PCoutRet[:, :3] *= indexer.bandDetectPlan.patDim[1]
            PCoutRet[:, :3] *= np.atleast_2d(patdimnorm)
            PCoutRet[:, 0] *= -1.0
            PCoutRet[:, 1] -= 0.5 * patDim[0]
            PCoutRet[:, 2] *= delta[3]
            newout[:, :3] = PCoutRet
            newout[:, 3] = delta[3]
            PCoutRet = newout
        else:
            newout = np.zeros(4)
            PCoutRet[0] -= 0.5
            PCoutRet[:3] *= patdimnorm
            #PCoutRet[:3] *= indexer.bandDetectPlan.patDim[1]
            PCoutRet[1] -= 0.5 * patDim[0]
            PCoutRet[0] *= -1.0
            PCoutRet[2] *= delta[3]
            newout[:3] = PCoutRet
            newout[3] = delta[3]
            PCoutRet = newout

    return PCoutRet


def optimize_pso(pats, indexer, PC0=None, batch=False, search_limit = 0.2,
                 nswarmparticles=30, pswarmpar=None, niter=50, verbose=1):
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
    npoints, nbands = banddat.shape[:2]
    if pswarmpar is None:
        #pswarmpar = {"c1": 3.05, "c2": 1.05, "w": 0.8}
        pswarmpar = {"c1": 3.5, "c2": 3.5, "w": 0.8}
    if nswarmparticles is None:
        #nswarmpoints = int(np.array(search_limit).max() * (10.0/0.2))
        nswarmparticles = 30

    nswarmparticles = max(5, nswarmparticles)

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



    # optimizer = pso.single.GlobalBestPSO(
    #     n_particles=nswarmpoints,
    #     dimensions=3,
    #     options=pswarmpar,
    #     bounds=(PC0 - np.array(search_limit), PC0 + np.array(search_limit)),
    # )
    optimizer = PSOOpt(dimensions=3, n_particles=nswarmparticles,
                       c1=pswarmpar['c1'],
                       c2 = pswarmpar['c2'], w = pswarmpar['w'], hyperparammethod='auto')

    if not batch:
        # cost, PCoutRet = optimizer.optimize(
        #     _optfunction, niter, indexer=indexer, banddat=banddat
        # )
        cost, PCoutRet = optimizer.optimize(_optfunction, indexer=indexer, banddat=banddat,
                                            start=PC0, bounds=(PC0 - np.array(search_limit), PC0 + np.array(search_limit)),
                                            niter=niter, verbose=verbose)

        #print(cost)
    else:
        PCoutRet = np.zeros((npoints, 3))
        if verbose >= 1:
            print('', end='\n')
        for i in range(npoints):
            # cost, PCoutRet[i, :] = optimizer.optimize(
            #     _optfunction, niter, indexer=indexer, banddat=banddat[i, :, :]
            # )


            cost, newPC = optimizer.optimize(_optfunction, indexer=indexer,
                                    banddat=banddat[i, :].reshape(1, nbands),
               start=PC0, bounds=(PC0 - np.array(search_limit), PC0 + np.array(search_limit)),
               niter=niter, verbose=0)

            PCoutRet[i, :] = newPC
            progress = int(round(10 * float(i) / npoints))
            if verbose >= 1:
                print('', end='\r')
                print('PC found: [',
                      '*' * progress, ' ' * (10 - progress), '] ', i + 1, '/', npoints,
                      '  global best:', "{0:.3g}".format(cost),
                      '  PC opt:', np.array_str(PCoutRet[i,:], precision=4, suppress_small=True),
                      sep='', end='')
        if verbose >= 1:
            print('', end='\n')

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


class PSOOpt():
    def __init__(self,
                dimensions=3,
                n_particles=50,
                c1 = 2.05,
                c2 = 2.05,
                w = 0.8,
                hyperparammethod = 'static',
                boundmethod = 'bounce'):
        self.n_particles = int(n_particles)
        self.dimensions = int(dimensions)
        self.c1 = c1
        self.c2 = c2
        self.c1i = None
        self.c2i = None
        self.w = w
        self.wi = None
        self.hyperparammethod = hyperparammethod
        self.boundmethod = boundmethod
        self.vellimit = None
        self.start = None
        self.bounds = None
        self.range = None
        self.niter = None
        self.pos = None
        self.vel = None


    def initializeswarm(self, start=None, bounds=None):


        if start is None:
            if bounds is not None:
                start = 0.5*(bounds[0]+bounds[1])
            else:
                start = np.zeros(self.dimensions, dtype=np.float32)

        self.start = start

        if bounds is None:
            bounds = (-1*np.ones(self.dimensions, dtype=np.float32),np.ones(self.dimensions, dtype=np.float32) )

        self.bounds = bounds
        self.range = self.bounds[1] - self.bounds[0]

        self.pos = np.random.uniform(low=bounds[0], high=bounds[1], size=(self.n_particles, self.dimensions))
        self.pos[0, :] = start

        self.vel = np.random.normal(size=(self.n_particles, self.dimensions), loc=0.0, scale=1.0)
        meanv = np.mean(np.sqrt(np.sum(self.vel**2, axis=1)))
        self.vel *= np.sqrt(np.sum(self.range**2))/(20. * meanv)
        self.vellimit = 4*np.mean(np.sqrt(np.sum(self.vel**2, axis=1)))


        self.pbest = np.zeros(self.n_particles) + np.infty
        self.pbest_loc = np.copy(self.pos)
        self.gbest = np.infty
        self.gbest_loc = start




    def updateswarmbest(self, fun2opt, pool, **kwargs):

        val = np.zeros(self.n_particles)

        #tic = timer()
        for part_i in range(self.n_particles):
            val[part_i] = fun2opt(self.pos[part_i, :], **kwargs)
        #print(timer()-tic)
        #pos = self.pos.copy()
        #tic = timer()
        #results = pool.map(partial(fun2opt, **kwargs),list(pos) )
        #print(timer()-tic)
        #print(len(results[0]), type(results[0]))
        #print(len(results))
        #val = np.concatenate(results)

        wh_newpbest = np.nonzero(val < self.pbest)[0]

        self.pbest[wh_newpbest] = val[wh_newpbest]
        self.pbest_loc[wh_newpbest, :] = self.pos[wh_newpbest, :]

        wh_minpbest = np.argmin(self.pbest)
        if self.pbest[wh_minpbest] < self.gbest:
            self.gbest = self.pbest[wh_minpbest]
            self.gbest_loc = self.pbest_loc[wh_minpbest, :]


    def updateswarmvelpos(self):

        w = self.wi
        c1 = self.c1i
        c2 = self.c2i
        r1 = np.random.random((self.n_particles,1))
        r2 = np.random.random((self.n_particles,1))
        nvel = self.vel.copy()
        nvel = w * nvel + \
               c1 * r1 * (self.pbest_loc - self.pos) + \
               c2 * r2 * (self.gbest_loc - self.pos)

        mag = np.expand_dims(np.sqrt(np.sum(nvel**2, axis=1)), axis=1)
        wh_toofast = np.nonzero(mag > self.vellimit)[0]
        #print(nvel.shape, wh_toofast.shape, mag.shape)
        nvel[wh_toofast, :] *= self.vellimit/mag[wh_toofast]

        self.vel = nvel
        self.pos += nvel

        self.boundarycheck()




    def boundarycheck(self):

        if str.lower(self.boundmethod) == 'bounce':
            self.boundarybounce()


    def boundarybounce(self):

        lb,ub = self.bounds
        for d in range(self.dimensions):
            wh_under = np.nonzero(self.pos[:,d] < lb[d])[0]
            self.pos[wh_under,d] = lb[d]
            self.vel[wh_under,d] *= -1.0

            wh_over = np.nonzero(self.pos[:, d] > ub[d])[0]
            self.pos[wh_over, d] = ub[d]
            self.vel[wh_over, d] *= -1.0

    def updatehyperparam(self, iter):
        if str.lower(self.hyperparammethod) == 'auto':

            N = float(self.niter)-1
            self.c1i = (self.c1 - self.c1/7) * (N-iter)/N + self.c1 / 7.0
            self.c2i = (self.c1 - self.c1 / 7) * (iter) / N + self.c1 / 7.0
            self.wi = self.w/2 * ((N - iter)/N)**2 + self.w/2

        else:
            self.c1i = self.c1
            self.c2i = self.c2
            self.wi = self.w


        pass
    def printprogress(self, iter):

        progress = int(round(10*float(iter)/self.niter))
        print('',end='\r' )
        print('Progress [',
              '*' * progress, ' '*(10-progress),'] ', iter+1 , '/', self.niter,
              '  global best:', "{0:.3g}".format(self.gbest),
              '  best loc:', np.array_str(self.gbest_loc, precision=4, suppress_small=True),
              sep='', end='')
    def optimize(self, function, start=None, bounds=None, niter=50, verbose = 1, **kwargs):

        self.initializeswarm(start, bounds)

        with multiprocessing.Pool(min(multiprocessing.cpu_count(), self.n_particles)) as pool:
            if verbose >= 1:
                print('n_particle:', self.n_particles, 'c1:', self.c1, 'c2:', self.c2, 'w:', self.w )

            self.niter = niter
            for iter in range(niter):
                self.updatehyperparam(iter)
                self.updateswarmbest(function, pool, **kwargs)
                if verbose >= 1:
                    self.printprogress(iter)
                self.updateswarmvelpos()


        pool.close()
        pool.terminate()
        final_best = self.gbest
        final_loc = self.gbest_loc
        if verbose >= 1:
            print('', end='\n')
            print("Optimization finished | best cost: {}, best pos: {}".format(
                final_best, final_loc))
            print(' ')
        return final_best, final_loc



