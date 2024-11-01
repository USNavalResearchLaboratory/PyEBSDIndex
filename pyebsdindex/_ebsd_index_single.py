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
# The US Naval Research Laboratory Date: 22 May 2024

# For further information see:
# David J. Rowenhorst, Patrick G. Callahan, Håkon W. Ånes. Fast Radon transforms for
# high-precision EBSD orientation determination using PyEBSDIndex.
# Journal of Applied Crystallography, 57(1):3–19, 2024.
# DOI: 10.1107/S1600576723010221


"""Setup and handling of Radon indexing runs of EBSD patterns on a
single thread.
"""

from timeit import default_timer as timer

import numpy as np
import h5py

from pyebsdindex import tripletvote as bandindexer  # use triplet voting as the default indexer.
from pyebsdindex.tripletvote import BandIndexer
from pyebsdindex import (
    ebsd_pattern,
    rotlib,
    _pyopencl_installed,
)

if _pyopencl_installed:
    from pyebsdindex.opencl import band_detect_cl as band_detect
else:
    from pyebsdindex import band_detect as band_detect


RADEG = 180.0 / np.pi


def index_pats(
    patsin=None,
    filename=None,
    phaselist=["FCC"],
    vendor=None,
    PC=None,
    sampleTilt=70.0,
    camElev=5.3,
    bandDetectPlan=None,
    nRho=90,
    nTheta=180,
    tSigma=None,
    rSigma=None,
    rhoMaskFrac=0.1,
    patternmask = None,
    patternmaskindex = None,
    nBands=9,
    backgroundSub=False,
    patstart=0,
    npats=-1,
    return_indexer_obj=False,
    ebsd_indexer_obj=None,
    clparams=None,
    verbose=0,
    chunksize=528,
    gpu_id=None,
):
    """Index EBSD patterns on a single thread.

    Parameters
    ----------
    patsin : numpy.ndarray, optional
        EBSD patterns in an array of shape (n points, n pattern
        rows, n pattern columns). If not given, these are read from
        ``filename``.
    filename : str, optional
        Name of file with EBSD patterns. If not given, ``patsin`` must
        be passed.
    phaselist : list of str, optional
        Options are ``"FCC"`` and ``"BCC"``. Default is ``["FCC"]``.
    vendor : str, optional
        Which vendor convention to use for the pattern center (PC) and
        the returned orientations. The available options are ``"EDAX"``
        (default), ``"BRUKER"``, ``"OXFORD"``, ``"EMSOFT"``,
        ``"KIKUCHIPY"``.
    PC : list, optional
        Pattern center (PCx, PCy, PCz) in the :attr:`indexer.vendor` or
        ``vendor`` convention. For EDAX TSL, this is (x*, y*, z*),
        defined in fractions of pattern width with respect to the lower
        left corner of the detector. If not passed, this is set to (x*,
        y*, z*) = (0.471659, 0.675044, 0.630139). If
        ``vendor="EMSOFT"``, the PC must be four numbers, the final
        number being the pixel size.
    sampleTilt : float, optional
        Sample tilt towards the detector in degrees. Default is 70
        degrees. Unused if ``ebsd_indexer_obj`` is passed.
    camElev : float, optional
        Camera elevation in degrees. Default is 5.3 degrees. Unused
        if ``ebsd_indexer_obj`` is passed.
    bandDetectPlan : pyebsdindex.band_detect.BandDetect, optional
        Collection of parameters using in band detection. Unused if
        ``ebsd_indexer_obj`` is passed.
    nRho : int, optional
        Default is 90 degrees. Unused if ``ebsd_indexer_obj`` is
        passed.
    nTheta : int, optional
        Default is 180 degrees. Unused if ``ebsd_indexer_obj`` is
        passed.
    tSigma : float, optional
        Unused if ``ebsd_indexer_obj`` is passed.
    rSigma : float, optional
        Unused if ``ebsd_indexer_obj`` is passed.
    rhoMaskFrac : float, optional
        Default is 0.1. Unused if ``ebsd_indexer_obj`` is passed.
    nBands : int, optional
        Number of detected bands to use in triplet voting. Default
        is 9. Unused if ``ebsd_indexer_obj`` is passed.
    backgroundSub : bool, optional
        Whether to subtract a static background prior to indexing.
        Default is ``False``.
    patstart : int, optional
        Starting index of the patterns to index. Default is ``0``.
    npats : int, optional
        Number of patterns to index. Default is ``-1``, which will
        index up to the final pattern in ``patsin``.
    return_indexer_obj : bool, optional
        Whether to return the EBSD indexer. Default is ``False``.
    ebsd_indexer_obj : EBSDIndexer, optional
        EBSD indexer. If not given, many of the above parameters must be
        passed. Otherwise, these parameters are retrieved from this
        indexer.
    clparams : list, optional
        OpenCL parameters passed to :mod:`pyopencl` if the package is
        installed.
    verbose : int, optional
        0 - no output (default), 1 - timings, 2 - timings and the Radon
        transform of the first pattern with detected bands highlighted.
    chunksize : int, optional
        Default is 528.
    gpu_id : int, optional
        ID of GPU to use if :mod:`pyopencl` is installed.

    Returns
    -------
    indxData : numpy.ndarray
        Complex numpy array (or array of structured data), that is
        [nphases + 1, npoints]. The data is stored for each phase used
        in indexing and the ``indxData[-1]`` layer uses the best guess
        on which is the most likely phase, based on the fit, and number
        of bands matched for each phase. Each data entry contains the
        orientation expressed as a quaternion (quat) (using the
        convention of ``vendor`` or :attr:`indexer.vendor`), Pattern
        Quality (pq), Confidence Metric (cm), Phase ID (phase), Fit
        (fit) and Number of Bands Matched (nmatch). There are some other
        metrics reported, but these are mostly for debugging purposes.
    bandData : numpy.ndarray
        Band identification data from the Radon transform. Stored
        as a structured numpy array, of dimensions [npoints, nbands].

        With fields that include:
            - id: band ID
            - max: peak max intesensity (used to calculate pattern quality)
            - maxloc: nearest integer location of the Radon peak
            - avemax: nearest neighbor average of the max peak intensity
            - aveloc: sub-pixel location of the Radon peak
            - width: a metric of the band width
            - theta: the theta value of the sub-pixel location on the Radon (lower-left origin)
            - rho: the rho value of the sub-pixel location on the Radon (lower-left origin)
            - valid: was the peak detected
            - band_match_index: index for phase number and pole number that indexed to this band
              (use :meth:`~EBSDIndexer.getmatchedpole`)

    indexer : EBSDIndexer
        EBSD indexer, returned if ``return_indexer_obj=True``.
    """
    pats = None
    if patsin is None:
        pdim = None
    else:
        if isinstance(patsin, ebsd_pattern.EBSDPatterns):
            pats = patsin.patterns
        elif isinstance(patsin, np.ndarray):
            pats = patsin
        elif isinstance(patsin, h5py.Dataset):
            shp = patsin.shape
            if len(shp) == 3:
                pats = patsin
            elif len(shp) == 2:  # Just read off disk now
                pats = patsin[()]
                pats = pats.reshape(1, shp[0], shp[1])
        else:
            raise ValueError("Unrecognized input data type")

        pdim = pats.shape[-2:]

    if ebsd_indexer_obj is None:
        indexer = EBSDIndexer(
            filename=filename,
            phaselist=phaselist,
            vendor=vendor,
            PC=PC,
            sampleTilt=sampleTilt,
            camElev=camElev,
            bandDetectPlan=bandDetectPlan,
            nRho=nRho,
            nTheta=nTheta,
            patternmask = patternmask,
            patternmaskindex = patternmaskindex,
            tSigma=tSigma,
            rSigma=rSigma,
            rhoMaskFrac=rhoMaskFrac,
            nBands=nBands,
            patDim=pdim,
            gpu_id=gpu_id,
        )
    else:
        indexer = ebsd_indexer_obj

    if filename is not None:
        indexer.update_file(filename)
    if pats is not None:
        if not np.all(indexer.bandDetectPlan.patDim == np.array(pdim)):
            indexer.update_file(patDim=pats.shape[-2:])

    if backgroundSub:
        indexer.bandDetectPlan.collect_background(
            fileobj=indexer.fID, patsIn=pats, nsample=1000
        )

    #indexer.bandDetectPlan.radonPlan.masksetup(mask=patternmask, maskindex=patternmaskindex)

    dataout, banddata, indxstart, indxend = indexer.index_pats(
        patsin=pats,
        patstart=patstart,
        npats=npats,
        clparams=clparams,
        verbose=verbose,
        chunksize=chunksize,
    )

    if not return_indexer_obj:
        return dataout, banddata
    else:
        return dataout, banddata, indexer


class EBSDIndexer:
    """Setup of Radon indexing of EBSD patterns.

    Parameters
    ----------
    filename : str, optional
        Name of file with EBSD patterns.
    phaselist : list of str, optional
        Options are ``"FCC"`` and ``"BCC"``. Default is ``["FCC"]``.
    vendor : str, optional
        Which vendor convention to use for the pattern center (PC) and
        the returned orientations. The available options are ``"EDAX"``
        (default), ``"BRUKER"``, ``"OXFORD"``, ``"EMSOFT"``,
        ``"KIKUCHIPY"``.
    PC : list, optional
        Pattern center (PCx, PCy, PCz) in the ``vendor`` convention. For
        EDAX TSL, this is (x*, y*, z*), defined in fractions of pattern
        width with respect to the lower left corner of the detector. If
        not passed, this is set to (x*, y*, z*) = (0.471659, 0.675044,
        0.630139). If ``vendor="EMSOFT"``, the PC must be four numbers,
        the final number being the pixel size.
    sampleTilt : float, optional
        Sample tilt towards the detector in degrees. Default is 70
        degrees. Unused if ``ebsd_indexer_obj`` is passed.
    camElev : float, optional
        Camera elevation in degrees. Default is 5.3 degrees. Unused
        if ``ebsd_indexer_obj`` is passed.
    bandDetectPlan : pyebsdindex.band_detect.BandDetect, optional
        Collection of parameters using in band detection. Unused if
        ``ebsd_indexer_obj`` is passed.
    nRho : int, optional
        Default is 90 degrees. Unused if ``ebsd_indexer_obj`` is
        passed.
    nTheta : int, optional
        Default is 180 degrees. Unused if ``ebsd_indexer_obj`` is
        passed.
    tSigma : float, optional
        Unused if ``ebsd_indexer_obj`` is passed.
    rSigma : float, optional
        Unused if ``ebsd_indexer_obj`` is passed.
    rhoMaskFrac : float, optional
        Default is 0.1. Unused if ``ebsd_indexer_obj`` is passed.
    nBands : int, optional
        Number of detected bands to use in triplet voting. Default
        is 9. Unused if ``ebsd_indexer_obj`` is passed.
    patDim : int, optional
        Number of dimensions of pattern array.
    **kwargs
        Keyword arguments passed on to
        :class:`~pyebsdindex.band_detect.BandDetect`.
    """

    def __init__(
        self,
        filename=None,
        phaselist=["FCC"],
        vendor=None,
        PC=None,
        sampleTilt=70.0,
        camElev=5.3,
        bandDetectPlan=None,
        nRho=90,
        nTheta=180,
        patternmask = None,
        patternmaskindex=None,
        tSigma=1.0,
        rSigma=1.2,
        rhoMaskFrac=0.15,
        nBands=9,
        patDim=None,
        nband_earlyexit=None,
        **kwargs
    ):
        """Create an EBSD indexer."""
        self.filein = filename
        if self.filein is not None:
            self.fID = ebsd_pattern.get_pattern_file_obj(self.filein)
        else:
            self.fID = None

        self.phaselist = phaselist
        self.phaseLib = []
        for ph in self.phaselist:
            if ph is None:
                self.phaseLib.append(None)
            if isinstance(ph, str):
                self.phaseLib.append(bandindexer.addphase(libtype=ph))
            if isinstance(ph, BandIndexer):
                self.phaseLib.append(ph)

        self.vendor = "EDAX"
        if vendor is None:
            if self.fID is not None:
                self.vendor = self.fID.vendor
        else:
            self.vendor = vendor

        if PC is None:
            self.PC = np.array([0.471659, 0.675044, 0.630139])  # A default value
        else:
            self.PC = np.asarray(PC)

        self.PCcorrectMethod = None
        self.PCcorrectParam = None

        self.sampleTilt = sampleTilt
        self.camElev = camElev

        if bandDetectPlan is None:
            self.bandDetectPlan = band_detect.BandDetect(
                nRho=nRho,
                nTheta=nTheta,
                tSigma=tSigma,
                rSigma=rSigma,
                rhoMaskFrac=rhoMaskFrac,
                nBands=nBands,
                nPhases = len(self.phaseLib),
                **kwargs
            )
        else:
            self.bandDetectPlan = bandDetectPlan

        if self.fID is not None:
            self.bandDetectPlan.band_detect_setup(
                patDim=[self.fID.patternW, self.fID.patternH],
                patternmask=patternmask, patternmaskindex=patternmaskindex
            )
        elif patDim is not None:
            self.bandDetectPlan.band_detect_setup(
                patDim=patDim,
                patternmask=patternmask, patternmaskindex=patternmaskindex
            )

        self.nband_earlyexit = nband_earlyexit
        self.dataTemplate = np.dtype(
            [
                ("quat", np.float64, 4),
                ("iq", np.float32),
                ("pq", np.float32),
                ("cm", np.float32),
                ("phase", np.int32),
                ("fit", np.float32),
                ("nmatch", np.int32),
                ("matchattempts", np.int32, 4),
                ("totvotes", np.int32),
            ]
        )

    def update_file(self, filename=None, patDim=np.array([120, 120], dtype=np.int32)):
        """Update file with patterns to index.

        Parameters
        ----------
        filename : str, optional
            Name of file with EBSD patterns.
        patDim : numpy.ndarray
            1D array with two values, the pattern height and width.
        """
        if filename is None:
            self.filein = None
            self.bandDetectPlan.band_detect_setup(patDim=patDim)
        elif isinstance(filename, ebsd_pattern.EBSDPatternFile):
            self.fID = filename  
            self.bandDetectPlan.band_detect_setup(
                patDim=[self.fID.patternH, self.fID.patternW]
            )       
            self.filein = filename.filepath   
        else:
            self.filein = filename
            self.fID = ebsd_pattern.get_pattern_file_obj(self.filein)
            self.bandDetectPlan.band_detect_setup(
                patDim=[self.fID.patternH, self.fID.patternW]
            )

    def index_pats(
        self,
        patsin=None,
        patstart=0,
        npats=-1,
        xyloc=None,
        clparams=None,
        PC=None,
        verbose=0,
        chunksize=512,
    ):
        """Index EBSD patterns.

        Parameters
        ----------
        patsin : numpy.ndarray, optional
            EBSD patterns in an array of shape (n points, n pattern
            rows, n pattern columns). If not given, these are read from
            :attr:`self.filename`.
        patstart : int, optional
            Starting index of the patterns to index. Default is ``0``.
        npats : int, optional
            Number of patterns to index. Default is ``-1``, which will
            index up to the final pattern in ``patsin``.
        clparams : list, optional
            OpenCL parameters passed to :mod:`pyopencl`.
        PC : list, optional
            Pattern center (PC) parameters (PCx, PCy, PCz) in the vendor
            convention. For EDAX TSL, this is (x*, y*, z*), defined in
            fractions of pattern width with respect to the lower left
            corner of the detector. If not given, this is read from
            :attr:`self.PC`. If :attr:`vendor` is ``"EMSOFT"``, the PC
            must be four numbers, the final number being the pixel size.
        verbose : int, optional
            0 - no output (default), 1 - timings, 2 - timings and the
            Radon transform of the first pattern with detected bands
            highlighted.
        chunksize : int, optional
            Default is 512.

        Returns
        -------
        indxData : numpy.ndarray
            Structured numpy array, that is
            [nphases + 1, npoints]. The data is stored for each phase used
            in indexing and the ``indxData[-1]`` layer uses the best guess
            on which is the most likely phase, based on the fit, and number
            of bands matched for each phase. Each data entry contains the
            orientation expressed as a quaternion ('quat') (using the
            convention of ``vendor`` or :attr:`indexer.vendor`), Pattern
            Quality ('pq'), Confidence Metric ('cm'), Phase ID ('phase'), Fit
            ('fit') and Number of Bands Matched ('nmatch'). There are some other
            metrics reported, but these are mostly for debugging purposes.
            The number and order of fields are not guaranteed to remain the same, but
            fields listed here are stable.
            (phase) parameter will be set to -1 for any no-solution point.
        bandData : numpy.ndarray
            Band identification data from the Radon transform. Stored
            as a structured numpy array, of dimensions [npoints, nbands].

            With fields that include:
                - id: band ID
                - max: peak max intesensity (used to calculate pattern quality)
                - maxloc: nearest integer location of the Radon peak
                - avemax: nearest neighbor average of the max peak intensity
                - aveloc: sub-pixel location of the Radon peak
                - width: a metric of the band width
                - theta: the theta value of the sub-pixel location on the Radon (lower-left origin)
                - rho: the rho value of the sub-pixel location on the Radon (lower-left origin)
                - valid: was the peak detected
                - band_match_index: index for phase number and pole number that indexed to this band
                  (use :meth:`~EBSDIndexer.getmatchedpole`)

        patstart : int
            Starting index of the indexed patterns.
        npats : int
            Number of patterns indexed. This and `patstart` are useful
            for the distributed indexing procedures.
        """

        pats, xyloc = self._getpats(patsin=patsin, patstart=patstart, npats=npats, xyloc=xyloc)
        # just a check that the band_detect is ready for this size pattern.
        if self.bandDetectPlan.patDim is None:
            self.bandDetectPlan.band_detect_setup(patterns=pats)

        npoints = pats.shape[0]
        if npats == -1:
            npats = npoints

        banddata, bandnorm = self._detectbands(pats, PC, xyloc=xyloc, clparams=clparams, verbose=verbose,
                                               chunksize=chunksize)
        tic = timer()

        indxData, banddata = self._indexbandsphase(banddata, bandnorm, verbose=verbose)

        if verbose > 0:
            print("Band Vote Time: ", timer() - tic)

        return indxData, banddata, patstart, npats

    def getmatchedpole(self, ebsddata, banddata, phasenumber = -1, float_out=False):
        """Return the pole from the library that was matched to the
        detected band.

        Parameters
        ----------
        ebsddata: numpy.ndarray
            Output structured ebsd data array from
            :meth:`~pyebsdindex.ebsd_index.index_pats` or
            :meth:`~pyebsdindex.ebsd_index.index_pats_distributed`.
        banddata : numpy.ndarray
            Output structured band data array from
            :meth:`~pyebsdindex.ebsd_index.index_pats` or
            :meth:`~pyebsdindex.ebsd_index.index_pats_distributed`.
        phasenumber: int, optional
            Default( -1). Set this to which phase the poles should be returned for.
            The default is to return the match to the best-fit phase.
        float_out : bool, optional
            Default (False) is to return an array of ints with Miller
            indices. If set to True, then floats, with unit length, will
            be returned in the sample Cartesian reference frame.
            (Length is only valid for cubic systems).

        Returns
        -------
        numpy.ndarray
            The default is an array [npoints, nbands, 3] that contains
            the Miller indices (as ints) of the matching pole (note that
            hexagonal will also return only three-index notation). If
            the float_out is set to True, then the output will be
            floating point vectors of length one, within the sample
            Cartesian reference frame.

            If the pole was unindexed, then for that entry, this will return
            [0,0,0].  Note - this might be a single band that was unindexed.
            If all bands return unindexed, this might be because the pattern did
            not have enough bands to index, or because another phase forced an early
            exit (solution was good enough that no other phases were tested).
            Setting ``EBSDIndexer.nband_earlyexit`` to a value that is greater than the
            number of bands will avoid this.
        """
        nphases = len(self.phaseLib)

        bnddat = banddata
        shpbdndat = bnddat.shape
        if len(shpbdndat) == 0:
            bnddat = np.array(bnddat).reshape(1)
        shpbdndat = bnddat.shape
        nbands = shpbdndat[-1]
        if len(shpbdndat) == 1:
            bnddat = bnddat.reshape(1, nbands)
        shpbdndat = bnddat.shape
        npoints = shpbdndat[0]

        polesout = np.zeros((npoints,nbands,3))
        if float_out is False:
            polekey = 'poles'
        else:
            polekey = 'polesCart'

        if phasenumber == -1: # use the best-fit phase ...
            for ph in range(nphases):
                #wh = np.nonzero(bnddat['band_match_index'][:,0,0] == ph)[0]
                wh = np.nonzero(ebsddata[-1, :]['phase'] == ph)[0]
                if len(wh) == 0:
                    continue
                pindex = (bnddat[wh]['band_match_index'][:,:, ph]).flatten()
                wh2 = np.nonzero(pindex >= 0)[0]
                if len(wh2) == 0:
                    continue

                poles = self.phaseLib[ph].completelib[polekey][pindex[wh2],:]
                temp = np.zeros((pindex.shape[0],3))

                temp[wh2,:] = poles
                polesout[wh, :, :] = temp.reshape(wh.shape[0], nbands, 3)
        else:
            pindex = (bnddat[:]['band_match_index'][:,:, phasenumber]).flatten()
            wh2 = np.nonzero(pindex >= 0)[0]
            if len(wh2) > 0:
                poles = self.phaseLib[phasenumber].completelib[polekey][pindex[wh2], :]
                temp = np.zeros((npoints*nbands, 3))
                temp[wh2, :] = poles
                polesout[:, :, :] = temp.reshape(npoints,nbands, 3)


        if float_out is False:
            polesout = np.round(polesout).astype(int)

        return polesout

    def _getpats(self, patsin=None, patstart=0, npats=-1, xyloc=None):
        if patsin is None:
            pats, xylocin = self.fID.read_data(
                returnArrayOnly=True,
                patStartCount=[patstart, npats],
                convertToFloat=False,
            )
            if xyloc is None:
                xyloc = xylocin
        else:
            pshape = patsin.shape
            if len(pshape) == 2:
                pats = np.reshape(patsin, (1, pshape[0], pshape[1]))
            else:
                pats = patsin  # [patStart:patEnd, :,:]
            pshape = pats.shape

            if self.bandDetectPlan.patDim is None:
                self.bandDetectPlan.band_detect_setup(patDim=pshape[1:3])
            else:
                if np.all((np.array(pshape[1:3]) == self.bandDetectPlan.patDim)) == False:
                    self.bandDetectPlan.band_detect_setup(patDim=pshape[1:3])
        return pats, xyloc

    def _detectbands(self, pats, PC, xyloc=None, clparams=None, verbose=0, chunksize=528):
        banddata = self.bandDetectPlan.find_bands(
            pats, clparams=clparams, verbose=verbose, chunksize=chunksize
        )
        #  shpBandDat = banddata.shape
        if PC is None:
            PC_0 = self.PC
        else:
            PC_0 = PC
        bandnorm = self.bandDetectPlan.radonPlan.radon2pole(
            banddata, PC=PC_0, vendor=self.vendor
        )
        return banddata, bandnorm

    def _indexbandsphase(self, banddata, bandnorm, verbose=0):

#
        rhomax = self.bandDetectPlan.rhoMax * (1-self.bandDetectPlan.rhoMaskFrac)
        shpBandDat = banddata.shape
        npoints = int(banddata.size/(shpBandDat[-1])+0.1)
        nPhases = len(self.phaseLib)
        nBands = shpBandDat[-1]
        q = np.zeros((nPhases, npoints, 4))
        indxData = np.zeros((nPhases + 1, npoints), dtype=self.dataTemplate)
        #bandmatchindex = np.zeros((nPhases, npoints,shpBandDat[-1],2), dtype=np.int32)-100
        bandmatchindex = np.zeros((npoints,nBands, nPhases), dtype=np.int32)-100
        banddataout = banddata.copy()

        indxData["phase"] = -1
        indxData["fit"] = 180.0
        indxData["totvotes"] = 0
        indxData["nmatch"] = 0


        if self.phaseLib[0] is None:
            return indxData, banddata

        if self.nband_earlyexit is None:
            earlyexit = -1
            for ph in self.phaselist:
                if hasattr(ph, 'nband_earlyexit'):
                    earlyexit = max(earlyexit, ph.nband_earlyexit)
            if earlyexit < 0:
                earlyexit = shpBandDat[1]  # default to all the poles.
            self.nband_earlyexit = earlyexit
        else:
            earlyexit = self.nband_earlyexit

        # the adj_intensity is used to weight the peaks in the quest fit.
        #adj_intensity =  banddata["max"].copy()
        adj_intensity = (-1 * np.abs(banddata["rho"]) * 0.5 / rhomax + 1) * banddata["max"]
        adj_intensity *= ((banddata["theta"] > (2 * np.pi / 180)).astype(np.float32) + 0.5) / 2
        adj_intensity *= ((banddata["theta"] < (178.0 * np.pi / 180)).astype(np.float32) + 0.5) / 2

        adj_intensity *= banddata["valid"]

        #print(adj_intensity.shape)


        for j in range(len(self.phaseLib)):

            indxData['pq'][j, :] = np.mean(banddata['max'] * banddata['valid'], axis=1) #/ shpBandDat[-1]


            p2do = np.ravel(np.nonzero(np.max(indxData["nmatch"], axis=0) < earlyexit)[0])

            if p2do.size ==0:
                break
            (
                avequat,
                fit,
                cm,
                bandmatch,
                nMatch,
                matchAttempts,
                totvotes,
            ) = self.phaseLib[j].bandindex(
                        bandnorm[p2do, ...],
                        band_intensity=adj_intensity[p2do, ...],
                        band_widths=banddata["width"][p2do, ...],
                        verbose=verbose)

            whgood = np.nonzero(nMatch >= 3 )[0]
            if whgood.size > 0:
                whgood2 = p2do[whgood]
                q[j, whgood2, :] = avequat[whgood, ...]
                indxData["fit"][j, whgood2] = fit[whgood]
                indxData["cm"][j, whgood2] = cm[whgood]
                indxData["phase"][j, whgood2] = j
                indxData["nmatch"][j, whgood2] = nMatch[whgood]
                indxData["matchattempts"][j, whgood2] = matchAttempts[whgood, ...]
                indxData["totvotes"][j, whgood2] = totvotes[whgood]
                bandmatchindex[whgood2, ..., j] = bandmatch[whgood, ...].reshape(whgood.size,nBands )




        qref2detect = self._detector2refframe()
        q = q.reshape(nPhases * npoints, 4)
        q = rotlib.quat_multiply(q, qref2detect)
        q = rotlib.quatnorm(q)
        q = q.reshape(nPhases, npoints, 4)
        indxData["quat"][0:nPhases, :, :] = q
        indxData[-1, :] = indxData[0, :]
        banddataout['band_match_index'][:,:, 0:nPhases] = bandmatchindex[:,:,:]#.squeeze()
        if nPhases > 1:
            for j in range(1, nPhases):
                # indxData[-1, :] = np.where(
                #    (indxData[j, :]["cm"] * indxData[j, :]["nmatch"])
                #    > (indxData[j + 1, :]["cm"] * indxData[j + 1, :]["nmatch"]),
                #    indxData[j, :],
                #    indxData[j + 1, :],
                phasetest = ((3.0 - indxData[j, :]["fit"]) * indxData[j, :]["nmatch"]) \
                            > ((3.0 - indxData[-1, :]["fit"]) * indxData[-1, :]["nmatch"])
                whbetter = np.nonzero(phasetest)
                indxData[-1, whbetter] = indxData[j, whbetter]
                #banddataout['band_match_index'][whbetter,:] =  bandmatchindex[j,whbetter,:,:].squeeze()
        return indxData, banddataout

    def _indexbandsphase_old(self, banddata, bandnorm, verbose=0):

#
        rhomax = self.bandDetectPlan.rhoMax * (1-self.bandDetectPlan.rhoMaskFrac)
        shpBandDat = banddata.shape
        npoints = int(banddata.size/(shpBandDat[-1])+0.1)
        nPhases = len(self.phaseLib)
        q = np.zeros((nPhases, npoints, 4))
        indxData = np.zeros((nPhases + 1, npoints), dtype=self.dataTemplate)
        bandmatchindex = np.zeros((nPhases, npoints,shpBandDat[-1],2), dtype=np.int32)-100
        banddataout = banddata.copy()

        indxData["phase"] = -1
        indxData["fit"] = 180.0
        indxData["totvotes"] = 0
        if self.phaseLib[0] is None:
            return indxData, banddata

        if self.nband_earlyexit is None:
            earlyexit = -1
            for ph in self.phaselist:
                if hasattr(ph, 'nband_earlyexit'):
                    earlyexit = max(earlyexit, ph.nband_earlyexit)
            if earlyexit < 0:
                earlyexit = shpBandDat[1]  # default to all the poles.
            self.nband_earlyexit = earlyexit
        else:
            earlyexit = self.nband_earlyexit

        for i in range(npoints):
            bandNorm1 = bandnorm[i, :, :]
            bDat1 = banddata[i, :]
            whgood = np.nonzero(bDat1["max"] > -1.0e6)[0]
            if whgood.size >= 3:
                bDat1 = bDat1[whgood]
                bandNorm1 = bandNorm1[whgood, :]
                indxData["pq"][0:nPhases, i] = np.sum(bDat1["max"], axis=0)
                adj_intensity = (-1*np.abs(bDat1["rho"]) * 0.5 / rhomax + 1) * bDat1["max"]
                adj_intensity *= ((bDat1["theta"] > (2*np.pi/180)).astype(np.float32)+0.5)/2
                adj_intensity *= ((bDat1["theta"] < (178.0 * np.pi / 180)).astype(np.float32)+0.5)/2
                #print(bDat1["max"])
                #print(adj_intensity)
                for j in range(len(self.phaseLib)):
                    bandmatchindex[j,i, :, 0] = j

                    (
                        avequat,
                        fit,
                        cm,
                        bandmatch,
                        nMatch,
                        matchAttempts,
                        totvotes,
                    ) = self.phaseLib[j].bandindex(
                        bandNorm1, band_intensity=adj_intensity, band_widths=bDat1["width"], verbose=verbose,
                    )
                    # avequat,fit,cm,bandmatch,nMatch, matchAttempts = self.phaseLib[j].pairVoteOrientation(bandNorm1,goNumba=True)
                    if nMatch >= 3:
                        q[j, i, :] = avequat
                        indxData["fit"][j, i] = fit
                        indxData["cm"][j, i] = cm
                        indxData["phase"][j, i] = j
                        indxData["nmatch"][j, i] = nMatch
                        indxData["matchattempts"][j, i] = matchAttempts
                        indxData["totvotes"][j, i] = totvotes
                        bandmatchindex[j,i, whgood, 1] = bandmatch

                    if nMatch >= earlyexit:
                        break

        qref2detect = self._detector2refframe()
        q = q.reshape(nPhases * npoints, 4)
        q = rotlib.quat_multiply(q, qref2detect)
        q = rotlib.quatnorm(q)
        q = q.reshape(nPhases, npoints, 4)
        indxData["quat"][0:nPhases, :, :] = q
        indxData[-1, :] = indxData[0, :]
        banddataout['band_match_index'][:,:,:] = bandmatchindex[0,:,:,:].squeeze()
        if nPhases > 1:
            for j in range(1, nPhases):
                # indxData[-1, :] = np.where(
                #    (indxData[j, :]["cm"] * indxData[j, :]["nmatch"])
                #    > (indxData[j + 1, :]["cm"] * indxData[j + 1, :]["nmatch"]),
                #    indxData[j, :],
                #    indxData[j + 1, :],
                phasetest = ((3.0 - indxData[j, :]["fit"]) * indxData[j, :]["nmatch"]) \
                            > ((3.0 - indxData[-1, :]["fit"]) * indxData[-1, :]["nmatch"])
                whbetter = np.nonzero(phasetest)
                indxData[-1, whbetter] = indxData[j, whbetter]
                banddataout['band_match_index'][whbetter,:] =  bandmatchindex[j,whbetter,:,:].squeeze()
        return indxData, banddataout
    def _detector2refframe(self):
        ven = str.upper(self.vendor)
        if ven in ["EDAX", "EMSOFT", "KIKUCHIPY"]:
            q0 = np.array([np.sqrt(2.0) * 0.5, 0.0, 0.0, -1.0 * np.sqrt(2.0) * 0.5])
            tiltang = -1.0 * (90.0 - self.sampleTilt + self.camElev) / RADEG
            q1 = np.array([np.cos(tiltang * 0.5), np.sin(tiltang * 0.5), 0.0, 0.0])
            quatref2detect = rotlib.quat_multiply(q1, q0)
        elif ven in ["OXFORD", "BRUKER"]:
            tiltang = -1.0 * (90.0 - self.sampleTilt + self.camElev) / RADEG
            q1 = np.array([np.cos(tiltang * 0.5), np.sin(tiltang * 0.5), 0.0, 0.0])
            quatref2detect = q1
        else:
            raise ValueError("`self.vendor` unknown")

        return quatref2detect

#    def pcCorrect(self, xy=[[0.0, 0.0]]):
#        # TODO: At somepoint we will put some methods here for
#        #  correcting the PC depending on the location within the scan.
#        #  Need to correct band_detect.radon2pole to accept a PC for
#        #  each point.
#        pass
