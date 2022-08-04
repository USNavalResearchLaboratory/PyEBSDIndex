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

"""Setup and handling of Hough indexing runs of EBSD patterns on a
single thread.
"""

from timeit import default_timer as timer

import numpy as np
import h5py

from pyebsdindex import (
    band_vote,
    ebsd_pattern,
    rotlib,
    tripletlib,
    _pyopencl_installed,
)

if _pyopencl_installed:
    from pyebsdindex.opencl import band_detect_cl as band_detect
else:
    from pyebsdindex import band_detect as band_detect


# if sys.platform == 'darwin':
#   if ray.__version__ < '1.1.0':  # this fixes an issue when running locally on a VPN
#     ray.services.get_node_ip_address = lambda: '127.0.0.1'
#   else:
#     ray._private.services.get_node_ip_address = lambda: '127.0.0.1'

RADEG = 180.0 / np.pi


def index_pats(
    patsIn=None,
    filename=None,
    filenameout=None,
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

    pats = None
    if patsIn is None:
        pdim = None
    else:
        if isinstance(patsIn, ebsd_pattern.EBSDPatterns):
            pats = patsIn.patterns
        if type(patsIn) is np.ndarray:
            pats = patsIn
        if isinstance(patsIn, h5py.Dataset):
            shp = patsIn.shape
            if len(shp) == 3:
                pats = patsIn
            if len(shp) == 2:  # just read off disk now.
                pats = patsIn[()]
                pats = pats.reshape(1, shp[0], shp[1])

        if pats is None:
            print("Unrecognized input data type")
            return
        pdim = pats.shape[-2:]

    if ebsd_indexer_obj == None:
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

    if backgroundSub == True:
        indexer.bandDetectPlan.collect_background(
            fileobj=indexer.fID, patsIn=pats, nsample=1000
        )

    dataout, banddata, indxstart, indxend = indexer.index_pats(
        patsin=pats,
        patstart=patstart,
        npats=npats,
        clparams=clparams,
        verbose=verbose,
        chunksize=chunksize,
    )

    if return_indexer_obj == False:
        return dataout, banddata
    else:
        return dataout, banddata, indexer


class EBSDIndexer:
    """Setup of Hough indexing of EBSD patterns."""

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
        tSigma=1.0,
        rSigma=1.2,
        rhoMaskFrac=0.15,
        nBands=9,
        patDim=None,
        **kwargs
    ):
        """Create an EBSD indexer.

        Parameters
        ----------
        filename : str, optional
            Name of file with EBSD patterns.
        phaselist : list of str, optional
            Options are ``"FCC"`` and ``"BCC"``. Default is ``["FCC"]``.
        vendor : str, optional
            This string determines the pattern center (PC) convention to
            use. The available options are ``"EDAX"`` (default),
            ``"BRUKER"``, ``"OXFORD"``, ``"EMSOFT"``, ``"KIKUCHIPY"``
            (equivalent to ``"BRUKER"``).
        PC : list, optional
            (PCx, PCy, PCz) in the vendor convention. For EDAX TSL, this
            is x*, y*, z*, defined in fractions of pattern width with
            respect to the lower left corner of the detector. If not
            passed, this is set to (x*, y*, z*) = (0.471659, 0.675044,
            0.630139). If ``vendor="EMSOFT"``, the PC must be four
            numbers, the final number being the pixel size.
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
            Keyword arguments passed on to ``BandDetect``.
        """
        self.filein = filename
        if self.filein is not None:
            self.fID = ebsd_pattern.get_pattern_file_obj(self.filein)
        else:
            self.fID = None

        # self.fileout = filenameout
        # if self.fileout is None:
        #     self.fileout = str.lower(Path(self.filein).stem)+'.ang'

        self.phaselist = phaselist
        self.phaseLib = []
        for ph in self.phaselist:
            self.phaseLib.append(band_vote.BandVote(tripletlib.triplib(libType=ph)))

        self.vendor = "EDAX"
        if vendor is None:
            if self.fID is not None:
                self.vendor = self.fID.vendor
        else:
            self.vendor = vendor

        if PC is None:
            self.PC = np.array([0.471659, 0.675044, 0.630139])  # a default value
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
                **kwargs
            )
        else:
            self.bandDetectPlan = bandDetectPlan

        if self.fID is not None:
            self.bandDetectPlan.band_detect_setup(
                patDim=[self.fID.patternW, self.fID.patternH]
            )
        else:
            if patDim is not None:
                self.bandDetectPlan.band_detect_setup(patDim=patDim)

        self.dataTemplate = np.dtype(
            [
                ("quat", np.float64, 4),
                ("iq", np.float32),
                ("pq", np.float32),
                ("cm", np.float32),
                ("phase", np.int32),
                ("fit", np.float32),
                ("nmatch", np.int32),
                ("matchattempts", np.int32, 2),
                ("totvotes", np.int32),
            ]
        )

    def update_file(self, filename=None, patDim=np.array([120, 120], dtype=np.int32)):
        if filename is None:
            self.filein = None
            self.bandDetectPlan.band_detect_setup(patDim=patDim)
        else:
            self.filein = filename
            self.fID = ebsd_pattern.get_pattern_file_obj(self.filein)
            self.bandDetectPlan.band_detect_setup(
                patDim=[self.fID.patternW, self.fID.patternH]
            )

    def index_pats(
        self,
        patsin=None,
        patstart=0,
        npats=-1,
        clparams=None,
        PC=[None, None, None],
        verbose=0,
        chunksize=528,
    ):
        """Hough indexing of EBSD patterns.

        Parameters
        ----------
        patsin : numpy.ndarray, optional
            EBSD patterns in an array of shape (n points, n pattern
            rows, n pattern columns). If not given, these are read from
            `self.filename`.
        patstart : int, optional
            Starting index of the patterns to index. Default is 0.
        npats : int, optional
            Number of patterns to index. Default is -1, which will index
            up to the final pattern in `patsin`.
        clparams : list, optional
            OpenCL parameters passed to pyopencl.
        PC : list, optional
            (PCx, PCy, PCz) in the vendor convention. For EDAX TSL, this
            is (x*, y*, z*), defined in fractions of pattern width with
            respect to the lower left corner of the detector. If not
            given, this is read from ``self.PC``. If
            ``vendor="EMSOFT"``, the PC must be four numbers, the final
            number being the pixel size.
        verbose : int, optional
            0 - no output, 1 - timings, 2 - timings and the Hough
            transform of the first pattern with detected bands
            highlighted.
        chunksize : int, optional
            Default is 528.

        Returns
        -------
        indxData : numpy.ndarray
            Complex numpy array (or array of structured data), that is
            [nphases + 1, npoints]. The data is stored for each phase
            used in indexing and the `indxData[-1]` layer uses the best
            guess on which is the most likely phase, based on the fit,
            and number of bands matched for each phase. Each data entry
            contains the orientation expressed as a quaternion (quat)
            (using `self.vendor`'s convention), Pattern Quality (pq),
            Confidence Metric (cm), Phase ID (phase), Fit (fit) and
            Number of Bands Matched (nmatch). There are some other
            metrics reported, but these are mostly for debugging
            purposes.
        bandData : numpy.ndarray
            Band identification data from the Radon transform.
        patstart : int
            Starting index of the indexed patterns.
        npats : int
            Number of patterns indexed. This and `patstart` are useful
            for the distributed indexing procedures.

        """
        tic = timer()

        if patsin is None:
            pats = self.fID.read_data(
                returnArrayOnly=True,
                patStartCount=[patstart, npats],
                convertToFloat=True,
            )
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
                if np.all((np.array(pshape[1:3]) - self.bandDetectPlan.patDim) == 0):
                    self.bandDetectPlan.band_detect_setup(patDim=pshape[1:3])

        if self.bandDetectPlan.patDim is None:
            self.bandDetectPlan.band_detect_setup(patterns=pats)

        npoints = pats.shape[0]
        if npats == -1:
            npats = npoints

        # print(timer() - tic)
        tic = timer()
        bandData = self.bandDetectPlan.find_bands(
            pats, clparams=clparams, verbose=verbose, chunksize=chunksize
        )
        shpBandDat = bandData.shape
        if PC[0] is None:
            PC_0 = self.PC
        else:
            PC_0 = PC
        bandNorm = self.bandDetectPlan.radonPlan.radon2pole(
            bandData, PC=PC_0, vendor=self.vendor
        )
        # print('Find Band: ', timer() - tic)

        # return bandNorm,patStart,patEnd
        tic = timer()
        # bv = []
        # for tl in self.phaseLib:
        #  bv.append(band_vote.BandVote(tl))
        nPhases = len(self.phaseLib)
        q = np.zeros((nPhases, npoints, 4))
        indxData = np.zeros((nPhases + 1, npoints), dtype=self.dataTemplate)

        indxData["phase"] = -1
        indxData["fit"] = 180.0
        indxData["totvotes"] = 0
        earlyexit = max(7, shpBandDat[1])
        for i in range(npoints):
            bandNorm1 = bandNorm[i, :, :]
            bDat1 = bandData[i, :]
            whgood = np.nonzero(bDat1["max"] > -1.0e6)[0]
            if whgood.size >= 3:
                bDat1 = bDat1[whgood]
                bandNorm1 = bandNorm1[whgood, :]
                indxData["pq"][0:nPhases, i] = np.sum(bDat1["max"], axis=0)

                for j in range(len(self.phaseLib)):
                    (
                        avequat,
                        fit,
                        cm,
                        bandmatch,
                        nMatch,
                        matchAttempts,
                        totvotes,
                    ) = self.phaseLib[j].tripvote(
                        bandNorm1,
                        band_intensity=bDat1["avemax"],
                        goNumba=True,
                        verbose=verbose,
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
                    if nMatch >= earlyexit:
                        break

        qref2detect = self.refframe2detector()
        q = q.reshape(nPhases * npoints, 4)
        q = rotlib.quat_multiply(q, qref2detect)
        q = rotlib.quatnorm(q)
        q = q.reshape(nPhases, npoints, 4)
        indxData["quat"][0:nPhases, :, :] = q
        if nPhases > 1:
            for j in range(nPhases - 1):
                indxData[-1, :] = np.where(
                    (indxData[j, :]["cm"] * indxData[j, :]["nmatch"])
                    > (indxData[j + 1, :]["cm"] * indxData[j + 1, :]["nmatch"]),
                    indxData[j, :],
                    indxData[j + 1, :],
                )
        else:
            indxData[-1, :] = indxData[0, :]

        if verbose > 0:
            print("Band Vote Time: ", timer() - tic)
        return indxData, bandData, patstart, npats

    def refframe2detector(self):
        ven = str.upper(self.vendor)
        if ven in ["EDAX", "EMSOFT", "KIKUCHIPY"]:
            q0 = np.array([np.sqrt(2.0) * 0.5, 0.0, 0.0, -1.0 * np.sqrt(2.0) * 0.5])
            tiltang = -1.0 * (90.0 - self.sampleTilt + self.camElev) / RADEG
            q1 = np.array([np.cos(tiltang * 0.5), np.sin(tiltang * 0.5), 0.0, 0.0])
            quatref2detect = rotlib.quat_multiply(q1, q0)

        if ven in ["OXFORD", "BRUKER"]:
            tiltang = -1.0 * (90.0 - self.sampleTilt + self.camElev) / RADEG
            q1 = np.array([np.cos(tiltang * 0.5), np.sin(tiltang * 0.5), 0.0, 0.0])
            quatref2detect = q1

        return quatref2detect

    def pcCorrect(
        self, xy=[[0.0, 0.0]]
    ):  # at somepoint we will put some methods here for correcting the PC
        # depending on the location within the scan.  Need to correct band_detect.radon2pole to accept a
        # PC for each point
        pass
