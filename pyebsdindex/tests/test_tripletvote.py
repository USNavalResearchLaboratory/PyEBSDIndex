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

from itertools import product

import numpy as np

from pyebsdindex import tripletvote


class TestAddPhase:
    def test_add_phase_fcc(self):
        phase = tripletvote.addphase("FCC")
        assert np.allclose(
            phase.polefamilies, [[0, 0, 2], [1, 1, 1], [0, 2, 2], [1, 1, 3]]
        )
        angles = phase.angpairs["angles"]
        assert angles.size == 21
        assert np.unique(angles).size == 17

    def test_add_phase_bcc(self):
        phase = tripletvote.addphase("BCC")
        assert np.allclose(
            phase.polefamilies, [[0, 1, 1], [0, 0, 2], [1, 1, 2], [0, 1, 3]]
        )
        angles = phase.angpairs["angles"]
        assert angles.size == 34
        assert np.unique(angles).size == 28

    def test_add_phase_hcp(self):
        phase = tripletvote.addphase("HCP")
        assert np.allclose(
            phase.polefamilies,
            [
                [1, 0, -1, 0],
                [0, 0,  0, 2],
                [1, 0, -1, 1],
                [1, 0, -1, 2],
                [1, 1, -2, 0],
                [1, 0, -1, 3],
                [1, 1, -2, 2],
                [2, 0, -2, 1],
            ]
        )
        angles = phase.angpairs["angles"]
        assert angles.size == 82
        assert np.unique(angles).size == 74

    def test_add_phase_triclinic(self):
        # Build our own reflector list
        hkl = [1, 1, 1]
        hkl_ranges = [np.arange(-i, i + 1) for i in hkl]
        hkl = np.asarray(list(product(*hkl_ranges)), dtype=int)
        hkl = hkl[~np.all(hkl == 0, axis=1)]  # Remove (000)

        phase = tripletvote.addphase(
            spacegroup=1,
            latticeparameter=[2, 3, 4, 70, 100, 120],
            nband_earlyexit=5,
            polefamilies=hkl,
        )

        assert phase.spacegroup == 1
        assert np.allclose(phase.latticeparameter, [2, 3, 4, 70, 100, 120])
        assert phase.nband_earlyexit == 5
        assert np.allclose(phase.polefamilies, hkl)

        angles = phase.angpairs["angles"]
        assert angles.size == 312
        assert np.unique(angles).size == 77
