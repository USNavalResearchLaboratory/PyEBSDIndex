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

import numpy as np

from pyebsdindex import tripletvote


class TestAddPhase:
    def test_add_phase_triclinic(self):
        reflectors = np.array(
            [
                [ 1,  1,  1],
                [-1, -1, -1],
                [ 1,  0,  0],
                [-1,  0,  0],
            ],
            dtype=np.int32,
        )
        phase = tripletvote.addphase(
            spacegroup=1,
            latticeparameter=[2, 3, 4, 70, 100, 120],
            nband_earlyexit=5,
            polefamilies=reflectors,
        )

        assert phase.spacegroup == 1
        assert np.allclose(phase.latticeparameter, [2, 3, 4, 70, 100, 120])
        assert phase.nband_earlyexit == 5
        assert np.allclose(phase.polefamilies, reflectors)
