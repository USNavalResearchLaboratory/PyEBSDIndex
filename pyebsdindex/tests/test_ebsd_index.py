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
import pytest

from pyebsdindex import _ray_installed
from pyebsdindex.ebsd_index import EBSDIndexer
from pyebsdindex.rotlib import qu2eu


class TestEBSDIndexer:
    # Pattern used in test is simulated with an identity rotation, but
    # indexing have returned these values in various environments
    # (operating systems and Python versions)
    _possible_euler = [(0, 0, 0), (0, 18, 72), (0, 90, 90), (0, 90, 360)]

    def test_init(self):
        """Test creation of an indexer instance and its default values.
        """
        indexer = EBSDIndexer()
        assert indexer.phaselist == ["FCC"]
        assert indexer.sampleTilt == 70
        assert indexer.camElev == 5.3
        assert indexer.vendor == "EDAX"

    def test_index_pats(self, pattern_al_sim_20kv):
        """Test Hough indexing and setting/passing projection center
        values.
        """
        pc = (0.4, 0.72, 0.6)

        # Set PC upon initialization of indexer
        indexer = EBSDIndexer(PC=pc, patDim=pattern_al_sim_20kv.shape)
        data = indexer.index_pats(pattern_al_sim_20kv)[0]
        assert np.allclose(data[0]["quat"], data[1]["quat"])

        # Pass PC upon indexing
        indexer2 = EBSDIndexer(patDim=pattern_al_sim_20kv.shape)
        data2 = indexer2.index_pats(pattern_al_sim_20kv, PC=pc)[0]

        # Results are the same in both examples
        assert np.allclose(data2[0]["quat"], data[0]["quat"])

        # Expected rotation
        euler = np.rad2deg(qu2eu(data[0]["quat"]))
        assert np.isclose(euler, self._possible_euler, atol=2).any()

    @pytest.mark.skipif(not _ray_installed, reason="ray is not installed")
    def test_index_pats_multi(self, pattern_al_sim_20kv):
        """Test Hough indexing parallelized with ray."""
        from pyebsdindex.ebsd_index import index_pats_distributed

        patterns = np.repeat(pattern_al_sim_20kv[None, ...], 4, axis=0)
        indexer = EBSDIndexer(PC=(0.4, 0.6, 0.5), patDim=patterns.shape[1:])
        data = index_pats_distributed(patterns, ebsd_indexer_obj=indexer)

        # Expected rotation
        euler = np.rad2deg(qu2eu(data[0]["quat"]))
        assert np.isclose(euler[0], self._possible_euler, atol=2).any()
        assert np.allclose(euler[0], euler[1:])
