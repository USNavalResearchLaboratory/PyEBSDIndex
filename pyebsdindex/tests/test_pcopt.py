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

from pyebsdindex import ebsd_index, pcopt


class TestPCOptimization:
    def test_pc_optimize(self, pattern_al_sim_20kv):
        pc0 = (0.4, 0.6, 0.5)
        indexer = ebsd_index.EBSDIndexer(patDim=pattern_al_sim_20kv.shape)
        new_pc = pcopt.optimize(pattern_al_sim_20kv, indexer, PC0=pc0)
        assert np.allclose(new_pc, pc0, atol=0.05)

    def test_pc_optimize_pso(self, pattern_al_sim_20kv):
        pc0 = (0.4, 0.6, 0.5)
        indexer = ebsd_index.EBSDIndexer(patDim=pattern_al_sim_20kv.shape)
        new_pc = pcopt.optimize_pso(pattern_al_sim_20kv, indexer, PC0=pc0)
        assert np.allclose(new_pc, pc0, atol=0.05)
