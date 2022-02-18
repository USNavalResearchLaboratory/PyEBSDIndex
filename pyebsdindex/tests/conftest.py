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

import os

import matplotlib.pyplot as plt
import pytest

DIR_PATH = os.path.dirname(__file__)
PATTERN_FILE = os.path.join(DIR_PATH, "data/al_sim_20kv/al_sim_20kv.png")


@pytest.fixture
def pattern_al_sim_20kv():
    """20 kV Al pattern dynamically simulated with EMsoft 4.3.

    The pattern has shape (n rows, n columns) = (100, 120),
    pattern/projection center (PC) of (0.4, 0.6, 0.5) in EDAX'
    convention, the sample is tilted 70 degrees and the camera elevation
    is 5.3 degrees. The crystal has the identity rotation, (0, 0, 0) in
    Euler angles.

    See the data/al_sim_20kv/generate_al_sim_20kv.py file for how it is
    generated.
    """
    return plt.imread(PATTERN_FILE)
