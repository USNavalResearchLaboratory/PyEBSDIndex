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

"""Tests for internal package technicalities and package
distribution.
"""

import pytest

from pyebsdindex import _pyopencl_installed, _ray_installed


@pytest.mark.skipif(not _pyopencl_installed, reason="pyopencl is not installed")
def test_available_functionality_without_pyopencl():
    from pyebsdindex.band_detect import BandDetect
    from pyebsdindex.opencl.band_detect_cl import BandDetect as BandDetectCL
    assert issubclass(BandDetectCL, BandDetect)


@pytest.mark.skipif(_pyopencl_installed, reason="pyopencl is installed")
def test_unavailable_functionality_without_pyopencl():
    with pytest.raises(ImportError):
        from pyebsdindex.opencl.band_detect_cl import BandDetect


@pytest.mark.skipif(not _ray_installed, reason="ray is not installed")
def test_available_functionality_with_ray():
    from pyebsdindex.ebsd_index import index_pats_distributed
    #from pyebsdindex.ebsd_index import IndexerRay

    #assert callable(index_pats_distributed)
    #_ = IndexerRay.remote()


@pytest.mark.skipif(_ray_installed, reason="ray is installed")
def test_unavailable_functionality_without_ray():
    with pytest.raises(ImportError):
        from pyebsdindex.ebsd_index import index_pats_distributed
    #with pytest.raises(ImportError):
    #    from pyebsdindex.ebsd_index import IndexerRay
