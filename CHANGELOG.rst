=========
Changelog
=========

All notable changes to PyEBSDIndex will be documented in this file. The format is based
on `Keep a Changelog <https://keepachangelog.com/en/1.1.0>`_.

Unreleased
==========

Added
-----

Changed
-------

Deprecated
----------

Removed
-------

Fixed
-----
- Hough transform figure when ``verbose=2`` is passed to various indexing methods is now
  plotted in its own figure.

Security
--------

0.1.1 (2022-10-25)
==================

Added
-----
- Explanation that the pixel size must be passed as the forth PC value whenever
  ``vendor=EMSOFT`` is used.

Changed
-------
- Changed the parameter name ``patsIn`` to ``patsin`` in functions ``index_pats()`` and
  ``index_pats_distributed()``, to be in line with ``EBSDIndex.index_pats()``, and
  ``peakDetectPlan`` to ``bandDetectPlan`` in ``index_pats_distributed()``, to be in
  line with the other two functions.
- Reversed the order of the pattern height and width in the ``patDim`` parameter passed
  to ``EBSDIndex.update_file()``: the new order is (height, width).

Removed
-------
- Parameter ``filenameout`` in functions ``index_pats()`` and
  ``index_pats_distributed()``, as it is unused.

Fixed
-----
- OpenCL kernels and test data are also included in the built distribution (wheel), not
  only the source distribution.

0.1.0 (2022-07-12)
==================

Added
-----

- Installation from Anaconda on Linux and Windows for Python 3.8 and 3.9.
- Make ``ray`` for parallel indexing an optional dependency, installable via the ``pip``
  selector ``pyebsdindex[parallel]``.
- Add ``pip`` selector ``pyebsdindex[all]`` for installing both ``ray`` and ``pyopencl``
  to get parallel and GPU supported indexing.
- Support for Python 3.10.
- ``ebsd_index`` functions return both the orientation data and band identification data
  from the Radon transform.
- QUEST algorithm to get a best fit for the orientation.
- Many small improvements to Radon peak detection.
- PC conventions for Bruker, EDAX, EMsoft, kikuchipy, and Oxford.

Fixed
-----
- Minimum version of ``ray`` package set to >= 1.13.
- Maximum version of ``ray`` package set to < 1.12.0 to avoid an import error on
  Windows.
