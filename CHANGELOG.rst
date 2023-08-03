=========
Changelog
=========

All notable changes to PyEBSDIndex will be documented in this file. The format is based
on `Keep a Changelog <https://keepachangelog.com/en/1.1.0>`_.

Unreleased
==========

Added
-----
- Initial support for uncompressed EBSP files from Oxford systems.
- Significant improvement in the particle swarm optimization for pattern center optimization.
- Initial support for non-cubic phases. Hexagonal verified with EDAX convention.  Others are untested.
- Significant improvements in phase differentiation.
- NLPAR support for Oxford HDF5 and EBSP.
- Initial support for Oxford .h5oina files
- Added IPF coloring/legends for hexagonal phases
- Data output files in .ang and EDAX .oh5 files


Changed
-------
- CRITICAL! All ``ebsd_pattern.EBSDPatternFiles.read_data()`` calls will now return TWO arguments.
  The patterns (same as previous), and an nd.array of the x,y location within the scan of the patterns. The origin is
  the center of the scan, and reported in microns.
- ``ebsd_index.index_pats_distributed()`` now will auto optimize the number of patterns processed at a time depending on GPU
    capability, and is set as the default.
- Updated tutorials for new features.


Deprecated
----------

Removed
-------
- Removed requirement for installation of pyswarms.
- Removed any references to np.floats and replaced with float() or np.float32/64.
Fixed
-----
- Radon transform figure when ``verbose=2`` is passed to various indexing methods is now
  plotted in its own figure.
- Several bug fixes with NLPAR file reading/writing.
- Complete rewrite of the scheduling for ``ebsd_index.index_pats_distributed()`` function to be compatible
  with NVIDIA cards.

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
