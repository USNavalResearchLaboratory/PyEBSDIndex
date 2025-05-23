=========
Changelog
=========

All notable changes to PyEBSDIndex will be documented in this file. The format is based
on `Keep a Changelog <https://keepachangelog.com/en/1.1.0>`_.

0.3.8 (2025-04-01)
==================

Added
-----
- Ability to add micron bars to IPF and scalar values maps. Use the ``addmicronbar`` keyword
to ``makeipf`` and ``scalarimage`` functions.
- When using ``ebsd_index`` function, if the machine has multiple GPUs, the desired GPU
can be chosen using the ``gpu_id`` keyword.
- When making IPF maps, a grayscale mix can be added using ``graychannel`` keyword.
- New pattern quality parameter, ``iq`` which is the mean intensity of the convolved peaks
divided by the mean intensity of the radon.  Typical values are 1.8--2.0
- Initial support for Thermo-Fisher ``.pat`` files.

Changed
-------
- Minimum official support is now python 3.9
- pyebsdinex[parallel] now uses a minimum Ray v2.9
- oh5 files are currently written with OIM 8.6.
OIM 9.1 oh5 files can be specified using ``version=9.1``
- The ``fit`` value now is the _unweighted_ mean angular deviation. Previously this was
the weighted eigen value from the QUEST algorithm.
- Automatic CPU scheduling is changed for distributed indexing, avoiding spinning up many
processes on large workstations.


Fixed
-----
- ``ebsd_index_distributed`` should have better scheduling for multiple NVIDIA GPUs.
- Removed warnings around OpenCL builds
- Removed warnings with ray distributed NLPAR and indexing.



0.3.7 (2024-10-16)
==================

Fixed
-----
- Added a very hacky fix to NLPAR not working consistently on Apple-Si chips.
    For reasons I do not understand, the OpenCL routine would return without executing the NLPAR
    processing, returning patterns filled with zeros.  This attempts to detect such behavior, and will
    resubmit the job. It will attempt to run the job three times, and then it will just return the zero patterns.
    This appears to be only an issue with the Apple Mx chips/architecture.



0.3.6 (2024-08-06)
==================

Fixed
-----
- Fixed issue with newer versions of Ray and NVIDIA cards (maybe exclusively on Linux).

Changed
-------
- Small adjustment on peak fitting suggested by W. Lenthe to put better limits on the peak fit.
  Maybe especially useful for fitting noisy peaks.


0.3.5 (2024-06-07)
==================

Fixed
-----
- Further tweaking of NLPAR GPU memory limits for Apple-ARM.
- Many small type fixes for numpy 2.0 compatibility.
- Corrected GPU detection for distributed indexing.
- Fixed issue where slower machines would erroneously detect a GPU timeout.


0.3.4 (2024-06-07)
==================

Fixed
-----
- This time I think that edge case for NLPAR chunking of scans is really fixed.
- Wrote on a chalkboard 100 times, "I will run ALL the unit tests before release."

0.3.3 (2024-06-07)
==================

Fixed
-----
- Fixed edge case for NLPAR chunking of scans that would lead to a crash.
- Fixed issue where PyEBSDIndex would not use all GPUs by default.
- ``IPFColor.makeipf()`` will now automatically read the number of columns/rows in the scan from the file defined in the indexer object.



0.3.2 (2024-05-31)
==================

Fixed
-----
- Fixed issues with smaller GPUs and NLPAR.
- Improved the initial write of NLPAR file under Windows.
- Fixed issue where user sends in non-numpy array of patterns to be indexed.


0.3.1 (2024-05-24)
==================

Fixed
-----
- Fixed issue when multiple OpenCL platforms are detected.  Will default to discrete GPUs, with whatever platform has the most discrete GPUs attached.  Otherwise, will fall back to integrated graphics.


0.3.0 (2024-05-23)
==================
Added
-----
- NLPAR should now use GPU if pyopencl is installed, and a GPU is found. Expect 2-10x improvement in speed.
- Faster band indexing. Should lead to increased pattern indexing speed.

Changed
-------
- PyEBSDIndex will now automatically select discrete GPUs if both integrated and discrete GPUs are found. If no discrete GPUs are found, it will use the integrated GPU.
- Numba will now cache in the directory ~/.pyebsdindex/  This *might* help with less recompilinging after restarts.

Removed
-------
- Removed ``band_vote`` modual as that is now wrapped into triplevote.

Fixed
-----


0.2.1 (2024-01-29)
==================
Added
-----


Changed
-------
- ``nlpar.NLPAR.opt_lambda()`` method will now return the array of
  the three optimal lambdas [less, medium, more] smoothing. The
  defualt lambda is still set to [medium].  Previous return was ``None``
- ``nlpar.NLPAR.calcnlpar()`` will now return a string of the new file
  that was made with the NLPARed patterns. Previous return was ``None``


Removed
-------

Fixed
-----
- ``ebsd_pattern``: Reading HDF5 manufacturing strings, and proper identification of
  the vendors within get_pattern_file_obj
- ``ebsd_pattern``:Proper reading of parameters from Bruker HDF5 files.
- Corrected writing of oh5 files with ``ebsdfile``

0.2.0 (2023-08-08)
==================

Added
-----
- Initial support for uncompressed EBSP files from Oxford systems.
- Significant improvement in the particle swarm optimization for pattern center
  optimization.
- Initial support for non-cubic phases. Hexagonal verified with EDAX convention.
  Others are untested.
- Significant improvements in phase differentiation.
- NLPAR support for Oxford HDF5 and EBSP.
- Initial support for Oxford .h5oina files
- Added IPF coloring/legends for hexagonal phases
- Data output files in .ang and EDAX .oh5 files
- Explicit support for Python 3.11.

Changed
-------
- CRITICAL! All ``ebsd_pattern.EBSDPatternFiles.read_data()`` calls will now return TWO
  arguments. The patterns (same as previous), and an nd.array of the x,y location within
  the scan of the patterns. The origin is the center of the scan, and reported in
  microns.
- ``ebsd_index.index_pats_distributed()`` now will auto optimize the number of patterns
  processed at a time depending on GPU capability, and is set as the default.
- Updated tutorials for new features.

Removed
-------
- Removed requirement for installation of pyswarms.
- Removed any references to np.floats and replaced with float() or np.float32/64.

Fixed
-----
- Radon transform figure when ``verbose=2`` is passed to various indexing methods is now
  plotted in its own figure.
- Several bug fixes with NLPAR file reading/writing.
- Complete rewrite of the scheduling for ``ebsd_index.index_pats_distributed()``
  function to be compatible with NVIDIA cards.

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
