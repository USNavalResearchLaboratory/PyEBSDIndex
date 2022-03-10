# PyEBSDIndex

Python based tool for Hough/Radon based EBSD orientation indexing.

[![Build status](https://github.com/USNavalResearchLaboratory/PyEBSDIndex/actions/workflows/build.yml/badge.svg)](https://github.com/USNavalResearchLaboratory/PyEBSDIndex/actions/workflows/build.yml)
[![Documentation status](https://readthedocs.org/projects/pyebsdindex/badge/?version=latest)](https://pyebsdindex.readthedocs.io/en/latest/)

The pattern processing is based on a GPU pipeline, and is based on the work of S. I.
Wright and B. L. Adams. Metallurgical Transactions A-Physical Metallurgy and Materials
Science, 23(3):759–767, 1992, and N. Krieger Lassen. Automated Determination of Crystal
Orientations from Electron Backscattering Patterns. PhD thesis, The Technical University
of Denmark, 1994.

The band indexing is achieved through triplet voting using the methods outlined by A.
Morawiec. Acta Crystallographica Section A Foundations and Advances, 76(6):719–734,
2020.

Additionally NLPAR pattern processing is included (original distribution
[NLPAR](https://github.com/USNavalResearchLaboratory/NLPAR); P. T. Brewick, S. I.
Wright, and D. J. Rowenhorst. Ultramicroscopy, 200:50–61, May 2019.).

Documentation with a user guide, API reference, changelog, and contributing guide is
available at https://pyebsdindex.readthedocs.io.

## Installation

The package can be installed from the
[Python Package Index](https://pypi.org/project/pyebsdindex) (`pip`) or from source on
all operating systems:

```bash
pip install pyebsdindex
```

Installing with optional GPU support via `pyopencl`:

```bash
pip install pyebsdindex[gpu]
```

Please refer to the [pyopencl](https://documen.tician.de/pyopencl/misc.html)
installation documentation in case installation fails.

Installing the package from source with optional dependencies for running tests

```bash
git clone https://github.com/USNavalResearchLaboratory/PyEBSDIndex
cd PyEBSDIndex
pip install --editable .[tests]
```