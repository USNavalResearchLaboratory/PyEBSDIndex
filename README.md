# PyEBSDIndex

Python based tool for Hough/Radon based EBSD orientation indexing.

[![Build status](https://github.com/USNavalResearchLaboratory/PyEBSDIndex/actions/workflows/build.yml/badge.svg)](https://github.com/USNavalResearchLaboratory/PyEBSDIndex/actions/workflows/build.yml)
[![Documentation status](https://readthedocs.org/projects/pyebsdindex/badge/?version=latest)](https://pyebsdindex.readthedocs.io/en/latest/)
[![PyPI version](https://img.shields.io/pypi/v/pyebsdindex.svg)](https://pypi.python.org/pypi/pyebsdindex)

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
In order to avoid potential conflicts with other system python packages, it is strongly recommended 
to use a virtual environment, such as venv or conda environments.  

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

Also, if you want to run the example jupyter notebooks in the documentation, 
you will need to install jypterlab:

```bash
pip install jupyterlab
```
or 
```bash
conda install jupyterlab
```

## Additional installation notes
### MacOS
The latest versions of pyopencl installed from conda-forge do not automatically include linking
to the MacOS OpenCL framework. If using a conda environment, it may be necessary to install: 
```bash
conda install -c conda-forge ocl_icd_wrapper_apple
```

Apple in recent installs has switched to zsh as the default shell.  It should be noted that zsh sees \[...\]  as a pattern.  Thus commands like: 
```bash
pip install pyebsdindex[gpu]
```
Will return an error.  "zsh: no matches found: [gpu]".  The solution is to put the comand within '...' such as:
```bash
pip install 'pyebsdindex[gpu]'
```


### MacOS with Apple Silicon
The Ray package used for distributed multi-processing only experimentally supports Apple's ARM64 architecture. More info is available [here](https://docs.ray.io/en/latest/ray-overview/installation.html).  In brief, to run on Apple ARM64, PyEBSDIndex should be installed in a conda environment.  Assuming that Ray has already been installed (perhaps as a dependency) one has activated the conda environment in the terminal, run the commands below (the first two commands are to guarantee that grpcio is fully removed, they may send a message that the packages are not installed.):
```bash
pip uninstall ray
pip uninstall grpcio
conda install -c conda-forge grpcio
pip install 'ray[default]'
```
