============
Installation
============

The package can be installed with `pip <https://pypi.org/project/pyebsdindex>`__,
`conda <https://anaconda.org/conda-forge/pyebsdindex>`__, or from source, and supports
Python >= 3.7. All alternatives are available on Windows, macOS and Linux.

In order to avoid potential conflicts with other system Python packages, it is strongly
recommended to use a virtual environment, such as ``venv`` or ``conda`` environments.

With pip
========

Installing all of PyEBSDIndex' functionalities with ``pip``::

    pip install pyebsdindex[all]

To install only the strictly required rependencies and limited functionalities, use::

    pip install pyebsdindex

See the following list of selectors to select the installation of optional dependencies
required for specific functionality:

- ``gpu`` - GPU support from `pyopencl
  <https://documen.tician.de/pyopencl/misc.html>`__. Please refer to the pyopencl
  installation documentation in case installation fails.
- ``parallel`` - Parallel indexing from `ray[default]
  <https://docs.ray.io/en/latest/>`__.
- ``all`` - Install the dependencies in the above selectors.
- ``doc`` - Dependencies to build the documentation.
- ``tests`` - Dependencies to run tests.
- ``dev`` - Install dependencies in the above selectors.

With conda
==========

GPU support is included when installing from Anaconda. On Linux or Windows::

    conda install pyebsdindex -c conda-forge

On macOS (without ``ray[default]``, which has to be installed separately)::

    conda install pyebsdindex-base -c conda-forge

From source
===========

Installing the package from source with optional dependencies for running tests::

    git clone https://github.com/USNavalResearchLaboratory/PyEBSDIndex
    cd PyEBSDIndex
    pip install -e .[tests]

Also, if you want to run the example Jupyter notebooks in the documentation, you will
need to install ``jupyterlab``::

    pip install jupyterlab

or::

    conda install jupyterlab

Additional installation notes
=============================

MacOS
-----

The latest versions of ``pyopencl`` installed from Anaconda do not automatically include
linking to the MacOS OpenCL framework. If using a ``conda`` environment, it may be
necessary to install::

    conda install -c conda-forge ocl_icd_wrapper_apple

Apple in recent installs has switched to ``zsh`` as the default shell. It should be
noted that ``zsh`` sees ``\[...\]`` as a pattern. Thus commands like::

    pip install pyebsdindex[gpu]

will return an error ``"zsh: no matches found: [gpu]"``. The solution is to put the
command within ``'...'`` such as::

    pip install 'pyebsdindex[gpu]'

MacOS with Apple Silicon
------------------------

The ``ray`` package used for distributed multi-processing only experimentally supports
Apple's ARM64 architecture. More info is available `here
<https://docs.ray.io/en/latest/ray-overview/installation.html>`_. In brief, to run on
Apple ARM64, PyEBSDIndex should be installed in a ``conda`` environment. Assuming that
``ray`` has already been installed (perhaps as a dependency) and one has activated the
conda environment in the terminal, run the commands below (the first two commands are to
guarantee that ``grpcio`` is fully removed, they may send a message that the packages
are not installed)::

    pip uninstall ray
    pip uninstall grpcio
    conda install -c conda-forge grpcio
    pip install 'ray[default]'
