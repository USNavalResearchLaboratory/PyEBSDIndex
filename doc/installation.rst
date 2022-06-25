============
Installation
============

In order to avoid potential conflicts with other system python packages, it is strongly
recommended to use a virtual environment, such as venv or conda environments.

The package can be installed from the `Python Package Index
<https://pypi.org/project/pyebsdindex>`_ (``pip``) or from source on all operating
systems with Python >= 3.8::

    pip install pyebsdindex

Installing with optional GPU support via ``pyopencl``::

    pip install pyebsdindex[gpu]

Please refer to the `pyopencl <https://documen.tician.de/pyopencl/misc.html>`_
installation documentation in case installation fails.

Installing the package from source with optional dependencies for running tests::

    git clone https://github.com/USNavalResearchLaboratory/PyEBSDIndex
    cd PyEBSDIndex
    pip install --editable .[tests]

Also, if you want to run the example Jupyter notebooks in the documentation, you will
need to install ``jypterlab``::

    pip install jupyterlab

or::

    conda install jupyterlab

Additional installation notes
=============================

MacOS
-----

The latest versions of ``pyopencl`` installed from conda-forge do not automatically
include linking to the MacOS OpenCL framework. If using a conda environment, it may be
necessary to install::

    conda install -c conda-forge ocl_icd_wrapper_apple

Apple in recent installs has switched to ``zsh`` as the default shell.  It should be
noted that ``zsh`` sees ``\[...\]`` as a pattern. Thus commands like::

    pip install pyebsdindex[gpu]

Will return an error ``"zsh: no matches found: [gpu]"``. The solution is to put the
command within ``'...'`` such as::

    pip install 'pyebsdindex[gpu]'

MacOS with Apple Silicon
------------------------

The ``ray`` package used for distributed multi-processing only experimentally supports
Apple's ARM64 architecture. More info is available `here
<https://docs.ray.io/en/latest/ray-overview/installation.html>`_. In brief, to run on
Apple ARM64, PyEBSDIndex should be installed in a conda environment. Assuming that
``ray`` has already been installed (perhaps as a dependency) and one has activated the
conda environment in the terminal, run the commands below (the first two commands are to
guarantee that ``grpcio`` is fully removed, they may send a message that the packages
are not installed)::

    pip uninstall ray
    pip uninstall grpcio
    conda install -c conda-forge grpcio
    pip install 'ray[default]'
