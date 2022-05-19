===========
PyEBSDIndex
===========

Python based tool for Hough/Radon based EBSD orientation indexing.

.. GitHub Actions
.. image:: https://github.com/USNavalResearchLaboratory/PyEBSDIndex/actions/workflows/build.yml/badge.svg
    :target: https://github.com/USNavalResearchLaboratory/PyEBSDIndex/actions
    :alt: Build status

.. Read the Docs
.. image:: https://readthedocs.org/projects/pyebsdindex/badge/?version=latest
    :target: https://pyebsdindex.readthedocs.io/en/latest/
    :alt: Documentation status

.. PyPI version
.. image:: https://img.shields.io/pypi/v/pyebsdindex.svg
    :target: https://pypi.python.org/pypi/pyebsdindex
    :alt: PyPI version

The pattern processing is based on a GPU pipeline, and is based on the work of S. I.
Wright and B. L. Adams. Metallurgical Transactions A-Physical Metallurgy and Materials
Science, 23(3):759–767, 1992, and N. Krieger Lassen. Automated Determination of Crystal
Orientations from Electron Backscattering Patterns. PhD thesis, The Technical University
of Denmark, 1994.

The band indexing is achieved through triplet voting using the methods outlined by A.
Morawiec. Acta Crystallographica Section A Foundations and Advances, 76(6):719–734,
2020.

Additionally NLPAR pattern processing is included (original distribution
`NLPAR <https://github.com/USNavalResearchLaboratory/NLPAR>`_); P. T. Brewick, S. I.
Wright, and D. J. Rowenhorst. Ultramicroscopy, 200:50–61, May 2019.).

.. toctree::
    :hidden:

    installation.rst

User guide
==========

.. nbgallery::
    :caption: User guide

    ebsd_index_demo.ipynb
    NLPAR_demo.ipynb

.. toctree::
    :hidden:
    :caption: Help & reference

    reference.rst
    changelog.rst
    contributing.rst
