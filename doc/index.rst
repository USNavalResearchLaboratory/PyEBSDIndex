===================================
PyEBSDIndex |release| documentation
===================================

Python based tool for Radon based EBSD orientation indexing.

The pattern processing is based on a GPU pipeline, and is based on the work of S. I.
Wright and B. L. Adams. Metallurgical Transactions A-Physical Metallurgy and Materials
Science, 23(3):759–767, 1992, and N. C. Krieger Lassen. Automated Determination of
Crystal Orientations from Electron Backscattering Patterns. PhD thesis, The Technical
University of Denmark, 1994.

The band indexing is achieved through triplet voting using the methods outlined by A.
Morawiec. Acta Crystallographica Section A Foundations and Advances, 76(6):719–734,
2020.

Additionally NLPAR pattern processing is included (original distribution
`NLPAR <https://github.com/USNavalResearchLaboratory/NLPAR>`_); P. T. Brewick, S. I.
Wright, and D. J. Rowenhorst. Ultramicroscopy, 200:50–61, May 2019.).

.. toctree::
    :hidden:
    :titlesonly:

    user/index.rst
    reference/index.rst
    dev/index.rst
    changelog.rst

Installation
============

PyEBSDIndex can be installed with `pip <https://pypi.org/project/pyebsdindex>`__ or
`conda <https://anaconda.org/conda-forge/pyebsdindex>`__:

.. tab-set::

    .. tab-item:: pip

        .. code-block:: bash

            pip install pyebsdindex[all]

    .. tab-item:: conda

        .. code-block:: bash

            conda install pyebsdindex -c conda-forge

Further details are available in the :doc:`installation guide <user/installation>`.

Learning resources
==================

.. See: https://sphinx-design.readthedocs.io/en/furo-theme/grids.html
.. grid:: 2
    :gutter: 2

    .. grid-item-card::
        :link: tutorials/index
        :link-type: doc

        :octicon:`book;2em;sd-text-info` Tutorials
        ^^^

        In-depth guides for using PyEBSDIndex.

    .. grid-item-card::
        :link: reference/index
        :link-type: doc

        :octicon:`code;2em;sd-text-info` API reference
        ^^^

        Descriptions of functions, modules, and objects in PyEBSDIndex.

Contributing
============

PyEBSDIndex is a community project maintained for and by its users. There are many ways
you can help!

- Report a bug or request a feature `on GitHub
  <https://github.com/USNavalResearchLaboratory/PyEBSDIndex>`__
- or improve the :doc:`documentation or code <dev/index>`
