=========
Reference
=========

.. This reference is (unfortunately) created manually.

This reference manual details the public modules, classes, and functions
in PyEBSDIndex, as generated from their docstrings. Also see the user
guide for how to use PyEBSDIndex.

.. caution::

    PyEBSDIndex is in development. This means that some breaking changes
    and changes to this reference are likely with each release.

.. module:: pyebsdindex

The list of top modules:

.. autosummary::
    pyebsdindex.ebsd_index
    pyebsdindex.pcopt

....

ebsd_index
==========

.. currentmodule:: pyebsdindex.ebsd_index
.. automodule:: pyebsdindex.ebsd_index

.. autosummary::
    EBSDIndexer

.. autoclass:: EBSDIndexer
    :members:

    .. automethod:: __init__

....

pcopt
=====

.. currentmodule:: pyebsdindex.pcopt
.. automodule:: pyebsdindex.pcopt

.. autosummary::
    optimize
    optimize_pso

.. autofunction:: optimize
.. autofunction:: optimize_pso
