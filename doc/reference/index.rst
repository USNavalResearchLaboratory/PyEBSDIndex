=============
API reference
=============

**Release**: |version|

**Date**: |today|

This reference manual details the public functions, modules, and objects in PyEBSDIndex.
For learning how to use PyEBSDIndex, see the :doc:`/tutorials/index`.

.. caution::

    PyEBSDIndex is in development. This means that some breaking changes and changes to
    this reference are likely with each release.

Functionality is inteded to be imported like this:

.. code-block:: python

    >>> from pyebsdindex import ebsd_index, pcopt
    >>> indexer = ebsd_index.EBSDIndexer()

.. currentmodule:: pyebsdindex

.. rubric:: Modules

.. autosummary::
    :toctree: generated
    :template: custom-module-template.rst

    ebsd_index
    nlpar
    pcopt
