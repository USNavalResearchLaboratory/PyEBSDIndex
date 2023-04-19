============
Contributing
============

PyEBSDIndex is a community maintained project. We welcome contributions in the form of
bug reports, documentation, code, feature requests, and more. The source code is hosted
on `GitHub <https://github.com/USNavalResearchLaboratory/PyEBSDIndex>`_. This guide
provides some tips on how to build documentation and run tests when contributing.

Building and writing documentation
==================================

We use `Sphinx <https://www.sphinx-doc.org/en/master/>`_ for documenting functionality.
Install necessary dependencies to build the documentation::

    pip install --editable .[doc]

Then, build the documentation from the ``doc`` directory::

    cd doc
    make html

The documentation's HTML pages are built in the ``doc/build/html`` directory from files
in the `reStructuredText (reST)
<https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_ plaintext
markup language. They should be accessible in the browser by typing
``file:///your/absolute/path/to/pyebsdindex/doc/build/html/index.html`` in the address
bar.

Tips for writing Jupyter Notebooks that are meant to be converted to reST text files by
`nbsphinx <https://nbsphinx.readthedocs.io/en/latest/>`_:

- Use ``_ = ax[0].imshow(...)`` to disable Matplotlib output if a Matplotlib command is
  the last line in a cell.
- Refer to our reference with this general MD
  ``[pcopt.optimize()](../reference.rst#pyebsdindex.pcopt.optimize)``. Remember to add
  the parentheses ``()`` for functions and methods.
- Refer to external references via standard MD like
  ``[Signal2D](http://hyperspy.org/hyperspy-doc/current/api/hyperspy._signals.signal2d.html)``.
- The Sphinx gallery thumbnail used for a notebook is set by adding the
  ``nbsphinx-thumbnail`` tag to a code cell with an image output. The notebook must be
  added to the gallery in the `index.rst` to be included in the documentation pages.
- The PyData Sphinx theme displays the documentation in a light or dark theme, depending
  on the browser/OS setting. It is important to make sure the documentation is readable
  with both themes. This means displaying all figures with a white background for axes
  labels and ticks and figure titles etc. to be readable.

Running and writing tests
=========================

Some functionality in PyEBSDIndex is tested via the `pytest <https://docs.pytest.org>`_
framework. The tests reside in the ``pyebsdindex/tests`` directory. Tests are short
methods that call functions in PyEBSDIndex and compare resulting output values with
known answers. Install necessary dependencies to run the tests::

    pip install --editable .[tests]

Some useful `fixtures <https://docs.pytest.org/en/latest/fixture.html>`_, like a
dynamically simulated Al pattern, are available in the ``conftest.py`` file.

To run the tests::

    pytest --cov --pyargs pyebsdindex

The ``--cov`` flag makes `coverage.py <https://coverage.readthedocs.io/en/latest/>`_
print a nice report in the terminal. For an even nicer presentation, you can use
``coverage.py`` directly::

    coverage html

Then, you can open the created ``htmlcov/index.html`` in the browser and inspect the
coverage in more detail.

To run only a specific test function or class, .e.g the ``TestEBSDIndexer`` class::

    pytest -k TestEBSDIndexer

This is useful when you only want to run a specific test and not the full test suite,
e.g. when you're creating or updating a test. But remember to run the full test suite
before pushing!

Tips for writing tests of Numba decorated functions:

- A Numba decorated function ``numba_func()`` is only covered if it is called in the
  test as ``numba_func.py_func()``.
- Always test a Numba decorated function calling ``numba_func()`` directly, in addition
  to ``numba_func.py_func()``, because the machine code function might give different
  results on different OS with the same Python code.

Continuous integration (CI)
===========================

We use `GitHub Actions
<https://github.com/USNavalResearchLaboratory/PyEBSDIndex/actions>`_ to ensure that
PyEBSDIndex can be installed on macOS and Linux (Ubuntu). After a successful
installation of the package, the CI server runs the tests. Add "[skip ci]" to a commit
message to skip this workflow on any commit to a pull request.