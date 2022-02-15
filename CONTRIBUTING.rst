============
Contributing
============

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