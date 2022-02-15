# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a
# full list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from datetime import datetime
import os
import sys

from pyebsdindex import __author__, __version__


# If extensions (or modules to document with autodoc) are in another
# directory, add these directories to sys.path here. If the directory is
# relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
sys.path.insert(0, os.path.abspath('.'))

project = "PyEBSDIndex"
copyright = f"{datetime.now().year}, {__author__}"
author = __author__
version = __version__

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
#    "sphinxcontrib.bibtex",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
#    "sphinx.ext.linkcode",
#    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_gallery.load_style",
    "nbsphinx",
]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The theme to use for HTML and HTML Help pages.  See the documentation
# for a list of builtin themes.
#
html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets
# here, relative to this directory. They are copied after the builtin
# static files, so a file named "default.css" will overwrite the builtin
# "default.css".
# html_static_path = ['_static']

# Create links to references within the documentation to these packages
intersphinx_mapping = {
    "h5py": ("https://docs.h5py.org/en/stable", None),
    "matplotlib": ("https://matplotlib.org", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
}

# -- nbsphinx configuration --------------------------------------------
nbsphinx_execute = "auto"
nbsphinx_execute_arguments = [
    "--InlineBackend.rc=figure.facecolor='w'",
    "--InlineBackend.rc=font.size=15",
]
