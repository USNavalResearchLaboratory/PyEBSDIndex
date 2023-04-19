# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a
# full list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from datetime import datetime
import inspect
import os
from os.path import relpath, dirname
import re
import sys

from pyebsdindex import __author__, __file__, __version__


# If extensions (or modules to document with autodoc) are in another
# directory, add these directories to sys.path here. If the directory is
# relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
sys.path.insert(0, os.path.abspath('.'))

project = "PyEBSDIndex"
copyright = f"{datetime.now().year}, {__author__}"
author = __author__
version = __version__
release = __version__

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "numpydoc",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_gallery.load_style",
]

# Create links to references within the documentation to these packages
intersphinx_mapping = {
    "h5py": ("https://docs.h5py.org/en/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]

# HTML theming: pydata-sphinx-theme
# https://pydata-sphinx-theme.readthedocs.io
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "logo": {
        "alt_text": project,
    },
    "github_url": "https://github.com/USNavalResearchLaboratory/PyEBSDIndex",
    "header_links_before_dropdown": 6,
    "navigation_with_keys": False,
    "show_toc_level": 2,
    "use_edit_page_button": True,
}
html_context = {
    "github_user": "USNavalResearchLaboratory",
    "github_repo": "PyEBSDIndex",
    "github_version": "main",
    "doc_path": "doc",
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Syntax highlighting
pygments_style = "friendly"

# -- nbsphinx configuration --------------------------------------------
nbsphinx_execute = "auto"
nbsphinx_execute_arguments = [
    "--InlineBackend.rc=figure.facecolor='w'",
    "--InlineBackend.rc=font.size=15",
]


def linkcode_resolve(domain, info):
    """Determine the URL corresponding to Python object.
    This is taken from SciPy's ``conf.py``:
    https://github.com/scipy/scipy/blob/main/doc/source/conf.py.
    """
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    try:
        fn = inspect.getsourcefile(obj)
    except Exception:
        fn = None
    if not fn:
        try:
            fn = inspect.getsourcefile(sys.modules[obj.__module__])
        except Exception:
            fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        lineno = None

    if lineno:
        linespec = "#L%d-L%d" % (lineno, lineno + len(source) - 1)
    else:
        linespec = ""

    startdir = os.path.abspath(os.path.join(dirname(__file__), ".."))
    fn = relpath(fn, start=startdir).replace(os.path.sep, "/")

    if fn.startswith("pyebsdindex/"):
        m = re.match(r"^.*dev0\+([a-f\d]+)$", __version__)
        pre_link = "https://github.com/USNavalResearchLaboratory/PyEBSDIndex/blob/"
        if m:
            return pre_link + "%s/%s%s" % (m.group(1), fn, linespec)
        elif "dev" in __version__:
            return pre_link + "main/%s%s" % (fn, linespec)
        else:
            return pre_link + "v%s/%s%s" % (__version__, fn, linespec)
    else:
        return None


# -- Copy button customization (taken from PyVista)
# Exclude traditional Python prompts from the copied code
copybutton_prompt_text = r">>> ?|\.\.\. "
copybutton_prompt_is_regexp = True

# -- sphinx.ext.autodoc
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
autosummary_ignore_module_all = False
autosummary_imported_members = True
autodoc_typehints_format = "short"
autodoc_default_options = {
    "show-inheritance": True,
}

# -- numpydoc
# https://numpydoc.readthedocs.io
numpydoc_show_class_members = False
numpydoc_use_plots = True
numpydoc_xref_param_type = True
numpydoc_validation_checks = {
    "all",   # All but the following:
    "ES01",  # Not all docstrings need an extended summary
    "EX01",  # Examples: Will eventually enforce
    "GL01",  # Contradicts numpydoc examples
    "GL02",  # Appears to be broken?
    "GL07",  # Appears to be broken?
    "GL08",  # Methods can be documented in super class
    "PR01",  # Parameters can be documented in super class
    "PR02",  # Properties with setters might have docstrings w/"Returns"
    "PR04",  # Doesn't seem to work with type hints?
    "RT01",  # Abstract classes might not have return sections
    "SA01",  # Not all docstrings need a "See Also"
    "SA04",  # "See Also" section does not need descriptions
    "SS06",  # Not possible to make all summaries one line
    "YD01",  # Yields: No plan to enforce
}
