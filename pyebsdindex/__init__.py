__author__ = "Dave Rowenhorst"
__author_email__ = ""
# Initial committer first, then sorted by line contributions
__credits__ = [
    "Dave Rowenhorst",
    "Håkon Wiik Ånes",
]
__description__ = "Python based tool for Hough/Radon based EBSD indexing"
__name__ = "pyebsdindex"
__version__ = "0.1.1"


# Try to import only once
try:
    import pyopencl
    _pyopencl_installed = True
except ImportError:
    _pyopencl_installed = False

try:
    import ray
    _ray_installed = True
except ImportError:
    _ray_installed = False
