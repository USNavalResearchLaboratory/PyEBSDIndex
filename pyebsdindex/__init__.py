__author__ = "Dave Rowenhorst"
__author_email__ = ""
# Initial committer first, then sorted by line contributions
__credits__ = [
    "Dave Rowenhorst",
    "Håkon Wiik Ånes",
]
__description__ = "Python based tool for Radon based EBSD indexing"
__name__ = "pyebsdindex"
__version__ = "0.3.7"


# Try to import only once - also will perform check that at least one GPU is found.
_pyopencl_installed = False
try:
    import pyopencl as cl
    try:
        platform = cl.get_platforms()
        for p in platform:
            gpu = p.get_devices(device_type=cl.device_type.GPU)
            if len(gpu) > 0:
                _pyopencl_installed = True
    except:
        _pyopencl_installed = False

except ImportError:
    _pyopencl_installed = False

try:
    import ray
    _ray_installed = True
except ImportError:
    _ray_installed = False
