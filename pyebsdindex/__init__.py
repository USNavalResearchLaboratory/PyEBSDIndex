__author__ = "Dave Rowenhorst"
__author_email__ = ""
# Initial committer first, then sorted by line contributions
__credits__ = [
    "Dave Rowenhorst",
    "Håkon Wiik Ånes",
]
__description__ = "Python based tool for Radon based EBSD indexing"
__name__ = "pyebsdindex"
__version__ = "0.3.5"


# Try to import only once - also will perform check that at least one GPU is found.
_pyopencl_installed = False
try:
    import pyopencl as cl
    plt = cl.get_platforms()
    if len(plt) > 0:
        for p in plt:
            g = p.get_devices(device_type=cl.device_type.GPU)
            if len(g) > 0:
                _pyopencl_installed = True
                break
    #from pyebsdindex.opencl import openclparam
    #testcl = openclparam.OpenClParam()
    #try:
    #    gpu = testcl.get_gpu()
    #    if len(gpu) > 0:
    #        _pyopencl_installed = True
    #except:
    #    raise ImportError('pyopencl could not find GPU')
except ImportError:
    _pyopencl_installed = False

try:
    import ray
    _ray_installed = True
except ImportError:
    _ray_installed = False
