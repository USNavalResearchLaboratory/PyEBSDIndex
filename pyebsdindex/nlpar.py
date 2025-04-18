
# This software was developed by employees of the US Naval Research Laboratory (NRL), an
# agency of the Federal Government. Pursuant to title 17 section 105 of the United States
# Code, works of NRL employees are not subject to copyright protection, and this software
# is in the public domain. PyEBSDIndex is an experimental system. NRL assumes no
# responsibility whatsoever for its use by other parties, and makes no guarantees,
# expressed or implied, about its quality, reliability, or any other characteristic. We
# would appreciate acknowledgment if the software is used. To the extent that NRL may hold
# copyright in countries other than the United States, you are hereby granted the
# non-exclusive irrevocable and unconditional right to print, publish, prepare derivative
# works and distribute this software, in any medium, or authorize others to do so on your
# behalf, on a royalty-free basis throughout the world. You may improve, modify, and
# create derivative works of the software or any portion of the software, and you may copy
# and distribute such modifications or works. Modified works should carry a notice stating
# that you changed the software and should note the date and nature of any such change.
# Please explicitly acknowledge the US Naval Research Laboratory as the original source.
# This software can be redistributed and/or modified freely provided that any derivative
# works bear some notice that they are derived from it, and any modified versions bear
# some notice that they have been modified.
#
# Author: David Rowenhorst;
# The US Naval Research Laboratory Date: 22 May 2024

# For more info see
# Patrick T. Brewick, Stuart I. Wright, David J. Rowenhorst. Ultramicroscopy, 200:50–61, May 2019.




"""Non-local pattern averaging and re-indexing (NLPAR)."""


from pyebsdindex import _ray_installed
from pyebsdindex import _pyopencl_installed

gpuisthere = False
if _pyopencl_installed:
  # check for at least one gpu
  import pyopencl as cl
  try:
    plt = cl.get_platforms()
    if len(plt) > 0:
      for p in plt:
        g = p.get_devices(device_type=cl.device_type.GPU)
        if len(g) > 0:
          gpuisthere = True
          g = None
          break
    plt = None
  except:
    pass


if _ray_installed and gpuisthere:
  from pyebsdindex.opencl.nlpar_clray import NLPAR
elif gpuisthere and not _ray_installed:
  from pyebsdindex.opencl.nlpar_cl import NLPAR
else:
  from pyebsdindex.nlpar_cpu import NLPAR

__all__ = [
    "NLPAR",
]

class DIFF_NLPAR(NLPAR):
  def __init__(self, **kwargs):
    pass
