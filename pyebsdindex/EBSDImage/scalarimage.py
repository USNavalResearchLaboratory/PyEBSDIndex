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

# Author: David Rowenhorst;
# The US Naval Research Laboratory Date: 29 Oct 2024

"""Generate a 2D map from a scalar field in the indexed EBSD data."""

import os

import matplotlib.pyplot as plt
import scipy.ndimage as scipyndim


import numpy as np
from pyebsdindex.EBSDImage import micronbar

def scalarimage(ebsddata, indexer,
                datafield='pq',
                ncols = None,
                nrows = None,
                addmicronbar=False,
                cmap='viridis',
                norescalegray=False,
                rescalenice = False,
                datafieldindex=0,
                gamma=1.0,
                xsize=None,  # these are kept for backwards compatibility.
                ysize=None,
                **kwargs):
  """Generate a 2D map from a scalar field in the indexed EBSD data.

  Parameters
  ----------
  ebsddata : numpy.ndarray
      Structured array of EBSD indexing results returned by ``EBSDIndexer.index_pats``.
  indexer : pyebsdindex.ebsd_index.EBSDIndexer
      The indexer instance containing scan dimensions and step size in ``indexer.fID``.
  datafield : str, optional
      Scalar field name in ``ebsddata`` to plot (e.g., ``'pq'``, ``'iq'``, ``'fit'``,
      ``'fitinv'``, ``'phase'``, ``'nmatch'``). Default is ``'pq'``.
  ncols : int, optional
      Number of columns in the scan grid. If ``None``, read from ``indexer.fID.nCols``.
  nrows : int, optional
      Number of rows in the scan grid. If ``None``, calculated from total points and ``ncols``.
  addmicronbar : bool, optional
      If ``True``, appends a calibrated micron scale bar to the bottom of the image using
      ``indexer.fID.xStep`` via :func:`~pyebsdindex.EBSDImage.micronbar.addmicronbar`.
      Default is ``False``.
  cmap : str, optional
      Matplotlib colormap name. Default is ``'viridis'``.
  norescalegray : bool, optional
      If ``True`` when ``cmap='gray'`` or ``addmicronbar=False``, skips normalization.
      Default is ``False``.
  rescalenice : bool, optional
      If ``True``, normalizes intensity limits to ``[mean - 4*std, mean + 3*std]``.
      Default is ``False``.
  datafieldindex : int, optional
      Column index if the target ``datafield`` is 2D. Default is ``0``.
  gamma : float, optional
      Gamma exponent applied to the output image values. Default is ``1.0``.
  **kwargs
      Additional keyword arguments passed to
      :func:`~pyebsdindex.EBSDImage.micronbar.addmicronbar` (such as ``addminor=True``
      or ``zoom_xsize``).

  Returns
  -------
  numpy.ndarray
      2D grayscale or 3D RGB float32 array representing the scalar map.
  """


  # kept around for backwards compatability.
  if xsize is not None:
    ncols=xsize
  if ysize is not None:
    nrows = ysize

  npoints = ebsddata.shape[-1]
  if datafield != 'fitinv':
    imagedata = ebsddata[-1][datafield]
  else:
    imagedata = ebsddata[-1]['fit']

  if len(imagedata.shape) > 1:
    imagedata = imagedata[:,datafieldindex]

  imagedata = imagedata.astype(np.float32)

  if ncols is not None:
    ncols = int(ncols)

  else:
    ncols = indexer.fID.nCols

  if nrows is not None:
    nrows = int(nrows)
  else:
    nrows = int(npoints // ncols + np.int64((npoints % ncols) > 0))


  if datafield == 'fit':
    mn = imagedata[imagedata < 179].mean()
    std = imagedata[imagedata < 179].std()
    norm = plt.Normalize(vmin=max(0.0, mn-3*std), vmax=mn+3*std)
  elif datafield == 'fitinv':
    mn = imagedata[imagedata < 179].mean()
    std = imagedata[imagedata < 179].std()
    imagedata *= -1
    norm = plt.Normalize(vmin= (-mn-4*std), vmax=min(0.0, -mn+3*std))
  elif datafield == 'phase':
    norm = plt.Normalize(vmin=-1)
  elif rescalenice == True:
    mn = imagedata.mean()
    std = imagedata.std()
    norm = plt.Normalize(vmin= (mn - 4 * std), vmax= (mn + 3 * std))
  else:
    norm = plt.Normalize()

  if (cmap == 'gray' and norescalegray == True) or (addmicronbar == False and norescalegray == True):
    imagedata = np.array(imagedata)
  else:
    imagedata = np.array(norm(imagedata))
    cm = plt.colormaps[cmap]
    imagedata = cm(imagedata)




  if len(imagedata.shape) > 1:
    image_out = np.zeros((nrows, ncols, 3), dtype=np.float32)
    image_out = image_out.flatten()
    npts = min(int(npoints), int(ncols * nrows))

    image_out[0:npts * 3] = imagedata[0:npts, 0:3].flatten()
    image_out = image_out.reshape(nrows, ncols, 3)
    # perform desired image resize
  else:
    image_out = np.zeros((nrows, ncols), dtype=np.float32)
    image_out = image_out.flatten()
    npts = min(int(npoints), int(ncols * nrows))
    image_out[0:npts] = imagedata[0:npts].flatten()
    image_out = image_out.reshape(nrows, ncols)

  image_out = image_out**gamma

  if addmicronbar == True:
    image_out = micronbar.addmicronbar(image_out, indexer.fID.xStep, rescale=False, **kwargs)
  return image_out

