'''This software was developed by employees of the US Naval Research Laboratory (NRL), an
agency of the Federal Government. Pursuant to title 17 section 105 of the United States
Code, works of NRL employees are not subject to copyright protection, and this software
is in the public domain. PyEBSDIndex is an experimental system. NRL assumes no
responsibility whatsoever for its use by other parties, and makes no guarantees,
expressed or implied, about its quality, reliability, or any other characteristic. We
would appreciate acknowledgment if the software is used. To the extent that NRL may hold
copyright in countries other than the United States, you are hereby granted the
non-exclusive irrevocable and unconditional right to print, publish, prepare derivative
works and distribute this software, in any medium, or authorize others to do so on your
behalf, on a royalty-free basis throughout the world. You may improve, modify, and
create derivative works of the software or any portion of the software, and you may copy
and distribute such modifications or works. Modified works should carry a notice stating
that you changed the software and should note the date and nature of any such change.
Please explicitly acknowledge the US Naval Research Laboratory as the original source.
This software can be redistributed and/or modified freely provided that any derivative
works bear some notice that they are derived from it, and any modified versions bear
some notice that they have been modified.

Author: David Rowenhorst;
The US Naval Research Laboratory Date: 29 Oct 2024'''


import os

import matplotlib.pyplot as plt
import scipy.ndimage as scipyndim
#from matplotlib.font_manager import findfont, FontProperties
#FONT = findfont(FontProperties(family='sans-serif', weight='bold'), fontext='ttf', )

FONT = os.path.join(os.path.dirname(__file__), 'OpenSans-Bold.ttf')

import numpy as np
from pyebsdindex.EBSDImage import scalebar

def scalarimage(ebsddata, indexer,
                datafield='pq',
                xsize = None,
                ysize = None,
                addscalebar=False,
                cmap='viridis',
                norescalegray=False,
                rescalenice = False,
                datafieldindex=0,
                **kwargs):
  npoints = ebsddata.shape[-1]
  if datafield != 'fitinv':
    imagedata = ebsddata[-1][datafield]
  else:
    imagedata = ebsddata[-1]['fit']

  if len(imagedata.shape) > 1:
    imagedata = imagedata[:,datafieldindex]

  imagedata = imagedata.astype(np.float32)

  if xsize is not None:
    xsize = int(xsize)
    # if ysize is None:
    # print(ysize)
  else:
    xsize = indexer.fID.nCols
    # xsize = int(npoints)
    # ysize = 1
  if ysize is not None:
    ysize = int(ysize)
  else:
    ysize = int(npoints // xsize + np.int64((npoints % xsize) > 0))


  if datafield == 'fit':
    mn = imagedata[imagedata < 179].mean()
    std = imagedata[imagedata < 179].std()
    norm = plt.Normalize(vmin=max(0.0, mn-3*std), vmax=mn+3*std)
  elif datafield == 'fitinv':
    mn = imagedata[imagedata < 179].mean()
    std = imagedata[imagedata < 179].std()
    imagedata *= -1
    norm = plt.Normalize(vmin= (-mn-3*std), vmax=min(0.0, -mn+3*std))
  elif datafield == 'phase':
    norm = plt.Normalize(vmin=-1)
  elif rescalenice == True:
    mn = imagedata.mean()
    std = imagedata.std()
    norm = plt.Normalize(vmin= (mn - 4 * std), vmax= (mn + 3 * std))
  else:
    norm = plt.Normalize()

  if (cmap == 'gray' and norescalegray == True) or (addscalebar==False and norescalegray == True):
    if datafield == 'fit':
      imagedata = np.array(imagedata).clip(max(0.0, mn-4*std),mn+4*std )
    imagedata = np.array(imagedata)
  else:
    imagedata = np.array(norm(imagedata))
    cm = plt.colormaps[cmap]
    imagedata = cm(imagedata)




  if len(imagedata.shape) > 1:
    image_out = np.zeros((ysize, xsize, 3), dtype=np.float32)
    image_out = image_out.flatten()
    npts = min(int(npoints), int(xsize * ysize))
    # if int(xsize*ysize) < npoints:
    #   npts = int(xsize*ysize)
    image_out[0:npts * 3] = imagedata[0:npts, 0:3].flatten()
    image_out = image_out.reshape(ysize, xsize, 3)
    # perform desired image resize
  else:
    image_out = np.zeros((ysize, xsize), dtype=np.float32)
    image_out = image_out.flatten()
    npts = min(int(npoints), int(xsize * ysize))
    # if int(xsize*ysize) < npoints:
    #   npts = int(xsize*ysize)
    image_out[0:npts] = imagedata[0:npts].flatten()
    image_out = image_out.reshape(ysize, xsize)

  if addscalebar == True:
    image_out = scalebar.addscalebar(image_out, indexer.fID.xStep, rescale=False, **kwargs)
  return image_out

