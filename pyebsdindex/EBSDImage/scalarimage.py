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
The US Naval Research Laboratory Date: 21 Aug 2020'''


'''This package uses the Google Font Open Sans.  
Copyright 2020 The Open Sans Project Authors (https://github.com/googlefonts/opensans)
This Font Software is licensed under the SIL Open Font License, Version 1.1 . 
This license available with a FAQ at: https://openfontlicense.org
SIL OPEN FONT LICENSE Version 1.1 - 26 February 2007
'''

from PIL import  Image, ImageDraw, ImageFont
import os
import matplotlib.colors as pltcolors
import matplotlib.pyplot as plt
#from matplotlib.font_manager import findfont, FontProperties
#FONT = findfont(FontProperties(family='sans-serif', weight='bold'), fontext='ttf', )

FONT = os.path.join(os.path.dirname(__file__), 'OpenSans-Bold.ttf')


import numpy as np

def scalarimage(ebsddata, indexer, datafield='pq', xsize = None, ysize = None, addscalebar=False, cmap='viridis', datafieldindex=0):
  npoints = ebsddata.shape[-1]
  imagedata = ebsddata[-1][datafield]
  if len(imagedata.shape) > 1:
    imagedata = imagedata[:,datafieldindex]

  imagedata = imagedata.astype(np.float32)
  if datafield == 'fit':
    mn = imagedata[imagedata < 179].mean()
    std = imagedata[imagedata < 179].std()
    norm = plt.Normalize(vmin=max(0.0, mn-3*std), vmax=mn+3*std)
  else:
    norm = plt.Normalize()
  imagedata = np.array(norm(imagedata))
  #imagedata -= imagedata.min()
  #imagedata *= 1.0/imagedata.max()

  cm = plt.colormaps[cmap]
  imagedata = cm(imagedata)

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

  image_out = np.zeros((ysize, xsize, 3), dtype=np.float32)
  image_out = image_out.flatten()
  npts = min(int(npoints), int(xsize * ysize))
  # if int(xsize*ysize) < npoints:
  #   npts = int(xsize*ysize)
  image_out[0:npts * 3] = imagedata[0:npts, 0:3].flatten()
  image_out = image_out.reshape(ysize, xsize, 3)
  if addscalebar == True:
    image_out = add_scalebar(image_out, indexer.fID.xStep, rescale=False)
  return image_out

def add_scalebar(image, stepsize, rescale=True):
  # image: grayscale or color image to add scale bar to.
  # stepsize: size of a pixel in microns.
  imshape = image.shape
  channels = 1
  if len(imshape) > 2:
    channels = imshape[-1]

  scale_bar_size, scale_bar_width_px, units = _round_scalebar(imshape[1], stepsize, imfract=0.33)
  #print(scale_bar_size, scale_bar_width_px, units)
  #scale_bar_height_px = np.int64(scale_bar_width_px / (16.18/2) ) # use golden ratio.
  #underbar_size = (scale_bar_height_px*3, imshape[1])
  underbar_size = (int(imshape[1] * 0.04) , imshape[1])
  scale_bar_height_px = int(underbar_size[0] * 0.5)


  underbar = np.zeros(underbar_size, dtype=np.uint8)+255
  #yxoffset = (int(0.25*underbar_size[0]), int(0.01*imshape[1]))
  yxoffset = (int(0.25 * underbar_size[0]), 0)
  # add our scale bar.
  underbar[yxoffset[0]:yxoffset[0]+scale_bar_height_px,
              yxoffset[1]:yxoffset[1]+scale_bar_width_px]  = 0

  underbarim = Image.fromarray(underbar, mode='L')
  draw = ImageDraw.Draw(underbarim)
  fontsize = scale_bar_height_px * 1.4

  imfont = ImageFont.truetype(FONT, fontsize)
  #imfont = ImageFont.truetype(FONT, fontsize)
  imtext = ' ' + str(scale_bar_size) + ' ' + units
  text_color = 0
  text_length = draw.textlength(imtext, imfont)
  txoffset = (yxoffset[1]+scale_bar_width_px, yxoffset[0]-fontsize*0.35)
  draw.text(txoffset, imtext, fill=text_color, font=imfont)

  underbar = np.array(underbarim, dtype=np.float32)
  underbar -= underbar.min()
  underbar *= 1.0/underbar.max()



  newshp = (imshape[0]+underbar_size[0], imshape[1], channels)
  scalebarim = np.zeros(newshp, dtype=np.float32)

  rescaleim = image.astype(np.float32)
  if rescale == True:
    rescaleim -= rescaleim.min()
    rescaleim *= (1.0 / (rescaleim.max()))*0.999


  rescaleim = rescaleim.reshape((imshape[0], imshape[1], channels))
  scalebarim[0:imshape[0], 0:imshape[1], :] = rescaleim
  for i in range(channels):
    scalebarim[imshape[0]:, 0:imshape[1], i] = underbar
  scalebarim = np.squeeze(scalebarim)
  return scalebarim

def _round_scalebar(image_width, pixel_width, imfract=0.333):
    # image width is the width of the image in pixels (steps) in microns.
    # pixel_width is the width of one pixel (aka step size).
    # imfract is the maximum fraction of the image width that is desired for the scale bar size.
    units = 'μm'
    # these are the acceptable scale bar sizes
    sequence = np.array([1,2,5,10,20,25,50,100,200,250,500], dtype=np.int64)
    max_scale_bar_size = image_width*pixel_width*imfract
    if max_scale_bar_size < 1:
      units = 'nm'
      max_scale_bar_size *= 1000.0
      pixel_width *= 1000.0
    if max_scale_bar_size > 1000:
      units = 'mm'
      max_scale_bar_size *= 0.001
      pixel_width *= 0.001
    scale_bar_size = sequence[sequence < max_scale_bar_size].max()
    scale_bar_size_px = np.int64(np.float32(scale_bar_size/np.float32(pixel_width)))
    return scale_bar_size, scale_bar_size_px, units