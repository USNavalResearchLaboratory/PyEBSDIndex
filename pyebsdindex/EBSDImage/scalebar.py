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


'''This package uses the Google Font Open Sans.  
Copyright 2020 The Open Sans Project Authors (https://github.com/googlefonts/opensans)
This Font Software is licensed under the SIL Open Font License, Version 1.1 . 
This license available with a FAQ at: https://openfontlicense.org
SIL OPEN FONT LICENSE Version 1.1 - 26 February 2007
'''



from PIL import  Image, ImageDraw, ImageFont
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as scipyndim
from matplotlib.font_manager import findfont, FontProperties
#FONT = findfont(FontProperties(family='sans-serif', weight='bold'), fontext='ttf', )
#FONT = findfont(FontProperties(family='Dejavu Sans', style='normal', weight='bold'), fontext='ttf', )

FONT = os.path.join(os.path.dirname(__file__), 'OpenSans-Bold.ttf')

def addscalebar(image,
                stepsize=1.0,
                addminor = False,
                rescale=True,
                upscale_xsize=None,
                **kwargs):
  """Automatically add a scalebar to the bottom of a micrograph
  when given the dimension of a pixel within the image in microns. It will automatically
  choose the appropriate length of the scale bar, and will autoscale the units from
  nm, μm, or mm depending on the image width and provided ``stepsize``.
  The microbar is burned-in within the rasterized array, or in otherwords, it is not
  a vectorized shapes/text.

  Parameters
      ----------
      image : numpy.ndarray
          numpy array, of shape
          ``(n image rows, n image columns)``,
          or ``(n image rows, n image columns, n image channels)``.
          If no channel dimension is given, it is assumed that the image
          is a gray-scale image.
      stepsize : float
          The width of one pixel in microns.  Default is ``1.0``.
      addminor: bool, optional [``False``]
          Set equal to ``True`` if you want to automatically add minor tick lines
          to the scale bar. Default is ``False``
      rescale: bool, optional [``False``]
          Set to ``True`` to scale the output image values between
          ``[0,1.0]``.  If set to ``False`` the background of the scale
          bar area will be set to the maximum data value, and the bar/text
          will be set to the minumum value.
      upscale_xsize: int, optional
          Set this to be the output array number of columns in pixels.  This can be useful
          for small images, with < 500 pixels across the image.  In these cases the text
          for the scalebar can become pixelated. The image will be interpolated
          (using bi-cubic interpolation) to the entered xsize (keeping the aspect ratio
          the same).  We're not going to tell you how to live your life
          but this will only use nearest-neighbor interpolation for the resizing,
           thus even muliples of the original image size (2x, 3x, ...) is ideal for up-sizing.
          Note: setting ``upscale_xsize`` to a value that is smaller
          than the input image number of columns will provide a down-sized image,
          and obviously data points will be removed.

  Returns
  -------
  numpy.ndarray
      A copy of the numpy image-like array where the scalebar and notation is appended to
      the bottom of the image.  The output image will have dimensions,
       ``(round(1.04*image.shape[0], image.shape[1], {image.shape[2]})``, depending on if
       the original image had multiple channels.

  Example:
  -------
  Make a random micrograph that has a stepsize of 0.15 microns, add a scale bar,
  and write it out to a png and show it.
  >>> import os
  >>> import numpy as np
  >>> import matplotlib.pyplot as plt
  >>> from pyebsdindex.EBSDImage import scalebar
  >>> a = np.random.random((800, 1024))
  >>> abar = scalebar.addscalebar(a, 0.15)
  >>> plt.imsave(os.path.expanduser('~/random_image.png'), abar, cmap='gray')
  >>> plt.imshow(abar)

  Notes:
  -------
  If the input image has four channels, with the notion that the channels
  represnet ``[R,G,B,A]`` where ``A`` is the alpha channel, the current behavior of
  ```addscalebar``` will output the bar/text as ``[0.0,0.0,0.0,0.0]`` (black, transparent)
  and the background as ``[1.0,1.0,1.0,1.0]`` (white, opaque).  Currently, this is seen as a
  feature and not a bug.  The behavior can easily be altered as in this example:
  >>> import os
  >>> import numpy as np
  >>> import matplotlib.pyplot as plt
  >>> from pyebsdindex.EBSDImage import scalebar
  >>> a = np.random.random((800, 1024, 4))
  >>> abar = scalebar.addscalebar(a, 0.15)
  >>> abar[a.shape[0]:, :, 3] = 1.0  # if ``a`` is [R,G,B,A] ubyte8, replace with 255
  >>> plt.imsave(os.path.expanduser('~/random_image.png'), abar, cmap='gray')
  >>> plt.imshow(abar)
  """

  imshape = image.shape
  channels = 1
  if len(imshape) > 2:
    channels = imshape[-1]

  rescaleim = image.astype(np.float32)
  if rescale == True:
    rescaleim -= rescaleim.min()
    rescaleim *= (1.0 / (rescaleim.max())) * 0.999

  rescaleim = rescaleim.reshape((imshape[0], imshape[1], channels))
  stepadjust = 1.0
  if upscale_xsize is not None:
    upscale_xsize = np.int64(upscale_xsize)
    #aspect = np.float32(imshape[1]) / np.float32(imshape[0])
    #upscale_ysize = np.int64(upscale_xsize / aspect)
    zoomfact = upscale_xsize / np.float32(imshape[1])
    rescaleim = scipyndim.zoom(rescaleim, (zoomfact, zoomfact, 1),
                               order=0, grid_mode=True, mode='mirror')
    stepadjust = 1.0 / zoomfact

  imshape = rescaleim.shape

  scale_bar_size, scale_bar_width_px, units = _round_scalebar(imshape[1], stepsize*stepadjust)#, imfract=0.33)
  #print(scale_bar_size, scale_bar_width_px, units)
  #scale_bar_height_px = np.int64(scale_bar_width_px / (16.18/2) ) # use golden ratio.
  #underbar_size = (scale_bar_height_px*3, imshape[1])
  underbar_size = (max(int(np.round(imshape[1] * 0.04)), 10) , imshape[1])
  scale_bar_height_px = int(underbar_size[0] * 0.5)


  underbar = np.zeros(underbar_size, dtype=np.uint8)+255
  # set the scalebar to start slightly off of the edge of the image.
  bump = max(1, int(np.floor(0.001* underbar_size[1])))
  yxoffset = (int(0.25 * underbar_size[0]), bump)
  # add our scale bar.
  underbar[yxoffset[0]:yxoffset[0]+scale_bar_height_px,
              yxoffset[1]:yxoffset[1]+scale_bar_width_px]  = 0

  if addminor == True:
    minorgray = 64
    minorw = int(np.floor(0.002*scale_bar_width_px))

    underbar[yxoffset[0]:yxoffset[0] + scale_bar_height_px, yxoffset[1]: yxoffset[1]+minorw] = minorgray
    underbar[yxoffset[0]:yxoffset[0] + scale_bar_height_px,
          yxoffset[1]+scale_bar_width_px-minorw:yxoffset[1]+scale_bar_width_px] = minorgray

    underbar[yxoffset[0]:yxoffset[0] + minorw,
        yxoffset[1]:yxoffset[1] + scale_bar_width_px] = minorgray

    underbar[yxoffset[0] + scale_bar_height_px - minorw: yxoffset[0] + scale_bar_height_px,
        yxoffset[1]:yxoffset[1] + scale_bar_width_px] = minorgray

    scalefact = 1 - np.int64(np.floor(np.log10(scale_bar_size)))
    if ((scale_bar_size*10.0**scalefact) % 4) == 0:
      minorlines = np.int64(scale_bar_width_px*np.array([ 0.25, 0.5, 0.75, ]))
    else:
      minorlines = np.int64(scale_bar_width_px*np.array([ 0.2, 0.4, 0.6, 0.8 ]))
    for l in minorlines:
      underbar[yxoffset[0]:yxoffset[0] + scale_bar_height_px, l-minorw:l+minorw+1] = minorgray
      #print(scale_bar_width_px, minorw)

  underbarim = Image.fromarray(underbar, mode='L')
  draw = ImageDraw.Draw(underbarim)
  fontsize = scale_bar_height_px * 1.4 #Open sans
  #fontsize = scale_bar_height_px * 1.32 #dejavu sans

  imfont = ImageFont.truetype(FONT, fontsize)
  #imfont = ImageFont.truetype(FONT, fontsize)
  imtext = ' ' + str(scale_bar_size) + ' ' + units
  text_color = 0
  text_length = draw.textlength(imtext, imfont)
  txoffset = (yxoffset[1]+scale_bar_width_px, yxoffset[0]-fontsize*0.35) #Open sans
  #txoffset = (yxoffset[1]+scale_bar_width_px, yxoffset[0]-fontsize*0.19) #dejavu sans
  draw.text(txoffset, imtext, fill=text_color, font=imfont)

  underbar = np.array(underbarim, dtype=np.float32)
  underbar -= underbar.min()
  underbar *= 1.0/underbar.max()
  underbar *= rescaleim.max()-rescaleim.min()
  underbar += rescaleim.min()


  newshp = (imshape[0]+underbar_size[0], imshape[1], channels)
  scalebarim = np.zeros(newshp, dtype=np.float32)


  scalebarim[0:imshape[0], 0:imshape[1], :] = rescaleim
  for i in range(channels):
    scalebarim[imshape[0]:, 0:imshape[1], i] = underbar
  scalebarim = np.squeeze(scalebarim)
  return scalebarim

def _round_scalebar(image_width, pixel_width, imfract=0.333):
  """ Internal function that when given the number of columns in an image, and the
  width of a pixel in microns, will return the appropriate scalebar for the image from
  1 nm -- 500 mm.


  Parameters
    ----------
  image_width: int
      The width (number of columns) of the image in pixels (steps).
  pixel_width: float
   The width of one pixel (aka step size) in microns.
  imfract: float, optional [0.333]
   The maximum fraction of the image width that is allowed for the scale bar size.
   Default value is 0.333 of the image width.

  Returns
  -------
  (np.int64, np.int64, str)
      scale_bar_size: size of the scale bar in the ``units`` provided.
      scale_bar_size_px: size of the scale bar on the image in pixels.
      units: the units for the scale bar: {nm, μm, mm}.
  """

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