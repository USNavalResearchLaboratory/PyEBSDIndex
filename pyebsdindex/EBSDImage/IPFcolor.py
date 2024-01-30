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



import matplotlib.colors as pltcolors
import matplotlib.pyplot as plt
import numpy as np

from pyebsdindex import rotlib


def makeipf(ebsddata, indexer, vector=np.array([0,0,1.0]), xsize = None, ysize = None):
  nphase = len(indexer.phaseLib)

  npoints = ebsddata.shape[-1]
  ipfphase = np.zeros((nphase,npoints,3), dtype =np.float32)+1

  phcount = 0
  for ph in indexer.phaseLib:
    quat = ebsddata[phcount]['quat']
    if ph.lauecode == 43:
      ipfphase[phcount, :, :] = qu2ipf_cubic(quat, vector=vector)
    if ph.lauecode == 62:
      ipfphase[phcount, :, :] = qu2ipf_hex(quat, vector=vector)
    phcount += 1
  phase = ((ebsddata[-1]['phase']).copy()).clip(0).reshape(npoints,1)
  ipfout = np.choose(phase, ipfphase).squeeze()
  ipfout[ebsddata[-1]['fit'] > 179,:] = 0


  if xsize is not None:
    xsize = int(xsize)
    if ysize is None:
      ysize = int(npoints // xsize + np.int64((npoints % xsize) > 0))
      #print(ysize)
  else:
    xsize = int(npoints)
    ysize = 1

  npts = int(npoints)
  if int(xsize*ysize) < npoints:
    npts = int(xsize*ysize)
  ipf_out = ipfout[0:npts,:].reshape(ysize, xsize,3)
  return ipf_out



def qu2ipf_cubic(quats, vector=np.array([0,0,1.0])):
  xstalvect = rotlib.quat_vector(quats,vector)
  return ipf_color_cubic(xstalvect).clip(0.0, 1.0)

def ipf_color_cubic(xstalvect):
  shp = xstalvect.shape
  if len(shp) == 1:
    xstalv = np.copy(xstalvect.reshape(1,shp))
  else:
    xstalv = np.copy(xstalvect)
  npoints = shp[0]

  xstalv = np.abs(xstalv)
  xstalv = np.sort(xstalv, axis=1)
  # project to sterographic triangle
  xP = (2 * xstalv[:,1]) / (1 + xstalv[:,2])
  yP = (2 * xstalv[:,0]) / (1 + xstalv[:,2])

  # cubic unit tri center
  triPts = np.array( [[0,0],
                      [2. / np.sqrt(2.) / (1. + 1. / np.sqrt(2.)),0],
                      [2. / np.sqrt(3.) / (1. + 1. / np.sqrt(3.)),2. / np.sqrt(3.) / (1. + 1. / np.sqrt(3.))]], dtype = np.float32)

  middle = np.tan(1. / 2. * np.arctan(triPts[2,1] / triPts[2,0]))
  a = np.sqrt( (triPts[2,1] - triPts[1,1]) ** 2. + (triPts[2,0] - triPts[1,0]) ** 2.)
  b = np.sqrt((triPts[1,0]) ** 2. + (triPts[1,1]) ** 2.)
  c = np.sqrt(triPts[2,0] ** 2. + triPts[2,1] ** 2.)

  #y0 = 1/2. * np.sqrt( ((b+c-a)*(c+a-b)*(a+b-c)) / (a+b+c) )
  #x0 = y0 / middle
  y0 = np.mean(triPts[:, 1])
  x0 = np.mean(triPts[:, 0])

  S = np.sqrt((xP - x0) ** 2. + (yP - y0) ** 2.)
  H = np.arctan2((yP - y0) , (xP - x0)) *180.0/np.pi
  V = np.ones(npoints)

  #H =  (xP < x0).astype(np.float)*180.0+H
  H = H + 240.0 - np.arctan2((triPts[2,1] - y0) , (triPts[2,0] - x0)) * 180.0/np.pi
  #H = H - np.arctan2(-y0 , -x0) * 180.0 / np.pi
  sMax = np.sqrt(x0**2+y0**2)
  S = S / (sMax) * 0.8 + 0.2


  H = H % (360.0)
  H = H / 360.0

  RGB = pltcolors.hsv_to_rgb(np.array([H,S,V]).T)

  return RGB

def ipf_ledgend_cubic_old(size=512):
  szx = size
  aspect = 0.94
  szy = np.round(size*aspect).astype(int)
  triangleWX = np.round(size*0.86).astype(int)
  triangleWY = np.round(triangleWX * aspect).astype(int)

  triOrigin = np.round(np.array([0.1,0.1])*size).astype(int)

  triScale = 0.82842708/triangleWX
  np0 = triangleWX*triangleWY
  triXY = np.indices([triangleWY,triangleWX])
  triYX_stereo = (triXY*triScale).reshape(2,np0)
  xt = triYX_stereo[1,:]
  yt = triYX_stereo[0,:]
  xyz = np.zeros((np0, 3))
  xyz[:,2] = (4. - (xt**2.+ yt**2.))/(4. + (xt**2.+ yt**2.))
  xyz[:,0] = (xyz[:,2] + 1.) * (xt / 2.)
  xyz[:,1] = (xyz[:,2] + 1.) * (yt / 2.)
  pltest = (np.sqrt(2.0) * (-xyz[:,0]+xyz[:,2])) >= 0.0

  wh = np.nonzero( (xyz[:,1] <= xyz[:,0]) & pltest )[0]
  rgbaTri = np.full((np0, 4), 1.0, dtype = np.float32)
  rgbaTri[:,3] = 0
  rgbaTri[wh,3] = 1.0
  rgbaTri[wh,0:3] = ipf_color_cubic(xyz[wh,:])
  rgbaTri = rgbaTri.reshape(triangleWY,triangleWX,4)

  palette = np.full((szy,szx, 4), 1.0)
  trim = palette.shape[0] - triOrigin[0] - rgbaTri.shape[0]
  palette[triOrigin[1]:triOrigin[1]+triangleWY, triOrigin[0]:triOrigin[0]+triangleWX,:] = rgbaTri[:, :,:]

  dpi = size
  fsize = 6.0

  fig = plt.figure(figsize=(1.0,1.0),dpi=size/1.0)
  ax = plt.Axes(fig,[0.,0.,1.,aspect])
  ax.set_axis_off()
  fig.add_axes(ax)
  img = plt.imshow(palette, origin='lower', extent=[0,szx,0,szy])
  anno001 = plt.text(triOrigin[0] - 8*fsize*size/512,triOrigin[1] - 6*fsize*size/512.0, '001', fontsize = fsize)
  anno011 = plt.text(size - 14*fsize*size/512,triOrigin[1] - 6*fsize*size/512.0,'011',fontsize=fsize)
  anno111 = plt.text(size - 14*fsize*size/512,(triangleWY+triOrigin[1])*0.95,'111',fontsize=fsize)
  #ax = img.gca()
  fig.savefig("IPFCubic.pdf",bbox_inches='tight')
  #plt.show()
  #img.axes.get_xaxis().set_visible(False)
  #img.axes.get_yaxis().set_visible(False)

  #plt.savefig("test.png",bbox_inches='tight')
  #plt.savefig("test.png",bbox_inches = 0)

def ipf_ledgend_cubic(size=512):
  szx = size
  aspect = 0.89
  szy = np.round(size*aspect).astype(int)
  triangleWX = np.round(size*1.0).astype(int)
  triangleWY = np.round(triangleWX * aspect).astype(int)

  #triOrigin = np.round(np.array([0.1,0.1])*size).astype(int)
  triOrigin = np.array([0,0]).astype(int)

  triScale = 0.82842708/triangleWX
  np0 = triangleWX*triangleWY
  triXY = np.indices([triangleWY,triangleWX])
  triYX_stereo = (triXY*triScale).reshape(2,np0)
  xt = triYX_stereo[1,:]
  yt = triYX_stereo[0,:]
  xyz = np.zeros((np0, 3))
  xyz[:,2] = (4. - (xt**2.+ yt**2.))/(4. + (xt**2.+ yt**2.))
  xyz[:,0] = (xyz[:,2] + 1.) * (xt / 2.)
  xyz[:,1] = (xyz[:,2] + 1.) * (yt / 2.)
  pltest = (np.sqrt(2.0) * (-xyz[:,0]+xyz[:,2])) >= 0.0

  wh = np.nonzero( (xyz[:,1] <= xyz[:,0]) & pltest )[0]
  rgbaTri = np.full((np0, 4), 1.0, dtype = np.float32)
  rgbaTri[:,3] = 0.0
  rgbaTri[wh,3] = 1.0
  rgbaTri[wh,0:3] = ipf_color_cubic(xyz[wh,:])
  rgbaTri = rgbaTri.reshape(triangleWY,triangleWX,4)
  dpi = size
  figsz = 1.0
  fsize = 5.0#/512*size

  fig = plt.figure(1001, figsize=(figsz,figsz),dpi=size/figsz*0.5)
  ax = plt.Axes(fig,[0.05,0.1,0.9,aspect*0.9])
  ax.set_axis_off()
  fig.add_axes(ax)

  img = plt.imshow(rgbaTri, origin='lower', extent=[0,szx,0,szy])
  anno001 = plt.text(triOrigin[0] - 5*fsize*size/512/figsz,triOrigin[1] - 7*fsize*size/512.0/figsz, '001', fontsize = fsize)
  anno011 = plt.text(size - 10*fsize*size/512/figsz,triOrigin[1] - 7*fsize*size/512.0/figsz,'011',fontsize=fsize)
  anno111 = plt.text(size - 10*fsize*size/512/figsz,(triangleWY+triOrigin[1])*1.0,'111',fontsize=fsize)
  fig.savefig("IPFCubic.pdf",bbox_inches=0, transparent=True)
  plt.close(1001)


def qu2ipf_hex(quats, vector=np.array([0,0,1.0])):
  xstalvect = rotlib.quat_vector(quats,vector)
  return ipf_color_hex(xstalvect).clip(0.0, 1.0)

def ipf_color_hex(xstalvect):
  shp = xstalvect.shape
  if len(shp) == 1:
    xstalv = np.copy(xstalvect.reshape(1,shp))
  else:
    xstalv = np.copy(xstalvect)
  npoints = shp[0]

  xstalv /= np.sqrt((xstalv ** 2).sum(-1))[..., np.newaxis]
  xstalv[xstalv[:, 2] < 0, :] *= -1

  theta = np.arctan2(xstalv[:,1], xstalv[:,0])
  wh = np.where(theta >= np.pi/3.)[0]
  q60 = rotlib.quatnorm(np.array([ np.cos(np.pi/6.0),0, 0, -0.50000000]))

  while wh.size > 0:
    xstalv[wh,:] = rotlib.quat_vector(q60,xstalv[wh,:] )
    theta = np.arctan2(xstalv[:, 1], xstalv[:, 0])
    wh = np.where(theta >= np.pi / 3.)[0]



  theta = np.arctan2(xstalv[:, 1], xstalv[:, 0])
  wh = np.where(theta < 0.0)[0]
  q60 = np.array([np.cos(np.pi / 6.0), 0, 0, 0.50000000])
  while wh.size > 0:
    xstalv[wh, :] = rotlib.quat_vector(q60, xstalv[wh, :])
    theta = np.arctan2(xstalv[:, 1], xstalv[:, 0])
    wh = np.where(theta < 0.0)[0]


  theta = np.arctan2(xstalv[:, 1], xstalv[:, 0])
  wh = np.where(theta >= np.pi / 6.)[0]
  if wh.size > 0:
    nx = -np.sin(np.pi / 6.)
    ny = np.cos(np.pi / 6.)
    const = 2. * (nx * xstalv[wh,0] + ny * xstalv[wh, 1])
    xstalv[wh, 0] -= const * nx
    xstalv[wh, 1] -= const * ny



  xP = (xstalv[:,0]) / (1 + xstalv[:,2])
  yP = (xstalv[:,1]) / (1 + xstalv[:,2])

  # cubic unit tri center
  triPts = np.array( [[0,0],
                      [1.0 ,0],
                      [np.sqrt(3.)/2.0, 0.5 ]], dtype = np.float32)

  middle = np.tan(1. / 2. * np.arctan(triPts[2,1] / triPts[2,0]))

  a = np.sqrt( (triPts[2,1] - triPts[1,1]) ** 2. + (triPts[2,0] - triPts[1,0]) ** 2.)
  b = np.sqrt((triPts[1,0]) ** 2. + (triPts[1,1]) ** 2.)
  c = np.sqrt(triPts[2,0] ** 2. + triPts[2,1] ** 2.)

  #y0 = 0.4 * np.sqrt( ((b+c-a)*(c+a-b)*(a+b-c)) / (a+b+c) )
  #x0 = y0 / middle
  y0 = np.mean(triPts[:,1])
  x0 = np.mean(triPts[:, 0])


  S = np.sqrt((xP - x0) ** 2. + (yP - y0) ** 2.)
  H = np.arctan2((yP - y0) , (xP - x0)) * 180.0 / np.pi
  V = np.ones(npoints)

  #H = (xP < x0).astype(np.float) * 180.0 + H
  H = H + 240 - np.arctan2((triPts[2, 1] - y0) , (triPts[2, 0] - x0)) * 180.0 / np.pi
  #H = H  - np.arctan2((- y0), ( - x0)) * 180.0 / np.pi
  sMax = np.sqrt((triPts[:,0] - x0) ** 2 + (triPts[:,1] - y0) ** 2).max()
  S = S / (sMax) * 0.75 + 0.25

  H = H % (360.0)
  H = H / 360.0

  RGB = pltcolors.hsv_to_rgb(np.array([H,S,V]).T)

  return RGB


def ipf_ledgend_hex(size=512):
  szx = size
  aspect = 0.6
  szy = np.round(size*aspect).astype(int)
  triangleWX = np.round(size*1.0).astype(int)
  triangleWY = np.round(triangleWX * aspect).astype(int)

  #triOrigin = np.round(np.array([0.1,0.1])*size).astype(int)
  triOrigin = np.array([0,0]).astype(int)

  triScale = 1.0/triangleWX #0.82842708/triangleWX
  np0 = triangleWX*triangleWY
  triXY = np.indices([triangleWY,triangleWX])
  triYX_stereo = (triXY*triScale).reshape(2,np0)
  xt = triYX_stereo[1,:]*2
  yt = triYX_stereo[0,:]*2

  xyz = np.zeros((np0, 3))

  xyz[:,2] = (4. - (xt**2+ yt**2))/(4. + (xt**2+ yt**2))
  xyz[:,0] = (xyz[:,2] + 1.) * (xt / 2.)
  xyz[:,1] = (xyz[:,2] + 1.) * (yt / 2.)
  pltest = np.sqrt(xt ** 2 + yt**2) < 2.0
  pltest2 = (xyz[:,2] >= 0.0).squeeze()

  theta  =  np.arctan2(xyz[:,1],xyz[:,0]) < np.pi/6
  theta = np.logical_and( theta, pltest)
  #theta = np.logical_and(theta, pltest2)

  wh = np.nonzero( theta)[0]
  #return xyz[wh,:]
  rgbaTri = np.full((np0, 4), 1.0, dtype = np.float32)
  rgbaTri[:,3] = 0.0
  rgbaTri[wh,3] = 1.0
  rgbaTri[wh,0:3] = ipf_color_hex(xyz[wh,:])

  rgbaTri = rgbaTri.reshape(triangleWY,triangleWX,4)
  dpi = size
  figsz = 1.0
  fsize = 4.0#/512*size

  fig = plt.figure(1001, figsize=(figsz,figsz*aspect),dpi=size/figsz*0.5)
  ax = plt.Axes(fig,[-0.2,0.15,1.4,aspect*1.4])
  ax.set_axis_off()
  fig.add_axes(ax)

  img = plt.imshow(rgbaTri, origin='lower', extent=[0,szx,0,szy])
  anno001 = plt.text(triOrigin[0] - 5*fsize*size/512/figsz,triOrigin[1] - 8*fsize*size/512.0/figsz, r'0001', fontsize = 0.9*fsize)
  anno011 = plt.text(size - 10*fsize*size/512/figsz,triOrigin[1] - 9.5*fsize*size/512.0/figsz,r'$2\bar{1}\bar{1}0$',fontsize=0.9*fsize)
  anno111 = plt.text(size - 25*fsize*size/512/figsz,(triangleWY+triOrigin[1])*0.85,r'$10\bar{1}0$',fontsize=0.9*fsize)
  fig.savefig("IPFHex.pdf",bbox_inches=0, transparent=True)
  plt.close(1001)