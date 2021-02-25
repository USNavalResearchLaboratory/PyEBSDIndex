import numpy as np
import matplotlib.colors as pltcolors
import matplotlib.pyplot as plt
import rotlib


def qu2ipf_cubic(quats, vector=np.array([0,0,1.0])):
  xstalvect = rotlib.quat_vector(quats,vector)
  return ipf_color_cubic(xstalvect)

def ipf_color_cubic(xstalvect):
  shp = xstalvect.shape
  if len(shp) == 1:
    xstalv = xstalvect.reshape(1,shp)
  else:
    xstalv = xstalvect
  npoints = shp[0]

  xstalv = np.abs(xstalv)
  xstalv = np.sort(xstalv, axis=1)
  # project to sterographic triangle
  xP = (2 * xstalv[:,1]) / (1 + xstalv[:,2])
  yP = (2 * xstalv[:,0]) / (1 + xstalv[:,2])

  # cubic unit tri center
  triPts = np.array( [[0,0],
                      [2. / np.sqrt(2.) / (1. + 1. / np.sqrt(2.)),0],
                      [2. / np.sqrt(3.) / (1. + 1. / np.sqrt(3.)),2. / np.sqrt(3.) / (1. + 1. / np.sqrt(3.))]], dtype = np.float)

  middle = np.tan(1. / 2. * np.arctan(triPts[2,1] / triPts[2,0]))
  a = np.sqrt( (triPts[2,1] - triPts[1,1]) ** 2. + (triPts[2,0] - triPts[1,0]) ** 2.)
  b = np.sqrt((triPts[1,0]) ** 2. + (triPts[1,1]) ** 2.)
  c = np.sqrt(triPts[2,0] ** 2. + triPts[2,1] ** 2.)

  y0 = 1/2. * np.sqrt( ((b+c-a)*(c+a-b)*(a+b-c)) / (a+b+c) )
  x0 = y0 / middle

  S = np.sqrt((xP - x0) ** 2. + (yP - y0) ** 2.)
  H = np.arctan((yP - y0) / (xP - x0)) *180.0/np.pi
  V = np.ones(npoints)

  H =  (xP < x0).astype(np.float)*180.0+H
  H = H + 240.0 - np.arctan((triPts[2,1] - y0) / (triPts[2,0] - x0)) * 180.0/np.pi
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
  fig.savefig("IPFCubic.png",bbox_inches=0, transparent=True)
  plt.close(1001)
