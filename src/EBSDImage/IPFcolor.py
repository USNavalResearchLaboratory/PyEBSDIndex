import numpy as np
import matplotlib as plt
import rotlib


def ipf_color_cubic(quats, vector=np.array([0,0,1.0])):
  shp = quats.shape
  npoints = shp[0]
  xstalvect = rotlib.quat_vector(quats, vector)
  xstalvect = np.abs(xstalvect)
  xstalvect = np.sort(xstalvect, axis=1)
  # project to sterographic triangle
  xP = (2 * xstalvect[:,1]) / (1 + xstalvect[:,2])
  yP = (2 * xstalvect[:,0]) / (1 + xstalvect[:,2])

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

  RGB = plt.colors.hsv_to_rgb(np.array([H,S,V]).T)

  return RGB
