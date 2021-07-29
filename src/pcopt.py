import numpy as np
import scipy.optimize as opt
import copy
from timeit import default_timer as timer
import time
import ebsd_index
RADEG = 180.0/np.pi


def optfunction(PC_i,indexer,banddat):
  bandNorm = indexer.bandDetectPlan.radon2pole(banddat,PC=PC_i,vendor=indexer.vendor)
  npoints = banddat.shape[0]
  nave = 0.0
  fitave = 0.0
  for i in range(npoints):

    bandNorm1 = bandNorm[i,:,:]
    bDat1 = banddat[i,:]
    whgood = np.nonzero(bDat1['max'] > -1.0e6)[0]

    if whgood.size >= 3:
      bDat1 = bDat1[whgood]
      bandNorm1 = bandNorm1[whgood,:]

      avequat,fit,cm,bandmatch,nMatch,matchAttempts = indexer.phaseLib[0].tripvote(bandNorm1,goNumba=True)
      fitave += fit
      nave += 1.0


  #dat, start, end = indexer.index_pats(patsin=pats, PC=PC_i)
  fitave /= nave
  #print(fitave)
  return fitave

def optimize(pats, indexer, batch = False):
  ndim = pats.ndim
  if ndim == 2:
    patterns = np.expand_dims(pats,axis=0)
  else:
    patterns = pats


  banddat = indexer.bandDetectPlan.find_bands(pats)
  npoints = banddat.shape[0]
  PC0 = indexer.PC
  if batch == False:

    PCopt = opt.minimize(optfunction,PC0,args=(indexer,banddat),method='Nelder-Mead')
    PCoutRet = PCopt['x']
  else:
    PCoutRet = np.zeros((npoints, 3))
    for i in range(npoints):
      PCopt = opt.minimize(optfunction,PC0,args=(indexer,banddat[i,:,:]),method='Nelder-Mead')
      PCoutRet[i,:] = PCopt['x']
  return PCoutRet


def file_opt(fobj, indexer):

  stat = fobj.read_header()
  nCols = fobj.nCols
  nRows = fobj.nRows
  stride=20
  pcopt = np.zeros((int(nRows/stride),int(nCols/stride),3), dtype=np.float32)

  for i in range(int(nRows/stride)):
    ii = i*stride
    print(ii)
    for j in range(int(nCols/stride)):
      jj = j*stride

      pats = fobj.read_data(returnArrayOnly=True, convertToFloat=True, patStartEnd=[ii*nCols+jj,ii*nCols+jj+1])

      pc = optimize(pats, indexer)
      #print(pc, pc.shape)
      pcopt[int(i),int(j),:] = pc


  return pcopt








