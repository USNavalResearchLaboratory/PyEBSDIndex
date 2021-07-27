import numpy as np
import scipy.optimize as opt
import copy
from timeit import default_timer as timer
import time
import ebsd_index
RADEG = 180.0/np.pi


def optfunction(PC_i,indexer,pats):
  dat, start, end = indexer.index_pats(patsin=pats, PC=PC_i)
  return dat['fit'][0]

def optimzie(pats, indexer):
  ndim = pats.ndim
  if ndim == 2:
    patterns = np.expand_dims(pats,axis=0)
  else:
    patterns = pats

  shape = patterns.shape
  nPats = shape[0]

  PCopts = np.zeros((nPats,3), dtype = np.float32)
  clops = indexer.bandDetectPlan.CLOps
  indexer.bandDetectPlan.CLOps = [False, False, False, False]
  for i in range(nPats):
    PC0 = indexer.PC
    PCopt = opt.minimize(optfunction, PC0, args =(indexer,patterns), method = 'Nelder-Mead'  )
    PCopts[i,:] = PCopt['x']

  indexer.bandDetectPlan.CLOps = clops
  return PCopts









