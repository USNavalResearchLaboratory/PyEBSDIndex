



from timeit import default_timer as timer
import numpy as np
import pyopencl as cl

from pyebsdindex import nlpar
from pyebsdindex.opencl import openclparam
import scipy
class NLPAR(nlpar.NLPAR):
  def __init__( self, **kwargs):
    nlpar.NLPAR.__init__(self, **kwargs)
    self.useCPU = False


  def calcsigmacl(self,chunksize=0,nn=1,saturation_protect=True,automask=True):
    pass

  def _calcchunks(patdim, ncol, nrow, target_bytes=4e9, col_overlap=0, row_overlap=0, col_offset=0, row_offset=0):

    col_overlap = min(col_overlap, ncol - 1)
    row_overlap = min(row_overlap, nrow - 1)

    byteperpat = patdim[-1] * patdim[-2] * 4 * 2  # assume a 4 byte float input and output array
    byteperdataset = byteperpat * ncol * nrow
    nchunks = int(np.ceil(byteperdataset / target_bytes))

    print(nchunks)

    ncolchunks = (max(np.round(np.sqrt(nchunks * float(ncol) / nrow)), 1))
    colstep = max((ncol / ncolchunks), 1)
    ncolchunks = max(ncol / colstep, 1)
    colstepov = min(colstep + 2 * col_overlap, ncol)
    ncolchunks = max(int(np.ceil(ncolchunks)), 1)
    colstep = max(int(np.round(colstep)), 1)
    colstepov = min(colstep + 2 * col_overlap, ncol)

    nrowchunks = max(np.round(nchunks / ncolchunks), 1)
    rowstep = max((nrow / nrowchunks), 1)
    nrowchunks = max(nrow / rowstep, 1)
    rowstepov = min(rowstep + 2 * row_overlap, nrow)
    nrowchunks = max(int(np.ceil(nrowchunks)), 1)
    rowstep = max(int(np.round(rowstep)), 1)
    rowstepov = min(rowstep + 2 * row_overlap, nrow)

    # colchunks = np.round(np.arange(ncolchunks+1)*ncol/ncolchunks).astype(int)
    colchunks = np.zeros((ncolchunks, 2), dtype=int)
    colchunks[:, 0] = (np.arange(ncolchunks) * colstep).astype(int)
    colchunks[:, 1] = colchunks[:, 0] + colstepov - int(col_overlap)
    colchunks[:, 0] -= col_overlap
    colchunks[0, 0] = 0;

    for i in range(ncolchunks - 1):
      if colchunks[i + 1, 0] >= ncol:
        colchunks = colchunks[0:i + 1, :]

    ncolchunks = colchunks.shape[0]
    colchunks[-1, 1] = ncol

    colchunks += col_offset

    colproc = np.zeros((ncolchunks, 2), dtype=int)
    if ncolchunks > 1:
      colproc[1:, 0] = col_overlap
    if ncolchunks > 1:
      colproc[0:, 1] = colchunks[:, 1] - colchunks[:, 0] - col_overlap
    colproc[-1, 1] = colchunks[-1, 1] - colchunks[-1, 0]

    # rowchunks = np.round(np.arange(nrowchunks + 1) * nrow / nrowchunks).astype(int)
    rowchunks = np.zeros((nrowchunks, 2), dtype=int)
    rowchunks[:, 0] = (np.arange(nrowchunks) * rowstep).astype(int)
    rowchunks[:, 1] = rowchunks[:, 0] + rowstepov - int(row_overlap)
    rowchunks[:, 0] -= row_overlap
    rowchunks[0, 0] = 0;

    for i in range(nrowchunks - 1):
      if rowchunks[i + 1, 0] >= nrow:
        rowchunks = rowchunks[0:i + 1, :]

    nrowchunks = rowchunks.shape[0]
    rowchunks[-1, 1] = nrow

    rowchunks += row_offset

    rowproc = np.zeros((nrowchunks, 2), dtype=int)
    if nrowchunks > 1:
      rowproc[1:, 0] = row_overlap
    if nrowchunks > 1:
      rowproc[0:, 1] = rowchunks[:, 1] - rowchunks[:, 0] - row_overlap
    rowproc[-1, 1] = rowchunks[-1, 1] - rowchunks[-1, 0]

    return ncolchunks, nrowchunks, colchunks, rowchunks, colproc, rowproc


