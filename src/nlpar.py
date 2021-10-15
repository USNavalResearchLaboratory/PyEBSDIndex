import numpy as np
import numba
import scipy.optimize as opt
import matplotlib.pyplot as plt
from pathlib import Path
from os import environ
from timeit import default_timer as timer


import ebsd_pattern
#environ["NUMBA_CACHE_DIR"] = str(tempdir)

class NLPAR():
  def __init__(self, filename=None, hdfdatapath=None):
    self.lam = 0.7
    self.searchradius = 3
    self.dthresh = 0.0
    self.filepath = None
    self.hdfdatapath = None
    self.filepathout = None
    self.hdfdatapathout = None
    self.patternfile = None
    self.patternfileout = None
    self.setfile(filename, hdfdatapath)
    self.mask = None
    self.sigma = None


  def setfile(self, filename=None,  hdfdatapath=None):
    if filename is not None:
      self.filepath = Path(filename)
      self.hdfdatapath = hdfdatapath
      self.getfileobj(True)

  def setoutfile(self, filename=None,  hdfdatapath=None):
    if filename is not None:
      self.filepathout = Path(filename)
      self.hdfdatapathout = hdfdatapath
      self.getfileobj(False)
    else:
      if self.patternfile is not None:
        if self.patternfile.file_type == 'UP':
          p = Path(self.filepath)
          appnd = "lam{:1.2f}".format(self.lam) + "sr{:d}".format(self.searchradius)\
                  + "dt{:1.1f}".format(self.dthresh)
          newfilepath = str(p.parent / Path(p.stem + appnd + p.suffix))
          self.patternfile.copy_file(newfilepath)
          self.filepathout = newfilepath
          self.getfileobj(False)
        if self.patternfile.file_type == 'HDF5':
          pass

  def getfileobj(self, inout=True):
      if inout == True:
        if self.filepath is not None:
          self.patternfile = ebsd_pattern.get_pattern_file_obj(self.filepath, hdfDataPath=self.hdfdatapath)
      else:
        if self.filepathout is not None:
          self.patternfileout = ebsd_pattern.get_pattern_file_obj(self.filepathout, hdfDataPath=self.hdfdatapathout)

  def opt_lambda(self,chunksize=0,saturation_protect=True,automask=True, backsub = False,
                 target_weights=[0.5, 0.375, 0.25], dthresh=0.0):

    target_weights = np.asarray(target_weights)

    def loptfunc(lam,d2,tw,dthresh):
      temp = (d2 > dthresh).choose(dthresh, d2)
      dw = np.exp(-(temp) / lam ** 2)
      w = np.sum(dw, axis=2) + 1e-12

      metric = np.mean(np.abs(tw - 1.0 / w))
      return metric


    @numba.njit(fastmath=True, cache=True,parallel=True)
    def d2norm(d2, n2, dij, sigma):
      shp = d2.shape
      s2 = sigma**2
      for j in numba.prange(shp[0]):
        for i in range(shp[1]):
          for q in range(shp[2]):
            if n2[j,i,q] > 0:
              ii = dij[j,i,q,1]
              jj = dij[j,i,q,0]
              s2_12 = (s2[j,i]+s2[jj,ii])
              d2[j,i,q] -= n2[j,i,q] * s2_12
              d2[j,i,q] /= s2_12*np.sqrt(2.0*n2[j,i,q])

    nrows = np.uint64(self.patternfile.nRows)
    ncols = np.uint64(self.patternfile.nCols)

    pwidth = np.uint64(self.patternfile.patternW)
    pheight = np.uint64(self.patternfile.patternH)
    phw = pheight * pwidth

    nn = 1
    if chunksize <= 0:
      chunksize = np.round(1e9 / phw / ncols)  # keeps the chunk at about 4GB
      print("Chunk size set to nrows:", int(chunksize))
    chunksize = np.int64(chunksize)

    nn = np.uint64(nn)

    if (automask is True) and (self.mask is None):
      self.mask = (self.automask(pheight,pwidth))
    if self.mask is None:
      self.mask = np.ones((pheight,pwidth),dytype=np.uint8)

    indices = np.asarray((self.mask.flatten().nonzero())[0],np.uint64)

    sigma = np.zeros((nrows,ncols),dtype=np.float32)+1e24
    colstartcount = np.asarray([0,ncols],dtype=np.int64)


    dthresh = np.float32(dthresh)
    lamopt_values = []
    for j in range(0,nrows,chunksize):
      #rowstartread = np.int64(max(0,j - nn))
      rowstartread = np.int64(j)
      rowend = min(j + chunksize + nn,nrows)
      rowcountread = np.int64(rowend - rowstartread)
      data = self.patternfile.read_data(patStartCount=[[0,rowstartread],[ncols,rowcountread]],
                                        convertToFloat=True,returnArrayOnly=True)

      shp = data.shape
      data = data.reshape(shp[0],phw)
      if backsub is True:
        back = np.mean(data, axis=0)
        back -= np.mean(back)
        data -= back

      rowstartcount = np.asarray([0,rowcountread],dtype=np.int64)
      sigchunk, (d2,n2, dij) = self.sigma_numba(data,nn,rowcountread,ncols,rowstartcount,colstartcount,indices,saturation_protect)
      tmp = (sigma[j:j + rowstartcount[1],:] < sigchunk).choose( sigchunk, sigma[j:j + rowstartcount[1],:])
      sigma[j:j + rowstartcount[1],:] = tmp


      d2norm(d2, n2, dij, sigchunk)
      lamopt_values_chnk = []
      for tw in target_weights:
        lam = 1.0
        lambopt1 = opt.minimize(loptfunc,lam,args=(d2,tw,dthresh),method='Nelder-Mead',
                                bounds = [[0.0, 10.0]],options={'fatol': 0.0001})
        lamopt_values_chnk.append(lambopt1['x'])


      lamopt_values.append(lamopt_values_chnk)
    lamopt_values = np.asarray(lamopt_values)
    print("Range of lambda values: ", np.mean(lamopt_values, axis = 0).flatten())
    print("Optimal Choice: ", np.mean(lamopt_values))
    if self.sigma is None:
      self.sigma = sigma

  def calcnlpar(self,chunksize=0,searchradius=None,lam = None, dthresh = None, saturation_protect=True,automask=True,
                filename=None, fileout=None, backsub = False,  hdfdatapath=None, hdfdatapathout=None):


    if filename is not None:
      self.setfile(filename=filename,  hdfdatapath=hdfdatapath)

    if fileout is not None:
      self.setoutfile(filename=fileout,hdfdatapath=hdfdatapathout)

    if self.patternfileout is None:
      self.setoutfile()

    nrows = np.int64(self.patternfile.nRows)
    ncols = np.int64(self.patternfile.nCols)

    pwidth = np.int64(self.patternfile.patternW)
    pheight = np.int64(self.patternfile.patternH)
    phw = pheight*pwidth

    if chunksize <= 0:
      chunksize = np.round(1e9/phw/ncols) # keeps the chunk at about 8GB
      print("Chunk size set to nrows:", int(chunksize))
    chunksize = np.int64(chunksize)

    if lam is not None:
      self.lam = lam

    if dthresh is not None:
      self.dthresh = dthresh

    if searchradius is not None:
      self.searchradius = searchradius

    lam = np.float32(self.lam)
    dthresh = np.float32(self.dthresh)
    sr = np.int64(self.searchradius)

    if (automask is True) and (self.mask is None):
      self.mask = (self.automask(pheight,pwidth))
    if self.mask is None:
      self.mask = np.ones((pheight,pwidth), dytype=np.uint8)

    indices = np.asarray( (self.mask.flatten().nonzero())[0], np.uint64)
    calcsigma = False
    if self.sigma is None:
      calcsigma = True
      self.sigma = np.zeros((nrows, ncols), dtype=np.float32)+1e24


    if np.asarray(self.sigma).size == 1:
      tmp = np.asarray(self.sigma)[0]
      self.sigma =  np.zeros((nrows, ncols), dtype=np.float32)+tmp
      calcsigma = False

    shpsigma = np.asarray(self.sigma).shape
    if (shpsigma[0] != nrows) and (shpsigma[1] != ncols):
      self.sigma = np.zeros((nrows,ncols),dtype=np.float32) + 1e24
      calcsigma = True


    sigma = np.asarray(self.sigma)


    colstartcount = np.asarray([0,ncols],dtype=np.int64)
    print(lam, sr, dthresh)
    for j in range(0,nrows,chunksize):
      rowstartread = np.int64(max(0, j-sr))
      rowend = min(j + chunksize+sr,nrows)
      rowcountread = np.int64(rowend-rowstartread)
      data = self.patternfile.read_data(patStartCount = [[0,rowstartread], [ncols,rowcountread]],
                                        convertToFloat=True,returnArrayOnly=True)

      shpdata = data.shape
      data = data.reshape(shpdata[0], phw)

      if backsub is True:
        back = np.mean(data,axis=0)
        back -= np.mean(back)
        data -= back

      rowstartcount = np.asarray([0,rowcountread],dtype=np.int64)
      if calcsigma is True:
        sigchunk, tmp = self.sigma_numba(data,1,rowcountread,ncols,rowstartcount,colstartcount,indices,saturation_protect)
        del tmp
        tmp = (sigma[rowstartread:rowend,:] < sigchunk).choose(sigchunk,sigma[rowstartread:rowend,:])
        sigma[rowstartread:rowend,:] = tmp
      else:
        sigchunk = sigma[rowstartread:rowend,:]


      dataout = self.nlpar_nb(data,lam, sr, dthresh, sigchunk,
                              rowcountread,ncols,indices,saturation_protect)

      dataout = dataout.reshape(rowcountread, ncols, phw)
      dataout = dataout[j-rowstartread:, :, : ]
      shpout = dataout.shape
      dataout = dataout.reshape(shpout[0]*shpout[1], pheight, pwidth)

      self.patternfileout.write_data(newpatterns=dataout,patStartCount = [[0,j], [ncols, shpout[0]]],
                                     flt2int='clip',scalevalue=1.0 )

      #return dataout
      #sigma[j:j+rowstartcount[1],:] += \
      #  sigchunk[rowstartcount[0]:rowstartcount[0]+rowstartcount[1],:]



  def calcsigma(self,chunksize=0,nn=1,saturation_protect=True,automask=True):

    nrows = np.uint64(self.patternfile.nRows)
    ncols = np.uint64(self.patternfile.nCols)

    pwidth = np.uint64(self.patternfile.patternW)
    pheight = np.uint64(self.patternfile.patternH)
    phw = pheight*pwidth

    if chunksize <= 0:
      chunksize = np.round(2e9/phw/ncols) # keeps the chunk at about 8GB
      print("Chunk size set to nrows:", int(chunksize))
    chunksize = np.int64(chunksize)


    nn = np.uint64(nn)

    if (automask is True) and (self.mask is None):
      self.mask = (self.automask(pheight,pwidth))
    if self.mask is None:
      self.mask = np.ones((pheight,pwidth), dytype=np.uint8)

    indices = np.asarray( (self.mask.flatten().nonzero())[0], np.uint64)

    sigma = np.zeros((nrows, ncols), dtype=np.float32)
    #d_nn = np.zeros((nrows, ncols, int((2*nn+1)**2)), dtype=np.float32)
    colstartcount = np.asarray([0,ncols],dtype=np.int64)
    dave = 0.0
    for j in range(0,nrows,chunksize):
      rowstartread = np.int64(max(0, j-nn))
      rowend = min(j + chunksize+nn,nrows)
      rowcountread = np.int64(rowend-rowstartread)
      data = self.patternfile.read_data(patStartCount = [[0,rowstartread], [ncols,rowcountread]],
                                        convertToFloat=True,returnArrayOnly=True)

      shp = data.shape
      data = data.reshape(data.shape[0], phw)

      #data = None
      if rowend == nrows:
        rowstartcount = np.asarray([j-rowstartread,rowcountread - (j-rowstartread) ], dtype=np.int64)
      else:
        rowstartcount = np.asarray([j-rowstartread,chunksize ], dtype=np.int64)
      dtic = timer()
      sigchunk, temp = self.sigma_numba(data,nn, rowcountread,ncols,rowstartcount,colstartcount,indices,saturation_protect)
      dave += (timer() - dtic)
      sigma[j:j+rowstartcount[1],:] += \
        sigchunk[rowstartcount[0]:rowstartcount[0]+rowstartcount[1],:]

    return sigma

  @staticmethod
  def automask( h,w ):
    r = (min(h,w)*0.98*0.5)
    x = np.arange(w, dtype=np.float32)
    x = np.minimum(x , (w-x))
    x = x.reshape(1,w)
    y = np.arange(h, dtype=np.float32)
    y = np.minimum(y , (h - y))
    y = y.reshape(h,1)

    mask =np.sqrt(y**2 + x**2)
    mask = (mask < r).astype(np.uint8)
    mask = np.roll(mask, (int(h/2), int(w/2) ), axis=(0,1))

    return mask

  @staticmethod
  @numba.jit(nopython=True,cache=True,fastmath=False,parallel=True)
  def sigma_numba(data, nn, nrows, ncols, rowstartcount, colstartcount, indices, saturation_protect=True):
    sigma = np.zeros((nrows,ncols), dtype = np.float32)
    dout = np.zeros((nrows,ncols,(nn*2+1)**2), dtype=np.float32)
    nout = np.zeros((nrows,ncols,(nn * 2 + 1) ** 2),dtype=np.float32)
    dij = np.zeros((nrows,ncols,(nn * 2 + 1) ** 2, 2),dtype=np.uint64)
    shpdata = data.shape

    n0 = np.float32(shpdata[-1])
    shpind = indices.shape
    mxval = np.max(data)
    if saturation_protect == False:
      mxval += 1.0
    else:
      mxval *= 0.9961

    for j in numba.prange(rowstartcount[0], rowstartcount[0]+rowstartcount[1]):
      nn_r_start = j - nn if (j - nn) >= 0 else 0
      nn_r_end = (j + nn if (j + nn) < nrows else nrows-1)+1
      for i in numba.prange(colstartcount[0], colstartcount[0]+colstartcount[1]):
        nn_c_start = i - nn if (i - nn) >= 0 else 0
        nn_c_end = (i + nn if (i + nn) < ncols else ncols - 1) + 1

        mind = np.float32(1e24)
        indx_0 = i+ncols*j
        count = 0
        for j_nn in range(nn_r_start,nn_r_end ):
          for i_nn in range(nn_c_start,nn_c_end):
            dij[j,i,count,0] = np.uint64(j_nn)
            dij[j,i,count,1] = np.uint64(i_nn) # want to save this for labmda optimization
            indx_nn = i_nn+ncols*j_nn
            d2 = np.float32(0.0)
            n2 = np.float32(0.0)
            nout[j,i,count] = n0 # want to save this for labmda optimization
            if not((i == i_nn) and (j == j_nn)):
              for q in range(shpind[0]):
                d0 = data[indx_0, indices[q]]
                d1 = data[indx_nn, indices[q]]
                if (d1 < mxval) and (d0 < mxval):
                  n2 += 1.0
                  d2 += (d0 - d1)**2
              nout[j,i,count] = n2
              s0 = d2 / np.float32(n2 * 2.0)
              if d2 >= 1.e-3: #sometimes EDAX collects the same pattern twice
                if s0 < mind:
                  mind = s0
            dout[j,i,count] = d2 # want to save this for labmda optimization

            count += 1

        sigma[j,i] = np.sqrt(mind)
    return sigma,( dout, nout, dij)

  @staticmethod
  @numba.jit(nopython=True,cache=True,fastmath=True,parallel=True)
  def nlpar_nb(data,lam, sr, dthresh, sigma, nrows,ncols,indices,saturation_protect=True):

    def getpairid(idx0, idx1):
      idx0_t = int(idx0)
      idx1_t = int(idx1)
      if idx0 < idx1:
        pairid = idx0_t + (idx1_t << 32)
      else:
        pairid = idx1_t + (idx0_t << 32)
      return numba.uint64(pairid)

    lam2 = 1.0 / lam**2
    dataout = np.zeros_like(data, np.float32)
    shpdata = data.shape
    shpind = indices.shape
    winsz = np.int32((2*sr+1)**2)


    mxval = np.max(data)

    if saturation_protect == False:
      mxval += 1.0
    else:
      mxval *= 1.0#0.9961

    for i in numba.prange(ncols):
      winstart_x = max((i - sr),0) - max((i + sr - (ncols - 1)),0)
      winend_x = min((i + sr),(ncols - 1)) + max((sr - i),0) + 1
      pairdict = numba.typed.Dict.empty(key_type=numba.core.types.uint64,value_type=numba.core.types.float32)
      for j in range(nrows):
        winstart_y = max((j - sr),0) - max((j + sr - (nrows - 1)),0)
        winend_y = min((j + sr),(nrows - 1)) + max((sr - j),0) + 1

        weights = np.zeros(winsz,dtype=np.float32)
        pindx = np.zeros(winsz,dtype=np.uint64)

        indx_0 = i + ncols * j
        sigma0 = sigma[j,i]**2

        counter = 0
        for j_nn in range(winstart_y,winend_y):
          for i_nn in range(winstart_x,winend_x):
            indx_nn = i_nn + ncols * j_nn
            pindx[counter] = indx_nn
            #print(indx_0, indx_nn)
            if indx_nn == indx_0:
              pass
            else:
              pairid = getpairid(indx_0, indx_nn)
              if pairid in pairdict:
                pass
              else:
                sigma1 = sigma[j_nn,i_nn]**2
                d2 = np.float32(0.0)
                n2 = np.float32(0.0)
                for q in range(shpind[0]):
                  d0 = data[indx_0,indices[q]]
                  d1 = data[indx_nn,indices[q]]
                  if (d1 < mxval) and (d0 < mxval):
                    n2 += 1.0
                    d2 += (d0 - d1) ** 2
                d2 -= n2*(sigma0+sigma1)
                dnorm = (sigma1 + sigma0)*np.sqrt(2.0*n2)
                if dnorm > 1.e-8:
                  d2 /= dnorm
                else:
                  d2 = 1e6*n2
                pairdict[pairid] = numba.float32(d2)
            counter += 1
        #print('________________')
        # end of window scanning
        for i_nn in range(winsz):
          indx_nn = pindx[i_nn]
          if indx_nn == indx_0:
            weights[i_nn] = 1.0
          else:
            pairid = getpairid(indx_0, indx_nn)

            #print(indx_0, indx_nn, pairid, pairid in pairdict)
            weights[i_nn] = pairdict[pairid]
            weights[i_nn] = np.maximum(weights[i_nn]-dthresh, 0.0)
            weights[i_nn] = np.exp(-1.0 * weights[i_nn] * lam2)

        sum = np.sum(weights)
        weights /= sum
        for i_nn in range(winsz):
          indx_nn = pindx[i_nn]
          for q in range(shpdata[1]):
            dataout[indx_0, q] += data[indx_nn, q]*weights[i_nn]



    return dataout


