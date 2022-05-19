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


from pathlib import Path
from timeit import default_timer as timer

import numba
import numpy as np
import scipy.optimize as opt

from pyebsdindex import ebsd_pattern

#from os import environ
#environ["NUMBA_CACHE_DIR"] = str(tempdir)


class NLPAR():
  def __init__(self, filename=None,  lam=0.7, searchradius=3,dthresh=0.0, nrows = None, ncols = None):
    self.lam = lam
    self.searchradius = searchradius
    self.dthresh = dthresh
    self.filepath = None
    self.hdfdatapath = None
    self.filepathout = None
    self.hdfdatapathout = None
    #self.patternfile = None
    #self.patternfileout = None
    self.setfile(filename)
    self.mask = None
    self.sigma = None
    self.nrows = None
    self.ncols = None
    if nrows is not None:
      self.nrows = nrows
    if ncols is not None:
      self.ncols = ncols


  def setfile(self,filepath=None):
    self.filepath = None
    self.hdfdatapath = None
    pathtemp = np.atleast_1d(filepath)
    fpath = pathtemp[0]
    hdf5path = None
    if pathtemp.size > 1:
      hdf5path = pathtemp[1]
    if fpath is not None:
      self.filepath = Path(fpath)
      self.hdfdatapath = hdf5path




  def setoutfile(self,patternfile, filepath=None):
    '''patternfile is an input pattern file object from ebsd_pattern.  Filepath is a string.
    In the future I want to be able to specify the HDF5 data path to store the output data, but that
    is proving to be a bit of a mess.  For now, a copy of the original HDF5 is made, and the NLPAR patterns will be
    overwritten on top of the originals. '''
    self.filepathout = None
    self.hdfdatapathout = None
    pathtemp = np.atleast_1d(filepath)
    fpath = pathtemp[0]
    hdf5path = None
    #if pathtemp.size > 1:
    #  hdf5path = pathtemp[1]
    #print(fpath, hdf5path)
    if fpath is not None: # the user has set an output file path.
      self.filepathout = Path(fpath).expanduser().resolve()
      self.hdfdatapathout =  hdf5path
      if patternfile.filetype != 'HDF5': #check that the input and output are not the same.
        pathok = self.filepathout.exists()
        if pathok:
          pathok = self.filepathout.samefile(patternfile.filepath)
        if pathok:
          raise ValueError('Error: File input and output are exactly the same.')
          return

        patternfile.copy_file([self.filepathout,self.hdfdatapathout] )
        return  # fpath and (maybe) hdf5 path were set manually.
      else: # this is a hdf5 file
        #if self.filepathout.exists():
        #  print([self.filepathout,self.hdfdatapathout])
        patternfile.copy_file([self.filepathout, self.hdfdatapathout])

    if patternfile is not None: # the user has set no path.
      hdf5path = None
      if patternfile.filetype == 'UP':
        p = Path(patternfile.filepath)
        appnd = "_NLPAR_l{:1.2f}".format(self.lam) + "sr{:d}".format(self.searchradius)
        newfilepath = str(p.parent / Path(p.stem + appnd + p.suffix))
        patternfile.copy_file(newfilepath)

      if patternfile.filetype == 'HDF5':
        hdf5path_tmp = str(patternfile.h5patdatpth).split('/')
        if hdf5path_tmp[0] == '':
          hdf5path_org =  hdf5path_tmp[1]
        else:
          hdf5path_org = hdf5path_tmp[0]
        p = Path(patternfile.filepath)
        appnd = "_NLPAR_l{:1.2f}".format(self.lam) + "sr{:d}".format(self.searchradius)
        hdf5path = hdf5path_org+appnd
        newfilepath = str(p.parent / Path(p.stem + appnd + p.suffix))
        #patternfile.copy_file([newfilepath, hdf5path_org], newh5path=hdf5path)
        patternfile.copy_file([newfilepath])
        hdf5path = patternfile.h5patdatpth

      self.filepathout = newfilepath
      self.hdfdatapathout = hdf5path
      return

  def getinfileobj(self, inout=True):
    if self.filepath is not None:
      fID = ebsd_pattern.get_pattern_file_obj([self.filepath, self.hdfdatapath])
      if self.nrows is None:
        self.nrows = fID.nRows
      else:
        fID.nRows = self.nrows
      if self.ncols is None:
        self.ncols = fID.nCols
      else:
        fID.nCols = self.ncols
      return fID
    else:
      return None

  def getoutfileobj(self, inout=True):
    if self.filepathout is not None:
      return ebsd_pattern.get_pattern_file_obj([self.filepathout, self.hdfdatapathout])
    else:
      return None

  def opt_lambda(self,chunksize=0,saturation_protect=True,automask=True, backsub = False,
                 target_weights=[0.5, 0.34, 0.25], dthresh=0.0, autoupdate=True):

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

    patternfile = self.getinfileobj()
    patternfile.read_header()
    nrows = np.uint64(self.nrows) #np.uint64(patternfile.nRows)
    ncols = np.uint64(self.ncols) #np.uint64(patternfile.nCols)

    pwidth = np.uint64(patternfile.patternW)
    pheight = np.uint64(patternfile.patternH)
    phw = pheight * pwidth

    nn = 1
    if chunksize <= 0:
      chunksize = np.maximum(1, np.round(1e9 / phw / ncols) ) # keeps the chunk at about 4GB
      chunksize = np.minimum(nrows,chunksize)
      print("Chunk size set to nrows:", int(chunksize))
    chunksize = np.int64(chunksize)

    nn = np.uint64(nn)

    if (automask is True) and (self.mask is None):
      self.mask = (self.automask(pheight,pwidth))
    if self.mask is None:
      self.mask = np.ones((pheight,pwidth),dtype=np.uint8)

    indices = np.asarray((self.mask.flatten().nonzero())[0],np.uint64)

    sigma = np.zeros((nrows,ncols),dtype=np.float32)+1e24
    colstartcount = np.asarray([0,ncols],dtype=np.int64)


    dthresh = np.float32(dthresh)
    lamopt_values = []
    for j in range(0,nrows,chunksize):
      print('Block',j)
      #rowstartread = np.int64(max(0,j - nn))
      rowstartread = np.int64(j)
      rowend = min(j + chunksize + nn,nrows)
      rowcountread = np.int64(rowend - rowstartread)
      data = patternfile.read_data(patStartCount=[[0,rowstartread],[ncols,rowcountread]],
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
    print("Optimal Choice: ", np.median(np.mean(lamopt_values, axis = 0)))
    if autoupdate == True:
      self.lam = np.median(np.mean(lamopt_values, axis = 0))
    if self.sigma is None:
      self.sigma = sigma

  def calcnlpar(self,chunksize=0,searchradius=None,lam = None,dthresh = None,saturation_protect=True,automask=True,
                filepath=None,filepathout=None,reset_sigma=True,backsub = False):

    if lam is not None:
      self.lam = lam

    if dthresh is not None:
      self.dthresh = dthresh

    if searchradius is not None:
      self.searchradius = searchradius

    lam = np.float32(self.lam)
    dthresh = np.float32(self.dthresh)
    sr = np.int64(self.searchradius)

    if filepath is not None:
      self.setfile(filepath=filepath)

    if reset_sigma:
      self.sigma = None

    patternfile = self.getinfileobj()

    #if filepathout is not None:
    self.setoutfile(patternfile, filepath=filepathout)

    patternfileout = self.getoutfileobj()


    nrows = np.int64(self.nrows)#np.int64(patternfile.nRows)
    ncols = np.int64(self.ncols)#np.int64(patternfile.nCols)
    if patternfileout.nCols is None:
      patternfileout.nCols = ncols
    if patternfileout.nRows is None:
      patternfileout.nRows = nrows


    pwidth = np.int64(patternfile.patternW)
    pheight = np.int64(patternfile.patternH)
    phw = pheight*pwidth

    if chunksize <= 0:
      chunksize = np.maximum(1, np.round(1e9 / phw / ncols) ) # keeps the chunk at about 8GB
      chunksize = np.minimum(nrows, chunksize)
      print("Chunk size set to nrows:", int(chunksize))
    chunksize = np.int64(chunksize)



    if (automask is True) and (self.mask is None):
      self.mask = (self.automask(pheight,pwidth))
    if self.mask is None:
      self.mask = np.ones((pheight,pwidth), dtype=np.uint8)

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

    nthreadpos = numba.get_num_threads()
    #numba.set_num_threads(36)
    colstartcount = np.asarray([0,ncols],dtype=np.int64)
    print(lam, sr, dthresh)
    for j in range(0,nrows,chunksize):
      rowstartread = np.int64(max(0, j-sr))
      rowend = min(j + chunksize+sr,nrows)
      rowcountread = np.int64(rowend-rowstartread)
      data = patternfile.read_data(patStartCount = [[0,rowstartread], [ncols,rowcountread]],
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

      print('Block', j)
      dataout = self.nlpar_nb(data,lam, sr, dthresh, sigchunk,
                              rowcountread,ncols,indices,saturation_protect)

      dataout = dataout.reshape(rowcountread, ncols, phw)
      dataout = dataout[j-rowstartread:, :, : ]
      shpout = dataout.shape
      dataout = dataout.reshape(shpout[0]*shpout[1], pheight, pwidth)

      patternfileout.write_data(newpatterns=dataout,patStartCount = [[0,j], [ncols, shpout[0]]],
                                     flt2int='clip',scalevalue=1.0 )
      #self.patternfileout.write_data(newpatterns=dataout,patStartCount=[j*ncols,shpout[0]*shpout[1]],
      #                               flt2int='clip',scalevalue=1.0 )
      #return dataout
      #sigma[j:j+rowstartcount[1],:] += \
      #  sigchunk[rowstartcount[0]:rowstartcount[0]+rowstartcount[1],:]
    numba.set_num_threads(nthreadpos)


  def calcsigma(self,chunksize=0,nn=1,saturation_protect=True,automask=True):

    patternfile = self.getinfileobj()


    nrows = np.int64(self.nrows)#np.uint64(patternfile.nRows)
    ncols = np.int64(self.ncols)#np.uint64(patternfile.nCols)

    pwidth = np.uint64(patternfile.patternW)
    pheight = np.uint64(patternfile.patternH)
    phw = pheight*pwidth

    if chunksize <= 0:
      chunksize = np.round(2e9/phw/ncols) # keeps the chunk at about 8GB
      chunksize = np.minimum(nrows,chunksize)
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
      data = patternfile.read_data(patStartCount = [[0,rowstartread], [ncols,rowcountread]],
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
      mxval += np.float32(1.0)
    else:
      mxval *= np.float32(0.999)
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

            if indx_nn == indx_0:
              weights[counter] = np.float32(-1.0e6)
            else:
              pairid = getpairid(indx_0, indx_nn)
              if pairid in pairdict:
                weights[counter] = pairdict[pairid]
              else:
                sigma1 = sigma[j_nn,i_nn]**2
                d2 = np.float32(0.0)
                n2 = np.float32(0.0)
                for q in range(shpind[0]):
                  d0 = data[indx_0,indices[q]]
                  d1 = data[indx_nn,indices[q]]
                  if (d1 < mxval) and (d0 < mxval):
                    n2 += np.float32(1.0)
                    d2 += (d0 - d1) ** np.int32(2)
                d2 -= n2*(sigma0+sigma1)
                dnorm = (sigma1 + sigma0)*np.sqrt(np.float32(2.0)*n2)
                if dnorm > np.float32(1.e-8):
                  d2 /= dnorm
                else:
                  d2 = np.float32(1e6)*n2
                weights[counter] = d2
                pairdict[pairid] = numba.float32(d2)
            counter += 1
        #print('________________')
        # end of window scanning
        sum = np.float(0.0)
        for i_nn in range(winsz):

          weights[i_nn] = np.maximum(weights[i_nn]-dthresh, numba.float32(0.0))
          weights[i_nn] = np.exp(-1.0 * weights[i_nn] * lam2)
          sum += weights[i_nn]

        for i_nn in range(winsz):
          indx_nn = pindx[i_nn]
          weights[i_nn] /= sum
          #print(weights[i_nn], ' \n')
          for q in range(shpdata[1]):
            dataout[indx_0, q] += data[indx_nn, q]*weights[i_nn]
        #print('_______', '\n')
    return dataout


