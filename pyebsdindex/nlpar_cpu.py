# This software was developed by employees of the US Naval Research Laboratory (NRL), an
# agency of the Federal Government. Pursuant to title 17 section 105 of the United States
# Code, works of NRL employees are not subject to copyright protection, and this software
# is in the public domain. PyEBSDIndex is an experimental system. NRL assumes no
# responsibility whatsoever for its use by other parties, and makes no guarantees,
# expressed or implied, about its quality, reliability, or any other characteristic. We
# would appreciate acknowledgment if the software is used. To the extent that NRL may hold
# copyright in countries other than the United States, you are hereby granted the
# non-exclusive irrevocable and unconditional right to print, publish, prepare derivative
# works and distribute this software, in any medium, or authorize others to do so on your
# behalf, on a royalty-free basis throughout the world. You may improve, modify, and
# create derivative works of the software or any portion of the software, and you may copy
# and distribute such modifications or works. Modified works should carry a notice stating
# that you changed the software and should note the date and nature of any such change.
# Please explicitly acknowledge the US Naval Research Laboratory as the original source.
# This software can be redistributed and/or modified freely provided that any derivative
# works bear some notice that they are derived from it, and any modified versions bear
# some notice that they have been modified.
#
# Author: David Rowenhorst;
# The US Naval Research Laboratory Date: 22 May 2024

# For more info see
# Patrick T. Brewick, Stuart I. Wright, David J. Rowenhorst. Ultramicroscopy, 200:50–61, May 2019.


import psutil
from pathlib import Path
from timeit import default_timer as timer

import numba
import numpy as np
import scipy.optimize as opt

from pyebsdindex import ebsd_pattern


#from os import environ
#environ["NUMBA_CACHE_DIR"] = str(tempdir)


__all__ = [
    "NLPAR",
]


class NLPAR:
  def __init__(self, filename=None,
               lam=0.7,
               searchradius=3,
               dthresh=0.0,
               diff_offset=0.0,
               saturation_protect=True,
               stem_scale=False,
               automask=True,
               nrows = None, ncols = None, **kwargs):
    self.lam = lam
    self.searchradius = searchradius
    self.dthresh = dthresh
    self.diff_offset = diff_offset,
    self.saturation_protect = saturation_protect,
    self.stem_scale = stem_scale,
    self.automask = automask,
    self.filepath = None
    self.hdfdatapath = None
    self.filepathout = None
    self.hdfdatapathout = None
    #self.patternfile = None
    #self.patternfileout = None
    self.setfile(filename)
    self.mask = None
    self.sigma = None
    self.sigmann = 1
    self.nrows = None
    self.ncols = None
    if nrows is not None:
      self.nrows = nrows
    if ncols is not None:
      self.ncols = ncols

  def auto_nlpar(self, filename = None, fileout=None, lindex = 1, **kwargs):
    if filename is not None:
      self.setfile(filename)
    lam = self.opt_lambda(autoupdate=True, **kwargs)
    if 'lam' in kwargs:
      pass
    else:
      kwargs['lam'] = lam[int(lindex)]
    nlparfile = self.calcnlpar(fileout=fileout, **kwargs)
    return nlparfile



  def opt_lambda_cpu(self, target_weights=(0.5, 0.34, 0.25), autoupdate=True, dthresh=None,
                # see __init__ for default dthresh values
                verbose = 2, **kwargs):
    # will accept all keywords to calcsigma_cpu. See NLPAR __init__ for default values

    target_weights = np.asarray(target_weights)

    if dthresh is not None:
      self.dthresh = dthresh
    dthresh = self.dthresh if np.isscalar(self.dthresh) else self.dthresh[0]
    dthresh = np.float64(dthresh)

    def loptfunc(lam,d2,tw,dthresh):
      temp = np.maximum(d2, dthresh)
      dw = np.exp(-(temp) / lam ** 2)
      w = np.sum(dw, axis=2) + 1e-12

      metric = np.mean(np.abs(tw - 1.0 / w))
      return metric

    patternfile = self.getinfileobj()
    patternfile.read_header()
    nrows = np.uint64(self.nrows) #np.uint64(patternfile.nRows)
    ncols = np.uint64(self.ncols) #np.uint64(patternfile.nCols)

    pwidth = np.uint64(patternfile.patternW)
    pheight = np.uint64(patternfile.patternH)
    phw = pheight * pwidth

    nn = 1
    nn = np.uint64(nn)
    dthresh = np.float32(dthresh)


    # for j in range(0,nrows,chunksize):
    #
    #   if verbose >= 2:
    #     print("begin row: ", j, "/", nrows, sep='', end='\r')
    #   #print('Block',j)
    #
    #   #rowstartread = np.int64(max(0,j - nn))
    #   rowstartread = np.int64(j)
    #   rowend = min(j + chunksize + nn,nrows)
    #   rowcountread = np.int64(rowend - rowstartread)
    #   data, xyloc = patternfile.read_data(patStartCount=[[0,rowstartread],[ncols,rowcountread]],
    #                                     convertToFloat=True,returnArrayOnly=True)
    #
    #   shp = data.shape
    #
    #   if backsub is True:
    #     data = self.backsub(data)
    #     #back = np.mean(data, axis=0)
    #     #back -= np.mean(back)
    #     #data -= back
    #   data = data.reshape(shp[0], phw)
    #
    #   rowstartcount = np.asarray([0,rowcountread],dtype=np.int64)
    #   sigchunk, (d2,n2, dij) = self.sigma_numba(data,nn,rowcountread,ncols,rowstartcount,colstartcount,indices,saturation_protect)
    #   tmp = (sigma[j:j + rowstartcount[1],:] < sigchunk).choose( sigchunk, sigma[j:j + rowstartcount[1],:])
    #   sigma[j:j + rowstartcount[1],:] = tmp


    sigma, d2,n2 = self.calcsigma_cpu(nn=nn, **kwargs)

    #print(d2.max(), d2.min())
    #d2 = d2norm(d2, n2, dij, sigma)
    #print(d2.max(), d2.min())


    lamopt_values = []
    stride = 1 if sigma.size < 1e6 else 2 #for large scans cut down on the optimization time.
    for tw in target_weights:

        lam = 1.0
        lambopt1 = opt.minimize(loptfunc,lam,args=(d2[0::stride,0::stride,:],tw,dthresh),method='Nelder-Mead',
                                bounds = [[0.001, 10.0]],options={'fatol': 0.0001})

        lamopt_values.append(lambopt1['x'])


    if verbose >= 2:
      print('', end='')
    lamopt_values = np.asarray(lamopt_values)

    print("Range of lambda values: ", lamopt_values.flatten())
    print("Optimal Choice: ", np.median(lamopt_values))
    if autoupdate == True:
        self.lam = np.median(lamopt_values)
    if self.sigma is None:
      self.sigma = sigma
    return lamopt_values.flatten()

  def calcsigma_cpu(self,chunksize=0,nn=1,saturation_protect=None,automask=None, stem_scale=None,
                    # See NLPAR __init__ for default values
                    verbose = 2, **kwargs):

    self.sigmann = nn

    if saturation_protect is not None:
      self.saturation_protect = saturation_protect
    saturation_protect = self.saturation_protect if np.isscalar(self.saturation_protect) else self.saturation_protect[0]

    if automask is not None:
      self.automask = automask
    automask = self.automask if np.isscalar(self.automask) else self.automask[0]

    if stem_scale is not None:
      self.stem_scale = stem_scale
    stem_scale = self.stem_scale if np.isscalar(self.stem_scale) else self.stem_scale[0]



    patternfile = self.getinfileobj()


    nrows = np.int64(self.nrows)#np.uint64(patternfile.nRows)
    ncols = np.int64(self.ncols)#np.uint64(patternfile.nCols)

    pwidth = np.uint64(patternfile.patternW)
    pheight = np.uint64(patternfile.patternH)
    phw = pheight*pwidth

    nn = np.uint64(nn)
    nnn = np.uint64((2*nn+1)**2)
    if chunksize <= 0:
        sysram = (psutil.virtual_memory()).total
        chunksize = np.minimum(32e9, sysram // 4)
    chunksize = np.int64(chunksize)
    chunks = self._calcchunks([pwidth, pheight], ncols, nrows, target_bytes=chunksize,
                              col_overlap=nn, row_overlap=nn)



    if (automask is True) and (self.mask is None):
      self.mask = (self.makeautomask(pheight,pwidth))
    if self.mask is None:
      self.mask = np.ones((pheight,pwidth), dtype=np.uint8)

    indices = np.asarray( (self.mask.flatten().nonzero())[0], np.uint64)

    sigma = np.zeros((nrows, ncols), dtype=np.float32)+1e18

    n2 = np.zeros((nrows, ncols, nnn), dtype=np.float32)
    d2 = np.zeros((nrows, ncols, nnn), dtype=np.float32)

    ndone = 0
    nchunks = int(chunks[1] * chunks[0])

    for rowchunk in range(chunks[1]):
        rstart = chunks[3][rowchunk, 0]
        rend = chunks[3][rowchunk, 1]
        nrowchunk = rend - rstart

        for colchunk in range(chunks[0]):
            cstart = chunks[2][colchunk, 0]
            cend = chunks[2][colchunk, 1]
            ncolchunk = cend - cstart
            data, xyloc = patternfile.read_data(patStartCount=[[cstart, rstart], [ncolchunk, nrowchunk]],
                                                convertToFloat=True, returnArrayOnly=True)

            if stem_scale is True:
                #data = data - data.min() + 1
                #data = np.log(data)
                data = data - data.min()
                data = np.sqrt(data)
            shp = data.shape
            data = data.reshape(data.shape[0], phw)


            sigchunk, d2chunk, n2chunk = self.sigma_numba(data,nn, nrowchunk,ncolchunk,
                                                       np.array([0,nrowchunk ], dtype=np.uint64),
                                                                    np.array([0,ncolchunk],dtype=np.uint64),
                                                                    indices,saturation_protect)

            sigma[rstart:rend, cstart:cend] = np.minimum(sigma[rstart:rend, cstart:cend], sigchunk)
            # temp = (d2 > thresh).choose(dthresh, d2)
            n2[rstart:rend, cstart:cend,:] = np.select( [n2chunk >  0], [n2chunk], default=n2[rstart:rend, cstart:cend,:])
            d2[rstart:rend, cstart:cend,:] = np.select( [n2chunk >  0], [d2chunk], default=d2[rstart:rend, cstart:cend,:])
            #dij[rstart:rend, cstart:cend, :] = dijchunk
            ndone += 1
            if verbose >= 2:
                print("tiles complete: ", ndone, "/", nchunks, sep='', end='\r')

    return sigma, d2, n2

  def calcnlpar_cpu(self, chunksize=0, searchradius=None, lam = None, dthresh = None,
               saturation_protect=None, automask=None, stem_scale = None, # see NLPAR __init__ for default values
               filename=None, fileout=None, reset_sigma=False, backsub = False, rescale = False,verbose=2,
               diff_offset=None,
               **kwargs):

    if searchradius is not None:
      self.searchradius = searchradius
    sr = self.searchradius if np.isscalar(self.searchradius) else self.searchradius[0]
    sr = np.int64(sr)

    if lam is not None:
      self.lam = lam
    lam = self.lam if np.isscalar(self.lam) else self.lam[0]
    lam = np.float32(lam)


    if dthresh is not None:
      self.dthresh = dthresh
    dthresh = self.dthresh if np.isscalar(self.dthresh) else self.dthresh[0]
    dthresh = np.float32(dthresh)

    if diff_offset is not None:
      self.diff_offset = diff_offset
    diff_offset = self.diff_offset if np.isscalar(self.diff_offset) else self.diff_offset[0]

    if saturation_protect is not None:
      self.saturation_protect = saturation_protect
    saturation_protect = self.saturation_protect if np.isscalar(self.saturation_protect) else self.saturation_protect[0]

    if automask is not None:
      self.automask = automask
    automask = self.automask if np.isscalar(self.automask) else self.automask[0]

    if stem_scale is not None:
      self.stem_scale = stem_scale
    stem_scale = self.stem_scale if np.isscalar(self.stem_scale) else self.stem_scale[0]

    if filename is not None:
      self.setfile(filepath=filename)




    patternfile = self.getinfileobj()

    #if filepathout is not None:
    self.setoutfile(patternfile, filepath=fileout)

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

    # if chunksize <= 0:
    #   chunksize = np.maximum(1, np.round(1e9 / phw / ncols) ) # keeps the chunk at about 8GB
    #   chunksize = np.minimum(nrows, chunksize)
    #   print("Chunk size set to nrows:", int(chunksize))
    # chunksize = np.int64(chunksize)
    if chunksize <= 0:
        sysram = (psutil.virtual_memory()).total
        chunksize = np.minimum(32e9, sysram/4)


    chunks = self._calcchunks( [pwidth, pheight], ncols, nrows, target_bytes=chunksize,
                              col_overlap=sr, row_overlap=sr)
    chunksize = (chunks[2][:, 1] - chunks[2][:, 0]).reshape(1, -1) * \
                (chunks[3][:, 1] - chunks[3][:, 0]).reshape(-1, 1)
    nchunks = chunksize.size
    # return chunks, chunksize
    mxchunk = np.int64(chunksize.max())

    if (automask is True) and (self.mask is None):
      self.mask = (self.makeautomask(pheight,pwidth))
    if self.mask is None:
      self.mask = np.ones((pheight,pwidth), dtype=np.uint8)

    indices = np.asarray( (self.mask.flatten().nonzero())[0], np.int64)
    calcsigma = False
    if self.sigma is None:
      calcsigma = True
      self.sigma = np.zeros((nrows, ncols), dtype=np.float32)+1e24

    if reset_sigma:
      self.sigma = None

    if (np.asarray(self.sigma).size == 1) and (self.sigma is not None):
      tmp = np.asarray(self.sigma)[0]
      self.sigma =  np.zeros((nrows, ncols), dtype=np.float32)+tmp
      calcsigma = False

    shpsigma = np.asarray(self.sigma).shape
    if (shpsigma[0] != nrows) and (shpsigma[1] != ncols):
      self.sigma = np.zeros((nrows,ncols),dtype=np.float32) + 1e24
      calcsigma = True

    sigma = np.asarray(self.sigma)

    scalemethod = 'clip'
    if rescale == True:
      if np.issubdtype(patternfileout.filedatatype, np.integer):
        mxval = np.iinfo(patternfileout.filedatatype).max
        scalemethod = 'fullscale'
      else: # not int, so no rescale.
        rescale = False



    nthreadpos = numba.get_num_threads()
    #numba.set_num_threads(18)
    #numba.set_num_threads(18)
    colstartcount = np.asarray([0,ncols],dtype=np.int64)
    if verbose >= 1:
      print("lambda:", self.lam, "search radius:", self.searchradius, "dthresh:", self.dthresh)

    # set up  a job queue
    ndone = 0
    jqueue = []
    jobid = 0
    # if verbose >= 2:
    #   print('\n', end='')
    for rowchunk in range(chunks[1]):
        rstart = chunks[3][rowchunk, 0]
        rend = chunks[3][rowchunk, 1]
        nrowchunk = rend - rstart

        rstartcalc = sr if (rowchunk > 0) else 0
        rendcalc = nrowchunk - sr if (rowchunk < (chunks[1] - 1)) else nrowchunk
        nrowcalc = np.int64(rendcalc - rstartcalc)

        for colchunk in range(chunks[0]):
            cstart = chunks[2][colchunk, 0]
            cend = chunks[2][colchunk, 1]
            ncolchunk = cend - cstart

            cstartcalc = sr if (colchunk > 0) else 0
            cendcalc = ncolchunk - sr if (colchunk < (chunks[0] - 1)) else ncolchunk
            ncolcalc = np.int64(cendcalc - cstartcalc)

            job = {"rstart": rstart,
                   "rend": rend,
                   "nrowchunk": nrowchunk,
                   "rstartcalc": rstartcalc,
                   "rendcalc": rendcalc,
                   "nrowcalc": nrowcalc,
                   "cstart": cstart,
                   "cend": cend,
                   "ncolchunk": ncolchunk,
                   "cstartcalc": cstartcalc,
                   "cendcalc": cendcalc,
                   "ncolcalc": ncolcalc,
                   "nattempts": -1,
                   "jobid": jobid}
            jobid += 1
            jqueue.append(job)



    while len(jqueue) > 0:
        j = jqueue.pop(0)
        j["nattempts"] += 1

        rstart = j["rstart"]
        cstart = j["cstart"]
        rend = j["rend"]
        cend = j["cend"]
        cstartcalc = j["cstartcalc"]
        rstartcalc = j["rstartcalc"]
        ncolchunk = j["ncolchunk"]
        nrowchunk = j["nrowchunk"]
        ncolcalc = j["ncolcalc"]
        nrowcalc = j["nrowcalc"]

        calclim = np.array([cstartcalc, rstartcalc, ncolcalc, nrowcalc], dtype=np.int64)

        data, xyloc = patternfile.read_data(patStartCount=[[ cstart, rstart], [ncolchunk, nrowchunk]],
                                          convertToFloat=True, returnArrayOnly=True)
        if stem_scale is True:
            datamin = data.min()
            # data = data - datamin + 1
            # data = np.log(data)
            data = data - datamin
            data = np.sqrt(data)

        shpdata = data.shape

        if backsub is True:
            data = self.backsub(data)

        data = data.reshape(shpdata[0], phw)

        if calcsigma is True:
            sigchunk = self.sigma_numba(data, 1, nrowchunk, ncolchunk,
                                             [0,nrowchunk], [0,ncolchunk],
                                             indices, saturation_protect)[0]

            sigchunk = np.minimum(sigma[rstart:rend,cstart:cend], sigchunk)
            sigma[rstart:rend,cstart:cend] = sigchunk
        else:
            sigchunk = sigma[rstart:rend, cstart:cend ]



        dataout = self.nlpar_nb(data, lam, sr, dthresh, sigchunk,
                                nrowchunk, ncolchunk,
                                calclim=calclim, indices_in=indices,
                                saturation_protect=saturation_protect, diff_offset=diff_offset)

        # nlpar_nb(data, lam, sr, dthresh, sigma, nrows, ncols, calclim=np.array([-1, -1, -1, -1], dtype= np.int64),
        #         indices=np.array([-1], dtype= np.int64), saturation_protect=True,
        #         diff_offset=np.float32(0.0))
        dataout = dataout.reshape(nrowchunk, ncolchunk, -1)
        dataout = dataout[rstartcalc: rstartcalc + nrowcalc,
                            cstartcalc:cstartcalc + ncolcalc, :]
        if stem_scale is True:
            #dataout = np.exp(dataout) - 1 + datamin
            dataout = dataout**2 + datamin
        shpout = dataout.shape
        dataout = dataout.reshape(shpout[0] * shpout[1], pheight, pwidth)
        if rescale == True:
            for i in range(dataout.shape[0]):
                temp = dataout[i, :, :]
                temp -= temp.min()
                temp *= np.float32(mxval) / temp.max()
                dataout[i, :, :] = temp

        patternfileout.write_data(newpatterns=dataout,
                                  patStartCount=[[np.int64(cstart + cstartcalc), np.int64(rstart + rstartcalc)],
                                                 [ncolcalc, nrowcalc]],
                                  flt2int='clip', scalevalue=1.0)
        ndone += 1
        if verbose >= 2:
            print("tiles complete: ", ndone, "/", nchunks, sep='', end='\r')


    # for j in range(0,nrows,chunksize):
    #   #print('Row start', j)
    #   if verbose >= 2:
    #     print("begin row: ", j, "/", nrows, sep='', end='\r')
    #
    #   rowstartread = np.int64(max(0, j-sr))
    #   rowend = min(j + chunksize+sr,nrows)
    #
    #   if (rowend - rowstartread) < (2*sr+1):
    #     rowstartread = np.int64(max(0, rowend - (2*sr+1)))
    #   rowcountread = np.int64(rowend-rowstartread)
    #   data, xyloc = patternfile.read_data(patStartCount = [[0,rowstartread], [ncols,rowcountread]],
    #                                     convertToFloat=True,returnArrayOnly=True)
    #
    #   shpdata = data.shape
    #
    #   if backsub is True:
    #     data = self.backsub(data)
    #
    #
    #   data = data.reshape(shpdata[0], phw)
    #
    #   rowstartcount = np.asarray([0,rowcountread],dtype=np.int64)
    #   if calcsigma is True:
    #     sigchunk, tmp = self.sigma_numba(data,1,rowcountread,ncols,rowstartcount,colstartcount,indices,saturation_protect)
    #     del tmp
    #     tmp = (sigma[rowstartread:rowend,:] < sigchunk).choose(sigchunk,sigma[rowstartread:rowend,:])
    #     sigma[rowstartread:rowend,:] = tmp
    #   else:
    #     sigchunk = sigma[rowstartread:rowend,:]
    #
    #   #dataout = data
    #
    #   dataout = self.nlpar_nb(data,lam, sr, dthresh, sigchunk,
    #                           rowcountread,ncols,indices,saturation_protect, diff_offset=diff_offset)
    #
    #
    #   dataout = dataout.reshape(rowcountread, ncols, phw)
    #   dataout = dataout[j-rowstartread:, :, : ]
    #   shpout = dataout.shape
    #   dataout = dataout.reshape(shpout[0]*shpout[1], pheight, pwidth)
    #   if rescale == True:
    #     for i in range(dataout.shape[0]):
    #       temp = dataout[i,:,:]
    #       temp -= temp.min()
    #       temp *= np.float32(mxval)/temp.max()
    #       dataout[i,:,:] = temp
    #
    #   patternfileout.write_data(newpatterns=dataout,patStartCount = [[0,j], [ncols, shpout[0]]],
    #                                  flt2int='clip',scalevalue=1.0 )
    #   #self.patternfileout.write_data(newpatterns=dataout,patStartCount=[j*ncols,shpout[0]*shpout[1]],
    #   #                               flt2int='clip',scalevalue=1.0 )
    #   #return dataout
    #   #sigma[j:j+rowstartcount[1],:] += \
    #   #  sigchunk[rowstartcount[0]:rowstartcount[0]+rowstartcount[1],:]


    if verbose >= 2:
      print('', end='')

    numba.set_num_threads(nthreadpos)
    return str(patternfileout.filepath)

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

  def setoutfile(self, patternfile, filepath=None):
    """Set the output file.

    Parameters
    ----------
    patternfile
      Input pattern file object from ebsd_pattern.
    filepath
      String.

    Notes
    -----
    In the future I want to be able to specify the HDF5 data path to
    store the output data, but that is proving to be a bit of a mess.
    For now, a copy of the original HDF5 is made, and the NLPAR patterns
    will be overwritten on top of the originals.
    """
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
          pathok = not self.filepathout.samefile(patternfile.filepath)
          if not pathok:
            raise ValueError('Error: File input and output are exactly the same.')
            return

        patternfile.copy_file([self.filepathout,self.hdfdatapathout], empty_data=True)
        return  # fpath and (maybe) hdf5 path were set manually.
      else: # this is a hdf5 file
        if self.hdfdatapathout is None:
          patternfile.copy_file(self.filepathout, empty_data=True)
          self.hdfdatapathout = patternfile.h5patdatpth
          return
        else:
          patternfile.copy_file([self.filepathout, self.hdfdatapathout], empty_data=True)
          return

    if patternfile is not None: # the user has set no path.
      hdf5path = None

      if patternfile.filetype in ['UP', 'EBSP', 'TFPAT']:
        p = Path(patternfile.filepath)
        appnd = "_NLPAR_l{:1.2f}".format(self.lam) + "sr{:d}".format(self.searchradius)
        newfilepath = str(p.parent / Path(p.stem + appnd + p.suffix))
        emptydata = True
        if patternfile.filetype in ['EBSP']:
          if patternfile.version > 5:
            emptydata = False
        #print(emptydata)
        patternfile.copy_file(newfilepath,empty_data=emptydata)

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
        patternfile.copy_file([newfilepath], empty_data=True)
        hdf5path = patternfile.h5patdatpth

      self.filepathout = newfilepath
      self.hdfdatapathout = hdf5path
      return

  def getinfileobj(self):
    if self.filepath is not None:
      fID = ebsd_pattern.get_pattern_file_obj([self.filepath, self.hdfdatapath])
      if (fID.nRows is not None):
        if (self.nrows is None):
          self.nrows = fID.nRows
        else:
          fID.nRows = self.nrows

      if (fID.nCols is not None):
        if (self.ncols is None):
          self.ncols = fID.nCols
        else:
          fID.nCols = self.ncols

      if self.ncols == 1:
        print('The number of scan columns is set to one, which is unusual, and may indicate that')
        print('the number of columns is not saved as metadata in the pattern file. Consider manually')
        print('entering the number of columns/rows with ``nlobj.ncols={number of your scan columns}`` and ')
        print('``nlobj.nrows={number of your scan rows}``.')
      return fID

    else:
      return None

  def getoutfileobj(self):
    if self.filepathout is not None:
      fID = ebsd_pattern.get_pattern_file_obj([self.filepathout, self.hdfdatapathout])
      if self.nrows is not None:
        fID.nRows = self.nrows
      else:
        self.nrows = fID.nRows

      if self.ncols is not None:
        fID.nCols = self.ncols
      else:
        self.ncols = fID.nCols
      return fID
    else:
      return None






  def backsub(self, data):
    # This function will fit a 2D gaussian on top of a plane to the averaged set of patterns (data) that is provided.
    # It will automatically use whatever mask is defined for valid data.
    # If the gaussian fit fails to converge, it will fall back to just using the mean set of patterns for the background
    # with a warning.


    def gaussian_surf(x, y, a, x0, y0, sigx, sigy, c, d, e):
    # equation for 2D gaussian on top of a plane.
      return a * np.exp(- ((x - x0) ** 2) / (2.0 * sigx ** 2) - ((y - y0) ** 2) / (2.0 * sigy ** 2)) + c + d * x + e * y

    def fit_gauss(M, *args):
    # helper function
      x, y = M
      #arr = np.zeros(x.shape)
      return gaussian_surf(x, y, *args)

    back = np.mean(data, axis=0) # start with the mean of all the data
    # now fit a 2D gaussian sitting on a plane.  See fuction def above.
    nx = data.shape[-1]
    ny = data.shape[-2]
    x = np.arange(nx, dtype=float)
    x = (np.broadcast_to(x.reshape(1,nx), (ny, nx)))
    y = np.arange(ny, dtype=float)
    y = (np.broadcast_to(y, (nx, ny)).T)
    x = x.ravel()
    y = y.ravel()

    # need to grab only the values that are in the mask.
    wh = np.nonzero(self.mask.ravel())[0]
    xwh = x[wh]
    ywh = y[wh]
    xywh = np.vstack((xwh, ywh))
    zwh = (back.ravel())[wh]
    whmx = np.unravel_index(back.argmax(), back.shape)
    minz = zwh.min()
    # initialize a guess for the parameters.
    # [gauss amplitude, max loc x, max loc y, sigx, sigy, const offset, slope x, slope y]
    p0 = [(zwh.max() - zwh.min()), whmx[1], whmx[0], nx/2.355, ny/2.355, minz, 0, 0]
    try:
      popt, pcov = opt.curve_fit(fit_gauss, xywh, zwh, p0)
      backfit = (gaussian_surf(x, y, *popt)).reshape(ny, nx)
      #print(p0, popt)
    except RuntimeError:
      print('Warning: no convergence on back subtract ... using mean of the patterns.')
      print('This may not be ideal for scans with few grains across the width of the scan.')
      backfit = back
    backfit -= np.mean(backfit)
    #f, axarr = plt.subplots(1, 3)
    #f.set_size_inches(10, 4)
    #axarr[0].imshow(data[0,:,:].squeeze(), cmap='gray')
    #axarr[1].imshow(data[0,:,:].squeeze() - backfit, cmap='gray')
    #axarr[2].imshow(backfit, cmap='gray')

    data -= backfit
    return data

  @staticmethod
  def makeautomask(h, w):
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
            n2 = np.float32(1.0e-12)
            nout[j,i,count] = n0 # want to save this for labmda optimization
            if not((i == i_nn) and (j == j_nn)):
              for q in range(shpind[0]):
                d0 = data[indx_0, indices[q]]
                d1 = data[indx_nn, indices[q]]
                if (d1 < mxval) and (d0 < mxval): # this is a saturation protection
                  n2 += 1.0
                  d2 += (d0 - d1)**2
              nout[j,i,count] = n2

              if d2 >= 1.e-3: #sometimes EDAX collects the same pattern twice
                s0 = d2 / np.float32(n2 * 2.0)
                if s0 < mind:
                  mind = s0
            dout[j,i,count] = d2 # want to save this for labmda optimization

            count += 1

        sigma[j,i] = np.sqrt(mind)

    # normalize the distances for lambda optimization.
    shp = dout.shape
    s2 = sigma ** 2
    for j in numba.prange(shp[0]):
        for i in range(shp[1]):
            for q in range(shp[2]):
                if nout[j, i, q] > 0:
                    ii = dij[j, i, q, 1]
                    jj = dij[j, i, q, 0]
                    s2_12 = (s2[j, i] + s2[jj, ii])
                    dout[j, i, q] -= nout[j, i, q] * s2_12
                    dout[j, i, q] /= s2_12 * np.sqrt(2.0 * nout[j, i, q])
    return sigma, dout, nout

  @staticmethod
  @numba.jit(nopython=True,cache=True,fastmath=False,parallel=True)
  def nlpar_nb(data, lam, sr, dthresh, sigma, nrows, ncols, calclim = np.array([-1, -1, -1, -1], dtype=np.int64),
                indices_in=np.array([-1], dtype= np.int64), saturation_protect=True,
                diff_offset=np.float32(0.0)):
    def getpairid(idx0, idx1):
      idx0_t = int(idx0)
      idx1_t = int(idx1)
      if idx0 < idx1:
        pairid = idx0_t + (idx1_t << 32)
      else:
        pairid = idx1_t + (idx0_t << 32)
      return numba.uint64(pairid)

    # set some defaults
    # calclim = np.array([cstartcalc, rstartcalc, ncolcalc, nrowcalc], dtype=np.int64)
    if calclim[0] <= -1:
        calclim[0] = 0
    if calclim[1] <= -1:
        calclim[1] = 0
    if calclim[2] <= -1:
        calclim[2] = ncols-calclim[0]
    if calclim[3] <= -1:
        calclim[3] = nrows-calclim[1]



    lam2 = 1.0 / lam**2
    dataout = np.zeros_like(data, np.float32)
    shpdata = data.shape

    # set defaults ... normally will not be needed.
    if indices_in.size == 1:
        if indices_in[0] == -1:
            indices = np.arange(shpdata[1], dtype=np.int64)
        else:
            indices = indices_in.astype(np.int64)
    else:
        indices = indices_in.astype(np.int64)


    shpind = indices.shape
    winsz = np.int32((2*sr+1)**2)
    diff_step =  np.zeros((winsz), dtype=np.float32)

    mxval = np.max(data)
    if saturation_protect == False:
      mxval += np.float32(1.0)
    else:
      mxval *= np.float32(0.999)
    for ii in numba.prange(calclim[2]):
      i = calclim[0]+ii
      winstart_x = max((i - sr),0) - max((i + sr - (ncols - 1)),0)
      winend_x = min((i + sr),(ncols - 1)) + max((sr - i),0) + 1
      pairdict = numba.typed.Dict.empty(key_type=numba.core.types.uint64,value_type=numba.core.types.float32)
      for jj in range(calclim[3]):
        j = calclim[1]+jj
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
              diff_step[counter] +=  diff_offset
            else:
              diff_step[counter] =  0.0
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
                    d2 += np.float32(d0 - d1) ** np.int32(2)
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
        sum = np.float32(0.0)
        for i_nn in range(winsz):

          weights[i_nn] = np.maximum(weights[i_nn]-dthresh, numba.float32(0.0))
          weights[i_nn] = np.exp(-1.0 * weights[i_nn] * lam2)
          weights[i_nn] += diff_step[i_nn]
          sum += weights[i_nn]

        for i_nn in range(winsz):
          indx_nn = pindx[i_nn]
          weights[i_nn] /= sum
          #print(weights[i_nn], ' \n')
          for q in range(shpdata[1]):
            dataout[indx_0, q] += data[indx_nn, q]*weights[i_nn]
        #print('_______', '\n')
    return dataout

  def calcsigma(self,**kwargs): # helper function
    return self.calcsigma_cpu(**kwargs)


  def opt_lambda(self, **kwargs): # helper function
    return self.opt_lambda_cpu(**kwargs)

  def calcnlpar(self, **kwargs):  # helper function
    return self.calcnlpar_cpu(**kwargs)


  def _calcchunks(self, patdim, ncol, nrow, target_bytes=2e9, col_overlap=0, row_overlap=0, col_offset=0, row_offset=0):

    col_overlap = min(col_overlap, ncol - 1)
    row_overlap = min(row_overlap, nrow - 1)

    byteperpat = patdim[-1] * patdim[-2] * 4 * 2  # assume a 4 byte float input and output array
    byteperdataset = byteperpat * ncol * nrow
    nchunks = int(np.ceil(byteperdataset / target_bytes))

    ncolchunks = (max(np.round(np.sqrt(nchunks * float(ncol) / nrow)), 1))
    colstep = max((ncol / ncolchunks), 1)
    ncolchunks = max(ncol / colstep, 1)
    colstepov = min(colstep + 2 * col_overlap, ncol)
    ncolchunks = max(int(np.ceil(ncolchunks)), 1)
    colstep = max(int(np.round(colstep)), 1)
    colstepov = min(colstep + 2 * col_overlap, ncol)

    nrowchunks = max(np.ceil(nchunks / ncolchunks), 1)
    rowstep = max((nrow / nrowchunks), 1)
    nrowchunks = max(nrow / rowstep, 1)
    rowstepov = min(rowstep + 2 * row_overlap, nrow)
    nrowchunks = max(int(np.ceil(nrowchunks)), 1)
    rowstep = max(int(np.round(rowstep)), 1)
    rowstepov = min(rowstep + 2 * row_overlap, nrow)

    # colchunks = np.round(np.arange(ncolchunks+1)*ncol/ncolchunks).astype(int)
    # colchunks = np.zeros((ncolchunks, 2), dtype=int)
    # colchunks[:, 0] = (np.arange(ncolchunks) * colstep).astype(int)
    # colchunks[:, 1] = colchunks[:, 0] + colstepov - int(col_overlap)
    # colchunks[:, 0] -= col_overlap
    # colchunks[0, 0] = 0;

    colchunks = []
    col_overlap = int(col_overlap)
    for c in range(ncolchunks):
      cchunk = [int(c * colstep) - col_overlap, int(c * colstep + colstepov) - col_overlap]
      colchunks.append(cchunk)
      if cchunk[1] > ncol:
        break

    ncolchunks = len(colchunks)
    colchunks = np.array(colchunks, dtype=int)
    colchunks[0, 0] = 0
    colchunks[-1, 1] = ncol

    if ncolchunks > 1:
      colchunks[-1, 0] = max(0, colchunks[-2, 1] -  2*col_overlap-1)

    colchunks += col_offset

    # for i in range(ncolchunks - 1):
    #   if colchunks[i + 1, 0] >= ncol:
    #     colchunks = colchunks[0:i + 1, :]

    rowchunks = []
    row_overlap = int(row_overlap)
    for r in range(nrowchunks):
      rchunk = [int(r * rowstep) - row_overlap, int(r * rowstep + rowstepov) - row_overlap]
      rowchunks.append(rchunk)
      if rchunk[1] > nrow:
        break

    nrowchunks = len(rowchunks)
    rowchunks = np.array(rowchunks, dtype=int)
    rowchunks[0, 0] = 0
    # for i in range(ncolchunks - 1):
    #   if colchunks[i + 1, 0] >= ncol:
    #     colchunks = colchunks[0:i + 1, :]

    rowchunks = []
    row_overlap = int(row_overlap)
    for r in range(nrowchunks):
      rchunk = [int(r * rowstep) - row_overlap, int(r * rowstep + rowstepov) - row_overlap]
      rowchunks.append(rchunk)
      if rchunk[1] > nrow:
        break

    nrowchunks = len(rowchunks)
    rowchunks = np.array(rowchunks, dtype=int)
    rowchunks[0, 0] = 0
    rowchunks[-1, 1] = nrow

    if nrowchunks > 1:
      rowchunks[-1, 0] = max(0, rowchunks[-2, 1] -  2*row_overlap-1)

    rowchunks += row_offset

    return ncolchunks, nrowchunks, colchunks, rowchunks

  # def asciiupdate(self, nrow, ncol, completematrix):
  #   cm = completematrix
  #   ncdisplay = min(ncol, 80)
  #   nrow


