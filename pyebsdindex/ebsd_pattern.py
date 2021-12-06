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


import numpy as np
from pathlib import Path
import shutil
import copy
import os
import h5py




def get_pattern_file_obj(path,file_type=str(''),hdfDataPath=None):
  ''' this function will look at the path and return the correct EBSDPatterFile object
  if file_type is not specified, then it will be guessed based off of the extension'''
  ebsdfileobj = None
  pathtemp = np.atleast_1d(path)
  filepath = pathtemp[0]
  ftype = file_type
  if ftype == str(''):
    extension = str.lower(Path(filepath).suffix)
    if (extension == '.up1'):
      ftype = 'UP1'
    elif (extension == '.up2'):
      ftype = 'UP2'
    elif (extension == '.oh5'):
      ftype = 'OH5'
    elif (extension == '.h5'):
      ftype = 'H5'
    else:
      raise ValueError('Error: extension not recognized')

  if (ftype.upper() == 'UP1') or (ftype.upper() == 'UP2'):
    ebsdfileobj = UPFile(path)
  if (ftype.upper() == 'OH5'):
    ebsdfileobj = EDAXOH5(path)
    if pathtemp.size == 1: #automatically chose the first data group
      ebsdfileobj.get_data_paths()
      ebsdfileobj.set_data_path(pathindex=0)
  if (ftype.upper() == 'H5'):
    ebsdfileobj = HDF5PatFile(path) # if the path variable is a list, the second item is set to be the hdf5 path to
    # the patterns.
    if pathtemp.size == 1: #automatically chose the first data group
      ebsdfileobj.get_data_paths()
      ebsdfileobj.set_data_path(pathindex=0)
  ebsdfileobj.read_header()
  return ebsdfileobj

def pat_flt2int(patterns,typeout=None,method='clip',scalevalue=0.98,maxScale=None):
  ''' A helper function that will scale patterns that are floating point (or large ints) into
  unsigned integer values. There are several methods for scaling the values into an uint.
   If the typeout value is set to a np.floating type, no scaling will occur. '''

  if isinstance(patterns,EBSDPatterns):
    pats = patterns.patterns
    npats = patterns.nPatterns
  elif isinstance(patterns,np.ndarray):
    pats = patterns

  shp = pats.shape
  ndim = pats.ndim
  if ndim == 2:
    pats = pats.reshape(1,shp[0],shp[1])
  elif ndim == 3:
    pats = pats

  shp = pats.shape
  npats = shp[0]
  max = pats.max()
  min = pats.min()
  type = pats.dtype
  # make a guess if the bitdepth is not set
  if typeout is None:
    typeout = np.uint8
    if max > 258:
     typeout = np.uint16

  if (isinstance(typeout(0), np.floating )):
    return pats.astype(typeout)

  minval = 0
  maxval = 255
  if (isinstance(typeout(0), np.uint16)):
    typeout = np.uint16
    minval = 0
    maxval = 65535


  patsout = np.zeros(shp, dtype=typeout)

  if method=='clip':
    patsout[:,:,:] = pats.clip(minval, maxval).astype(dtype=typeout)
  elif method=='fullscale':
    temp = pats.astype(np.float32) - min
    if maxScale is None:
      maxScale = temp.max()
    temp *= scalevalue * maxval / maxScale
    temp = np.around(temp)
    patsout[:,:,:] = temp.astype(typeout)
  elif method=='scale': # here we assume that the min if not < 0 should not be scaled.
    temp = pats.astype(np.float32)
    if min < minval:
      temp += minval  - min
    if maxScale is None:
      maxScale = temp.max()
    temp *= scalevalue * maxval / maxScale
    temp = np.around(temp)
    patsout[:,:,:] = temp.astype(typeout)
  return patsout # note, this function uses multiple returns.

class EBSDPatterns():
  def __init__(self, path=None):
    self.vendor = None
    self.path = None
    self.filetype = None
    self.patternW = None
    self.patternH = None
    self.nFileCols = None
    self.nFileRows = None
    self.nPatterns = None
    self.hexFlag = None
    self.xStep = None
    self.yStep = None
    self.patStart = [0,0] #starting point of the pattern location in the file. len==1
    # if 2D, then it is the row/column starting points
    self.patterns = None




# a class template for any EBSD pattern file type.
# Any EBSD file class should inheret this class.
class EBSDPatternFile():
  def __init__(self,path, filetype=None):
    self.path = path
    self.vendor = None
    self.version = None
    self.nCols = None
    self.nRows = None
    self.nPatterns = None
    self.patternW = None
    self.patternH = None
    self.xStep = None
    self.yStep = None
    self.hexflag = False
    self.filetype = filetype
    self.filedatatype = np.uint8  # the data type of the patterns within the file


  def read_header(self, path=None):
    pass

  def read_data(self,path=None,convertToFloat=False,patStartCount = [0,-1],returnArrayOnly=False):
    if path is not None:
      self.path = path
      self.read_header()
    if self.version is None:
      self.read_header()

    # bitD = 8 * (self.bitdepth == 8) + 16 * (self.bitdepth == 16)

    # # this will allow for overriding the original file spec -- not sure why would want to but ...
    # if (bitdepth == 8):
    #   bitD = 8
    # if (bitdepth == 16):
    #   bitD = 16
    #
    # type = np.uint8
    # if bitD == 8:
    #   type = np.uint8
    # if bitD == 16:
    #   type = np.uint16

    if convertToFloat == True:
      typeout = np.float32
    else:
      typeout = self.filedatatype

    pStartEnd = np.asarray(patStartCount, dtype=np.int64)
    if pStartEnd.ndim == 1: # read a continuous set of patterns.
      patStart = patStartCount[0]
      nPatToRead = patStartCount[-1]
      if nPatToRead == -1:
        nPatToRead = self.nPatterns - patStart
      if nPatToRead == 0:
        nPatToRead = 1
      if (patStart + nPatToRead) > self.nPatterns:
        nPatToRead = self.nPatterns - patStart


      # this function does the actual reading from the file.
      readpats = self.pat_reader(patStart, nPatToRead)
      patterns = readpats.astype(typeout)



    elif pStartEnd.ndim == 2: # read a slab of patterns.
        colstart = pStartEnd[0,0]
        ncolread = pStartEnd[1,0]
        rowstart = pStartEnd[0,1]
        nrowread = pStartEnd[1,1]

        patStart = [colstart, rowstart]
        if ncolread < 0:
          ncolread = self.nCols - colstart
        if nrowread < 0:
          nrowread = self.nRows - rowstart

        if (colstart+ncolread) > self.nCols:
          ncolread = self.nCols - colstart

        if (rowstart+nrowread) > self.nRows:
          nrowread = self.nRows - rowstart
        nrowread = np.uint64(nrowread)
        ncolread = np.uint64(ncolread)
        nPatToRead = [ncolread, nrowread]

        patterns = np.zeros([int(ncolread*nrowread),self.patternH,self.patternW],dtype=typeout)

        for i in range(nrowread):
          pstart = int(((rowstart+i)*self.nCols)+colstart)
          ptemp = self.read_data(convertToFloat=convertToFloat,patStartCount = [pstart,ncolread],returnArrayOnly=True)

          patterns[int(i*ncolread):int((i+1)*ncolread), :, :] = ptemp

    if returnArrayOnly == True:
      return patterns
    else:  # package this up in an EBSDPatterns Object
      patsout = EBSDPatterns()
      patsout.vendor = self.vendor
      patsout.file = Path(self.path).expanduser()
      patsout.filetype = self.filetype
      patsout.patternW = self.patternW
      patsout.patternH = self.patternH
      patsout.nFileCols = self.nCols
      patsout.nFileRows = self.nRows
      patsout.nPatterns = np.array(nPatToRead)
      patsout.hexFlag = self.hexFlag
      patsout.xStep = self.xStep
      patsout.yStep = self.yStep
      patsout.patStart = np.array(patStart)
      patsout.patterns = patterns
      return patsout # note this function uses multiple return statements

  def pat_reader(self, patStart, nPatToRead):
    '''Depending on the file type, it will return a numpy array of patterns.'''
    pass

  def write_header(self):
    pass

  def write_data(self, newpatterns=None, patStartCount = [0,-1], writeHead=False,
                 flt2int='clip', scalevalue = 0.98, maxScale = None):
    writeblank = False

    if not os.path.isfile(Path(self.path).expanduser()): # file does not exist
      writeHead = True
      writeblank = True

    if writeHead==True:
      self.write_header(writeBlank=writeblank)

    if isinstance(newpatterns,EBSDPatterns):
      pats = newpatterns.patterns
      npats = newpatterns.nPatterns
    elif isinstance(newpatterns, np.ndarray):
      shp = newpatterns.shape
      ndim = newpatterns.ndim
      if ndim == 2:
        pats = newpatterns.reshape(1,shp[0], shp[1])
      elif ndim == 3:
        pats = newpatterns
      npats = pats.shape[0]
    max = pats.max()

    if maxScale is not None:
      max = maxScale

    pStartEnd = np.asarray(patStartCount)
    # npats == number of patterns in the newpatterns
    # self.nPatterns == number of patterns in the file
    # nPats to write == number of patterns to write out
    if pStartEnd.ndim == 1:  # write a continuous set of patterns.
      patStart = patStartCount[0]
      nPatToWrite = patStartCount[-1]
      if nPatToWrite == -1:
        nPatToWrite = npats
      if nPatToWrite == 0:
        nPatToWrite = 1
      if (patStart + nPatToWrite) > self.nPatterns:
        nPatToWrite = self.nPatterns - patStart

      typewrite = self.filedatatype
      pat2write = pat_flt2int(pats,typeout=typewrite,method=flt2int,scalevalue=scalevalue,maxScale=None)
      self.pat_writer(pat2write,patStart,nPatToWrite, typewrite)

    elif pStartEnd.ndim == 2: # write a slab of patterns.
        colstart = pStartEnd[0,0]
        ncolwrite = pStartEnd[1,0]
        rowstart = pStartEnd[0,1]
        nrowwrite = pStartEnd[1,1]

        patStart = [colstart, rowstart]
        if ncolwrite < 0:
          ncolwrite = self.nCols - colstart
        if nrowwrite < 0:
          nrowwrite = self.nRows - rowstart

        if (colstart+ncolwrite) > self.nCols:
          ncolwrite = self.nCols - colstart

        if (rowstart+nrowwrite) > self.nRows:
          nrowwrite = self.nRows - rowstart


        for i in range(nrowwrite):
          pstart = ((rowstart+i)*self.nCols)+colstart
          self.write_data(newpatterns = pats[i*ncolwrite:(i+1)*ncolwrite, :, :], patStartCount=[pstart,ncolwrite],writeHead=False,
                          flt2int=flt2int,scalevalue=0.98, maxScale = max)
  def pat_writer(self, pat2write, patStart, nPatToWrite, typewrite):
    pass


  def copy_file(self, newpath):
    src = Path(self.path).expanduser()
    if newpath is not None:
      dst = Path(newpath).expanduser()
    else:
      dst = Path(str(src.expanduser())+'.copy')
    shutil.copyfile(src.expanduser(),dst.expanduser())

  def copy_obj(self):
    return copy.deepcopy(self)

  def set_scan_rc(self, rc=(0,0)): # helper function for pattern files that don't record the scan rows and columns
    self.nCols = rc[1]
    self.nRows = rc[0]
    self.nPatterns = self.nCols * self.nRows


class UPFile(EBSDPatternFile):

  def __init__(self, path=None):
    EBSDPatternFile.__init__(self, path)
    self.filetype = 'UP'
    self.vendor = 'EDAX'
    self.filedatatype = None
    #UP only attributes
    #self.bitdepth = None
    self.filePos = None  # file location in bytes where pattern data starts
    self.extraPatterns = 0
    self.hexFlag = 0


  def read_header(self,path=None,bitdepth=None):  # readInterval=[0, -1], arrayOnly=False,
    if path is not None:
      self.path = path

    extension = str.lower(Path(self.path).suffix)
    try:
      if (extension == '.up1'):
        self.filedatatype = np.uint8
      elif (extension == '.up2'):
        self.filedatatype = np.uint16
      else:
        if (bitdepth is None) and (self.filedatatype is None):
          raise ValueError('Error: extension not recognized, set "bitdepth" parameter')
        elif (bitdepth == 8):
          self.filedatatype = np.uint8
        elif (bitdepth == 16):
          self.filedatatype = np.uint16

    except ValueError as exp:
      print(exp)
    except:
      print('Error: file extension not recognized')
      return -1

    try:
      f = open(Path(self.path).expanduser(), 'rb')
    except:
      print("File Not Found:", str(Path(self.path)))
      return -1

    self.version = np.fromfile(f, dtype=np.uint32, count=1)[0]
    if self.version == 1:
      dat = np.fromfile(f, dtype=np.uint32, count=3)
      self.patternW = dat[0]
      self.patternH = dat[1]
      self.filePos = dat[2]
      self.nPatterns = np.int((Path(self.path).expanduser().stat().st_size - 16) /
                              (self.patternW * self.patternH * self.bitdepth/8))
    elif self.version >= 3:
      dat = np.fromfile(f, dtype=np.uint32, count=3)
      self.patternW = dat[0]
      self.patternH = dat[1]
      self.filePos = dat[2]
      self.extraPatterns = np.fromfile(f, dtype=np.uint8, count=1)[0]
      dat = np.fromfile(f, dtype=np.uint32, count=2)
      self.nCols = dat[0]
      self.nRows = dat[1]
      self.nPatterns = np.int(self.nCols.astype(np.uint64) * self.nRows.astype(np.uint64))
      self.hexFlag = np.fromfile(f, dtype=np.uint8, count=1)[0]
      dat = np.fromfile(f, dtype=np.float64, count=2)
      self.xStep = dat[0]
      self.yStep = dat[1]
    f.close()
    return 0 #note this function uses multiple returns

  def pat_reader(self, patStart, nPatToRead):
    try:
      f = open(Path(self.path).expanduser(),'rb')
    except:
      print("File Not Found:",str(Path(self.path)))
      return -1

    f.seek(self.filePos)
    nPerPat = self.patternW * self.patternH
    typeread = self.filedatatype
    typebyte = self.filedatatype(0).nbytes

    f.seek(int(nPerPat * patStart * typebyte),1)
    readpats = np.fromfile(f,dtype=typeread,count=int(nPatToRead * nPerPat))
    readpats = readpats.reshape(nPatToRead,self.patternH,self.patternW)
    f.close()
    return readpats


  def write_header(self, writeBlank=False, bitdepth=None):

    filepath = self.path
    extension = str.lower(Path(filepath).suffix)
    try:
      if (extension == '.up1'):
        self.filedatatype = np.uint8
      elif (extension == '.up2'):
        self.filedatatype = np.uint16
      else:
        if (bitdepth is None) and (self.filedatatype is None):
          raise ValueError('Error: extension not recognized, set "bitdepth" parameter')
        elif (bitdepth == 8):
          self.filedatatype = np.uint8
        elif (bitdepth == 16):
          self.filedatatype = np.uint16

    except ValueError as exp:
      print(exp)
    except:
      print('Error: file extension not recognized')
      return -1

    try:
      if os.path.isfile(Path(self.path).expanduser()):
        f = open(Path(filepath).expanduser(), 'r+b')
        f.seek(0)
      else:
        f = open(Path(filepath).expanduser(),'w+b')
        f.seek(0)
    except:
      print("File Not Found:", str(Path(filepath)))
      return -1

    np.asarray(self.version, dtype=np.uint32).tofile(f)
    if self.version == 1:
      np.asarray(self.patternW,dtype=np.uint32).tofile(f)
      np.asarray(self.patternH,dtype=np.uint32).tofile(f)
      np.asarray(self.filePos,dtype=np.uint32).tofile(f)

    elif self.version >= 3:
      np.asarray(self.patternW,dtype=np.uint32).tofile(f)
      np.asarray(self.patternH,dtype=np.uint32).tofile(f)
      np.asarray(self.filePos,dtype=np.uint32).tofile(f)
      np.asarray(self.extraPatterns,dtype=np.uint8).tofile(f)
      np.asarray(self.nCols,dtype=np.uint32).tofile(f)
      np.asarray(self.nRows,dtype=np.uint32).tofile(f)
      np.asarray(self.hexFlag,dtype=np.uint8).tofile(f)
      np.asarray(self.xStep,dtype=np.float64).tofile(f)
      np.asarray(self.yStep,dtype=np.float64).tofile(f)

    if writeBlank == True:
      typewrite = self.filedatatype
      # if self.bitdepth == 8:
      #   type = np.uint8
      # if self.bitdepth == 16:
      #   type = np.uint16

      blank = np.zeros((self.patternH, self.patternW), dtype=typewrite)
      for j in range(self.nRows):
        for i in range(self.nCols):
          blank.tofile(f)
    f.close()


  def pat_writer(self, pat2write, patStart, nPatToWrite, typewrite):
    try:
      f = open(Path(self.path).expanduser(),'br+')
      f.seek(0,0)
    except:
      print("File Not Found:",str(Path(self.path)))
      return -1

    nPerPat = self.patternW * self.patternH
    nPerPatByte = nPerPat * typewrite(0).nbytes
    f.seek(int(nPerPatByte * (patStart) + self.filePos),0)
    pat2write[0:nPatToWrite,:,:].tofile(f)
    f.close()



  def file_from_pattern_obj(self, patternobj, filepath=None, bitdepth = None):
    if self.version is None:
      self.version = 3
    if self.xStep is None:
      self.xStep = 1.0
    if self.yStep is None:
      self.yStep = 1.0


    self.filePos = 42  # file location in bytes where pattern data starts
    self.extraPatterns = 0
    self.hexFlag = 0

    if isinstance(patternobj, EBSDPatterns):
      shp = (patternobj.nPatterns.prod(),patternobj.patternH,patternobj.patternW)
      mx = patternobj.patterns.max()
    elif isinstance(patternobj, np.ndarray):
      shp = patternobj.shape
      mx = patternobj.max()
      ndim = patternobj.ndim
      if ndim == 2: #
        shp = (1,shp[0], shp[1])
      elif ndim == 3:
        shp = shp
    self.patternH = shp[1]
    self.patternW = shp[2]

    self.nCols = shp[0]
    self.nRows = 1
    self.nPatterns = shp[0]

    if bitdepth is None: #make a guess
      self.bitdepth = 16
      if mx <= 256:
        self.bitdepth = 8


class HDF5PatFile(EBSDPatternFile):
  def __init__(self, path=None):
    filepath = None
    hdf5path = None
    if path is not None:
      ptemp = np.atleast_1d(path)
      filepath = ptemp[0]
      if ptemp.size > 1:
        hdf5path = ptemp[1]
    EBSDPatternFile.__init__(self, filepath)
    self.filetype = 'HDF5'
    self.vendor = 'PyEBSDIndex'
    #HDF only attributes
    self.filedatatype = np.dtype(np.uint8)
    self.h5datasetpath = hdf5path # This will be the h5 path to the patterns
    self.h5datagroups = [] # there can be multiple scans in one h5 file.  Potential data groups will be stored here.
    self.patternh5id = 'Pattern' #the name used for the pattern dataset array in the h5 file.

  def get_data_paths(self, verbose=0):
    '''Based on the H5EBSD spec this will search for viable Pattern Datasets '''
    try:
      f = h5py.File(self.path, 'r')
    except:
      print("File Not Found:",str(Path(self.path)))
      return -1
    self.h5datagroups = []
    groupsets = list(f.keys())
    for grpset in groupsets:
      if isinstance(f[grpset],h5py.Group):
        if 'EBSD' in f[grpset]:
          if self.patternh5id in f[grpset + '/EBSD/Data']:
            if (grpset  not in self.h5datagroups):
              self.h5datagroups.append(grpset)

    if len(self.h5datagroups) < 1:
      print("No viable EBSD patterns found:",str(Path(self.path)))
      return -2
    else:
      if verbose > 0:
        print(self.h5datagroups)
    return len(self.h5datagroups)

  def set_data_path(self, datapath=None):
    if datapath is not None:
      self.h5datasetpath = datapath

  def pat_reader(self,patStart,nPatToRead):
    '''This is a basic function that will read a chunk of patterns from the HDF5 file.
    It assumes that patterns are laid out in a HDF5 dataset as an array of N patterns x pattern height x pattern width '''
    try:
      f = h5py.File(Path(self.path).expanduser(),'r')
    except:
      print("File Not Found:",str(Path(self.path)))
      return -1

    patterndset = f[self.h5datasetpath]
    readpats = np.array(patterndset[patStart:patStart+nPatToRead, :, :])
    readpats = readpats.reshape(nPatToRead,self.patternH,self.patternW)
    f.close()
    return readpats

  def pat_writer(self, pat2write, patStart, nPatToWrite, typewrite):
    '''This is a basic function that will write a chunk of patterns to the HDF5 file.
        It assumes that patterns are laid out in a HDF5 dataset as an array of
        N patterns x pattern height x pattern width.  It also assumes the full dataset exists.
        Using the parent write_data method, it will check that a header exists, and that a dataset
        exists and fill it with blank data if needed.'''
    try:
      f = h5py.File(Path(self.path).expanduser(),'r+')
    except:
      print("File Not Found:",str(Path(self.path)))
      return -1

    patterndset = f[self.h5datasetpath]
    patterndset[patStart:patStart+nPatToWrite, :, :] = pat2write[0:nPatToWrite,:,:]
    f.close()

class EDAXOH5(HDF5PatFile):
  def __init__(self, path=None):
    HDF5PatFile.__init__(self, path)
    self.filetype = 'OH5'
    self.vendor = 'EDAX'
    #EDAXOH5 only attributes
    self.filedatatype = None # np.dtype(np.uint8)
    self.patternh5id = 'Pattern'
    self.activegroupid = None
    if self.path is not None:
      self.get_data_paths()

  def set_data_path(self, datapath=None, pathindex=0): #overloaded from parent - will default to first group.
    if datapath is not None:
      self.h5datasetpath = datapath
    else:
      if len(self.h5datagroups) > 0:
        self.activegroupid = pathindex
        self.h5datasetpath = self.h5datagroups[self.activegroupid] + '/EBSD/Data/' + self.patternh5id


  def read_header(self, path=None):
    if path is not None:
      self.path = path

    try:
      f = h5py.File(Path(self.path).expanduser(), 'r')
    except:
      print("File Not Found:", str(Path(self.path)))
      return -1

    self.version = str(f['Version'][()][0])

    if self.version  >= 'OIM Analysis 8.6.00':
      ngrp = self.get_data_paths()
      if ngrp <= 0:
        f.close()
        return -2 # no data groups with patterns found.
      if self.h5datasetpath is None: # default to the first datagroup
        self.set_data_path(pathindex=0)

      dset = f[self.h5datasetpath]
      shp = np.array(dset.shape)
      self.patternW = shp[-1]
      self.patternH = shp[-2]
      self.nPatterns = shp[-3]
      self.filedatatype = dset.dtype.type
      headerpath = self.h5datagroups[self.activegroupid]+'/EBSD/Header/'
      self.nCols = np.int32(f[headerpath+'nColumns'][()][0])
      self.nRows = np.int32(f[headerpath+'nRows'][()][0])
      self.hexFlag = np.int32(f[headerpath+'Grid Type'][()][0] == 'HexGrid')

      self.xStep = np.float32(f[headerpath+'Step X'][()][0])
      self.yStep = np.float32(f[headerpath+'Step Y'][()][0])

    return 0 #note this function uses multiple returns


