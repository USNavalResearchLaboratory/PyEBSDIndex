import numpy as np
from pathlib import Path
import shutil
import copy
import os


# this function will look at the path and return the correct EBSDPatterFile object
# if file_type is not specified, then it will be guessed based off of the extension
def get_pattern_file_obj(path,file_type=None,hdfDataPath=None):
  ebsdfileobj = None

  ftype = file_type
  if ftype is None:
    extension = str.lower(Path(path).suffix)
    if (extension == '.up1'):
      ftype = 'UP1'
    elif (extension == '.up2'):
      ftype = 'UP2'
    else:
      raise ValueError('Error: extension not recognized')

  if (ftype == 'UP1') or (ftype == 'UP2'):
    ebsdfileobj = UPFile(path)

  ebsdfileobj.read_header()
  return ebsdfileobj

def pat_flt2int(patterns, method='clip', scalvalue=0.98, bitdepth=None, maxScale=None):
  if isinstance(EBSDPatterns):
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
  if bitdepth is None:
    bitdepth = 8
    if max > 258:
     bitdepth = 16

  outtype = np.uint8
  minval = 0
  maxval = 255
  if bitdepth == 16:
    outtype = np.uint16
    minval = 0
    maxval = 65535

  patsout = np.zeros(shp, dtype=outtype)

  if method=='clip':
    patsout[:,:,:] = pats.clip(minval, maxval, dtype=outtype)
  elif method=='fullscale':
    temp = pats.astype(np.float32) - min
    if maxScale is None:
      maxScale = temp.max()
    temp *= scalvalue*maxval/maxScale
    temp = np.around(temp)
    patsout[:,:,:] = temp.astype(outtype)
  elif method=='scale': # here we assume that the min if not < 0 should not be scaled.
    temp = pats.astype(np.float32)
    if min < minval:
      temp += minval  - min
    if maxScale is None:
      maxScale = temp.max()
    temp *= scalvalue * maxval / maxScale
    temp = np.around(temp)
    patsout[:,:,:] = temp.astype(outtype)
  return patsout

class EBSDPatterns():
  def __init__(self, path=None):
    self.vendor = None
    self.path = None
    self.filetype = None
    self.patternW = None
    self.patternH = None
    self.nFileCols = None
    self.nFileRows = None
    self.nPaterns = None
    self.hexFlag = None
    self.xStep = None
    self.yStep = None
    self.patStartEnd = None # the 1D index range of patterns that was read
    self.patterns = None




# a class template for any EBSD pattern file type.
# Any EBSD file class should inheret this class.
class EBSDPatternFile():
  def __init__(self,path, file_type=None):
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
    self.file_type = file_type


  def read_header(self, path=None):
    pass

  def read_data(self,path=None,convertToFloat=False,patStartEnd = [0,-1],returnArrayOnly=False):
    pass

  def write_header(self):
    pass

  def write_data(self):
    pass

  def copy_file(self, newpath):
    src = Path(self.path)
    dst = Path(newpath)
    shutil.copyfile(src,dst)

  def copy_obj(self):
    return copy.deepcopy(self)

  def set_scan_rc(self, rc=(0,0)):
    self.nCols = rc[1]
    self.nRows = rc[0]
    self.nPatterns = self.nCols * self.nRows


class UPFile(EBSDPatternFile):

  def __init__(self, path=None):
    EBSDPatternFile.__init__(self, path)
    self.file_type = 'UP'
    self.vendor = 'EDAX'
    #UP only attributes
    self.bitdepth = None
    self.filePos = None  # file location in bytes where pattern data starts
    self.extraPatterns = 0
    self.hexFlag = 0


  def read_header(self,path=None,bitdepth=None):  # readInterval=[0, -1], arrayOnly=False,
    if path is not None:
      self.path = path

    extension = str.lower(Path(self.path).suffix)
    try:
      if (extension == '.up1'):
        self.bitdepth = 8
      elif (extension == '.up2'):
        self.bitdepth = 16
      else:
        if (bitdepth is None) and (self.bitdepth is None):
          raise ValueError('Error: extension not recognized, set "bitdepth" parameter')
        elif (bitdepth == 8) or (bitdepth == 16):
          self.bitdepth = bitdepth

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
    return 0

  def read_data(self,path=None,convertToFloat=False,patStartCount = [0,-1],returnArrayOnly=False, bitdepth=None):  # readInterval=[0, -1], arrayOnly=False,
    if path is not None:
      self.path = path
      self.read_header(bitdepth=bitdepth)
    if self.version is None:
      self.read_header(bitdepth=bitdepth)
    bitD = None
    bitD = 8 * (self.bitdepth == 8) + 16 * (self.bitdepth == 16)

    # this will allow for overriding the original file spec -- not sure why would want to but ...
    if (bitdepth == 8):
      bitD = 8
    if (bitdepth == 16):
      bitD = 16

    type = np.uint8
    if bitD == 8:
      type = np.uint8
    if bitD == 16:
      type = np.uint16

    if convertToFloat == True:
      typeout = np.float32
    else:
      typeout = type

    pStartEnd = np.asarray(patStartCount)
    if pStartEnd.ndim == 1: # read a continuous set of patterns.
      patStart = patStartCount[0]
      nPatToRead = patStartCount[-1]
      if nPatToRead == -1:
        nPatToRead = self.nPatterns - patStart
      if nPatToRead == 0:
        nPatToRead = 1
      if (patStart + nPatToRead) > self.nPatterns:
        nPatToRead = self.nPatterns - patStart


      #nPatToRead = int(patEnd - patStart)

      patterns = np.zeros([nPatToRead,self.patternH,self.patternW],dtype=typeout)

      try:
        f = open(Path(self.path).expanduser(), 'rb')
      except:
        print("File Not Found:", str(Path(self.path)))
        return -1

      f.seek(self.filePos)
      nPerPat = self.patternW * self.patternH

      f.seek(int(nPerPat * patStart * bitD/8), 1)



      for p in np.arange(int(nPatToRead)):
        onePat = np.fromfile(f, dtype=type, count=nPerPat)
        onePat = onePat.reshape(self.patternH, self.patternW)
        patterns[p, :, :] = onePat.astype(typeout)


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

        nPatToRead = [ncolread, nrowread]
        patterns = np.zeros([ncolread*nrowread,self.patternH,self.patternW],dtype=typeout)

        for i in range(nrowread):
          pstart = ((rowstart+i)*self.nCols)+colstart
          ptemp = self.read_data(path=path,convertToFloat=convertToFloat,patStartCount = [pstart,ncolread],returnArrayOnly=True, bitdepth=bitdepth)
          patterns[i*ncolread:(i+1)*ncolread, :, :] = ptemp

    if returnArrayOnly == True:
      return patterns
    else:  # package this up in an EBSDPatterns Object
      patsout = EBSDPatterns()
      patsout.vendor = self.vendor
      patsout.file = Path(self.path).expanduser()
      patsout.filetype = 'UP1' if (bitD == 8) else 'UP2'
      patsout.patternW = self.patternW
      patsout.patternH = self.patternH
      patsout.nFileCols = self.nCols
      patsout.nFileRows = self.nRows
      patsout.nPaterns = np.array(nPatToRead)
      patsout.hexFlag = self.hexFlag
      patsout.xStep = self.xStep
      patsout.yStep = self.yStep
      patsout.patStart = np.array(patStart)
      patsout.patterns = patterns
      return patsout

  def write_header(self, writeBlank=False, bitdepth=None):

    filepath = self.path
    extension = str.lower(Path(filepath).suffix)
    try:
      if (extension == '.up1'):
        self.bitdepth = 8
      elif (extension == '.up2'):
        self.bitdepth = 16
      else:
        if (bitdepth is None) and (self.bitdepth is None):
          raise ValueError('Error: extension not recognized, set "bitdepth" parameter')
        elif (bitdepth == 8) or (bitdepth == 16):
          self.bitdepth = bitdepth

    except ValueError as exp:
      print(exp)
    except:
      print('Error: file extension not recognized')
      return -1

    try:
      f = open(Path(filepath).expanduser(), 'wb')
    except:
      print("File Not Found:", str(Path(filepath)))
      return -1

    np.asarray(self.version, dtype=np.uint32)[0].tofile(f)
    if self.version == 1:
      np.asarray(self.patternW,dtype=np.uint32)[0].tofile(f)
      np.asarray(self.patternH,dtype=np.uint32)[0].tofile(f)
      np.asarray(self.filePos,dtype=np.uint32)[0].tofile(f)

    elif self.version >= 3:
      np.asarray(self.patternW,dtype=np.uint32)[0].tofile(f)
      np.asarray(self.patternH,dtype=np.uint32)[0].tofile(f)
      np.asarray(self.filePos,dtype=np.uint32)[0].tofile(f)
      np.asarray(self.extraPatterns,dtype=np.uint8)[0].tofile(f)
      np.asarray(self.nCols,dtype=np.uint32)[0].tofile(f)
      np.asarray(self.nRows,dtype=np.uint32)[0].tofile(f)
      np.asarray(self.hexFlag,dtype=np.uint8)[0].tofile(f)
      np.asarray(self.xStep,dtype=np.float64)[0].tofile(f)
      np.asarray(self.yStep,dtype=np.float64)[0].tofile(f)

    if writeBlank == True:
      if self.bitdepth == 8:
        type = np.uint8
      if self.bitdepth == 16:
        type = np.uint16

      blank = np.zeros((self.patternH, self.patternW), dtype=type)
      for j in range(self.nRows):
        for i in range(self.nCols):
          blank.tofile(f)


  def write_data(self, newpatterns=None, patStartCount = [0,-1], writeHead=True,
                 flt2int='clip', scalevalue = 0.98, maxScale = None):
    castscale = 0.98
    writeblank = False
    if patStartCount != [0,-1]:
      writeblank = True

    if not os.path.isfile(Path(self.path)):
      writeHead=True

    if writeHead==True:
      self.write_header(writeBlank=writeblank)

    bitD = self.bitdepth

    if isinstance(EBSDPatterns):
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
    type = pats.dtype
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

      try:
        f = open(Path(self.path).expanduser(),'rb')
      except:
        print("File Not Found:",str(Path(self.path)))
        return -1

      f.seek(self.filePos)
      nPerPat = self.patternW * self.patternH
      nPerPatByte = nPerPat*bitD/8
      f.seek(int(nPerPatByte*patStart),1)
      for p in np.arange(int(nPatToWrite)):
        onePat = pats[p, :, :]
        onePatInt = pat_flt2int(onePat,method='clip',scalvalue=scalevalue,bitdepth=bitD,maxScale=max)
        onePatInt.tofile(f)

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
      shp = (patternobj.nPaterns.prod(), patternobj.patternH, patternobj.patternW)
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






if __name__ == "__main__":
  file = '~/Desktop/SLMtest/scan2v3nlparl09sw7.up1'
  f = UPFile(file)
  #f.ReadHeader()
  pat = f.read_data()
