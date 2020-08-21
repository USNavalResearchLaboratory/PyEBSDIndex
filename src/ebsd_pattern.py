import numpy as np
from pathlib import Path


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
    self.patStartEnd = None
    # the 1D index range of patterns that was read




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


class UPFile(EBSDPatternFile):

  def __init__(self, path=None):
    EBSDPatternFile.__init__(self, path)
    self.file_type = 'UP'
    self.vendor = 'EDAX'
    self.bitdepth = None
    self.filePos = None  # file location in bytes where pattern data starts
    self.extraPatterns = None
    self.hexFlag = None


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
      self.nPatterns = np.int((Path(self.path).expanduser().stat().st_size - 16) / (self.patternW * self.patternH))
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

  def read_data(self,path=None,convertToFloat=False,patStartEnd = [0,-1],returnArrayOnly=False, bitdepth=None):  # readInterval=[0, -1], arrayOnly=False,
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

    type = 8
    if bitD == 8:
      type = np.uint8
    if bitD == 16:
      type = np.uint16

    try:
      f = open(Path(self.path).expanduser(), 'rb')
    except:
      print("File Not Found:", str(Path(self.path)))
      return -1

    f.seek(self.filePos)
    patStart = patStartEnd[0]
    patEnd = patStartEnd[-1]
    if patEnd == -1:
      patEnd = self.nPatterns
    nPatToRead = int(patEnd - patStart)
    nPerPat = self.patternW * self.patternH
    f.seek(nPerPat * patStart, 1)

    if convertToFloat == True:
      typeout = np.float32
    else:
      typeout = type
    patterns = np.zeros([nPatToRead, self.patternH, self.patternW], dtype=typeout)
    for p in np.arange(int(nPatToRead)):
      onePat = np.fromfile(f, dtype=type, count=nPerPat)
      onePat = onePat.reshape(self.patternH, self.patternW)
      patterns[p, :, :] = onePat.astype(typeout)

    patterns =np.flip(patterns,axis=-2)  # vertical flip of the axis vertical axis
    if returnArrayOnly == True:
      return patterns
    else: #package this up in an EBSDPatterns Object
      patsout = EBSDPatterns()
      patsout.vendor = self.vendor
      patsout.file = Path(self.path).expanduser()
      patsout.filetype = 'UP1' if (bitD == 8) else 'UP2'
      patsout.patternW = self.patternW
      patsout.patternH = self.patternH
      patsout.nFileCols = self.nCols
      patsout.nFileRows = self.nRows
      patsout.nPaterns = nPatToRead
      patsout.hexFlag = self.hexFlag
      patsout.xStep = self.xStep
      patsout.yStep = self.yStep
      patsout.patStartEnd = [patStart, patEnd]
      patsout.patterns = patterns
      return patsout


if __name__ == "__main__":
  file = '~/Desktop/SLMtest/scan2v3nlparl09sw7.up1'
  f = UPFile(file)
  #f.ReadHeader()
  pat = f.read_data()
