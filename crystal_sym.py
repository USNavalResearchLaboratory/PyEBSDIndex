import numpy as np
#import numba
import rotlib
PI = np.pi

def cubicsym_q(quatin=None, low = False):


  symops = np.zeros((24,4), dtype=np.float32)
  #identity
  symops [0,:] = np.array([1.0,0.0,0.0,0.0])
  #four fold
  cang1 = np.cos(0.25 * PI)
  sang1 = np.sin(0.25 * PI)
  cang2 = np.cos(0.75 * PI)
  sang2 = np.sin(0.75 * PI)
  #[001]
  symops[1,:] = np.array([cang1, 0.0,0.0,sang1])
  symops[2,:] = np.array([cang2,0.0,0.0,sang2])
  #[010]
  symops[3,:] = np.array([cang1,0.0,sang1,0.0])
  symops[4,:] = np.array([cang2,0.0,sang2,0.0])
  #[100]
  symops[5,:] = np.array([cang1,sang1,0.0,0.0])
  symops[6,:] = np.array([cang2,sang2,0.0,0.0])

  #two fold
  cang1 = np.cos(PI*0.5)
  sang1 = np.sin(PI*0.5)
  #[001]
  symops[7,:] = np.array([cang1,0.0,0.0,sang1])
  #[010]
  symops[8,:] = np.array([cang1,0.0,sang1,0.0])
  #[100]
  symops[9,:] = np.array([cang1,sang1,0.0,0.0])

  t = 1.0/np.sqrt(2.0)
  sang1 *= t
  #[-101]
  symops[10,:] = np.array([cang1,-sang1,0.0,sang1])
  # [0-11]
  symops[11,:] = np.array([cang1,0.0,-sang1,sang1])
  # [101]
  symops[12,:] = np.array([cang1,sang1,0.0,sang1])
  #[011]
  symops[13,:] = np.array([cang1,0.0,sang1,sang1])
  #[-110]
  symops[14,:] = np.array([cang1,-1*sang1,sang1,0.0])
  #[110]
  symops[15,:] = np.array([cang1,sang1,sang1,0.0])

  #three fold axis
  t = 1.0 / np.sqrt(3.0)
  cang1 = np.cos(1.0/3.0 * PI)
  sang1 = np.sin(1.0/3.0 * PI) * t
  cang2 = np.cos(2.0/3.0 * PI)
  sang2 = np.sin(2.0/3.0 * PI) * t
  #[111]
  symops[16,:] = np.array([cang1,sang1,sang1,sang1])
  symops[17,:] = np.array([cang2,sang2,sang2,sang2])
  #[-111]
  symops[18,:] = np.array([cang1,-sang1,sang1,sang1])
  symops[19,:] = np.array([cang2,-sang2,sang2,sang2])
  #[-1-11]
  symops[20,:] = np.array([cang1,-sang1,-sang1,sang1])
  symops[21,:] = np.array([cang2,-sang2,-sang2,sang2])
  #[1-11]
  symops[22,:] = np.array([cang1,sang1,-sang1,sang1])
  symops[23,:] = np.array([cang2,sang2,-sang2,sang2])

  if low:
    symops = symops[[0,7,8,9,16,17,18,19,20,21,22,23],:]

  if quatin is None:
    return symops
  else:
    qsym = rotlib.quat_multiply(symops,quatin)
    return qsym







def triclinic_q(quatin=None):
  symops = np.zeros((1,4),dtype=np.float32)
  # identity
  symops[0,:] = np.array([1.0,0.0,0.0,0.0])

  if quatin is None:
    return symops
  else:
    qsym = rotlib.quat_multiply(symops,quatin)
    return qsym