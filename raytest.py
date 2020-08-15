import ray
import numpy as np
import numba

@ray.remote
def testfun(x,y):
  #z = numbaadd(x,y)#x+y
  z = x+y
  # a = np.zeros([1000,60,60])
  # for j in range(1000):
  #   for i in range(1000):
  #     b = np.fft.fft(a[i,:,:])
  if z < 10:
    return z
  else:
    d = 15
    return z,d

@numba.jit(nopython=True,fastmath=True,cache=True)
def numbaadd(x,y):
  return x+y