from timeit import default_timer as timer

import numpy as np
import numba
import ray


@numba.jit(nopython=True,fastmath=True,cache=True,parallel=False)
def numbanpflip(a, pad, iter):
  for i in range(iter):
    flipl = np.flipud(a[:,-2 * pad[1]:])
    rnflip = a[:,0: 2 * pad[1]]
    rnflip = np.where(rnflip >= flipl,rnflip,flipl)
    a[:,0: 2 * pad[1]] = rnflip

    flipr = np.flipud(a[:,0: 2 * pad[1]])
    lnflip = a[:,-2 * pad[1]:]
    lnflip = np.where(lnflip >= flipr,lnflip,flipr)
    a[:,-2 * pad[1]:] = lnflip
  return a


@numba.jit(nopython=True,fastmath=True,cache=True,parallel=False)
def numbaflip(a, pad, iter):
  flipr = np.zeros((a.shape[0], 2*pad[1]), dtype = np.float32)
  flipl = np.zeros((a.shape[0], 2*pad[1]), dtype = np.float32)

  for i in range(iter):
    for j in range(pad[0]):
      for k in range(2*pad[1]):
        flipl[-j,k] = a[j,a.shape[1] - k-1]


    #flipl = np.flipud(a[:,-2 * pad[1]:])
    rnflip = a[:,0: 2 * pad[1]]
    rnflip = np.where(rnflip >= flipl,rnflip,flipl)
    a[:,0: 2 * pad[1]] = rnflip

    for j in range(pad[0]):
      for k in range(2*pad[1]):
        flipl[-j,k] = a[j,k]
    #flipr = np.flipud(a[:,0: 2 * pad[1]])
    lnflip = a[:,-2 * pad[1]:]
    lnflip = np.where(lnflip >= flipr,lnflip,flipr)
    a[:,-2 * pad[1]:] = lnflip
  return a

@ray.remote
def numbanpflip_wrap(b,pad,iter):
  a = b.copy()
  tic = timer()
  c = numbanpflip(a,pad,iter)
  print(timer()-tic)
  return c

@ray.remote
def numbaflip_wrap(b,pad,iter):
  a = b.copy()
  tic = timer()
  a = numbaflip(a,pad,iter)
  print(timer()-tic)
  return a

@ray.remote
def numpyflip(b,pad, iter):
  a = b.copy()
  tic = timer()
  for i in range(iter):
    flipl = np.flipud(a[:,-2*pad[1]:])
    rnflip = a[:, 0: 2*pad[1]]
    rnflip = np.where(rnflip >= flipl,rnflip, flipl)
    a[:, 0: 2 * pad[1]] = rnflip

    flipr = np.flipud(a[:,0: 2*pad[1]])
    lnflip = a[:, -2*pad[1]:]
    lnflip = np.where(lnflip >= flipr, lnflip,flipr)
    a[:, -2 * pad[1]:] = lnflip
  print(timer()-tic)
  return a

a = np.random.random((110, 200))
pad =np.array([10, 20])
iter = 1000
ncpu = 1
b = numbanpflip(a, pad, iter)
#ray.shutdown()
#ray.init()
# ar = ray.put(a)
# pr = ray.put(pad)
#
#wrk = [numpyflip.remote(a,pad, iter) for i in range(ncpu)]
#trash = ray.get(wrk)


