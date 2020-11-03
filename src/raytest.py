import ray
ray.services.get_node_ip_address = lambda: '127.0.0.1'
import numpy as np
import numba
import time
from timeit import default_timer as timer
#1from ebsd_index import index_chunk

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

@ray.remote(num_cpus=1)
def donothing(pat = None, goon = None, start = 0, end = -1):
  time.sleep(10)
  return 1, start, end

def testqueue(ncpu):
  njobs = 400
  chunksize = 1000
  p_indx_start = [i * chunksize for i in range(njobs)]
  p_indx_end = [(i + 1) * chunksize for i in range(njobs)]
  ray.shutdown()
  ray.init()
  ndone = 0
  nsubmit = 0
  workers = []
  tic = timer()
  for i in range(ncpu):
    workers.append( donothing.remote(start = p_indx_start[i],end = p_indx_end[i]))
    #workers.append(index_chunk.remote(patStart=p_indx_start[nsubmit],patEnd=p_indx_end[nsubmit]))
    nsubmit += 1
  while ndone < njobs:

    wrker,busy = ray.wait(workers,num_returns=1,timeout=None)
    tic2 = timer()
    dave, indxstr,indxend = ray.get(wrker[0])
    print('Completed: ',str(indxstr),' -- ',str(indxend),'  ',str(indxend / (timer() - tic)),len(workers))
    # print('Completed:', str(p_indx_end[ndone]),str(p_indx_end[ndone]/(timer()-tic)) )
    workers.remove(wrker[0])
    ndone += 1
    if nsubmit < njobs:
      # pats = indexer.fID.read_data(convertToFloat=True,patStartEnd=[p_indx_start[nsubmit],p_indx_end[nsubmit]],
      #                             returnArrayOnly=True)
      # workers.append(index_chunk.remote(pats = pats, indexer = remote_indexer, patStart = p_indx_start[nsubmit], patEnd = p_indx_end[nsubmit]))
      workers.append(donothing.remote(start = p_indx_start[nsubmit],end = p_indx_end[nsubmit]))
      #workers.append(index_chunk.remote(patStart=p_indx_start[nsubmit],patEnd=p_indx_end[nsubmit]))
      time.sleep(0.048)
      nsubmit += 1
      print("time to launch: ",timer() - tic2)

  print(timer()-tic)
  ray.shutdown()
