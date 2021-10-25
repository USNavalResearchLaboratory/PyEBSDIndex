from os import path, environ
import multiprocessing
import queue
import sys
import time
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import ray

from pyebsdindex import (
    band_detect,
    band_vote,
    ebsd_pattern,
    rotlib,
    tripletlib,
)
from pyebsdindex.EBSDImage import IPFcolor


if ray.__version__ < '1.1.0':  # this fixes an issue when runnning locally on a VPN
  ray.services.get_node_ip_address = lambda: '127.0.0.1'
else:
  ray._private.services.get_node_ip_address = lambda: '127.0.0.1'

RADEG = 180.0 / np.pi


def index_pats(patsIn=None,filename=None,filenameout=None,phaselist=['FCC'], \
               vendor=None,PC=None,sampleTilt=70.0,camElev=5.3, \
               bandDetectPlan=None,nRho=90,nTheta=180,tSigma=None,rSigma=None,rhoMaskFrac=0.1,nBands=9, \
               backgroundSub=False,patstart=0,npats=-1, \
               return_indexer_obj=False,ebsd_indexer_obj=None,clparams=None,verbose=0):
  pats = None
  if patsIn is None:
    pdim = None
  else:
    if patsIn is ebsd_pattern.EBSDPatterns:
      pats = patsIn.patterns
    if type(patsIn) is np.ndarray:
      pats = patsIn
    pdim = pats.shape[-2:]

  if ebsd_indexer_obj == None:
    indexer = EBSDIndexer(filename=filename,phaselist=phaselist, \
                          vendor=None,PC=PC,sampleTilt=sampleTilt,camElev=camElev, \
                          bandDetectPlan=bandDetectPlan, \
                          nRho=nRho,nTheta=nTheta,tSigma=tSigma,rSigma=rSigma,rhoMaskFrac=rhoMaskFrac,nBands=nBands,
                          patDim=pdim)
  else:
    indexer = ebsd_indexer_obj

  if filename is not None:
    indexer.update_file(filename)
  if pats is not None:
    if not np.all(indexer.bandDetectPlan.patDim == np.array(pdim)):
      indexer.update_file(patDim=pats.shape[-2:])

  if backgroundSub == True:
    indexer.bandDetectPlan.collect_background(fileobj=indexer.fID,patsIn=pats,nsample=1000)

  dataout,indxstart,indxend = indexer.index_pats(patsin=pats,patstart=patstart,npats=npats, \
                                                 clparams=clparams,verbose=verbose)

  if return_indexer_obj == False:
    return dataout
  else:
    return dataout,indexer


@ray.remote(num_cpus=1,num_gpus=1)
class IndexerRay():
  def __init__(self,actorid=0):
    sys.path.append(path.dirname(__file__))  # do this to help Ray find the program files
    import openclparam # do this to help Ray find the program files
    # device, context, queue, program, mf
    # self.dataout = None
    # self.indxstart = None
    # self.indxend = None
    # self.rate = None
    self.actorID = actorid
    self.openCLParams = None
    try:
      if sys.platform != 'darwin':  # linux with NVIDIA (unsure if it is the os or GPU type) is slow to make a
        self.openCLParams = openclparam.OpenClParam()
        self.openCLParams.gpu_id = self.actorID % self.openCLParams.ngpu

      else:  # MacOS handles GPU memory conflicts much better when the context is destroyed between each
        # run, and has very low overhead for making the context.
        #pass
        self.openCLParams = openclparam.OpenClParam()
        # self.openCLParams.gpu_id = 0
        # self.openCLParams.gpu_id = 1
        self.openCLParams.gpu_id = self.actorID % self.openCLParams.ngpu
    except:
      self.openCLParams = None

  def index_chunk_ray(self,pats=None,indexer=None,patstart=0,npats=-1):
    try:
      tic = timer()
      dataout,indxstart,npatsout = indexer.index_pats(patsin=pats,patstart=patstart,npats=npats,
                                                      clparams=self.openCLParams)
      rate = np.array([timer() - tic,npatsout])
      return dataout,indxstart,indxstart + npatsout,rate
    except:
      indxstart = patstart
      indxend = patstart + npats
      return None,indxstart,indxend,[-1,-1]
  # def readdata(self):
  #   return self.dataout, self.indxstart,self.indxend, self.rate
  #
  # def cleardata(self):
  #   #self.dataout = None
  #   self.indxstart = None
  #   self.indxend = None
  #   self.rate = None


def index_chunk_MP(pats=None,queues=None,lock=None,id=None):
  # queues[0]: messaging queue -- will initially get the indexer object, and the mode.  Will listen for a "Done" after that.
  # queues[1]: job queue
  # queues[2]: results queue

  indexer,mode = queues[0].get()
  workToDo = True
  while workToDo:
    tic = timer()

    try:
      # lock.acquire()
      if mode == 'filemode':
        pats = None
        patStart,patEnd,jid = queues[1].get(False)
      elif mode == 'memory':  # the patterns were stored in memory and sent within the job queue
        pats,patStart,patEnd,jid = queues[1].get(False)
    except queue.Empty:  # apparently queue.Empty is really unreliable.
      # lock.release()
      time.sleep(0.01)
    else:
      # lock.release()
      dataout,indxstart,indxend = indexer.index_pats(patsin=pats,patStart=patStart,patEnd=patEnd)
      rate = np.array([timer() - tic,indxend - indxstart])

      # try:
      queues[2].send((dataout,indxstart,indxend,rate,id,jid))
      # except:
      #  print("Unexpected error:",sys.exc_info()[0])

    # We will wait until an explicit "Done" comes from the main thread.
    try:
      wait_for_quit = queues[0].get(False)
      if wait_for_quit == 'Done':
        workToDo = False
        queues[2].close()
        return
    except queue.Empty:
      time.sleep(0.01)
  return


def index_chunk_MP2(pats=None,queues=None,lock=None,id=None):
  # queues[0]: messaging queue -- will initially get the indexer object, and the mode.  Will listen for a "Done" after that.
  # queues[1]: job queue
  # queues[2]: results queue

  indexer,mode = queues[0].get()
  # print('here', id)
  workToDo = True
  while workToDo:
    tic = timer()
    lock[0].acquire()
    # print(queues[1].empty())
    if queues[1].empty() == False:
      if mode == 'filemode':
        pats = None
        patStart,patEnd,jid = queues[1].get()
      elif mode == 'memory':  # the patterns were stored in memory and sent within the job queue
        pats,patStart,patEnd,jid = queues[1].get()
      lock[0].release()
      dataout,indxstart,indxend = indexer.index_pats(patsin=pats,patStart=patStart,patEnd=patEnd)
      rate = np.array([timer() - tic,indxend - indxstart])

      queues[2].send((dataout,indxstart,indxend,rate,id,jid))
    else:
      lock[0].release()
      time.sleep(0.01)

    # We will wait until an explicit "Done" comes from the main thread.
    lock[1].acquire()
    if queues[0].empty() == False:
      wait_for_quit = queues[0].get()
      lock[1].release()
      if wait_for_quit == 'Done':
        workToDo = False
        queues[2].close()
        # queues[0].close()

        return
    else:
      lock[1].release()
      time.sleep(0.01)
  return


def index_pats_distributed(patsIn=None,filename=None,filenameout=None,phaselist=['FCC'], \
                           vendor=None,PC=None,sampleTilt=70.0,camElev=5.3, \
                           peakDetectPlan=None,nRho=90,nTheta=180,tSigma=None,rSigma=None,rhoMaskFrac=0.1,nBands=9,
                           patstart=0,npats=-1,chunksize=256,ncpu=-1,
                           return_indexer_obj=False,ebsd_indexer_obj=None,keep_log=False):
  n_cpu_nodes = int(multiprocessing.cpu_count())  # int(sum([ r['Resources']['CPU'] for r in ray.nodes()]))
  if ncpu != -1:
    n_cpu_nodes = ncpu

  try:
    clparam = openclparam.OpenClParam()
    if clparam.gpu is None:
      ngpu = 0
      ngpupnode = 0
    else:
      ngpu = len(clparam.gpu)
      ngpupnode = ngpu / n_cpu_nodes
  except:
    ngpu = 0
    ngpupnode = 0

  ray.shutdown()

  ray.init(num_cpus=n_cpu_nodes,num_gpus=ngpu,_system_config={"maximum_gcs_destroyed_actor_cached_count": n_cpu_nodes})
  # ray.init(num_cpus=n_cpu_nodes,num_gpus=ngpu)
  pats = None
  if patsIn is None:
    pdim = None
  else:
    if patsIn is ebsd_pattern.EBSDPatterns:
      pats = patsIn.patterns
    if type(patsIn) is np.ndarray:
      pats = patsIn
    pdim = pats.shape[-2:]

  if ebsd_indexer_obj == None:
    indexer = EBSDIndexer(filename=filename,phaselist=phaselist, \
                          vendor=None,PC=PC,sampleTilt=sampleTilt,camElev=camElev, \
                          bandDetectPlan=peakDetectPlan, \
                          nRho=nRho,nTheta=nTheta,tSigma=tSigma,rSigma=rSigma,rhoMaskFrac=rhoMaskFrac,nBands=nBands,
                          patDim=pdim)
  else:
    indexer = ebsd_indexer_obj

  if filename is not None:
    indexer.update_file(filename)
  else:
    indexer.update_file(patDim=pats.shape[-2:])

  # differentiate between getting a file to index or an array
  # Need to index one pattern to make sure the indexer object is fully initiated before
  #   placing in shared memory store.
  mode = 'memorymode'
  if pats is None:
    mode = 'filemode'
    temp,indexer = index_pats(patstart=0,npats=1,return_indexer_obj=True,ebsd_indexer_obj=indexer)

  if mode == 'filemode':
    npatsTotal = indexer.fID.nPatterns
  else:
    pshape = pats.shape
    if len(pshape) == 2:
      npatsTotal = 1
      pats = pats.reshape([1,pshape[0],pshape[1]])
    else:
      npatsTotal = pshape[0]
    temp,indexer = index_pats(pats[0,:,:],patstart=0,npats=1,return_indexer_obj=True,ebsd_indexer_obj=indexer)

  if patstart < 0:
    patstart = npatsTotal - patstart
  if npats <= 0:
    npats = npatsTotal - patstart

  # place indexer obj in shared memory store so all workers can use it.
  remote_indexer = ray.put(indexer)
  # set up the jobs
  njobs = (np.ceil(npats / chunksize)).astype(np.long)
  # p_indx_start = [i*chunksize+patStart for i in range(njobs)]
  # p_indx_end = [(i+1)*chunksize+patStart for i in range(njobs)]
  # p_indx_end[-1] = npats+patStart
  p_indx_start_end = [[i * chunksize + patstart,(i + 1) * chunksize + patstart,chunksize] for i in range(njobs)]
  p_indx_start_end[-1][1] = npats + patstart
  p_indx_start_end[-1][2] = p_indx_start_end[-1][1] - p_indx_start_end[-1][0]

  if njobs < n_cpu_nodes:
    n_cpu_nodes = njobs

  nPhases = len(indexer.phaseLib)
  dataout = np.zeros((nPhases + 1,npats),dtype=indexer.dataTemplate)
  ndone = 0
  nsubmit = 0
  tic0 = timer()
  npatsdone = 0.0

  if keep_log is True:
    newline = '\n'
  else:
    newline = '\r'
  if mode == 'filemode':
    # send out the first batch
    workers = []
    jobs = []
    timers = []
    jobs_indx = []
    chunkave = 0.0
    for i in range(n_cpu_nodes):
      job_pstart_end = p_indx_start_end.pop(0)
      workers.append(IndexerRay.options(num_cpus=1,num_gpus=ngpupnode).remote(i))
      jobs.append(workers[i].index_chunk_ray.remote(pats=None,indexer=remote_indexer, \
                                                    patstart=job_pstart_end[0],npats=job_pstart_end[2]))
      nsubmit += 1
      timers.append(timer())
      time.sleep(0.01)
      jobs_indx.append(job_pstart_end[:])

    while ndone < njobs:
      # toc = timer()
      wrker,busy = ray.wait(jobs,num_returns=1,timeout=None)

      # print("waittime: ",timer() - toc)
      jid = jobs.index(wrker[0])
      try:
        wrkdataout,indxstr,indxend,rate = ray.get(wrker[0])
      except:
        # print('a death has occured')
        indxstr = jobs_indx[jid][0]
        indxend = jobs_indx[jid][1]
        rate = [-1,-1]
      if rate[0] >= 0:  # job finished as expected

        ticp = timers[jid]
        dataout[:,indxstr - patstart:indxend - patstart] = wrkdataout
        npatsdone += rate[1]
        ndone += 1

        ratetemp = len(workers) * (rate[1]) / (timer() - ticp)
        chunkave += ratetemp
        totalave = npatsdone / (timer() - tic0)
        # print('Completed: ',str(indxstr),' -- ',str(indxend), '  ', npatsdone/(timer()-tic) )

        toc0 = timer() - tic0
        if keep_log is False:
          print('                                                                                             ',
                end='\r')
          time.sleep(0.00001)
        print('Completed: ',str(indxstr),' -- ',str(indxend),'  PPS:',"{:.0f}".format(ratetemp) + ';' +
              "{:.0f}".format(chunkave / ndone) + ';' + "{:.0f}".format(totalave),
              '  ',"{:.0f}".format((ndone / njobs) * 100) + '%',
              "{:.0f};".format(toc0) + "{:.0f}".format((njobs - ndone) / ndone * toc0) + ' running;remaining(s)',
              end=newline)

        if len(p_indx_start_end) > 0:
          job_pstart_end = p_indx_start_end.pop(0)
          jobs[jid] = workers[jid].index_chunk_ray.remote(pats=None,indexer=remote_indexer,
                                                          patstart=job_pstart_end[0],npats=job_pstart_end[2])
          nsubmit += 1
          timers[jid] = timer()
          jobs_indx[jid] = job_pstart_end[:]
        else:
          del jobs[jid]
          del workers[jid]
          del timers[jid]
          del jobs_indx[jid]

      else:  # something bad happened.  Put the job back on the queue and kill this worker
        p_indx_start_end.append([indxstr,indxend,indxend - indxstr])
        del jobs[jid]
        del workers[jid]
        del timers[jid]
        del jobs_indx[jid]
        if len(workers) < 1:  # rare case that we have killed all workers...
          job_pstart_end = p_indx_start_end.pop(0)
          workers.append(IndexerRay.options(num_cpus=1,num_gpus=ngpupnode).remote(jid))
          jobs.append(workers[0].index_chunk_ray.remote(pats=None,indexer=remote_indexer, \
                                                        patstart=job_pstart_end[0],npats=job_pstart_end[2]))
          nsubmit += 1
          timers.append(timer())
          time.sleep(0.01)
          jobs_indx.append(job_pstart_end[:])

  if mode == 'memorymode':
    workers = []
    jobs = []
    timers = []
    jobs_indx = []
    chunkave = 0.0
    for i in range(n_cpu_nodes):
      job_pstart_end = p_indx_start_end.pop(0)
      workers.append(IndexerRay.options(num_cpus=1,num_gpus=ngpupnode).remote(i))
      jobs.append(workers[i].index_chunk_ray.remote(pats=pats[job_pstart_end[0]:job_pstart_end[1],:,:],
                                                    indexer=remote_indexer, \
                                                    patstart=job_pstart_end[0],npats=job_pstart_end[2]))
      nsubmit += 1
      timers.append(timer())
      jobs_indx.append(job_pstart_end)
      time.sleep(0.01)

    # workers = [index_chunk.remote(pats = None, indexer = remote_indexer, patStart = p_indx_start[i], patEnd = p_indx_end[i]) for i in range(n_cpu_nodes)]
    # nsubmit += n_cpu_nodes

    while ndone < njobs:
      toc = timer()
      wrker,busy = ray.wait(jobs,num_returns=1,timeout=None)
      jid = jobs.index(wrker[0])
      # print("waittime: ",timer() - toc)
      try:
        wrkdataout,indxstr,indxend,rate = ray.get(wrker[0])
      except:
        indxstr = jobs_indx[jid][0]
        indxend = jobs_indx[jid][1]
        rate = [-1,-1]
      if rate[0] >= 0:
        ticp = timers[jid]
        dataout[:,indxstr - patstart:indxend - patstart] = wrkdataout
        npatsdone += rate[1]
        ratetemp = len(workers) * (rate[1]) / (timer() - ticp)
        chunkave += ratetemp
        totalave = npatsdone / (timer() - tic0)
        # print('Completed: ',str(indxstr),' -- ',str(indxend), '  ', npatsdone/(timer()-tic) )
        ndone += 1
        toc0 = timer() - tic0
        if keep_log is False:
          print('                                                                                             ',
                end='\r')
          time.sleep(0.0001)
        print('Completed: ',str(indxstr),' -- ',str(indxend),'  PPS:',"{:.0f}".format(ratetemp) + ';' +
              "{:.0f}".format(chunkave / ndone) + ';' + "{:.0f}".format(totalave),
              '  ',"{:.0f}".format((ndone / njobs) * 100) + '%',
              "{:.0f};".format(toc0) + "{:.0f}".format((njobs - ndone) / ndone * toc0) + ' running;remaining(s)',
              end=newline)

        if len(p_indx_start_end) > 0:
          job_pstart_end = p_indx_start_end.pop(0)
          jobs[jid] = workers[jid].index_chunk_ray.remote(pats=pats[job_pstart_end[0]:job_pstart_end[1],:,:]
                                                          ,indexer=remote_indexer,
                                                          patstart=job_pstart_end[0],npats=job_pstart_end[2])
          nsubmit += 1
          timers[jid] = timer()
          jobs_indx[jid] = job_pstart_end
        else:
          del jobs[jid]
          del workers[jid]
          del timers[jid]
          del jobs_indx[jid]
      else:  # something bad happened.  Put the job back on the queue and kill this worker
        p_indx_start_end.append([indxstr,indxend,indxend - indxstr])
        del jobs[jid]
        del workers[jid]
        del timers[jid]
        del jobs_indx[jid]
        if len(workers) < 1:  # rare case that we have killed all workers...
          job_pstart_end = p_indx_start_end.pop(0)
          workers.append(IndexerRay.options(num_cpus=1,num_gpus=ngpupnode).remote(jid))
          jobs.append(workers[0].index_chunk_ray.remote(pats=pats[job_pstart_end[0]:job_pstart_end[1],:,:],
                                                        indexer=remote_indexer, \
                                                        patstart=job_pstart_end[0],npats=job_pstart_end[2]))
          nsubmit += 1
          timers.append(timer())
          jobs_indx.append(job_pstart_end)

    del jobs
    del workers
    del timers
    # # send out the first batch
    # workers = [index_chunk_ray.remote(pats=pats[p_indx_start[i]:p_indx_end[i],:,:],indexer=remote_indexer,patStart=p_indx_start[i],patEnd=p_indx_end[i]) for i
    #            in range(n_cpu_nodes)]
    # nsubmit += n_cpu_nodes
    #
    # while ndone < njobs:
    #   wrker,busy = ray.wait(workers,num_returns=1,timeout=None)
    #   wrkdataout,indxstr,indxend, rate = ray.get(wrker[0])
    #   dataout[indxstr:indxend] = wrkdataout
    #   print('Completed: ',str(indxstr),' -- ',str(indxend))
    #   workers.remove(wrker[0])
    #   ndone += 1
    #
    #   if nsubmit < njobs:
    #     workers.append(index_chunk_ray.remote(pats=pats[p_indx_start[nsubmit]:p_indx_end[nsubmit],:,:],indexer=remote_indexer,patStart=p_indx_start[nsubmit],
    #                                       patEnd=p_indx_end[nsubmit]))
    #     nsubmit += 1

  ray.shutdown()
  if return_indexer_obj:
    return dataout,indexer
  else:
    return dataout


def index_patsMP(pats=None,filename=None,filenameout=None,phaselist=['FCC'], \
                 vendor=None,PC=None,sampleTilt=70.0,camElev=5.3, \
                 peakDetectPlan=None,nRho=90,nTheta=180,tSigma=None,rSigma=None,rhoMaskFrac=0.1,nBands=9,
                 patStart=0,patEnd=-1,chunksize=1000,ncpu=-1,
                 return_indexer_obj=False,ebsd_indexer_obj=None):
  n_cpu_nodes = int(multiprocessing.cpu_count())  # int(sum([ r['Resources']['CPU'] for r in ray.nodes()]))
  if ncpu != -1:
    n_cpu_nodes = ncpu

  if pats is None:
    pdim = None
  else:
    pdim = pats.shape[-2:]
  if ebsd_indexer_obj == None:
    indexer = EBSDIndexer(filename=filename,phaselist=phaselist, \
                          vendor=None,PC=PC,sampleTilt=sampleTilt,camElev=camElev, \
                          bandDetectPlan=peakDetectPlan, \
                          nRho=nRho,nTheta=nTheta,tSigma=tSigma,rSigma=rSigma,rhoMaskFrac=rhoMaskFrac,nBands=nBands,
                          patDim=pdim)
  else:
    indexer = ebsd_indexer_obj

  # differentiate between getting a file to index or an array
  # Need to index one pattern to make sure the indexer object is fully initiated before
  #   placing in shared memory store.
  mode = 'memorymode'
  if pats is None:
    mode = 'filemode'
    # just make sure indexer is fully initialized
    temp,indexer = index_pats(patStart=0,patEnd=1,return_indexer_obj=True,ebsd_indexer_obj=indexer)

  if mode == 'filemode':
    npats = indexer.fID.nPatterns
  else:
    pshape = pats.shape
    if len(pshape) == 2:
      npats = 1
      pats = pats.reshape([1,pshape[0],pshape[1]])
    else:
      npats = pshape[0]
    temp,indexer = index_pats(pats[0,:,:],patStart=0,patEnd=1,return_indexer_obj=True,ebsd_indexer_obj=indexer)

  if patEnd == 0:
    patEnd = 1
  if (patStart != 0) or (patEnd > 0):
    npats = patEnd - patStart

  # set up the jobs
  njobs = (np.ceil(npats / chunksize)).astype(np.long)
  p_indx_start = [i * chunksize + patStart for i in range(njobs)]
  p_indx_end = [(i + 1) * chunksize + patStart for i in range(njobs)]
  p_indx_end[-1] = npats + patStart
  if njobs < n_cpu_nodes:
    n_cpu_nodes = njobs

  dataout = np.zeros((npats),dtype=indexer.dataTemplate)
  returnqueue = []

  ndone = 0
  tic = timer()
  npatsdone = 0.0
  toc = 0.0
  messagequeue = []  # each worker will get its own message queue
  messagelock = []
  workers = []
  rateave = 0.0
  readlock = multiprocessing.Lock()
  jobqueue = multiprocessing.SimpleQueue()
  jcheck = np.ones(njobs)
  if mode == 'filemode':

    tic = timer()
    for i in range(n_cpu_nodes):  # launch the workers
      messagequeue.append(multiprocessing.SimpleQueue())
      messagelock.append(multiprocessing.Lock())
      returnqueue.append(multiprocessing.Pipe())
      workers.append(multiprocessing.Process(target=index_chunk_MP2,args=(None, \
                                                                          [messagequeue[i],jobqueue,returnqueue[i][1]],
                                                                          [readlock,messagelock[i]],i)))
      workers[i].start()

    readlock.acquire()
    for i in range(njobs):
      jobqueue.put((p_indx_start[i],p_indx_end[i],i))
    readlock.release()
    for i in range(n_cpu_nodes):
      messagelock[i].acquire()
      messagequeue[i].put((indexer,mode))
      messagelock[i].release()
    ndone = 0
    ntry = -1
    tryThresh = 400
    ntrysum = 0
    while ndone < njobs:
      # check if work is done.

      for i in range(n_cpu_nodes):
        # try:
        if (returnqueue[i][0]).poll():
          wrkdataout,indxstr,indxend,rate,wrkID,jid = returnqueue[i][0].recv()
          dataout[indxstr:indxend] = wrkdataout
          npatsdone += rate[1]
          ratetemp = n_cpu_nodes * (rate[1]) / (rate[0])
          rateave += ratetemp
          jcheck[jid] = 0
          ndone += 1  # np.count_nonzero(jcheck == 0)
          ntrysum += np.max((ntry,0))
          tryThresh = np.int(ntrysum / np.float(ndone))
          # print( tryThresh, ntry)
          ntry = 0
          print('Completed: ',str(indxstr),' -- ',str(indxend),'  ',np.int(ratetemp),'  ',
                np.int(npatsdone / (timer() - tic)),np.int(npatsdone / npats * 100),'%  ',njobs,ndone)
          whzero = jcheck.nonzero()[0]
          # if whzero.size < 10:
          #  print(jcheck.nonzero()[0])
        # except queue.Empty: # there is a strange race condition that I can not figure out.
        # else:
      time.sleep(0.005)
      ntry += 1
      if ntry > (10 * tryThresh):
        ntry = 0
        whbad = jcheck.nonzero()[0]
        if whbad.size > 0:
          print('adding back job',whbad[0])
          jobqueue.put((p_indx_start[whbad[0]],p_indx_end[whbad[0]],whbad[0]))
        # except:
        #  print("Unexpected error:",sys.exc_info()[0])

  if mode == 'memorymode':

    nsubmit = 0
    n2queue = n_cpu_nodes + 4 if (njobs > (n_cpu_nodes + 4)) else n_cpu_nodes  # going to pre-load a few extra jobs.
    # memory concerns are the only reason we do not load all into the queue at once.
    for i in range(n2queue):
      jobqueue.put(((pats[p_indx_start[nsubmit]:p_indx_end[nsubmit],:,:],p_indx_start[nsubmit],p_indx_end[nsubmit])))
      nsubmit += 1

    for i in range(n_cpu_nodes):  # launch the workers
      toc = timer()
      messagequeue.append(multiprocessing.Queue())
      # print('New q', timer()-toc)
      toc = timer()
      messagequeue[i].put((indexer,mode))  # provide the indexer and where the patterns are coming from
      workers.append(multiprocessing.Process(target=index_chunk_MP,args=(None, \
                                                                         [messagequeue[i],jobqueue,returnqueue],i)))
      workers[i].start()

    while ndone < njobs:
      # check if work is done.
      try:
        wrkdataout,indxstr,indxend,rate,wrkID = returnqueue.get(False)
        dataout[indxstr:indxend] = wrkdataout
        npatsdone += rate[1]
        ratetemp = n_cpu_nodes * (rate[1]) / (rate[0])
        rateave += ratetemp
        ndone += 1
        print('Completed: ',str(indxstr),' -- ',str(indxend),'  ',np.int(ratetemp),'  ',
              np.int(npatsdone / (timer() - tic)),np.int(npatsdone / npats * 100),'%  ',njobs,ndone)
        if nsubmit < njobs:  # still more to do.
          jobqueue.put(
            ((pats[p_indx_start[nsubmit]:p_indx_end[nsubmit],:,:],p_indx_start[nsubmit],p_indx_end[nsubmit])))
          nsubmit += 1
      except queue.Empty:
        pass

  for i in range(n_cpu_nodes):
    messagelock[i].acquire()
    messagequeue[i].put('Done')
    messagequeue[i].put('Done')
    messagequeue[i].put('Done')
    messagelock[i].release()
  for i in range(n_cpu_nodes):
    workers[i].terminate()
    workers[i].join()
    workers[i].close()

  if return_indexer_obj:
    return dataout,indexer
  else:
    return dataout


class EBSDIndexer():
  def __init__(self,filename=None,phaselist=['FCC'], \
               vendor=None,PC=None,sampleTilt=70.0,camElev=5.3, \
               bandDetectPlan=None,nRho=90,nTheta=180,tSigma=1.0,rSigma=1.2,rhoMaskFrac=0.15,nBands=9,patDim=None):
    self.filein = filename
    if self.filein is not None:
      self.fID = ebsd_pattern.get_pattern_file_obj(self.filein)
    else:
      self.fID = None

    # self.fileout = filenameout
    # if self.fileout is None:
    #   self.fileout = str.lower(Path(self.filein).stem)+'.ang'

    self.phaselist = phaselist
    self.phaseLib = []
    for ph in self.phaselist:
      self.phaseLib.append(band_vote.BandVote(tripletlib.triplib(libType=ph)))

    self.vendor = 'EDAX'
    if vendor is None:
      if self.fID is not None:
        self.vendor = self.fID.vendor
    else:
      if vendor is not None:
        self.vendor = vendor

    if PC is None:
      self.PC = np.array([0.471659,0.675044,0.630139])  # a default value
    else:
      self.PC = np.asarray(PC)

    self.PCcorrectMethod = None
    self.PCcorrectParam = None

    self.sampleTilt = sampleTilt
    self.camElev = camElev

    if bandDetectPlan is None:
      self.bandDetectPlan = band_detect.BandDetect(nRho=nRho,nTheta=nTheta, \
                                                   tSigma=tSigma,rSigma=rSigma, \
                                                   rhoMaskFrac=rhoMaskFrac,nBands=nBands)
    else:
      self.bandDetectPlan = bandDetectPlan

    if self.fID is not None:
      self.bandDetectPlan.band_detect_setup(patDim=[self.fID.patternW,self.fID.patternH])
    else:
      if patDim is not None:
        self.bandDetectPlan.band_detect_setup(patDim=patDim)

    self.dataTemplate = np.dtype([('quat',np.float32,(4)),('iq',np.float32), \
                                  ('pq',np.float32),('cm',np.float32),('phase',np.int32), \
                                  ('fit',np.float32),('nmatch',np.int32),('matchattempts',np.int32,(2))])

  def update_file(self,filename=None,patDim=np.array([120,120],dtype=np.int32)):
    if filename is None:
      self.filein = None
      self.bandDetectPlan.band_detect_setup(patDim=patDim)
    else:
      self.filein = filename
      self.fID = ebsd_pattern.get_pattern_file_obj(self.filein)
      self.bandDetectPlan.band_detect_setup(patDim=[self.fID.patternW,self.fID.patternH])

  def index_pats(self,patsin=None,patstart=0,npats=-1,clparams=None,PC=[None,None,None],verbose=0):
    tic = timer()

    if patsin is None:
      pats = self.fID.read_data(returnArrayOnly=True,patStartCount=[patstart,npats],convertToFloat=True)
    else:
      pshape = patsin.shape
      if len(pshape) == 2:
        pats = np.reshape(patsin,(1,pshape[0],pshape[1]))
      else:
        pats = patsin  # [patStart:patEnd, :,:]
      pshape = pats.shape

      if self.bandDetectPlan.patDim is None:
        self.bandDetectPlan.band_detect_setup(patDim=pshape[1:3])
      else:
        if np.all((np.array(pshape[1:3]) - self.bandDetectPlan.patDim) == 0):
          self.bandDetectPlan.band_detect_setup(patDim=pshape[1:3])

    if self.bandDetectPlan.patDim is None:
      self.bandDetectPlan.band_detect_setup(patterns=pats)

    npoints = pats.shape[0]
    if npats == -1:
      npats = npoints

    # print(timer() - tic)
    tic = timer()
    bandData = self.bandDetectPlan.find_bands(pats,clparams=clparams,verbose=verbose)

    if PC[0] is None:
      PC_0 = self.PC
    else:
      PC_0 = PC
    bandNorm = self.bandDetectPlan.radon2pole(bandData,PC=PC_0,vendor=self.vendor)
    # print('Find Band: ', timer() - tic)

    # return bandNorm,patStart,patEnd
    tic = timer()
    # bv = []
    # for tl in self.phaseLib:
    #  bv.append(band_vote.BandVote(tl))
    nPhases = len(self.phaseLib)
    q = np.zeros((nPhases,npoints,4))
    indxData = np.zeros((nPhases + 1,npoints),dtype=self.dataTemplate)

    indxData['phase'] = -1
    indxData['fit'] = 180.0
    for i in range(npoints):

      bandNorm1 = bandNorm[i,:,:]
      bDat1 = bandData[i,:]
      whgood = np.nonzero(bDat1['max'] > -1.0e6)[0]
      if whgood.size >= 3:
        bDat1 = bDat1[whgood]
        bandNorm1 = bandNorm1[whgood,:]
        indxData['pq'][0:nPhases,i] = np.sum(bDat1['max'],axis=0)

        for j in range(len(self.phaseLib)):
          avequat,fit,cm,bandmatch,nMatch,matchAttempts = self.phaseLib[j].tripvote(bandNorm1,goNumba=True)
          # avequat,fit,cm,bandmatch,nMatch, matchAttempts = self.phaseLib[j].pairVoteOrientation(bandNorm1,goNumba=True)
          if nMatch >= 3:
            fitmetric = nMatch * cm
            q[j,i,:] = avequat
            indxData['fit'][j,i] = fit
            indxData['cm'][j,i] = cm
            indxData['phase'][j,i] = j
            indxData['nmatch'][j,i] = nMatch
            indxData['matchattempts'][j,i] = matchAttempts
          if nMatch >= 9:
            break

    qref2detect = self.refframe2detector()
    q = q.reshape(nPhases * npoints,4)
    q = rotlib.quat_multiply(q,qref2detect)
    q = q.reshape(nPhases,npoints,4)
    indxData['quat'][0:nPhases,:,:] = q
    if nPhases > 1:
      for j in range(nPhases - 1):
        indxData[-1,:] = np.where((indxData[j,:]['cm'] * indxData[j,:]['nmatch']) >
                                  ((indxData[j + 1,:]['cm'] * indxData[j + 1,:]['nmatch'])),
                                  indxData[j,:],indxData[j + 1,:])
    else:
      indxData[-1,:] = indxData[0,:]

    if verbose > 0:
      print('Band Vote Time: ',timer() - tic)
    return indxData,patstart,npats

  def refframe2detector(self):
    if self.vendor == 'EDAX':
      q0 = np.array([np.sqrt(2.0) * 0.5,0.0,0.0,-1.0 * np.sqrt(2.0) * 0.5])
      tiltang = -1.0 * (90.0 - self.sampleTilt + self.camElev) / RADEG
      q1 = np.array([np.cos(tiltang * 0.5),np.sin(tiltang * 0.5),0.0,0.0])
      quatref2detect = rotlib.quat_multiply(q1,q0)

    return quatref2detect

  def pcCorrect(self,xy=[0.0,0.0]):  # at somepoint we will put some methods here for correcting the PC
    # depending on the location within the scan.  Need to correct band_detect.radon2pole to accept mulitple
    # PC for each point
    pass


def __main__(file=None,ncpu=-1):
  print('Hello')
  if file is None:
    file = '~/Desktop/SLMtest/scan2v3nlparl09sw7.up1'

  dat1,indxer = index_pats(filename=file,
                           patstart=0,npats=1,return_indexer_obj=True,
                           nTheta=180,nRho=90,
                           tSigma=1.0,rSigma=1.2,rhoMaskFrac=0.1,nBands=9, \
                           phaselist=['FCC'])

  dat = index_pats_distributed(filename=file,patstart=0,npats=-1,
                               chunksize=1008,ncpu=ncpu,ebsd_indexer_obj=indxer)
  imshape = (indxer.fID.nRows,indxer.fID.nCols)
  ipfim = IPFcolor.ipf_color_cubic(dat['quat']).reshape(imshape[0],imshape[1],3);
  plt.imshow(ipfim)