import numpy as np
import pyopencl as cl
import ray
if ray.__version__ < '1.1.0': # this fixes an issue when runnning locally on a VPN
  ray.services.get_node_ip_address = lambda: '127.0.0.1'
else:
  ray._private.services.get_node_ip_address = lambda: '127.0.0.1'
from pathlib import Path
from os import path
import multiprocessing
import queue
import ebsd_pattern
import band_detect2
import band_vote
#import band_vote_org
#import band_voteEX
import rotlib
import tripletlib
from timeit import default_timer as timer
import time
RADEG = 180.0/np.pi
import traceback



def index_pats(pats = None,filename=None,filenameout=None,phaselist=['FCC'], \
               vendor=None,PC = None,sampleTilt=70.0,camElev = 5.3, \
               bandDetectPlan = None,nRho=90,nTheta=180,tSigma= None,rSigma=None,rhoMaskFrac=0.1,nBands=9, \
               patStart = 0,patEnd = -1, \
               return_indexer_obj = False,ebsd_indexer_obj = None, clparams = [None, None, None, None]):

  if pats is None:
    pdim = None
  else:
    pdim = pats.shape[-2:]
  if ebsd_indexer_obj == None:
    indexer = EBSDIndexer(filename=filename,phaselist=phaselist, \
                          vendor=None,PC = PC,sampleTilt=sampleTilt,camElev = camElev, \
                          bandDetectPlan= bandDetectPlan, \
                          nRho=nRho,nTheta=nTheta,tSigma= tSigma,rSigma=rSigma,rhoMaskFrac=rhoMaskFrac,nBands=nBands,patDim = pdim)
  else:
    indexer = ebsd_indexer_obj


  dataout, indxstart, indxend = indexer.index_pats(patsin=pats, patStart=patStart, patEnd=patEnd, clparams = clparams)

  if return_indexer_obj == False:
    return dataout
  else:
    return dataout, indexer


@ray.remote(num_cpus=1, num_gpus=1)
class IndexerRay():
  def __init__(self):
    #device, context, queue, program, mf
    #self.dataout = None
    #self.indxstart = None
    #self.indxend = None
    #self.rate = None
    self.openCLParams = [None, None, None, None, None]
    try:
      self.openCLParams[0] = cl.get_platforms()[0].get_devices(device_type=cl.device_type.GPU)
      self.openCLParams[1] = cl.Context(devices = {self.openCLParams[0][0]})
      self.openCLParams[2] = None
      kernel_location = path.dirname(__file__)
      self.openCLParams[3] = cl.Program(self.openCLParams[1] ,
                                        open(path.join(kernel_location,'clkernels.cl')).read()).build()
      self.openCLParams[4] = cl.mem_flags
    except:
      self.openCLParams[0] = None
      self.openCLParams[1] = None
      self.openCLParams[2] = None
      self.openCLParams[3] = None
      self.openCLParams[4] = None



  def index_chunk_ray(self, pats = None, indexer = None, patStart = 0, patEnd = -1 ):
    tic = timer()
    dataout,indxstart,indxend = indexer.index_pats(patsin=pats,patStart=patStart,patEnd=patEnd, clparams = self.openCLParams)
    rate = np.array([timer()-tic, indxend-indxstart])
    return dataout, indxstart,indxend, rate

  # def readdata(self):
  #   return self.dataout, self.indxstart,self.indxend, self.rate
  #
  # def cleardata(self):
  #   #self.dataout = None
  #   self.indxstart = None
  #   self.indxend = None
  #   self.rate = None


def index_chunk_MP(pats = None, queues = None,lock=None, id = None):
  #queues[0]: messaging queue -- will initially get the indexer object, and the mode.  Will listen for a "Done" after that.
  #queues[1]: job queue
  #queues[2]: results queue

  indexer, mode = queues[0].get()
  workToDo = True
  while workToDo:
    tic = timer()

    try:
      #lock.acquire()
      if mode == 'filemode':
        pats = None
        patStart,patEnd, jid = queues[1].get(False)
      elif mode == 'memory': # the patterns were stored in memory and sent within the job queue
        pats, patStart,patEnd, jid = queues[1].get(False)
    except queue.Empty: # apparently queue.Empty is really unreliable.
      #lock.release()
      time.sleep(0.01)
    else:
      #lock.release()
      dataout,indxstart,indxend = indexer.index_pats(patsin=pats,patStart=patStart,patEnd=patEnd)
      rate = np.array([timer() - tic,indxend - indxstart])

      #try:
      queues[2].send((dataout,indxstart,indxend,rate,id,jid))
      #except:
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
def index_chunk_MP2(pats = None, queues = None,lock=None, id = None):
  #queues[0]: messaging queue -- will initially get the indexer object, and the mode.  Will listen for a "Done" after that.
  #queues[1]: job queue
  #queues[2]: results queue

  indexer, mode = queues[0].get()
  #print('here', id)
  workToDo = True
  while workToDo:
    tic = timer()
    lock[0].acquire()
    #print(queues[1].empty())
    if queues[1].empty() == False:
      if mode == 'filemode':
        pats = None
        patStart,patEnd, jid = queues[1].get()
      elif mode == 'memory': # the patterns were stored in memory and sent within the job queue
        pats, patStart,patEnd, jid = queues[1].get()
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
        #queues[0].close()

        return
    else:
      lock[1].release()
      time.sleep(0.01)
  return

def index_pats_distributed(pats = None, filename=None, filenameout=None, phaselist=['FCC'], \
               vendor=None, PC = None, sampleTilt=70.0, camElev = 5.3,\
               peakDetectPlan = None, nRho=90, nTheta=180, tSigma= None, rSigma=None, rhoMaskFrac=0.1, nBands=9,
               patStart = 0, patEnd = -1, chunksize = 1000, ncpu=-1,
               return_indexer_obj = False, ebsd_indexer_obj = None):



  n_cpu_nodes = int(multiprocessing.cpu_count()) #int(sum([ r['Resources']['CPU'] for r in ray.nodes()]))
  if ncpu != -1:
    n_cpu_nodes = ncpu

  try:
    plat = cl.get_platforms()
    gpu = plat[0].get_devices(device_type=cl.device_type.GPU)
    ngpu = len(gpu)
    ngpupnode = ngpu/n_cpu_nodes
  except:
    ngpu = 0
    ngpupnode = 0

  ray.shutdown()

  ray.init(num_cpus=n_cpu_nodes, num_gpus=ngpu )

  if pats is None:
    pdim = None
  else:
    pdim = pats.shape[-2:]
  if ebsd_indexer_obj == None:
    indexer = EBSDIndexer(filename=filename,phaselist=phaselist, \
                          vendor=None,PC = PC,sampleTilt=sampleTilt,camElev = camElev, \
                          bandDetectPlan= peakDetectPlan, \
                          nRho=nRho,nTheta=nTheta,tSigma= tSigma,rSigma=rSigma,rhoMaskFrac=rhoMaskFrac,nBands=nBands,patDim = pdim)
  else:
    indexer = ebsd_indexer_obj



  # differentiate between getting a file to index or an array
  # Need to index one pattern to make sure the indexer object is fully initiated before
  #   placing in shared memory store.
  mode = 'memorymode'
  if pats is None:
    mode = 'filemode'
    temp, indexer = index_pats(patStart = 0, patEnd = 1,return_indexer_obj = True, ebsd_indexer_obj = indexer)

  if mode == 'filemode':
    npats = indexer.fID.nPatterns
  else:
    pshape = pats.shape
    if len(pshape) == 2:
      npats = 1
      pats = pats.reshape([1,pshape[0],pshape[1]])
    else:
      npats = pshape[0]
    temp,indexer = index_pats(pats[0,:,:], patStart=0,patEnd=1,return_indexer_obj=True,ebsd_indexer_obj=indexer)

  if patEnd == 0:
    patEnd = 1
  if (patStart !=0) or (patEnd > 0):
    npats = patEnd - patStart

  #place indexer obj in shared memory store so all workers can use it.
  remote_indexer = ray.put(indexer)
  # set up the jobs
  njobs = (np.ceil(npats/chunksize)).astype(np.long)
  p_indx_start = [i*chunksize+patStart for i in range(njobs)]
  p_indx_end = [(i+1)*chunksize+patStart for i in range(njobs)]
  p_indx_end[-1] = npats+patStart
  if njobs < n_cpu_nodes:
    n_cpu_nodes = njobs

  dataout = np.zeros((npats),dtype=indexer.dataTemplate)
  ndone = 0
  nsubmit = 0
  nread = 0
  tic = timer()
  npatsdone = 0.0
  toc = 0.0
  if mode == 'filemode':
    # send out the first batch
    workers = []
    jobs = []
    timers = []
    rateave = 0.0
    for i in range(n_cpu_nodes):

      workers.append(IndexerRay.options(num_cpus=1, num_gpus=ngpupnode).remote())
      jobs.append(workers[i].index_chunk_ray.remote(pats = None, indexer = remote_indexer, \
                                        patStart=p_indx_start[nsubmit],patEnd=p_indx_end[nsubmit]))
      nsubmit += 1
      timers.append(timer())
      time.sleep(0.01)


    #workers = [index_chunk.remote(pats = None, indexer = remote_indexer, patStart = p_indx_start[i], patEnd = p_indx_end[i]) for i in range(n_cpu_nodes)]
    #nsubmit += n_cpu_nodes

    while ndone < njobs:
      toc = timer()
      wrker,busy = ray.wait(jobs,num_returns=1,timeout=None)

      # print("waittime: ",timer() - toc)
      wrkdataout,indxstr,indxend,rate = ray.get(wrker[0])
      jid = jobs.index(wrker[0])
      ticp = timers[jid]
      dataout[indxstr:indxend] = wrkdataout
      npatsdone += rate[1]
      ratetemp = n_cpu_nodes * (rate[1]) / (timer() - ticp)
      rateave += ratetemp
      # print('Completed: ',str(indxstr),' -- ',str(indxend), '  ', npatsdone/(timer()-tic) )
      ndone += 1
      print('Completed: ',str(indxstr),' -- ',str(indxend),'  ',np.int(ratetemp),'  ',np.int(rateave / ndone))

      if nsubmit < njobs:

        jobs[jid] = workers[jid].index_chunk_ray.remote(pats=None,indexer=remote_indexer,
                                                    patStart=p_indx_start[nsubmit],patEnd=p_indx_end[nsubmit])
        nsubmit += 1
        timers[jid] = timer()
      else: #start destroying workers, jobs, etc...
        del jobs[jid]
        del workers[jid]
        del timers[jid]




  if mode == 'memorymode':
    pass
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
    return dataout, indexer
  else:
    return dataout

def index_patsMP(pats = None, filename=None, filenameout=None, phaselist=['FCC'], \
               vendor=None, PC = None, sampleTilt=70.0, camElev = 5.3,\
               peakDetectPlan = None, nRho=90, nTheta=180, tSigma= None, rSigma=None, rhoMaskFrac=0.1, nBands=9,
               patStart = 0, patEnd = -1, chunksize = 1000, ncpu=-1,
               return_indexer_obj = False, ebsd_indexer_obj = None):



  n_cpu_nodes = int(multiprocessing.cpu_count()) #int(sum([ r['Resources']['CPU'] for r in ray.nodes()]))
  if ncpu != -1:
    n_cpu_nodes = ncpu


  if pats is None:
    pdim = None
  else:
    pdim = pats.shape[-2:]
  if ebsd_indexer_obj == None:
    indexer = EBSDIndexer(filename=filename,phaselist=phaselist, \
                          vendor=None,PC = PC,sampleTilt=sampleTilt,camElev = camElev, \
                          bandDetectPlan= peakDetectPlan, \
                          nRho=nRho,nTheta=nTheta,tSigma= tSigma,rSigma=rSigma,rhoMaskFrac=rhoMaskFrac,nBands=nBands,patDim = pdim)
  else:
    indexer = ebsd_indexer_obj

  # differentiate between getting a file to index or an array
  # Need to index one pattern to make sure the indexer object is fully initiated before
  #   placing in shared memory store.
  mode = 'memorymode'
  if pats is None:
    mode = 'filemode'
    #just make sure indexer is fully initialized
    temp, indexer = index_pats(patStart = 0, patEnd = 1,return_indexer_obj = True, ebsd_indexer_obj = indexer)

  if mode == 'filemode':
    npats = indexer.fID.nPatterns
  else:
    pshape = pats.shape
    if len(pshape) == 2:
      npats = 1
      pats = pats.reshape([1,pshape[0],pshape[1]])
    else:
      npats = pshape[0]
    temp,indexer = index_pats(pats[0,:,:], patStart=0,patEnd=1,return_indexer_obj=True,ebsd_indexer_obj=indexer)

  if patEnd == 0:
    patEnd = 1
  if (patStart !=0) or (patEnd > 0):
    npats = patEnd - patStart

  # set up the jobs
  njobs = (np.ceil(npats/chunksize)).astype(np.long)
  p_indx_start = [i*chunksize+patStart for i in range(njobs)]
  p_indx_end = [(i+1)*chunksize+patStart for i in range(njobs)]
  p_indx_end[-1] = npats+patStart
  if njobs < n_cpu_nodes:
    n_cpu_nodes = njobs



  dataout = np.zeros((npats),dtype=indexer.dataTemplate)
  returnqueue = []

  ndone = 0
  tic = timer()
  npatsdone = 0.0
  toc = 0.0
  messagequeue = [] # each worker will get its own message queue
  messagelock = []
  workers = []
  rateave = 0.0
  readlock = multiprocessing.Lock()
  jobqueue = multiprocessing.SimpleQueue()
  jcheck = np.ones(njobs)
  if mode == 'filemode':



    tic = timer()
    for i in range(n_cpu_nodes): # launch the workers
      messagequeue.append(multiprocessing.SimpleQueue())
      messagelock.append(multiprocessing.Lock())
      returnqueue.append(multiprocessing.Pipe())
      workers.append(multiprocessing.Process(target=index_chunk_MP2,args=(None, \
                                            [messagequeue[i], jobqueue, returnqueue[i][1]],[readlock, messagelock[i]], i)))
      workers[i].start()

    readlock.acquire()
    for i in range(njobs):
      jobqueue.put((p_indx_start[i],p_indx_end[i], i))
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
        #try:
        if (returnqueue[i][0]).poll():
          wrkdataout,indxstr,indxend, rate, wrkID, jid = returnqueue[i][0].recv()
          dataout[indxstr:indxend] = wrkdataout
          npatsdone += rate[1]
          ratetemp = n_cpu_nodes*(rate[1]) / (rate[0])
          rateave += ratetemp
          jcheck[jid] = 0
          ndone += 1#np.count_nonzero(jcheck == 0)
          ntrysum += np.max((ntry, 0))
          tryThresh = np.int( ntrysum/np.float(ndone) )
          #print( tryThresh, ntry)
          ntry = 0
          print('Completed: ',str(indxstr),' -- ',str(indxend),'  ',np.int(ratetemp), '  ',
                np.int(npatsdone/(timer()-tic)), np.int(npatsdone/npats*100), '%  ', njobs, ndone )
          whzero = jcheck.nonzero()[0]
          #if whzero.size < 10:
          #  print(jcheck.nonzero()[0])
        #except queue.Empty: # there is a strange race condition that I can not figure out.
        #else:
      time.sleep(0.005)
      ntry +=1
      if ntry > (10*tryThresh):
        ntry = 0
        whbad = jcheck.nonzero()[0]
        if whbad.size > 0:
          print('adding back job', whbad[0])
          jobqueue.put((p_indx_start[whbad[0]],p_indx_end[whbad[0]], whbad[0]))
        #except:
        #  print("Unexpected error:",sys.exc_info()[0])



  if mode == 'memorymode':

    nsubmit = 0
    n2queue = n_cpu_nodes+4 if (njobs > (n_cpu_nodes+4)) else  n_cpu_nodes # going to pre-load a few extra jobs.
    # memory concerns are the only reason we do not load all into the queue at once.
    for i in range(n2queue):
      jobqueue.put(((pats[p_indx_start[nsubmit]:p_indx_end[nsubmit],:,:], p_indx_start[nsubmit],p_indx_end[nsubmit])))
      nsubmit +=1

    for i in range(n_cpu_nodes): # launch the workers
      toc = timer()
      messagequeue.append(multiprocessing.Queue())
      #print('New q', timer()-toc)
      toc = timer()
      messagequeue[i].put((indexer, mode)) # provide the indexer and where the patterns are coming from
      workers.append(multiprocessing.Process(target=index_chunk_MP,args=(None, \
                                            [messagequeue[i], jobqueue, returnqueue], i)))
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
        if nsubmit < njobs: # still more to do.
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
    return dataout, indexer
  else:
    return dataout



class EBSDIndexer():
  def __init__(self,filename=None,phaselist=['FCC'], \
               vendor=None,PC =None,sampleTilt=70.0,camElev = 5.3, \
               bandDetectPlan = None,nRho=90,nTheta=180,tSigma= None,rSigma=None,rhoMaskFrac=0.1,nBands=9,patDim=None):
    self.filein = filename
    if self.filein is not None:
      self.fID = ebsd_pattern.get_pattern_file_obj(self.filein)
    else:
      self.fID = None
    # self.fileout = filenameout
    # if self.fileout is None:
    #   self.fileout = str.lower(Path(self.filein).stem)+'.ang'
    self.phaselist = phaselist

    self.vendor = 'EDAX'
    if vendor is None:
      if self.fID is not None:
        self.vendor = self.fID.vendor
    else:
      if vendor is not None:
        self.vendor = vendor



    if PC is None:
      self.PC = np.array([0.471659,0.675044,0.630139])
    else:
      self.PC = np.asarray(PC)
    self.sampleTilt = sampleTilt
    self.camElev = camElev
    # self.startPat = startPat
    # self.endPat = endPat
    # if (self.endPat == -1) and self.fID is not None:
    #   self.endPat = self.fID.nPatterns+1
    if bandDetectPlan is None:
        self.bandDetectPlan = band_detect2.BandDetect(nRho=nRho,nTheta=nTheta, \
                                                      tSigma=tSigma,rSigma=rSigma, \
                                                      rhoMaskFrac=rhoMaskFrac,nBands=nBands)
    else:
      self.bandDetectPlan = bandDetectPlan

    if self.fID is not None:
      self.bandDetectPlan.band_detect_setup(patDim=[self.fID.patternW,self.fID.patternH])
    else:
      if patDim is not None:
        self.bandDetectPlan.band_detect_setup(patDim=patDim)
    self.phaseLib =[]
    for ph in self.phaselist:
      self.phaseLib.append(band_vote.BandVote(tripletlib.triplib(libType=ph)))

    self.dataTemplate = np.dtype([('quat',np.float32, (4)),('iq',np.float32), \
                             ('pq',np.float32),('cm',np.float32),('phase',np.int32), \
                             ('fit',np.float32),('nmatch', np.int32), ('matchattempts', np.int32, (2))])


  def index_pats(self, patsin=None, patStart = 0, patEnd = -1,clparams = [None, None, None, None, None], PC=[None, None, None]):
    tic = timer()

    if patsin is None:
      pats = self.fID.read_data(returnArrayOnly=True,patStartEnd=[patStart,patEnd], convertToFloat=True)
    else:
      pshape = patsin.shape
      if len(pshape) == 2:
        pats = np.reshape(patsin, (1,pshape[0], pshape[1]))
      else:
        pats = patsin#[patStart:patEnd, :,:]
      pshape = pats.shape

      if self.bandDetectPlan.patDim is None:
        self.bandDetectPlan.band_detect_setup(patDim=pshape[1:3])
      else:
        if np.all((np.array(pshape[1:3])-self.bandDetectPlan.patDim) == 0):
          self.bandDetectPlan.band_detect_setup(patDim=pshape[1:3])

    if self.bandDetectPlan.patDim is None:
      self.bandDetectPlan.band_detect_setup(patterns = pats)

    npoints = pats.shape[0]
    if patEnd == -1:
      patEnd = npoints+1

    indxData = np.zeros((npoints),dtype=self.dataTemplate)
    #time.sleep(10.0)


    q = np.zeros((npoints, 4))
    #print(timer() - tic)
    tic = timer()
    bandData = self.bandDetectPlan.find_bands(pats, clparams = clparams)



    #indxData['iq'] = np.sum(bandData['pqmax'], axis = 1)
    if PC[0] is None:
      PC_0 = self.PC
    else:
      PC_0 = PC
    bandNorm = self.bandDetectPlan.radon2pole(bandData,PC=PC_0,vendor=self.vendor)
    #print('Find Band: ', timer() - tic)

    #return bandNorm,patStart,patEnd
    tic = timer()
    #bv = []
    #for tl in self.phaseLib:
    #  bv.append(band_vote.BandVote(tl))



    for i in range(npoints):
    #for i in range(10):
      phase = 0
      fitmetric = -1

      bandNorm1 = bandNorm[i,:,:]
      bDat1 = bandData[i,:]
      whgood = np.nonzero(bDat1['max'] > -1.0e6)[0]
      if whgood.size > 0:
        bDat1 = bDat1[whgood]
        bandNorm1 = bandNorm1[whgood,:]
        indxData['pq'][i] = np.sum(bDat1['max'],axis=0)
        #avequat, fit, cm, bandmatch, nMatch = bv[0].tripvote(bandNorm[i,:,:], goNumba = True)
        avequat,fit,cm,bandmatch,nMatch, matchAttempts = self.phaseLib[0].tripvote(bandNorm1,goNumba=True)

        if nMatch > 0:
          phase = 1
          fitmetric = nMatch * cm

        fitmetric1 = -1
        for j in range(1, len(self.phaseLib)):

          #avequat1,fit1,cm1,bandmatch1,nMatch1 = bv[j].tripvote(bandNorm[i,:,:], goNumba = True)
          avequat1,fit1,cm1,bandmatch1,nMatch1, matchAttempts1 = self.phaseLib[j].tripvote(bandNorm1,goNumba=True)
          if nMatch1 > 0:
            fitmetric1 = nMatch1 * cm1
          if fitmetric1 > fitmetric:
            fitmetric = fitmetric1
            avequat = avequat1
            fit = fit1
            cm = cm1
            bandmatch = bandmatch1
            nMatch = nMatch1
            matchAttempts = matchAttempts1
            phase = j+1

        #indxData['quat'][i] = avequat
        q[i,:] = avequat
        indxData['fit'][i] = fit
        indxData['cm'][i] = cm
        indxData['phase'][i] = phase
        indxData['nmatch'][i] = nMatch
        indxData['matchattempts'][i] = matchAttempts


    qref2detect = self.refframe2detector()
    q = rotlib.quat_multiply(q,qref2detect)
    indxData['quat'] = q
    #print('bandvote: ',timer() - tic)
    return indxData, patStart, patEnd

  def refframe2detector(self):
      if self.vendor == 'EDAX':
        q0 = np.array([np.sqrt(2.0)*0.5, 0.0, 0.0, -1.0 * np.sqrt(2.0)*0.5])
        tiltang = -1.0*(90.0-self.sampleTilt + self.camElev)/RADEG
        q1 = np.array([np.cos(tiltang*0.5), np.sin(tiltang*0.5), 0.0, 0.0])
        quatref2detect = rotlib.quat_multiply(q1,q0)

      return quatref2detect



def __main__():
  file = '~/Desktop/SLMtest/scan2v3nlparl09sw7.up1'