import numpy as np
from pathlib import Path
#from multiprocessing import Process, Queue
import ebsd_pattern
import band_detect
import band_vote
import rotlib
import tripletlib
import ray
ray.services.get_node_ip_address = lambda: '127.0.0.1'
from timeit import default_timer as timer
import time
import multiprocessing
RADEG = 180.0/np.pi



def index_pats(pats = None, filename=None, filenameout=None, phaselist=['FCC'], \
               vendor=None, PC = None, sampleTilt=70.0, camElev = 5.3,\
               peakDetectPlan = None, nRho=90, nTheta=180, tSigma= None, rSigma=None, rhoMaskFrac=0.1, nBands=9,\
               patStart = 0, patEnd = -1,\
               return_indexer_obj = False, ebsd_indexer_obj = None):

  if pats is None:
    pdim = None
  else:
    pdim = pats.shape[-2:]
  if ebsd_indexer_obj == None:
    indexer = EBSDIndexer(filename=filename, phaselist=phaselist, \
               vendor=None, PC = PC, sampleTilt=sampleTilt, camElev = camElev,\
               peakDetectPlan = peakDetectPlan, \
               nRho=nRho, nTheta=nTheta, tSigma= tSigma, rSigma=rSigma, rhoMaskFrac=rhoMaskFrac, nBands=nBands, patDim = pdim)
  else:
    indexer = ebsd_indexer_obj


  dataout, indxstart, indxend = indexer.index_pats(patsin=pats, patStart=patStart, patEnd=patEnd)

  if return_indexer_obj == False:
    return dataout
  else:
    return dataout, indexer


@ray.remote
def index_chunk(pats = None, indexer = None, patStart = 0, patEnd = -1 ):
  tic = timer()
  dataout,indxstart,indxend = indexer.index_pats(patsin=pats,patStart=patStart,patEnd=patEnd)
  rate = np.array([timer()-tic, indxend-indxstart])
  return dataout, indxstart,indxend, rate

def index_pats_distributed(pats = None, filename=None, filenameout=None, phaselist=['FCC'], \
               vendor=None, PC = None, sampleTilt=70.0, camElev = 5.3,\
               peakDetectPlan = None, nRho=90, nTheta=180, tSigma= None, rSigma=None, rhoMaskFrac=0.1, nBands=9,
               patStart = 0, patEnd = -1, chunksize = 1000, ncpu=-1,
               return_indexer_obj = False, ebsd_indexer_obj = None):



  n_cpu_nodes = int(multiprocessing.cpu_count()) #int(sum([ r['Resources']['CPU'] for r in ray.nodes()]))
  if ncpu != -1:
    n_cpu_nodes = ncpu

  ray.shutdown()

  ray.init(num_cpus=n_cpu_nodes, dashboard_host='0.0.0.0')

  if pats is None:
    pdim = None
  else:
    pdim = pats.shape[-2:]
  if ebsd_indexer_obj == None:
    indexer = EBSDIndexer(filename=filename, phaselist=phaselist, \
               vendor=None, PC = PC, sampleTilt=sampleTilt, camElev = camElev,\
               peakDetectPlan = peakDetectPlan, \
               nRho=nRho, nTheta=nTheta, tSigma= tSigma, rSigma=rSigma, rhoMaskFrac=rhoMaskFrac, nBands=nBands, patDim = pdim)
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
  #ray.shutdown()
  #return
  tic = timer()
  npatsdone = 0.0
  toc = 0.0
  if mode == 'filemode':
    # send out the first batch
    workers = []
    for i in range(n_cpu_nodes):
      pats = indexer.fID.read_data(convertToFloat=True, patStartEnd=[p_indx_start[i],p_indx_end[i]],returnArrayOnly=True)
      #pats = None
      workers.append(index_chunk.remote(pats = pats, indexer = remote_indexer, patStart=p_indx_start[nsubmit],
                                        patEnd=p_indx_end[nsubmit]))
      #workers.append(index_chunk.remote(patStart=p_indx_start[nsubmit],patEnd=p_indx_end[nsubmit]))
      nsubmit += 1

    #workers = [index_chunk.remote(pats = None, indexer = remote_indexer, patStart = p_indx_start[i], patEnd = p_indx_end[i]) for i in range(n_cpu_nodes)]
    #nsubmit += n_cpu_nodes
    while ndone < njobs:

      wrker,busy = ray.wait(workers, num_returns=1, timeout=None)
      wrkdataout,indxstr,indxend, rate = ray.get(wrker[0])
      dataout[indxstr:indxend] = wrkdataout
      npatsdone += rate[1]

      print('Completed: ',str(indxstr),' -- ',str(indxend), '  ', npatsdone/(timer()-tic) )
      workers.remove(wrker[0])
      ndone += 1
      if nsubmit < njobs:
        pats = indexer.fID.read_data(convertToFloat=True,patStartEnd=[p_indx_start[nsubmit],p_indx_end[nsubmit]],
                                     returnArrayOnly=True)

        workers.append(index_chunk.remote(pats = pats, indexer = remote_indexer, patStart = p_indx_start[nsubmit],
                                          patEnd = p_indx_end[nsubmit]))
        #workers.append(index_chunk.remote(pats=None,indexer=remote_indexer,patStart=p_indx_start[nsubmit],
        #                                  patEnd=p_indx_end[nsubmit]))
        nsubmit += 1


  if mode == 'memorymode':
    # send out the first batch
    workers = [index_chunk.remote(pats=pats[p_indx_start[i]:p_indx_end[i],:,:],indexer=remote_indexer,patStart=p_indx_start[i],patEnd=p_indx_end[i]) for i
               in range(n_cpu_nodes)]
    nsubmit += n_cpu_nodes

    while ndone < njobs:
      wrker,busy = ray.wait(workers,num_returns=1,timeout=None)
      wrkdataout,indxstr,indxend, rate = ray.get(wrker[0])
      dataout[indxstr:indxend] = wrkdataout
      print('Completed: ',str(indxstr),' -- ',str(indxend))
      workers.remove(wrker[0])
      ndone += 1

      if nsubmit < njobs:
        workers.append(index_chunk.remote(pats=pats[p_indx_start[nsubmit]:p_indx_end[nsubmit],:,:],indexer=remote_indexer,patStart=p_indx_start[nsubmit],
                                          patEnd=p_indx_end[nsubmit]))
        nsubmit += 1

  ray.shutdown()
  if return_indexer_obj:
    return dataout, indexer
  else:
    return dataout




class EBSDIndexer():
  def __init__(self, filename=None, phaselist=['FCC'], \
               vendor=None, PC =None, sampleTilt=70.0, camElev = 5.3,\
               peakDetectPlan = None, nRho=90, nTheta=180, tSigma= None, rSigma=None, rhoMaskFrac=0.1, nBands=9, patDim=None):
    self.filein = filename
    if self.filein is not None:
      self.fID = ebsd_pattern.get_pattern_file_obj(self.filein)
    else:
      self.fID = None
    # self.fileout = filenameout
    # if self.fileout is None:
    #   self.fileout = str.lower(Path(self.filein).stem)+'.ang'
    self.phaselist = phaselist

    if vendor is None:
      if self.fID is not None:
        self.vendor = self.fID.vendor
    else:
      if vendor is not None:
        self.vendor = vendor
      else:
        self.vendor = 'EDAX'

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
    if peakDetectPlan is None:
        self.peakDetectPlan = band_detect.BandDetect(nRho=nRho,nTheta=nTheta,\
                                                     tSigma=tSigma, rSigma=rSigma,\
                                                     rhoMaskFrac=rhoMaskFrac,nBands=nBands)
    else:
      self.peakDetectPlan = peakDetectPlan

    if self.fID is not None:
      self.peakDetectPlan.band_detect_setup(patDim=[self.fID.patternW,self.fID.patternH ])
    else:
      if patDim is not None:
        self.peakDetectPlan.band_detect_setup(patDim=patDim)
    self.phaseLib =[]
    for ph in self.phaselist:
      self.phaseLib.append(band_vote.BandVote(tripletlib.triplib(libType=ph)))

    self.dataTemplate = np.dtype([('quat',np.float32, (4)),('iq',np.float32), \
                             ('pq',np.float32),('cm',np.float32),('phase',np.int32), \
                             ('fit',np.float32),('nmatch', np.int32)])


  def index_pats(self, patsin=None, patStart = 0, patEnd = -1):
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

      if self.peakDetectPlan.patDim is None:
        self.peakDetectPlan.band_detect_setup(patDim=pshape[1:3])
      else:
        if np.all((np.array(pshape[1:3])-self.peakDetectPlan.patDim) == 0):
          self.peakDetectPlan.band_detect_setup(patDim=pshape[1:3])

    if self.peakDetectPlan.patDim is None:
      self.peakDetectPlan.band_detect_setup(patterns = pats)

    npoints = pats.shape[0]
    if patEnd == -1:
      patEnd = npoints+1

    indxData = np.zeros((npoints),dtype=self.dataTemplate)
    #time.sleep(10.0)
    #return indxData, patStart, patEnd

    q = np.zeros((npoints, 4))
    #print(timer() - tic)
    tic = timer()
    bandData = self.peakDetectPlan.find_bands(pats)


    indxData['pq'] = np.sum(bandData['max'], axis = 1)
    indxData['iq'] = np.sum(bandData['pqmax'], axis = 1)
    bandNorm = self.peakDetectPlan.radon2pole(bandData,PC=self.PC,vendor=self.vendor)
    #print('Radon: ', timer() - tic)

    tic = timer()
    #bv = []
    #for tl in self.phaseLib:
    #  bv.append(band_vote.BandVote(tl))



    for i in range(npoints):
    #for i in range(10):
      phase = 0
      fitmetric = 1e6

      #avequat, fit, cm, bandmatch, nMatch = bv[0].tripvote(bandNorm[i,:,:], goNumba = True)
      avequat,fit,cm,bandmatch,nMatch = self.phaseLib[0].tripvote(bandNorm[i,:,:],goNumba=True)

      if nMatch > 0:
        phase = 1
        fitmetric = fit/nMatch * (np.max([cm, 0.001]))

      fitmetric1 = 1e6
      for j in range(1, len(self.phaseLib)):

        #avequat1,fit1,cm1,bandmatch1,nMatch1 = bv[j].tripvote(bandNorm[i,:,:], goNumba = True)
        avequat1,fit1,cm1,bandmatch1,nMatch1 = self.phaseLib[j].tripvote(bandNorm[i,:,:],goNumba=True)
        if nMatch1 > 0:
          fitmetric1 = fit1/nMatch1 * (np.max([cm1, 0.001]))
        if fitmetric1 < fitmetric:
          fitmetric = fitmetric1
          avequat = avequat1
          fit = fit1
          cm = cm1
          bandmatch = bandmatch1
          nMatch = nMatch1
          phase = j+1

      #indxData['quat'][i] = avequat
      q[i,:] = avequat
      indxData['fit'][i] = fit
      indxData['cm'][i] = cm
      indxData['phase'][i] = phase
      indxData['nmatch'][i] = nMatch

    qref2detect = self.refframe2detector()
    q = rotlib.quat_multiply(q,qref2detect)
    indxData['quat'] = q
    #print('bandvote: ', timer() - tic)
    return indxData, patStart, patEnd

  def refframe2detector(self):
      if self.vendor == 'EDAX':
        q0 = np.array([np.sqrt(2.0)*0.5, 0.0, 0.0, -1.0 * np.sqrt(2.0)*0.5])
        tiltang = -1.0*(90.0-self.sampleTilt + self.camElev)/RADEG
        q1 = np.array([np.cos(tiltang*0.5), np.sin(tiltang*0.5), 0.0, 0.0])
        quatref2detect = rotlib.quat_multiply(q1,q0)

      return quatref2detect

