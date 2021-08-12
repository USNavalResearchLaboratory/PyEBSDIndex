import numpy as np
from os import path
import pyopencl as cl
from os import environ
environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

RADDEG = 180.0/np.pi
DEGRAD = np.pi/180.0



class OpenClParam():
  def __init__(self, gpu_id=0):
    self.platform = None
    self.gpu = None
    self.ngpu = 0
    self.gpu_id = gpu_id
    self.ctx = None
    self.prg = None
    self.queue = None
    self.memflags = cl.mem_flags

    try:
      self.get_context()

    except Exception as e:
      if hasattr(e,'message'):
        print(e.message)
      else:
        print(e)

  def get_platform(self):
    self.platform = cl.get_platforms()[0]
  def get_gpu(self):

    if self.platform is None:
      self.get_platform()

    self.gpu = self.platform.get_devices(device_type=cl.device_type.GPU)
    self.ngpu = len(self.gpu)
    if len(self.gpu)-1 < self.gpu_id:
      self.gpu_id = len(self.gpu)-1

  def get_context(self):
    if self.gpu is None:
      self.get_gpu()

    self.ctx = cl.Context(devices = self.gpu)
    kernel_location = path.dirname(__file__)
    self.prg = cl.Program(self.ctx,open(path.join(kernel_location,'clkernels.cl')).read()).build()

  def get_queue(self, gpu_id=None, random_gpu=False):

    if self.ctx is None:
      self.get_context()

    if gpu_id is None:
      gpu_id = self.gpu_id

    if random_gpu == True:
      gpu_id = np.random.randint(len(self.gpu))
    gpuindx = min(len(self.gpu)-1, gpu_id)
    self.gpu_id = gpuindx
    self.queue = cl.CommandQueue(self.ctx, device=self.gpu[gpuindx])


