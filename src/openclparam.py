import numpy as np
from os import path
import pyopencl as cl
from os import environ
environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

RADDEG = 180.0/np.pi
DEGRAD = np.pi/180.0



class OpenClParam():
  def __init__(self):
    self.platform = None
    self.gpu = None
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

  def get_context(self):
    if self.gpu is None:
      self.get_gpu()

    self.ctx = cl.Context(devices = self.gpu)
    kernel_location = path.dirname(__file__)
    self.prg = cl.Program(self.ctx,open(path.join(kernel_location,'clkernels.cl')).read()).build()

  def get_queue(self, gpu_id=0, random_gpu=False):

    if self.ctx is None:
      self.get_context()

    if random_gpu == True:
      gpu_id = np.random.randint(len(self.gpu))
    self.queue = cl.CommandQueue(self.ctx, device=self.gpu[gpu_id])


