'''This software was developed by employees of the US Naval Research Laboratory (NRL), an
agency of the Federal Government. Pursuant to title 17 section 105 of the United States
Code, works of NRL employees are not subject to copyright protection, and this software
is in the public domain. PyEBSDIndex is an experimental system. NRL assumes no
responsibility whatsoever for its use by other parties, and makes no guarantees,
expressed or implied, about its quality, reliability, or any other characteristic. We
would appreciate acknowledgment if the software is used. To the extent that NRL may hold
copyright in countries other than the United States, you are hereby granted the
non-exclusive irrevocable and unconditional right to print, publish, prepare derivative
works and distribute this software, in any medium, or authorize others to do so on your
behalf, on a royalty-free basis throughout the world. You may improve, modify, and
create derivative works of the software or any portion of the software, and you may copy
and distribute such modifications or works. Modified works should carry a notice stating
that you changed the software and should note the date and nature of any such change.
Please explicitly acknowledge the US Naval Research Laboratory as the original source.
This software can be redistributed and/or modified freely provided that any derivative
works bear some notice that they are derived from it, and any modified versions bear
some notice that they have been modified.

Author: David Rowenhorst;
The US Naval Research Laboratory Date: 21 Aug 2020'''


import numpy as np
from os import path
import pyopencl as cl
import warnings
from os import environ
environ['PYOPENCL_COMPILER_OUTPUT'] = '0'

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
      self.get_gpu()

    except Exception as e:
      if hasattr(e,'message'):
        print(e.message)
      else:
        print(e)

  def get_platform(self):
    self.platform = cl.get_platforms()
    return self.platform
  def get_gpu(self):
    if self.platform is None:
      self.get_platform()

    if type(self.platform) is not list:
      self.platform = [self.platform]

    pgpudiscrete = np.zeros(len(self.platform), dtype=int)

    for i in range(len(self.platform)):
      g = self.platform[i].get_devices(device_type=cl.device_type.GPU)
      if len(g) > 0:
        discrete = np.zeros(len(g), dtype=int)
        for j in range(len(g)):
          discrete[j] = g[j].host_unified_memory == False
          if discrete[j] > 0:
            pgpudiscrete[i] += 1
      else:
        pgpudiscrete[i] = -1
    gpu = []
    if pgpudiscrete.max() > 0:  # discrete graphics found
      self.platform = [self.platform[pgpudiscrete.argmax()]]
      g = self.platform[0].get_devices(device_type=cl.device_type.GPU)
      for g1 in g:
        if g1.host_unified_memory == False:
          gpu.append(g1)
    elif pgpudiscrete.max() == 0:  # only integrated graphics available
      self.platform = [self.platform[pgpudiscrete.argmax()]]
      gpu.extend(self.platform[0].get_devices(device_type=cl.device_type.GPU))
    else:
      pass
    self.gpu = gpu
    self.ngpu = len(gpu)
    return self.gpu


  def get_context(self, gpu_id=None, kfile = 'clkernels.cl' ):
    if self.gpu is None:
      self.get_gpu()

    if gpu_id is None:
      gpu_id = self.gpu_id

    gpuindx = min(len(self.gpu)-1, gpu_id)
    self.gpu_id = gpuindx
    self.ctx = cl.Context(devices = [self.gpu[self.gpu_id]])

    kernel_location = path.dirname(__file__)
    warnings.filterwarnings("ignore")
    self.prg = cl.Program(self.ctx,open(path.join(kernel_location,kfile)).read()).build(options=['-cl-std=CL1.2', '-w'])
    warnings.resetwarnings()
    #print('ctx', self.gpu_id)
    return self.ctx
  def get_queue(self, gpu_id=None):
    if self.ctx is None:
      self.get_context(gpu_id=None)
    if self.queue is None:
      self.queue = cl.CommandQueue(self.ctx)
    return self.queue


