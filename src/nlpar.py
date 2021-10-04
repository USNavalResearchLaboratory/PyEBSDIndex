import numpy as np
from timeit import default_timer as timer
from numba import jit, prange
import matplotlib.pyplot as plt

class NLPAR():
  def __init__(self):
  lamba = 0.7
  searchradius = 3
  ebsdfile = None