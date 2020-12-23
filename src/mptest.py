import my_queue
import multiprocessing as mp
import queue

import sys

def add(a,b, q):
  print(q.get())
  q.put(a+b)
  print('hello')
  q.put(a-b)
  print('goodbye')

def mptest():
  q = my_queue.MyQueue()
  q.put(100)
  p = mp.Process(target = add, args=(4,5, q))
  try:
    q.get(False)
  except queue.Empty:
    print('empty')
  p.start()


  print(q.get())
  print('what')
  print(q.get())

  #print(q.get(False))
  print(p.join())
  print(p.exitcode)
