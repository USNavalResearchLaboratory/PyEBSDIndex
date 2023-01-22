from pathlib import Path
import numpy as np
from pyebsdindex import rotlib


def writeang(filename, indexer, data,
             gridtype = 'SqrGrid', xstep=1.0, ystep=1.0,
             ncols = None, nrows=None):
  fpath = Path(filename).expanduser()
  with open(fpath,'w',encoding = 'utf-8') as f:
    f.write('# HEADER: Start \r\n')
    f.write('# TEM_PIXperUM          1.000000\r\n')
    f.write('# x-star                ' + str(indexer.PC[0])+'\r\n')
    f.write('# y-star                ' + str(indexer.PC[1])+'\r\n')
    f.write('# z-star                ' + str(indexer.PC[2])+'\r\n')
    f.write('# SampleTiltAngle       ' + str(indexer.sampleTilt)+'\r\n')
    f.write('# CameraElevationAngle  ' + str(indexer.camElev)+'\r\n')
    f.write('# '+'\r\n')
    pcount = 1
    nphase = len(indexer.phaseLib)
    for phase in reversed(indexer.phaseLib):
      f.write('# Phase '+str(nphase - pcount + 1)+'\r\n')
      f.write('# MaterialName \t' + str(phase.phasename)+'\r\n')
      f.write('# Formula '+'\t \r\n')
      f.write('# Info '+'\t\t \r\n')
      f.write('# Symmetry              ' + str(phase.lauecode) + '\r\n')
      #f.write('# PointGroupID              ' + str(phase.pointgroupid) + '\r\n')
      latticeparameter = np.array(phase.latticeparameter).astype(float) * np.array([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
      f.write('# LatticeConstants      '+ ' '.join(str(' {:.3f}'.format(x)) for x in latticeparameter)+'\r\n')
      f.write('# NumberFamilies             ' + str(phase.npolefamilies) + '\r\n')
      poles = np.array(phase.polefamilies).astype(int)
      if (phase.lauecode == 62) or (phase.lauecode == 6):
        if poles.shape[-1] == 4:
          poles = poles[:,[0,1,3]]

      for i in range(phase.npolefamilies):
        f.write('# hklFamilies   \t' + (' '.join(str(x).rjust(2,' ') for x in poles[i, :])) + ' 1 0.00000 1' + '\r\n')


      f.write('# '+'\r\n')
      pcount += 1

    f.write('# '+'\r\n')
    f.write('# GRID: '+gridtype+'\r\n')
    if indexer.fID is not None:
      if indexer.fID.xStep > 1e-6:
        xstep = indexer.fID.xStep
        ystep = indexer.fID.yStep

    f.write('# XSTEP: ' + str(xstep)+'\r\n')
    f.write('# YSTEP: ' + str(ystep)+'\r\n')
    if ncols is None:
      ncols = 1
      nrows = data.shape[-1]
      if indexer.fID is not None:
        if indexer.fID.nCols is not None:
          ncols = indexer.fID.nCols
          nrows = indexer.fID.nRows
    else:
      if nrows is None:
        nrows = np.ceil(data.shape[-1]/ncols)


    ncols = int(ncols)
    nrows = int(nrows)
    f.write('# NCOLS_ODD: ' + str(ncols)+'\r\n')
    f.write('# NCOLS_EVEN: ' + str(ncols)+'\r\n')
    f.write('# NROWS: ' + str(nrows)+'\r\n')
    f.write('# VERSION 7'+'\r\n')
    f.write('# COLUMN_COUNT: 10'+'\r\n')
    f.write('# HEADER: End'+'\r\n')

    nphase = data.shape[0]-1
    if nphase == 1:
      phaseIDadd = 0
    else:
      phaseIDadd = 1
    eulers = rotlib.qu2eu(data[-1]['quat'])
    for i in range(data.shape[-1]):
      line = '  '
      line += '   '.join('{:.5f}'.format(x) for x in eulers[i,:])
      line += ' '
      line += ('{:.5f}'.format((i % ncols)*float(xstep))).rjust(12,' ') + ' '
      line += ('{:.5f}'.format((int(i / ncols)) * float(ystep))).rjust(12, ' ') + ' '
      line += ('{:.1f}'.format(data[-1]['pq'][i])).rjust(8, ' ') + ' '
      if data[-1]['phase'][i] < 0: # this is an unindexed point.
        phase = 0
        ci = -1.0
        fit = 0.00
      else:
        phase = data[-1]['phase'][i]+phaseIDadd
        ci = data[-1]['cm'][i]
        fit = data[-1]['fit'][i]

      line += '{:.3f}'.format(ci).rjust(6, ' ') + ' '
      line += ' {:}'.format(phase) + ''
      line += '1'.rjust(7, ' ')+''
      line += ('{:.3f}'.format(fit)).rjust(7, ' ')
      f.write(line+'\r\n')