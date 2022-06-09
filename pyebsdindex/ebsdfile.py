from pathlib import Path
from pyebsdindex import rotlib


def writeang(filename, indexer, data,
             gridtype = 'SqrGrid', xstep=1.0, ystep=1.0,
             ncols = None, nrows=None):
  fpath = Path(filename).expanduser()
  with open(fpath,'w',encoding = 'utf-8') as f:
    f.write('# HEADER: Start \n')
    f.write('# TEM_PIXperUM          1.000000\n')
    f.write('# x-star          ' + str(indexer.PC[0])+'\n')
    f.write('# y-star          ' + str(indexer.PC[1])+'\n')
    f.write('# z-star          ' + str(indexer.PC[2])+'\n')
    f.write('# SampleTiltAngle       ' + str(indexer.sampleTilt)+'\n')
    f.write('# CameraElevationAngle  ' + str(indexer.camElev)+'\n')
    f.write('# '+'\n')
    pcount = 1
    for phase in indexer.phaseLib:
      f.write('# Phase '+str(pcount)+'\n')
      f.write('# MaterialName ' + str(phase.phase_name)+'\n')
      f.write('# Formula '+'\n')
      f.write('# Info '+'\n')
      f.write('# Symmetry              '+str(phase.tripLib.laue_code)+'\n')
      f.write('# PointGroupID              ' + str(phase.tripLib.symmetry_pgID)+'\n')
      f.write('# LatticeConstants      '+ ' '.join(str(x) for x in phase.tripLib.latticeParameter)+'\n')
      f.write('# NumberFamilies             ' + str(phase.tripLib.nfamily)+'\n')
      for i in range(phase.tripLib.nfamily):
        f.write('# hklFamilies   	 ' + ' '.join(str(x) for x in phase.tripLib.family[i,:]) + ' 1 0.00000 1'+'\n')
      f.write('# '+'\n')

    f.write('# '+'\n')
    f.write('# GRID: '+gridtype+'\n')
    if indexer.fID.xStep is not None:
      xstep = str(indexer.fID.xStep)
      ystep = str(indexer.fID.yStep)
    else:
      xstep = str(xstep)
      ystep = str(ystep)
    f.write('# XSTEP: ' + xstep+'\n')
    f.write('# YSTEP: ' + ystep+'\n')
    if ncols is None:
      if indexer.fID.nCols is not None:
        ncols = indexer.fID.nCols
        nrows = indexer.fID.nRows
      else:
        ncols = 1
        nrows = data.shape[-1]


    ncols = int(ncols)
    nrows = int(nrows)
    f.write('# NCOLS_ODD: ' + str(ncols)+'\n')
    f.write('# NCOLS_EVEN: ' + str(ncols)+'\n')
    f.write('# NROWS: ' + str(nrows)+'\n')
    f.write('# VERSION 5'+'\n')

    f.write('# HEADER: End'+'\n')

    nphase = data.shape[0]-1
    if nphase == 1:
      phaseIDadd = 0
    else:
      phaseIDadd = 1
    eulers = rotlib.qu2eu(data[-1]['quat'])
    for i in range(data.shape[-1]):
      line = ' '
      line += '   '.join('{:.5f}'.format(x) for x in eulers[i,:])
      line += ' '
      line += ('{:.5f}'.format((i % ncols)*float(xstep))).rjust(12,' ') + ' '
      line += ('{:.5f}'.format((int(i / ncols)) * float(ystep))).rjust(12, ' ') + ' '
      line += '{:.1f}'.format(data[-1]['pq'][i]) + ' '
      line += '{:.3f}'.format(data[-1]['cm'][i]) + ' '
      line += '{:}'.format(data[-1]['phase'][i]+phaseIDadd) + ' '
      line += '{:.3f}'.format(data[-1]['fit'][i])
      f.write(line+'\n')