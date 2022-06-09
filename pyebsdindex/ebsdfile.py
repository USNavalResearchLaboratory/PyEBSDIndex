def writeang(filename, indexer, data,
             gridtype = 'SqrGrid', xstep=1.0, ystep=1.0,
             ncols = None, nrows=None):

  with open(filename,'w',encoding = 'utf-8') as f:
    f.write('# HEADER: Start')
    f.write('# TEM_PIXperUM          1.000000')
    f.write('# x-star          ' + str(indexer.PC[0]))
    f.write('# y-star          ' + str(indexer.PC[1]))
    f.write('# z-star          ' + str(indexer.PC[2]))
    f.write('# SampleTiltAngle       ' + str(indexer.sampleTilt))
    f.write('# CameraElevationAngle  ' + str(indexer.camElev))
    f.write('# ')
    pcount = 1
    for phase in indexer.phaseLib:
      f.write('# Phase '+str(pcount))
      f.write('# MaterialName ' + str(phase.phase_name))
      f.write('# Formula ')
      f.write('# Info ')
      f.write('# Symmetry              '+str(phase.laue_code))
      f.write('# PointGroupID              ' + str(phase.symmetry_pgID))
      f.write('# LatticeConstants      '+ ' '.join(str(x) for x in phase.lattice_param))
      f.write('# NumberFamilies             ' + str(phase.trilib.nfamily))
      for i in range(phase.trilib.nfamily):
        f.write('# hklFamilies   	 ' + ' '.join(str(x) for x in phase.tripLib.family[i,:]) + ' 1 0.00000 1')
      f.write('# ')

    f.write('# ')
    f.write('# GRID: '+gridtype)
    if indexer.fID.xstep is not None:
      xstep = str(indexer.fID.xstep)
      ystep = str(indexer.fID.ystep)
    else:
      xstep = str(xstep)
      ystep = str(ystep)
    f.write('# XSTEP: ' + xstep)
    f.write('# YSTEP: ' + ystep)
    if indexer.fID.nCols is not None:
      ncols = indexer.fID.nCols
      nrows = indexer.fID.nRows
    else:
      if ncols is None:
        ncols = 1
        nrows = data.shape[-1]
    ncols = int(ncols)
    nrows = int(nrows)
    f.write('# NCOLS_ODD: ' + str(ncols))
    f.write('# NCOLS_EVEN: ' + str(ncols))
    f.write('# NROWS: ' + str(nrows))
    f.write('# VERSION 5')

    f.write('# HEADER: End')

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
      f.write(line)