from pathlib import Path
import numpy as np
import h5py
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

def writeoh5(filename, indexer, data,
               gridtype='SqrGrid', xstep=1.0, ystep=1.0,
               ncols=None, nrows=None, datasetname='Scan 1', version='8.6'):
    fpath = Path(filename).expanduser()

    nphase = data.shape[0] - 1
    phaseIDadd = 0
    if version == '8.6':
      if nphase == 1:
        phaseIDadd = 1
      else:
        phaseIDadd = 1

    with h5py.File(fpath, 'w') as f:

      f.create_dataset(datasetname +'/EBSD/Header/Camera Azimuthal Angle', data=np.array([np.float64(0.0)]))
      f.create_dataset(datasetname + '/EBSD/Header/Camera Diameter', data=np.array([np.float64(0.0)]))
      f.create_dataset(datasetname + '/EBSD/Header/Camera Elevation Angle', data=np.array([np.float64(indexer.camElev)]))
      comment = ' '
      chararray = np.chararray(1, itemsize=len(comment)+1)
      chararray[:] = comment
      f.create_dataset(datasetname + '/EBSD/Header/Comments', data=chararray)
      f.create_dataset(datasetname + '/EBSD/Header/Coordinate System/ID', data=np.array([np.int32(2)]))
      gtype = str(gridtype)
      chararray = np.chararray(1, itemsize=len(gtype)+1)
      chararray[:] = gtype
      f.create_dataset(datasetname + '/EBSD/Header/Grid Type', data=chararray)
      f.create_dataset(datasetname + '/EBSD/Header/Pattern Center Calibration/AdjMode', data=np.array([np.int32(3)]))
      f.create_dataset(datasetname + '/EBSD/Header/Pattern Center Calibration/x-star', data=np.array([np.float64(indexer.PC[0])]))
      f.create_dataset(datasetname + '/EBSD/Header/Pattern Center Calibration/xAdjCoeff0', data=np.array([np.float64(0.0)]))
      f.create_dataset(datasetname + '/EBSD/Header/Pattern Center Calibration/xAdjCoeff1', data=np.array([np.float64(0.0)]))
      f.create_dataset(datasetname + '/EBSD/Header/Pattern Center Calibration/xAdjCoeff2', data=np.array([np.float64(0.0)]))
      f.create_dataset(datasetname + '/EBSD/Header/Pattern Center Calibration/y-star', data=np.array([np.float64(indexer.PC[1])]))
      f.create_dataset(datasetname + '/EBSD/Header/Pattern Center Calibration/yAdjCoeff0', data=np.array([np.float64(0.0)]))
      f.create_dataset(datasetname + '/EBSD/Header/Pattern Center Calibration/yAdjCoeff1', data=np.array([np.float64(0.0)]))
      f.create_dataset(datasetname + '/EBSD/Header/Pattern Center Calibration/yAdjCoeff2', data=np.array([np.float64(0.0)]))
      f.create_dataset(datasetname + '/EBSD/Header/Pattern Center Calibration/z-star', data=np.array([np.float64(indexer.PC[2])]))
      f.create_dataset(datasetname + '/EBSD/Header/Pattern Center Calibration/zAdjCoeff0', data=np.array([np.float64(0.0)]))
      f.create_dataset(datasetname + '/EBSD/Header/Pattern Center Calibration/zAdjCoeff1', data=np.array([np.float64(0.0)]))
      f.create_dataset(datasetname + '/EBSD/Header/Pattern Center Calibration/zAdjCoeff2', data=np.array([np.float64(0.0)]))


      pcount = phaseIDadd
      nphase = len(indexer.phaseLib)


      for phase in indexer.phaseLib:
        f.create_dataset(datasetname + '/EBSD/Header/Phase/'+str(pcount)+'/LGsymID',
                         data=np.array([np.int32(phase.lauecode)]))
        f.create_dataset(datasetname + '/EBSD/Header/Phase/' + str(pcount) + '/SpaceGroupNumber',
                         data=np.array([np.int32(phase.spacegroup)]))

        f.create_dataset(datasetname + '/EBSD/Header/Phase/' + str(pcount) + '/Lattice Constant a',
                         data=np.array([np.float32(phase.latticeparameter[0]*10)]))

        f.create_dataset(datasetname + '/EBSD/Header/Phase/' + str(pcount) + '/Lattice Constant alpha',
                         data=np.array([np.float32(phase.latticeparameter[3])]))

        f.create_dataset(datasetname + '/EBSD/Header/Phase/' + str(pcount) + '/Lattice Constant b',
                         data=np.array([np.float32(phase.latticeparameter[1] * 10)]))

        f.create_dataset(datasetname + '/EBSD/Header/Phase/' + str(pcount) + '/Lattice Constant beta',
                         data=np.array([np.float32(phase.latticeparameter[4])]))

        f.create_dataset(datasetname + '/EBSD/Header/Phase/' + str(pcount) + '/Lattice Constant c',
                         data=np.array([np.float32(phase.latticeparameter[2] * 10)]))

        f.create_dataset(datasetname + '/EBSD/Header/Phase/' + str(pcount) + '/Lattice Constant gamma',
                         data=np.array([np.float32(phase.latticeparameter[5])]))
        pname = str(phase.phasename)
        chararray = np.chararray(1, itemsize=len(pname)+1)
        chararray[:] = pname
        f.create_dataset(datasetname + '/EBSD/Header/Phase/' + str(pcount) + '/MaterialName', data=chararray)
        f.create_dataset(datasetname + '/EBSD/Header/Phase/' + str(pcount) + '/NumberFamilies',
                         data=np.array([np.int32(phase.npolefamilies)]))


        poles = np.array(phase.polefamilies).astype(np.int32)
        if (phase.lauecode == 62) or (phase.lauecode == 6):
          if poles.shape[-1] == 4:
            poles = poles[:, [0, 1, 3]]

        famtype = np.ones((phase.npolefamilies), dtype=[('H', 'i4'),
                                      ('K', 'i4'),
                                      ('L', 'i4'),
                                      ('Diffraction Intensity', 'f4'),
                                      ('Use in Indexing', 'i1'),
                                      ('Show bands', 'i1'),
                                      ('Hough Rank', 'f4'),
                                      ('Beta Rank', 'f4')])
        famtype['Hough Rank'][:] = -1.0
        famtype['Beta Rank'][:] = -1.0
        famtype['H'] = np.squeeze(poles[:,0])
        famtype['K'] = np.squeeze(poles[:, 1])
        famtype['L'] = np.squeeze(poles[:, 2])

        f.create_dataset(datasetname + '/EBSD/Header/Phase/' + str(pcount) + '/hkl Families',
                         data=famtype)
        pcount +=1

      f.create_dataset(datasetname + '/EBSD/Header/Sample Tilt',
                         data=np.array([np.float64(indexer.sampleTilt)]))

      if indexer.fID is not None:
          if indexer.fID.xStep > 1e-6:
            xstep = np.array([np.float32(indexer.fID.xStep)])
            ystep = np.array([np.float32(indexer.fID.yStep)])

      xstep = np.atleast_1d(np.array([np.float32(xstep)]).squeeze())
      ystep = np.atleast_1d(np.array([np.float32(ystep)]).squeeze())

      f.create_dataset(datasetname + '/EBSD/Header/Step X',
                       data=xstep)
      f.create_dataset(datasetname + '/EBSD/Header/Step Y',
                       data=ystep)

      if ncols is None:
        ncols = 1
        nrows = data.shape[-1]
        if indexer.fID is not None:
            if indexer.fID.nCols is not None:
              ncols = indexer.fID.nCols
              nrows = indexer.fID.nRows
        else:
          if nrows is None:
            nrows = np.ceil(data.shape[-1] / ncols)

      ncols = np.atleast_1d(np.array([np.int32(ncols)]).squeeze())
      nrows = np.atleast_1d(np.array([np.int32(nrows)]).squeeze())

      f.create_dataset(datasetname + '/EBSD/Header/nColumns',
                       data=ncols)
      f.create_dataset(datasetname + '/EBSD/Header/nRows',
                       data=nrows)



      npoints = data[-1].shape[-1]

      eulers = rotlib.qu2eu(data[-1]['quat'])
      phi1 = np.squeeze(eulers[:,0]).astype(np.float32)
      phi = np.squeeze(eulers[:, 1]).astype(np.float32)
      phi2 = np.squeeze(eulers[:, 2]).astype(np.float32)

      f.create_dataset(datasetname + '/EBSD/Data/Phi1',
                       data=phi1)
      f.create_dataset(datasetname + '/EBSD/Data/Phi',
                       data=phi)
      f.create_dataset(datasetname + '/EBSD/Data/Phi2',
                       data=phi2)
      f.create_dataset(datasetname + '/EBSD/Data/IQ',
                       data=(data[-1]['iq']).astype(np.float32))
      f.create_dataset(datasetname + '/EBSD/Data/Pattern Quality',
                       data=(data[-1]['pq']).astype(np.float32))

      f.create_dataset(datasetname + '/EBSD/Data/Fit',
                       data=(data[-1]['fit']).astype(np.float32))



      phaseid = data[-1]['phase']+ phaseIDadd
      ci = data[-1]['cm']
      wh = np.nonzero(phaseid < 0)[0]
      if wh.shape[-1] > 0:
        ci[wh] = -1.0
      phaseid = (phaseid).clip(0)


      f.create_dataset(datasetname + '/EBSD/Data/Phase',
                       data=(phaseid).astype(np.int8))

      f.create_dataset(datasetname + '/EBSD/Data/CI',
                       data=(ci).astype(np.float32))

      x = (np.arange(ncols[0] * nrows[0], dtype=int) % ncols[0]).astype(np.float32) * xstep[0]
      y = (np.arange(ncols[0] * nrows[0], dtype=int) // ncols[0]).astype(np.float32) * ystep[0]

      f.create_dataset(datasetname + '/EBSD/Data/X Position', data=x)
      f.create_dataset(datasetname + '/EBSD/Data/Y Position', data=y)
      f.create_dataset(datasetname + '/EBSD/Data/Valid', data=np.zeros(npoints, dtype=np.int8))
      f.create_dataset(datasetname + '/EBSD/Data/SEM Signal', data=np.zeros(npoints, dtype=np.int32))

      if version == '8.6':
        versiontxt = 'OIM Analysis 8.6.103 x64 [29 Sep 2022]'
      else:
        versiontxt = 'OIM Analysis 9.1.0'
      chararray = np.chararray(1, itemsize=len(versiontxt)+1)
      chararray[:] = versiontxt
      f.create_dataset('Version', data=chararray)
      man = 'EDAX'
      chararray = np.chararray(1, itemsize=len(man)+1 )
      chararray[:] = man
      f.create_dataset('Manufacturer', data=chararray)
      f.close()