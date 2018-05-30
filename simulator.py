# Make the code agnostic to python version
# In Python 2 the print function is special and called without parenthesis
# also '/' is used for integer division (5/2 == 2) while in Python 3 that
# role is reserved for '//' (5/2 == 2.5 and 5//2 == 2)
from __future__ import print_function,division
import sys
if sys.version_info < (3,):
  # Python 2 automatically executes text passed in input()
  input = raw_input 
  # In Python 2 range is a full blown list instead of a generator
  range = xrange
  
import os
from datetime import datetime
from glob import glob
import numpy as np

import geometry

def safetysave(filename, numpyarray2d):
  lastInd = max(np.where(np.logical_not(np.isnan(numpyarray2d[:,-1]))))
  np.save(filename, numpyarray2d[0:lastInd+1,:])

def simulate(gen, objects, updaters, sampleTimes, simulationName, cleanStart=True):
  currentDir = os.path.basename(os.path.abspath('.'))
  if currentDir != simulationName:
    print(datetime.now(), 'Started outside simulation name directory')
    if not os.path.isdir(simulationName):
      os.makedirs(simulationName)
      print('Created new directory: ', simulationName)
    os.chdir(simulationName)
    print('Changed directory to: ', simulationName)
  else:
    print(datetime.now(), 'Started inside simulation name directory')
  
  if cleanStart and os.path.isfile('weight_cache.pkl'):
    os.remove('weight_cache.pkl')
  objects = geometry.compute_or_load_weights(gen, objects, cacheName='weight_cache.pkl')
    
  existingDataFiles = glob('_'.join([simulationName,'out_[0-9][0-9][0-9]*.npy']))
  if len(existingDataFiles) == 0:
    outFileName = '_'.join([simulationName,'out_000.npy'])
  else:
    numbers = [int(x.split('_')[-1][0:-4]) for x in existingDataFiles]
    numbers.sort()
    outFileName = '_'.join([simulationName, 'out_%03d.npy' % (numbers[-1]+1,)])
    
  # Create signal array initialized with NaN everywhere
  signals = np.full((len(sampleTimes), len(objects)+1,), np.nan)
  
  def safetysave():
    print('    Saving progress')
    lastInd = np.max(np.where(np.logical_not(np.isnan(signals[:,-1]))))
    np.save(outFileName, signals[0:lastInd+1,:])
  
  try:
    tlast = sampleTimes[0]
    for i in range(len(sampleTimes)):
      if i % 100 == 0 and i > 0:
        print(datetime.now(), 'Safety save')
        safetysave()
    
      print(datetime.now(), 'Generating sample #%d' % (i+1,))
      # Get time difference (ok even when i=0 since dt=0)
      dt = sampleTimes[i] - tlast
      tlast = sampleTimes[i]
      
      # Update the field
      updStart = datetime.now()
      for upd in updaters:
        upd.update(gen, dt) 
      updDone = datetime.now()
      print('    Updated in time ', updDone-updStart)
      
      # Get 3D realization and compute signals
      intStart = datetime.now()
      n = gen.get_current_field()
      oneDview = n.reshape((gen.Nx*gen.Ny*gen.Nz,)) # 1d pointer to the 'n'-data
      signals[i,0] = sampleTimes[i]
      for j in range(len(objects)):
        signals[i,j+1] = np.sum(objects[j].weights*oneDview[objects[j].inds])
      intDone = datetime.now()
      print('    Computed integrals in time ', intDone-intStart)
  except KeyboardInterrupt: # Doesn't work because of scipy
    print(datetime.now(), 'Interrupted by Ctrl-C')
  else:
    print(datetime.now(), 'Simulation completed')
  
  print(datetime.now(), 'Exiting')
  safetysave()
  