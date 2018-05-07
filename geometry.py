"""Some elaboration on the coordinate system may be useful. A grid
point with zero based index (ix,iy,iz) is located at the physical
coordinate (ix/Lx,iy/Ly,iz/Lz). This is consistent with (Lx,Ly,Lz)-
periodicity.

Due to periodicity, there is no physically well defined center. However,
in the indexing sense, the most reasonable horizontal center is (N-1)/2.

Below is an illustration with Lx=Nx=7 and Ly=Ny=4. The indexing centrum
is marked with 'C' and grid points with 'o'.
3 o---o---o---o---o---o---o P 
  |   |   |   |   |   |   | e ->
2 o---o---o---o---o---o---o r ->
  |   |   |   C   |   |   | i ->
1 o---o---o---o---o---o---o o
  |   |   |   |   |   |   | dic 
0 o---o---o---o---o---o---o---o
  0   1   2   3   4   5   6   0
"""

# Make the code agnostic to python version
# In Python 2 the print function is special and called without parenthesis
# also '/' is used for integer division (5/2 == 2) while in Python 3 that
# role is reserved for '//' (5/2 == 2.5 and 5//2 == 2)
from __future__ import print_function,division
import sys
if sys.version < (3,):
  # Python 2 automatically executes text passed in input()
  input = raw_input 
  # In Python 2 range is a full blown list instead of a generator
  range = xrange
  
import generator
import pickle
from datetime import datetime
import numpy as np

# Load actual geometry routines
import line
import cone

class GeometricalObject(object):
  def __init__(self, origin=[0.0,0.0,0.0], elevation=90, azimuth=0):
    self.origin = np.array(origin, dtype=np.float)
    self.elevation = float(elevation)
    self.azimuth = float(azimuth)
    
    # Initialize the index and weight data
    self.weights = None
    self.inds = None


class Line(GeometricalObject):
  def compute_weights(self, gen):
    self.weights, self.inds = line.line2weights(gen,self)
  
  def __init__(self, **kwargs):
    self.type = 'line'
    super(Line, self).__init__(**kwargs)


class Cone(GeometricalObject):
  def compute_weights(self, gen):
    self.weights, self.inds = cone.compute_weights(gen.Nx,gen.Ny,gen.Nz, gen.Lx,gen.Ly,gen.Lz, self)
  
  def __init__(self, hpbw=6, **kwargs):
    self.type = 'cone'
    self.hpbw = float(hpbw)
  
    super(Cone, self).__init__(**kwargs)

def compute_or_load_weights(gen, objects, cacheName=None):
  if cacheName != None:
    try:
      print(cacheName)
      with open(cacheName, 'rb') as file:
        loaded_objects = pickle.load(file)
      print(datetime.now(), 'Loaded precomputed integration weights from file', cacheName)
      # TODO: Error checking  if the file contains incompatible data
      return loaded_objects
    except (IOError, EOFError):
      print(datetime.now(), 
          'Couldn\'t read weights from %s. Computing new ones' % cacheName)
  
  for obj in objects:
    print('    Processing', obj.type, 'el=%.0f, az=%.0f' % (obj.elevation,obj.azimuth,))
    obj.compute_weights(gen)
  print(datetime.now(), 'Done computing weights')
  
  if cacheName != None:
    print('Saving in', cacheName)
    try:
      with open(cacheName, 'wb') as cacheFile:
        pickle.dump(objects, cacheFile)
    except IOError:
      print(datetime.now(), 'File operation failed', cacheName)
    else:
      print(datetime.now(), 'Done writing weights to file')
  
  return objects


