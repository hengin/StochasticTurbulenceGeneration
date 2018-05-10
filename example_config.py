from __future__ import division

import generator
import updater
from geometry import Cone, Line
import simulator
import numpy as np

# 1a)
#Nx,Ny,Nz = 256,256,128
Nx,Ny,Nz = 128,128,64 # half
Lx,Ly,Lz = 50e3,50e3,10e3

# 1b)
objects = []
#objects.append(Cone(origin=[Lx/2,Ly/2,0],hpbw=5,elevation=60,azimuth=30))
for el in (30,45,60):
  for az in range(0,360,15):
    #objects.append(Cone(elevation=el,azimuth=az,origin=(0,0,0),hpbw=3))
    objects.append(Line(elevation=el,azimuth=az,origin=(Lx/2+0.1,Ly/2+0.1,0)))
    
# 2a) Create the appropriate object
# Spatial specification
Cn2 = 1e-14
L = 1e3
H = 3e3
gen = generator.HomogeneousGenerator(Nx,Ny,Nz,Lx,Ly,Lz,
        covariance=generator.TreuhaftLanyiCovariance(Cn2, L),
        inhomogeneity=lambda z:np.exp(-z/H))

# Configure time evolution
wind_x = Lx/Nx # 1 square per unit time
wind_y = 0.1*wind_x
updaters = []
updaters.append(updater.FrozenFieldUpdate(wind_x,wind_y))
updaters.append(updater.IntrinsicUpdater(gen,
                updater.frequencyToAvoidPeriodicity(gen,wind_x,wind_y)))

# Creates a directory with output, caches and log files
# If such a directory already exists it tries to reuse old data
# Unless 'cleanStart' is set to True (default is False)
simulationName = 'example'
T = 0.5*np.arange(1000)
simulator.simulate(gen, objects, updaters, T, simulationName, cleanStart=False)
