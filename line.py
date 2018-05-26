# Make the code agnostic to python version.
# In Python 2 the print function is special and called without parenthesis
# also '/' is used for integer division (5/2 == 2) while in Python 3 that
# role is reserved for '//' (5/2 == 2.5 and 5//2 == 2).
from __future__ import print_function,division
import sys
if sys.version < (3,):
  # Python 2 automatically executes text passed in input().
  input = raw_input 
  # In Python 2, range is a full blown list instead of a generator.
  range = xrange

import sys
from collections import defaultdict
import numpy as np
from math import floor
from scipy.integrate import quad, romb
import numpy.linalg as la

def trilinearWeights(ind, xo,yo,zo, dx,dy,dz, dX, dY, dZ, weightDict):
  # Physical length of the cell. Needed to transform the integral from
  # index space to physical space.
  l = np.sqrt((dX*dx)**2 + (dY*dy)**2 + (dZ*dz)**2)
  # Loop over cell corners
  for ix in (0,1):
    for iy in (0,1):
      for iz in (0,1):
        # Define corner functions
        fx = lambda x: (1-ix) + (2*ix-1)*x
        fy = lambda y: (1-iy) + (2*iy-1)*y
        fz = lambda z: (1-iz) + (2*iz-1)*z
        # Multiply them together
        f = lambda t: fx(xo+t*dx)*fy(yo+t*dy)*fz(zo+t*dz)
        # Compute and store integral over cell
        weightDict[ind[0]+ix,ind[1]+iy,ind[2]+iz] += l*quad(f, 0, 1)[0]

def computeWeights(inds, pos, dX,dY,dZ):
  # The default value is important for the "+=" statements above
  weightDict = defaultdict(lambda: 0)
  
  for i in range(len(inds)):
    x1,y1,z1 = pos[i]
    x2,y2,z2 = pos[i+1]
    dx,dy,dz = x2-x1,y2-y1,z2-z1
    
    xo = x1-inds[i][0]
    yo = y1-inds[i][1]
    zo = z1-inds[i][2]
    #print xo,yo,zo
    
    trilinearWeights(inds[i], xo,yo,zo, dx,dy,dz, dX, dY, dZ, weightDict)
    #optimalWeights(inds[i], xo,yo,zo, dx,dy,dz, dX, dY, dZ, weightDict)
    
  return weightDict
  
# Below is a commented out block of code that employs optimal
# interpolation. This doesn't seem to work very well unless the
# corner covariance is sufficiently large.
# # If r1 is Mx3 and r2 is Nx3 then this function returns
# # a matrix of size MxN
# def covariance(r1,r2):
  # global L, H # Parameters for the covariance function
  # R1 = np.reshape(r1, (np.shape(r1)[0],1,3))
  # R2 = np.reshape(r2, (1,np.shape(r2)[0],3))
  # dr = la.norm(R1-R2, axis=2)
    
  # #iso = 1/(1 + (dr/L)**(2./3))
  # iso = 1/(1 + dr**(2./3))
  # Z1 = np.reshape(r1[:,2],(np.shape(r1)[0],1))
  # Z2 = np.reshape(r2[:,2],(1,np.shape(r2)[0]))
  # vrt = np.exp((Z1+Z2)/H)
  # return iso*vrt
  
# # Simple code to comparare integration routines:
# #>import numpy as np
# #>from np import trapz
# #>from scipy.integrate import simps, romb
# #>b = b = np.column_stack([(n+1)*x**n-1 for n in [0, 0.5, 1, 1.5, 2, 3, 4]])
# #>np.row_stack([trapz(b,x,axis=0),simps(b, x, axis=0),romb(b,dx=x[1]-x[0],axis=0)])
  
# def optimalWeights(ind,xo,yo,zo,dx,dy,dz, dX, dY, dZ, weights):
  # # Compute covariance matrix
  # C = np.zeros((8,8))
  # corners = [(ix,iy,iz) for ix in (0,1) for iy in (0,1) for iz in (0,1)]
  # #rc = [(dX*ix,dY*iy,dZ*iz) for ix in (0,1) for iy in (0,1) for iz in (0,1)]
  # rc = [(ix,iy,iz) for ix in (0,1) for iy in (0,1) for iz in (0,1)]
  # rc = np.array(rc, dtype=np.float)
  # C = covariance(rc, rc)
  # #for i in range(8):
  # #  for j in range(8):
  # #    C[i,j] = covariance(r[i],r[j])
  
  # # Define samples
  # t = np.linspace(0,1,64+1)
  # R = np.row_stack([xo+dx*t,yo+dy*t,zo+dz*t]).transpose()
  
  # # Compute covariance vectors (size 8x65)
  # X = covariance(rc, R)
  # # Compute "inversion coefficients" (size 8x65)
  # Y = la.solve(C,X)
  
  # # Compute integral by Romberg's method (repeated Richardson extrapolation)
  # l = np.sqrt((dx*dX)**2 + (dy*dY)**2 + (dz*dZ)**2)
  # values = romb(Y, dx=l*(t[1]-t[0]), axis=1)
  # #print l,sum(values)
  # for i,ixyz in enumerate(corners):
    # ix,iy,iz = ixyz
    # #print ind[0]+ix,ind[1]+iy,ind[2]+iz,values[i]
    # weights[ind[0]+ix,ind[1]+iy,ind[2]+iz] += values[i]#*l/sum(values)
    
def walk(x1, y1, z1, x2, y2, z2):
  tDeltaX = np.array(1.)/abs(x2-x1) # Works even if x1==x2 due to Inf
  tDeltaY = np.array(1.)/abs(y2-y1)
  tDeltaZ = np.array(1.)/abs(z2-z1)
  
  # int(floor(  )) is needed in case x1,y1 or z1 is negative
  X,Y,Z = int(floor(x1)),int(floor(y1)),int(floor(z1))
  
  if x2 > x1:
    stepX = 1
    tMaxX = np.array(X+1.-x1)*tDeltaX if X != x1 else tDeltaX
  else:
    stepX = -1
    tMaxX = -np.array(X+0.-x1)*tDeltaX if X != x1 else tDeltaX
    
  if y2 > y1:
    stepY = 1
    tMaxY = np.array(Y+1.-y1)*tDeltaY if Y != y1 else tDeltaY
  else:
    stepY = -1
    tMaxY = -np.array(Y+0.-y1)*tDeltaY if Y != y1 else tDeltaY
    
  if z2 > z1:
    stepZ = 1
    tMaxZ = np.array(Z+1.-z1)*tDeltaZ if Z != z1 else tDeltaZ
  else:
    stepZ = -1
    tMaxZ = -np.array(Z+0.-z1)*tDeltaZ if Z != z1 else tDeltaZ
  
  #print tDeltaX,tDeltaY,tDeltaZ
  #print tMaxX,tMaxY,tMaxZ
  
  Xe,Ye,Ze = floor(x2),floor(y2),floor(z2)
  
  inds = [(X,Y,Z)]
  pos = [(x1,y1,z1)]
  while (X,Y,Z) != (Xe,Ye,Ze):
    if tMaxX <= tMaxY and tMaxX <= tMaxZ:
      lastMaxT = tMaxX
      tMaxX += tDeltaX
      X += stepX
    elif tMaxY <= tMaxZ and tMaxY <= tMaxX:
      lastMaxT = tMaxY
      tMaxY += tDeltaY
      Y += stepY
    elif tMaxZ <= tMaxY and tMaxZ <= tMaxX:
      lastMaxT = tMaxZ
      tMaxZ += tDeltaZ
      Z += stepZ
    else: # Impossible unless something is NaN?
      assert False
    # If some of the t:s are equal, it must be handled.
    if tMaxX == lastMaxT:
      #print 'Impossible'
      tMaxX += tDeltaX
      X += stepX
    if tMaxY == lastMaxT:
      #print 'tMaxX == tMaxY'
      tMaxY += tDeltaY
      Y += stepY
    if tMaxZ == lastMaxT:
      #print 'tMaxX == tMaxZ or tMaxY =0 tMaxZ'
      tMaxZ += tDeltaZ
      Z += stepZ 
    
    inds.append((X,Y,Z))
    #print X,Y,Z
    x,y,z = x1+lastMaxT*(x2-x1),y1+lastMaxT*(y2-y1),z1+lastMaxT*(z2-z1)
    pos.append((x,y,z))
    if Z > 1000: #TODO: Put in proper error checking instead
      print('Error in the ray tracing module')
      sys.exit()
  pos.append((x2,y2,z2))
  
  return inds,pos
  
def line2weights(gen, line):
  #print 'Line: el: %d, az: %d' % (line.elevation,line.azimuth)
  el = line.elevation*np.pi/180
  az = line.azimuth*np.pi/180
  vz = np.sin(el)
  vx = np.cos(el)*np.cos(az)
  vy = np.cos(el)*np.sin(az)
  
  dx,dy,dz = gen.Lx/gen.Nx, gen.Ly/gen.Ny, gen.Lz/gen.Nz
  
  # Compute start and end points of the line
  x1,y1,z1 = line.origin
  # Vertical position just below the highest grid layer before wrap-around
  z2 = gen.Lz*(gen.Nz-1.0001)/gen.Nz
  y2 = y1 + vy*(z2-z1)/vz
  x2 = x1 + vx*(z2-z1)/vz
  
  # Find all intersected grid cells by ray-tracing in the integer grid
  inds,pos = walk(x1/dx,y1/dy,z1/dx, x2/dx,y2/dy,z2/dz)
  # Integrate up weights
  weightDict = computeWeights(inds, pos, dx,dy,dz)
    
  inds, weights = weightDict.keys(), weightDict.values()
  
  # Extract 1d indices
  inds = [ind[2] + (ind[1] % gen.Ny)*gen.Nz 
                 + (ind[0] % gen.Nx)*gen.Nz*gen.Ny
          for ind in inds]
  
  # Check for duplicates
  if len(inds) > len(set(inds)):
    raise ValueError('Infeasible line parameters')
    
  # Sort indices and weights so that they are as cache friendly as possible
  order = sorted(list(range(len(inds))), key=lambda i:inds[i])
  
  # Package into numpy arrays and return
  inds = np.array([inds[i] for i in order], dtype=np.int32)
  weights = np.array([weights[i] for i in order])
  return (weights, inds)
  