from __future__ import print_function,division
import sys
if sys.version < (3,):
  # Python 2 automatically executes text passed in input()
  input = raw_input 
  # In Python 2 range is a full blown list instead of a generator
  range = xrange

import numpy as np
import numpy.linalg as la
import scipy.integrate as integrate
from libc.math cimport sqrt, exp, sin, cos
from collections import defaultdict
import cython
  
cdef: # Constant parameters for the weight function
  double v1# = np.cos(el)*np.cos(az)
  double v2# = np.cos(el)*np.sin(az)
  double v3# = np.sin(el)
  double e1x,e1y,e1z,e2x,e2y,e2z
  double a # = 4*np.log(2)/hpbw**2
  double nfact # = a/pi
  int mx,my,mz
  int ix,iy,iz
  double kx,ky,kz # Corner function parameters
  # Position of the "lower left" corner of the cell being integrated
  # relative to the cone origin
  double x0,y0,z0
  double dx,dy,dz # Physical cube size
  double cutoff
  
cdef int i = 0
cdef double weight(double ix, double iy, double iz):
  global v1, v2, v3, a, mx,my,mz,kx,ky,kz, dx,dy,dz, i
  i += 1
  cdef:
    # Compute physical position wrt origin of cone
    double x = x0 + dx*ix
    double y = y0 + dy*iy
    double z = z0 + dz*iz
    # Compute scalar product with direction vector
    #phi = np.arccos((v1*x+v2*y+v3*z)/np.sqrt(r2))
    # Accurate to within 0.5% for phi < 10 degrees
    double tmp = v1*x+v2*y+v3*z
    double r2, phi2, f
  #print(v1,v2,v3,x,y,z,tmp)
  if tmp > 0:
    r2 = z*z + y*y + x*x # Distance from origin
    phi2 = 1 - tmp*tmp/r2 # Angle squared (assuming << 1 rad)
    f = (mx + kx*ix)*(my + ky*iy)*(mz + kz*iz) # interpolant
    return nfact*f*exp(-a*phi2)/r2
  else:
    return 0.0

def index2weight_accurate(cone, ix,iy,iz, weightDict):
  global x0,y0,z0
  x0,y0,z0 = ix*dx-cone.origin[0],iy*dy-cone.origin[1],iz*dx-cone.origin[2]
  #print(ix,iy,iz,x0,y0,z0)
  
  wMax = 0
  for cx in (0,1):
    for cy in (0,1):
      for cz in (0,1):
        # Define parameters of corner functions
        global mx,my,mz,kx,kt,kz
        mx,my,mz = 1-cx,1-cy,1-cz
        kx,ky,kz = 2*cx-1,2*cy-1,2*cz-1

        w = integrate.nquad(weight, [(0,1),(0,1),(0,1)],
                            opts={'epsabs': cutoff/(dx*dy*dz), 'epsrel': 1e-3})
        w = w[0]*dx*dy*dz
        #w = cubature(weight, 3, 1, (0,0,0), (1,1,1), abserr=cutoff, relerr=1e-3)
        if w > wMax:
          wMax = w
        weightDict[ix+cx,iy+cy,iz+cz] += w
  return wMax
  
def index2weight_cheap(cone, ix,iy,iz, weightDict):
  global x0,y0,z0
  x0,y0,z0 = ix*dx-cone.origin[0],iy*dy-cone.origin[1],iz*dx-cone.origin[2]
  #print(ix,iy,iz,x0,y0,z0)
  
  cdef:
    int n = 4
    double delta = 1./(n-1)
    double w = 0
    int iix,iiy,iiz
  
  wMax = 0
  for cx in (0,1):
    for cy in (0,1):
      for cz in (0,1):
        # Define parameters of corner functions
        global mx,my,mz,kx,kt,kz
        mx,my,mz = 1-cx,1-cy,1-cz
        kx,ky,kz = 2*cx-1,2*cy-1,2*cz-1

        # Gaussian integration extended to 3d
        w = 0
        #ip = [0.5*(1+x) for x in (-sqrt(3/5), 0, sqrt(3/5))]
        #iw = [0.5*x for x in (5/9, 8/9, 5/9)]
        ip = [0.5*(1+x) for x in (-0.861136,-0.339981,0.339981,0.861136)]
        iw = [0.5*x for x in (0.347855,0.652145,0.652145,0.347855)]
        for iix in range(len(ip)):
          for iiy in range(len(ip)):
            for iiz in range(len(ip)):
              w += iw[iix]*iw[iiy]*iw[iiz]*weight(ip[iix],ip[iiy],ip[iiz])
        w *= dx*dy*dz
              
        # w = (1*weight(0.0,0.0,0.0) + 2*weight(0.0,0.0,0.5) + 1*weight(0.0,0.0,1.0) \
           # + 2*weight(0.0,0.5,0.0) + 4*weight(0.0,0.5,0.5) + 2*weight(0.0,0.5,1.0) \
           # + 1*weight(0.0,1.0,0.0) + 2*weight(0.0,1.0,0.5) + 1*weight(0.0,1.0,1.0) \
           # + 2*weight(0.5,0.0,0.0) + 4*weight(0.5,0.0,0.5) + 2*weight(0.5,0.0,1.0) \
           # + 4*weight(0.5,0.5,0.0) + 8*weight(0.5,0.5,0.5) + 4*weight(0.5,0.5,1.0) \
           # + 2*weight(0.5,1.0,0.0) + 4*weight(0.5,1.0,0.5) + 2*weight(0.5,1.0,1.0) \
           # + 1*weight(1.0,0.0,0.0) + 2*weight(1.0,0.0,0.5) + 1*weight(1.0,0.0,1.0) \
           # + 2*weight(1.0,0.5,0.0) + 4*weight(1.0,0.5,0.5) + 2*weight(1.0,0.5,1.0) \
           # + 1*weight(1.0,1.0,0.0) + 2*weight(1.0,1.0,0.5) + 1*weight(1.0,1.0,1.0))
        #w *= dx*dy*dx/(1*8 + 2*12 + 4*6 + 8*1)
        
        # w2 = integrate.nquad(weight, [(0,1),(0,1),(0,1)],
                            # opts={'epsabs': cutoff/(dx*dy*dz), 'epsrel': 1e-3})
        # w2 = w2[0]*dx*dy*dz
        
        # print(w, w2, abs(w-w2)/w2)
        
        #w = cubature(weight, 3, 1, (0,0,0), (1,1,1), abserr=cutoff, relerr=1e-3)
        if w > wMax:
          wMax = w
        weightDict[ix+cx,iy+cy,iz+cz] += w
  return wMax
  
  
@cython.cdivision(True)
cdef double lineIntegral(double w1, double w2):
  #MUSTDO: Give kx/ky/kz units of 1/length (HAVEDONE?)
  cdef:
    # Compute offsetted direction vector
    double wx = v1 + w1*e1x + w2*e2x
    double wy = v2 + w1*e1y + w2*e2y
    double wz = v3 + w1*e1z + w2*e2z
    double wSqr = w1*w1 + w2*w2
  # Renormalize (second order accurate in 'w')
  wx *= (1 - 0.5*wSqr)
  wy *= (1 - 0.5*wSqr)
  wz *= (1 - 0.5*wSqr)
  cdef:
    # Compute intersection distances
    double tx = 1./0. if wx == 0 else ((ix+1)*dx - x0 if wx > 0 else ix*dx - x0)/wx
    double ty = 1./0. if wy == 0 else ((iy+1)*dy - y0 if wy > 0 else iy*dy - y0)/wy
    double tz = 1./0. if wz == 0 else ((iz+1)*dz - z0 if wz > 0 else iz*dz - z0)/wz
    double t = min(tx,ty,tz)
    double ax = mx + kx*(x0/dx - ix)
    double ay = my + ky*(y0/dy - iy)
    double az = mz + kz*(z0/dz - iz)
    
    double Kx = kx*wx/dx
    double Ky = ky*wy/dy
    double Kz = kz*wz/dz
    
    double p0 = ax*ay*az
    double p1 = ax*ay*Kz + ax*Ky*az + Kx*ay*az
    double p2 = ax*Ky*Kz + Kx*ay*Kz + Kx*Ky*az
    double p3 = Kx*Ky*Kz
    
  # if my == 0:
    # print('callA', mx,my,mz, t, p0, p1, p2, p3)
    # print('callB', ax,ay,az,Kx,Ky,Kz)
    # print('callC', kx,ky,kz)
  # Evaluate integrated polynomial by Horner's method
  # And multiply by the angular weighting function
  return nfact*exp(-a*wSqr)*t*(p0 + t*(p1/2 + t*(p2/3 + t*p3/4)))
  
def index2weight_originCell(cone, ixl,iyl,izl, weightDict):
  global x0,y0,z0
  x0,y0,z0 = cone.origin
  
  wMax = 0
  for cx in (0,1):
    for cy in (0,1):
      for cz in (0,1):
        # Define parameters of corner functions
        global mx,my,mz,kx,ky,kz,ix,iy,iz
        mx,my,mz = 1-cx,1-cy,1-cz
        kx,ky,kz = 2*cx-1,2*cy-1,2*cz-1
        ix,iy,iz = ixl,iyl,izl
        
        intLim = sqrt(5/2)*hpbw
        w = integrate.nquad(lineIntegral, [(-intLim,intLim),(-intLim,intLim)],
                            opts={'epsabs': cutoff, 'epsrel': 1e-6})
        w = w[0]
        #print(cx,cy,cz,w)
        if w > wMax:
          wMax = w
        weightDict[ixl+cx,iyl+cy,izl+cz] += w
  return wMax
  
def compute_weights(Nx,Ny,Nz, Lx,Ly,Lz, cone):
  global dx,dy,dz
  dx,dy,dz = Lx/Nx, Ly/Ny, Lz/Nz

  global hpbw
  hpbw = cone.hpbw*np.pi/180
  el = cone.elevation*np.pi/180
  az = cone.azimuth*np.pi/180

  # Compute the global variables needed by the weight function
  global a, nfact, v1, v2, v3
  a = 4*np.log(2)/hpbw**2
  #nfact = a/np.pi
  # TODO: Figure out why the sum of weights is a factor 2 too large
  nfact = 0.5*a/np.pi # The factor 0.5 compensates for unknown error
  v1 = cos(el)*cos(az)
  v2 = cos(el)*sin(az)
  v3 = sin(el)
  
  global e0,e1,e2
  e0 = np.array((v1,v2,v3))
  e1 = np.array((1.,0.,0.)) # x-hat
  e2 = np.array((0.,1.,0.)) # y-hat
  e1 -= np.dot(e1,e0)*e0
  e1 /= la.norm(e1)
  e2 -= np.dot(e2,e0)*e0
  e2 -= np.dot(e2,e1)*e1
  e2 /= la.norm(e2)
  
  # Convert to index space
  iv1,iv2,iv3 = v1/dx, v2/dy, v3/dz
  # Normalize the z-component
  iv1 /= iv3
  iv2 /= iv3
  
  # The unit for cutoff is length. The typical value for a cell near
  # the origin is the length of an intersection with a line divided by
  # 8 (the number of corners). But much smaller values are possible in
  # the case of grazing lines.
  # Far away the angular and radial dependencies are negligible and we
  # get a characteristic value of a/pi*dx*dy*dz/dist**2. This is a good
  # starting point for the cutoff.
  global cutoff,i
  cutoff = 1e-4*nfact*dx*dy*dz/(Lz/v3)**2
  
  # Initialize weight dictionary/map
  weightDict = defaultdict(lambda : 0)
  x1,y1,z1 = cone.origin
  x1,y1,z1 = x1/dx,y1/dy,z1/dz
  
  ix,iy,iz = int(x1),int(y1),int(z1)
  originCells = [(ix,iy,iz)]
  if x1 == ix:
    originCells.append((ix-1,iy,iz))
  if y1 == iy:
    for cell in list(originCells):
      originCells.append((cell[0],iy-1,iz))
  if z1 == iz:
    for cell in list(originCells):
      originCells.append((cell[0],cell[1],iz-1))    
  
  for iz in range(int(z1), Nz-1):
    x = x1 + (iz+0.5-z1)*iv1
    y = y1 + (iz+0.5-z1)*iv2
    
    ix = int(x)
    iy = int(y)
    print(ix,iy,iz)
    cellsToVisit = [(ix,iy)]
    cellsAdded = {(ix,iy): 1}
    # bfs loop over cells, stopping when cutoff is reached
    while len(cellsToVisit) > 0:
      ix,iy = cellsToVisit.pop()
      
      dist = sqrt((dx*(.5+ix-x1))**2 + (dy*(.5+iy-y1))**2 + (dz*(.5+iz-z1))**2)
      
      if (ix,iy,iz) in originCells:
        lasti = i
        w = index2weight_originCell(cone, ix,iy,iz, weightDict)
        #print('origin:', w, i-lasti)
      elif max(dx,dy,dz)/dist > 0.75*hpbw:
        lasti = i
        w = index2weight_accurate(cone, ix,iy,iz, weightDict)
        #print('accurate:', w, i-lasti)
      else:
        lasti = i
        w = index2weight_cheap(cone, ix,iy,iz, weightDict)
        #print('cheap:', w, i-lasti)
        
      if w > cutoff:
        for cx,cy in [(-1,0),(1,0),(0,-1),(0,1)]:
          pointToAdd = (ix+cx,iy+cy)
          if pointToAdd not in cellsAdded:
            cellsAdded[pointToAdd] = 1
            cellsToVisit.append(pointToAdd)
   
  for key in list(weightDict.keys()):
    if weightDict[key] < 0.1*cutoff:
      del weightDict[key]
  inds, weights = weightDict.keys(), weightDict.values()
  # Extract 1d indices
  inds = [ind[2] + (ind[1] % Ny)*Nz 
        + (ind[0] % Nx)*Nz*Ny for ind in inds]
  
  # Check for duplicates
  if len(inds) > len(set(inds)):
    raise ValueError('Infeasible cone parameters')
    
  # Sort indices and weights so that they are as cache friendly as possible
  order = sorted(list(range(len(inds))), key=lambda i:inds[i])
  
  # Package into numpy arrays and return
  inds = np.array([inds[i] for i in order], dtype=np.int32)
  weights = np.array([weights[i] for i in order])
  
  return (weights, inds)
  