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
  
import numpy as np

"""References:
Treuhaft & Lanyi (1987): "The effect of the dynamic wet troposphere on radio interferometric measurement", Radio Science, Volume 22, Number 2, Pages 251-265, March-April 1987

Ishimaru ...: TO BE ADDED

Gradinarsky, L. P. (2002), Sensing atmospheric water vapor using radio waves, Ph.D. thesis, Chalmers University of Technology, Technical Report No. 436.
"""

# TODO: Check if this really corresponds to Treuhaft & Lanyi in any respect
# This is really just a heuristic extension of the Fourier transform of
# Cn2*r**(2/3)
# which only exists in an asymptotic sense.
# See: https://en.wikipedia.org/wiki/Fourier_transform#Distributions
# [Link valid on 2018-05-03]
def IshimaruSpectrum(Cn2, L0, km):
  """See Nilsson p. 17"""
  return lambda kx,ky,kz: 0.033*Cn2*(kx*kx+ky*ky+kz*kz + 1/L0**2)**(-11/6) \
                          *np.exp(-(kx*kx+ky*ky+kz*kz)/(km*km))

def TreuhaftLanyiCovariance(Cn2, L):
  """Returns a function handle that computes the isotropic covariance
  in the model of Treuhaft & Lanyi (1987). Note that they specify the
  structure function instead.
  """
  return lambda x,y,z: Cn2*L**(4/3)/(L**(2/3) + (x*x+y*y+z*z)**(1/3))

def GradinarskyCovariance(Cn2, L, C0):
  """Returns a function handle that computes the non-isotropic
  covariance introduced by Gradinarsky (2002)
  """
  return lambda x,y,z: Cn2*L**(4/3)/(L**(2/3) + (x*x + y*y + C0*z*z)**(1/3))
  
def positionvectors(N,L):
    """Helper for computing DFT compatible positions.
    N -- number of grid points
    L -- physical length of box (more precisely the period)
    """
    # Vector of integers
    r = np.arange(N)
    # Make everything continuously periodic (assuming C(inf)=0)
    r[r > N//2] = r[r > N//2] - N
    # Give it physical dimensions
    r = r*L/N
    return r

def wavenumbers(N,L):
  """Helper for computing correct DFT wavenumbers.
  N -- number of grid points
  L -- physical length of box (more precisely the period)
  """
  # Vector of integers (essentially discrete frequencies)
  k = np.arange(N)
  # Make everything satisfy Nyquist, by making some negative
  k[k > N//2] = k[k > N//2] - N
  # Give it physical dimensions
  k = 2*np.pi*k/L
  return k

class HomogeneousGenerator:
  def compute_amplitude_from_covariance(self, covariance):
    """Computes the spectral amplitude based on an inputted covariance.
    
    covariance -- A function handle that computes covariance from three
                  numpy arrays of positions.
    """
    # Generate a "3D" grid (by exploiting broadcasting)
    x = positionvectors(self.Nx,self.Lx).reshape((self.Nx,1,1))
    y = positionvectors(self.Ny,self.Ly).reshape((1,self.Ny,1))
    z = positionvectors(self.Nz,self.Lz).reshape((1,1,self.Nz))
    
    C = covariance(x,y,z) # Compute covariance function
    
    # Compute the variance each independent mode should have
    PhiTilde = np.abs(np.fft.fftn(C, axes=(0,1,2)))
    # Correctly normalize for applications to come
    PhiTilde /= self.Nx*self.Ny*self.Nz
    del C # Deallocate
    
    # Store the amplitude instead of the amplitude squared
    self.A = np.sqrt(PhiTilde)  
    
  def compute_amplitude_from_spectrum(self, spectrum):
    # Generate a 3D grid (by broadcasting)
    kx = wavenumbers(self.Nx,self.Lx).reshape((self.Nx,1,1))
    ky = wavenumbers(self.Ny,self.Ly).reshape((1,self.Ny,1))
    kz = wavenumbers(self.Nz,self.Lz).reshape((1,1,self.Nz))
    
    # TODO: Test that this normalization is consistent
    # To prevent convergence issues, just use a Gaussian Phi/C
    dkx = 2*np.pi/self.Lx
    dky = 2*np.pi/self.Ly
    dkz = 2*np.pi/self.Lz
    self.A = np.sqrt(dkx*dky*dkz*spectrum(kx,ky,kz))
    
  def independent_realization(self):
    self.N = self.A*(np.random.randn(self.Nx,self.Ny,self.Nz)
                + 1j*np.random.randn(self.Nx,self.Ny,self.Nz))
                
  def get_current_field(self):
    n = np.fft.fftn(self.N, axes=(0,1,2))
    n = np.real(n)
    if self.inhomogeneity != None:
      n *= self.inhomogeneity
    return n
    
  def __init__(self, Nx,Ny,Nz, Lx,Ly,Lz,
               covariance=None, spectrum=None, inhomogeneity=None):
    """ Initialize a new HomogeneousGenerator. Needs exactly one of
    'covariance' and 'spectrum' to be set. Otherwise a ValueError is
    thrown.
    """
    # Handle errors
    if covariance != None and spectrum != None:
      raise ValueError('Accepts exactly one of the covariance and spectrum\n'
                     + 'parameters. Not both at the same time!')
    if covariance == None and spectrum == None:
      raise ValueError('Either the spectrum or covariance must be specified.')
    
    # If everything seems ok, construct the object
    self.Nx,self.Ny,self.Nz = Nx,Ny,Nz
    self.Lx,self.Ly,self.Lz = Lx,Ly,Lz
    # Compute frequency domain amplitudes
    if covariance != None:
      self.compute_amplitude_from_covariance(covariance)
    else:
      self.compute_amplitude_from_spectrum(spectrum)
      
    if inhomogeneity == None:
      self.inhomogeneity = None
    else:
      z = Lz*np.arange(Nz).reshape((1,1,Nz))/Nz
      self.inhomogeneity = np.sqrt(inhomogeneity(z))
      
    # Initial frequency domain realization
    self.independent_realization()  

