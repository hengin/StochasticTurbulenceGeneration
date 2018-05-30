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
from generator import wavenumbers
  
def frequencyToAvoidPeriodicity(gen, wind_x, wind_y):
  """Uses a the same decorrelation frequency for all modes, chosen such
  that one decorrelation time passes in the time it takes for a frozen
  field to loop back on itself.
  """
  tx = gen.Lx/wind_x
  ty = gen.Ly/wind_y
  return lambda K: 1/min(tx,ty)
    
def KolmogorovFrequency(gen, k0):
  """Uses dimensional analysis in the Kolmogorov way to compute
  characteristic decorrelation frequencies for each wavenumber.
  
  k0 -- The wavenumber at which the decorrelation time is one time unit
  """
  return lambda K: (K/k0)**(2/3)
  
def KraichnanFrequency(gen, w0):
  """Uses the approach of Kraichnan (1970): All frequencies are
  independent gaussians with given variance. Furthermore, the time
  evolution is deterministic.
  
  k0 -- The wavenumber at which the decorrelation time is one time unit
  """
  return lambda K: 1j*w0*np.random.standard_normal(K.shape)
    
def frequencyToAvoidPeriodicityAndKolmogorov(gen,wind_x,wind_y,k0):
  tx = gen.Lx/wind_x
  ty = gen.Ly/wind_y
  w_min = 1/min(tx,ty)
  
  return lambda K: w_min*(1 + (K/k0)**2)**(1/3)
    
class IndependentUpdater:
  """Computes an independent realization when called"""
  def update(self, gen, dt):
    gen.independent_realization()
  #def __init__(self):
    # Nothing
    
class IntrinsicUpdater:
  def compute_frequencies(self, gen, frequency):
    """Compute the frequencies of intrinsic variation. The supplied
    function takes a 3D numpy array of spatial frequencies and returns
    temporal frequencies for each. They are allowed to be complex. The
    real part is used as a decorrelation frequency while the imaginary
    part represents an oscillation.
    
    frequency -- Temporal frequency from wavenumber function handle
    """
    # Generate a 3D grid (by broadcasting)
    kx = wavenumbers(gen.Nx,gen.Lx).reshape((gen.Nx,1,1))
    ky = wavenumbers(gen.Ny,gen.Ly).reshape((1,gen.Ny,1))
    kz = wavenumbers(gen.Nz,gen.Lz).reshape((1,1,gen.Nz))
    
    # Compute the magnitude of the wavevector at each point
    K = np.sqrt(kx**2 + ky**2 + kz**2)
    
    # Compute and store the frequencies
    self.W = frequency(K)
    
  def update(self, gen, dt):
    """Update each wavemode as an Ornstein-Uhlenbeck process
       nk(t+dt) = b*nk(t) + a*sqrt(1-b*conj(b))*xi
    """
    # Split the calculation in an effort to reduce memory requirements
    dN = np.random.randn(gen.Nx,gen.Ny,gen.Nz) + 0j
    dN += 1j*np.random.randn(gen.Nx,gen.Ny,gen.Nz)
    # Note that having unit variance on both the imaginary and real
    # parts is correct. Because that is how 'N' is generated in the
    # first place. No 1/sqrt(2) "correction" should be applied.
    dN *= gen.A
    dN *= np.sqrt((1-np.exp(-2*dt*np.real(self.W))))
    
    gen.N *= np.exp(-dt*self.W)
    gen.N += dN
    
  def __init__(self, gen, frequency):
    if callable(frequency):
      self.compute_frequencies(gen, frequency)
    else:
      self.W = frequency
    
class FrozenFieldUpdate:
  def compute_wind_induced_phase(self, gen, dt):
    # Generate 2D grid that can be broadcast to 3D when needed
    kx = wavenumbers(gen.Nx,gen.Lx).reshape((gen.Nx,1,1))
    ky = wavenumbers(gen.Ny,gen.Ly).reshape((1,gen.Ny,1))
    # Note that all factors of 2pi are already included in the k's
    return np.exp(1j*dt*(kx*self.wind_x + ky*self.wind_y))
    
  def update(self, gen, dt):
    # Translate the field by Fourier interpolation
    gen.N *= self.compute_wind_induced_phase(gen, dt)
    
  def __init__(self, wind_x=0, wind_y=0):
    self.wind_x = wind_x
    self.wind_y = wind_y
    
class DynamicWindUpdate:
  def compute_wind_phase_from_velocity(self, gen, wind_x, wind_y):
    kx = wavenumbers(gen.Nx,gen.Lx).reshape((gen.Nx,1,1))
    ky = wavenumbers(gen.Ny,gen.Ly).reshape((1,gen.Ny,1))
    z = np.linspace(0, gen.Lz, gen.Nz)

    self.windPhasePerTime = kx*self.wind_x(z) + ky*self.wind_y(z)

  def update(self, gen, dt):
    # Transform to n(z,k)-space
    n = np.fft.fft(gen.N, axis=2)
    # Apply wind
    n *= np.exp(1j*dt*self.windPhasePerTime)
    # Transform back to use in next iteration
    gen.N = np.fft.ifft(n, axis=2)
  
  def __init__(self, gen, wind_x, wind_y):
    self.compute_wind_phase_from_velocity(self, gen, wind_x, wind_y)
