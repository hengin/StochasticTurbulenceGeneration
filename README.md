# StochasticTurbulenceGeneration
Python code to generate passive 3d+time fields representing a passive scalar being convected by Kolmogorov turbulence.

Dependencies:
Python 2 or 3 (should be version agnostic) with modules cython, numpy and scipy.
(Probably plus some more that I've forgotten abput)

To run the software first run
> python compile_cython.py build_ext --inplace
to compile the cython file "cone.pyx" used for generating integration weights for radiometer cones.

Then try
> python example_config.py
which will run a basic simulation.

More details will be added tomorrow.
2018-05-07
