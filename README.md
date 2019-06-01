# StochasticTurbulenceGeneration
Python code to generate passive 3d+time fields representing a passive scalar being convected by Kolmogorov turbulence.

The method and some background is described in the Master's thesis 
H. Gingsjö, "Modelling and simulation of tropospheric water vapour with gaussian random fields - Time dependence beyond the frozen flow hypothesis", M.S. thesis, Department of Space, Earth and Environment, Chalmers University of Technology, Gothenburg, Sweden, 2018. [Online]. Available: http://studentarbeten.chalmers.se/publication/255146-modelling-and-simulation-of-tropospheric-water-vapour-with-gaussian-random-fields-time-dependence-be [link valid 2019-06-01]

Dependencies:
Python 2 or 3 (should be version agnostic) with modules cython, numpy and scipy. The small visualization program additionally requires pyqtgraph and OpenGL.
(Probably plus some more that I've forgotten about)


A visualization program that can show some sample realizations of random 3D-fields with different statistics can be run with
```bash
python visualizingGUI.py
```


To run the software first run
```bash
python compile_cython.py build_ext --inplace
```

to compile the cython file "cone.pyx" used for generating integration weights for radiometer cones.

Then try
```bash
python example_config.py
```

which will run a basic simulation.


This creates a directory called 'example' in which a cache file, weight_cache.pkl, and an output file, example_out_000.npy, are written. If the code is run again it will generate an independent realization and store that in example_out_001.npy, and so on.

The files can be loaded using numpy as follows:
```python
import numpy as np
signals = np.load('example_out_000.npy')
```

Now signals is a 2d array where the first column, `signals[:,0]`, is the timestamps when the process was sampled and the other columns, `signals[:,1:]`, are line integrals over the refractivity field. Parameters for those lines are stored in weight_cache.pkl.

It could be nice to have the output in a '.mat'-file containing both the generated time series and information about the geometry. It would only take a few minutes to add that feature.
