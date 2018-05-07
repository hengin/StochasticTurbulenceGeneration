# To compile the cone.pyx-file, run the following:
# python compile_cython.py build_ext --inplace
from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("cone.pyx", annotate=True)
)