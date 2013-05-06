from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules = cythonize(
                               "load_redd.pyx",
                               sources=["load_data.cpp"],
                               language="c++",
     ))
