from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize(
                            "load_pwr_data.pyx",
                            sources=["load_pwr_data_cpp.cpp"],
                            language="c++",
     ))
