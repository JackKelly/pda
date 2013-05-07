# distutils: language = c++

"""
RESOURCES:
http://stackoverflow.com/questions/3046305/simple-wrapping-of-c-code-with-cython
http://stackoverflow.com/questions/4495420/passing-numpy-arrays-to-c-code-wrapped-with-cython
http://docs.cython.org/src/userguide/numpy_tutorial.html
http://docs.cython.org/src/userguide/wrapping_CPlusPlus.html
http://docs.cython.org/src/userguide/external_C_code.html
"""

from __future__ import print_function, division
# Uncomment the 2 lines below to use load_list() function from load_data.cpp
# from libcpp.pair cimport pair
# from libcpp.list cimport list
from libcpp.string cimport string
import numpy as np
cimport numpy as np
import pandas as pd
from os.path import join

# Data types for timestamps (TS = TimeStamp)
TS_DTYPE = np.uint64
ctypedef np.uint64_t TS_DTYPE_t

# Data types for power data (PW = PoWer)
PW_DTYPE = np.float32
ctypedef np.float32_t PW_DTYPE_t

cdef extern from 'load_pwr_data_cpp.h':
    Py_ssize_t count_lines(const string)
    void load_data(const string filename, const Py_ssize_t size,
                   TS_DTYPE_t* timestamps, PW_DTYPE_t* power)
#    list[pair[double, double]] load_list(string)

def load(filename, tz='Europe/London'):
    """Load a two-column, space-separated .dat file containing power data.

    Args:
        filename (str): filename including full path.
        tz (str): (optional) timezone.  Defaults to 'Europe/London'.

    Returns:
        pd.Series
    """

    # Declare C types
    cdef np.ndarray[TS_DTYPE_t, ndim=1, mode='c'] timestamps
    cdef np.ndarray[PW_DTYPE_t, ndim=1, mode='c'] powers
    cdef Py_ssize_t n_lines
    
    # Find the number of lines containing data
    n_lines = count_lines(filename)

    # Setup empty numpy arrays to store data
    timestamps = np.empty((n_lines, ), dtype=TS_DTYPE)
    powers = np.empty((n_lines, ), dtype=PW_DTYPE)

    # ascontiguousarray trick from http://stackoverflow.com/a/9116735/732596
    timestamps = np.ascontiguousarray(timestamps, dtype=TS_DTYPE)
    powers = np.ascontiguousarray(powers, dtype=PW_DTYPE)

    # use our C++ function load_data to pull the data from filename
    load_data(filename, n_lines, &timestamps[0], &powers[0])

    # Convert to pandas objects
    dti = pd.DatetimeIndex(timestamps, tz=tz)
    return pd.Series(powers, index=dti)
