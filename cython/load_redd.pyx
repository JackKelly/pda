# distutils: language = c++
# distutils: sources = load_data.cpp

from __future__ import print_function, division

from libcpp.pair cimport pair
from libcpp.list cimport list
from libcpp.string cimport string

# TODO: 
# NEEDS lots of tidying!
# is there a cleaner way to get number of lines (although it's pretty quick rightnow)
# convert to pandas series

# RESOURCES:
# http://stackoverflow.com/questions/3046305/simple-wrapping-of-c-code-with-cython
# http://stackoverflow.com/questions/4495420/passing-numpy-arrays-to-c-code-wrapped-with-cython
# http://docs.cython.org/src/userguide/numpy_tutorial.html
# http://docs.cython.org/src/userguide/wrapping_CPlusPlus.html
# http://docs.cython.org/src/userguide/external_C_code.html

import numpy as np
cimport numpy as np
import pandas as pd

DTYPE = np.uint64
ctypedef np.uint64_t DTYPE_t

cdef extern from "load_data.h":
    int count_lines(string)
    void load_data(string filename, int size, DTYPE_t* timestamps, np.float32_t* power)
    # list[pair[double, double]] load_list(string)

def load(filename="/data/mine/vadeec/jack-merged/channel_3.dat"):
    n_lines = count_lines(filename)
    print(n_lines, "lines found")

    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] timestamps  = np.empty((n_lines, ), dtype=DTYPE)
    cdef np.ndarray[np.float32_t, ndim=1, mode='c'] powers = np.empty((n_lines, ), dtype=np.float32)

    # ascontiguous array from http://stackoverflow.com/a/9116735/732596
    timestamps = np.ascontiguousarray(timestamps, dtype=DTYPE)
    powers = np.ascontiguousarray(powers, dtype=np.float32)

    load_data(filename, n_lines, &timestamps[0], &powers[0])

    dti = pd.DatetimeIndex(timestamps, tz='Europe/London')

    return pd.Series(powers, index=dti)
