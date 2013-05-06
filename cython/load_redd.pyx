# distutils: language = c++
# distutils: sources = load_data.cpp

from __future__ import print_function, division

from libcpp.pair cimport pair
from libcpp.list cimport list
from libcpp.string cimport string

# TODO: 
# pass both timestamps and power data back
# use np.datetime64 throughout
# NEEDS lots of tidying!
# is there a cleaner way to get number of lines (although it's pretty quick rightnow)
# convert to pandas series

# RESOURCES:
# http://stackoverflow.com/questions/3046305/simple-wrapping-of-c-code-with-cython
# http://stackoverflow.com/questions/4495420/passing-numpy-arrays-to-c-code-wrapped-with-cython
# http://docs.cython.org/src/userguide/numpy_tutorial.html
# http://docs.cython.org/src/userguide/wrapping_CPlusPlus.html

import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef extern from "load_data.h":
    int count_lines(string)
    # np.ndarray[DTYPE_t, ndim=1] load_data(string)
    void load_data(string filename, int size, double* array)
    # list[pair[double, double]] load_list(string)

def load(filename="/data/mine/vadeec/jack-merged/channel_3.dat"):
    n_lines = count_lines(filename)
    print(n_lines, "lines found")
    cdef np.ndarray[np.double_t, ndim=1] array = np.empty((n_lines, ), dtype=np.double)
    load_data(filename, n_lines, <double*>array.data)
    return array

    # cdef list[pair[double, double]] data
    # data = load_data(filename)

    # cdef np.ndarray[DTYPE_t, ndim=2] ndret = np.array(data)

    # cdef np.ndarray[DTYPE_t, ndim=2] ndret = np.empty([len(data), 2], dtype=DTYPE)


