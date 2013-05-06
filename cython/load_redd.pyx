# distutils: language = c++
# distutils: sources = load_data.cpp

from __future__ import print_function, division

from libcpp.pair cimport pair
from libcpp.list cimport list
from libcpp.string cimport string

# TODO: use np.datetime64 throughout

cdef extern from "load_data.h":
    list[pair[double, double]] load_data(string)

import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cimport cython
@cython.boundscheck(False)
def load(filename="/data/mine/vadeec/jack-merged/channel_3.dat"):

    cdef list[pair[double, double]] data
    data = load_data(filename)

    cdef np.ndarray[DTYPE_t, ndim=2] ndret = np.array(data)

#    cdef np.ndarray[DTYPE_t, ndim=2] ndret = np.empty([len(data), 2], dtype=DTYPE)

    return ndret
