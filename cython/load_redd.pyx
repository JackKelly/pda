# distutils: language = c++
# distutils: sources = load_data.cpp

from libcpp.pair cimport pair
from libcpp.list cimport list
from libcpp.string cimport string

cdef list[pair[double, double]] data

cdef extern from "load_data.h":
    list[pair[double, double]] load_data(string)

data = load_data("/data/mine/vadeec/jack-merged/channel_3.dat")

plist = []
for datum in data:
    plist.append(datum)
