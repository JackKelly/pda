#include <iostream>
#include <fstream>
#include <list>
#include <time.h>
#include <string>
#include "pyconfig.h"
#include "Python.h"
#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"

int count_lines(std::string filename);
void print_ts(const tm& timestamp);
std::list<std::pair<npy_float64, npy_float64> > load_list(std::string filename);
// PyObject* load_data(std::string filename);
// void load_data(std::string filename, int size, double* array);
void load_data(std::string filename, int size, npy_uint64* timestamps, npy_float32* powers);
