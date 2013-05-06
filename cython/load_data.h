#include <iostream>
#include <fstream>
#include <list>
#include <time.h>
#include <string>
#include "pyconfig.h"
#include "Python.h"
#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"

const size_t count_lines(std::fstream& fs);
void print_ts(const tm& timestamp);
std::list<std::pair<npy_float64, npy_float64> > load_data(std::string filename);
