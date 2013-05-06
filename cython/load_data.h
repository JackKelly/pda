#include <iostream>
#include <fstream>
#include <list>
#include <time.h>
#include <string>

const size_t count_lines(std::fstream& fs);
void print_ts(const tm& timestamp);
std::list<std::pair<double, double> > load_data(std::string filename);
