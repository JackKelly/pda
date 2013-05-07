#include "load_pwr_data_cpp.h"

/**
 * Interesting notes:
 * 
 * PYTHON
 * https://github.com/cython/cython/blob/master/Cython/Includes/cpython/datetime.pxd
 *
 * Apparently np.datetime64 internally stores the datetime as 
 * the number of nano/micro/milli/...-seconds since some epoch
 * (often the 1970 one), as an unsigned 64 bit int (np.uint64).
 *
 * https://groups.google.com/forum/?fromgroups=#!topic/cython-users/FlFBf9EJC28
 * http://docs.cython.org/src/tutorial/strings.html
 * http://grahamstratton.org/straightornamental/entries/cython
 * http://wesmckinney.com/blog/?p=278
 * http://hg.python.org/cpython/file/c3656dca65e7/Modules/datetimemodule.c
 *
 * C
 * http://en.cppreference.com/w/c/chrono/tm
 * http://en.cppreference.com/w/c/chrono/gmtime
 */

/**
 * @brief Find the number of numeric data points in file.
 *
 * @return number of data points.
 */
size_t count_lines(const std::string filename)
{
    std::fstream fs;
    fs.open(filename.c_str(), std::fstream::in);

    size_t count = 0;
    char line[255];

    while (!fs.eof()) {
        fs.getline(line, 255);
        if (isdigit(line[0])) {
            count++;
        }
    }

    fs.close();
    return count;
}

void print_ts(const tm& timestamp)
{
    std::cout << 1900 + timestamp.tm_year << "-" 
              << 1 + timestamp.tm_mon << "-"
              << timestamp.tm_mday << " "
              << timestamp.tm_hour << ":"
              << timestamp.tm_min << ":"
              << timestamp.tm_sec << " ";
}

std::list<std::pair<npy_float64, npy_float64> > load_list(const std::string filename)
{
    std::fstream fs;
    std::list<std::pair<npy_float64, npy_float64> > data;
    char ch;
    double timestamp, power;

    fs.open(filename.c_str(), std::fstream::in);
    
    while (!fs.eof()) {
        ch = fs.peek();
        if (isdigit(ch)) {
            fs >> timestamp;
            fs >> power;
            data.push_back(std::pair<npy_float64, npy_float64>(timestamp, power));
        }
        fs.ignore(255, '\n');  // skip to next line
    }
    fs.close();

    return data;
}

/**
 * @param filename: full path of filename
 * @param size: number of lines to load
 * @param timestamps: returned timestamp data
 * @param powers: returned power data
 */
void load_data(const std::string filename, const size_t size,
               npy_uint64* timestamps, npy_float32* powers)
{
    std::fstream fs;
    char ch;
    double timestamp;
    fs.open(filename.c_str(), std::fstream::in);

    size_t i=0;    
    while (!fs.eof()) {
        ch = fs.peek();
        if (isdigit(ch)) {
            fs >> timestamp;
            timestamps[i] = timestamp * 1000000000;
            fs >> powers[i];
            if (++i > size) {
                break;
            }
        }
        fs.ignore(255, '\n');  // skip to next line
    }
    fs.close();
}
