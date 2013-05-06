#include "load_data.h"

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
const size_t count_lines(std::fstream& fs)
{
    size_t count = 0;
    char line[255];

    while (!fs.eof()) {
        fs.getline(line, 255);
        if (isdigit(line[0])) {
            count++;
        }
    }

    // Return the read pointer to the beginning of the file
    fs.clear();
    fs.seekg(0, std::ios_base::beg); // seek to beginning of file

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

std::list<std::pair<npy_float64, npy_float64> > load_data(std::string filename)
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

// int main()
// {
//     std::list<std::pair<double, double> > data;
//     data = load_data("/data/mine/vadeec/jack-merged/channel_3.dat");
    
//     std::cout << data.front().first << " " << data.front().second << std::endl;

//     return 0;
// }
