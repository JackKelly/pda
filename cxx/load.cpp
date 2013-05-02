#include <iostream>
#include <fstream>
#include <time.h>

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
const size_t countDataPoints( std::fstream& fs )
{
    int count = 0;
    char line[255];

    while ( ! fs.eof() ) {
        fs.getline( line, 255 );
        if ( isdigit(line[0]) ) {
            count++;
        }
    }

    std::cout << "Found " << count << " data points in file." << std::endl;

    // Return the read pointer to the beginning of the file
    fs.clear();
    fs.seekg( 0, std::ios_base::beg ); // seek to beginning of file

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

int main()
{
    std::cout << "Start" << std::endl;
    std::fstream fs;
    fs.open("/data/mine/vadeec/jack-merged/channel_3.dat", std::fstream::in);

    size_t file_length = countDataPoints(fs);
    tm* timestamps = new tm[file_length];
    int* power = new int[file_length];

    size_t count = 0;
    time_t time;
    char ch;
    while (!fs.eof()) {
        ch = fs.peek();
        if (isdigit(ch)) {
            fs >> time;
            timestamps[count] = *gmtime(&time);
            fs >> power[count];
            count++;
        }
        fs.ignore(255, '\n');  // skip to next line
    }
    fs.close();

    print_ts(timestamps[0]);
    std::cout << " " << power[0] << std::endl;

    print_ts(timestamps[file_length-1]);
    std::cout << " " << power[file_length-1] << std::endl;

    std::cout << "done." << std::endl;
    return 0;
}
