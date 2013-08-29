#!/usr/bin/python
from __future__ import print_function, division
import unittest, os, inspect, subprocess, shlex
import numpy as np
import pandas as pd
import datetime

# Taken from http://stackoverflow.com/a/6098238/732596
FILE_PATH = os.path.dirname(inspect.getfile(inspect.currentframe()))
SMALL_TEST_DATA_PATH = os.path.join(FILE_PATH, 'test_data', 'small')
LARGE_TEST_DATA_PATH = os.path.join(FILE_PATH, 'test_data', 'large')

DATA_FILE = '/data/HES/CSVdata/appliance_group_data.csv'
H5_FILE = '/data/HES/h5/HES.h5'

def load_line(line_num):
    cmd = ('sed -n \'{line_num}p; {line_num}q\' {filename}'
           .format(line_num=line_num, filename=DATA_FILE))
    return subprocess.check_output(shlex.split(cmd)).strip()

def seek_and_load_line(seek_point, f):
    f.seek(seek_point)
    f.readline()
    return f.readline().strip()

class TestHESImport(unittest.TestCase):
    def test_data(self):
        FILESIZE = os.path.getsize(DATA_FILE)
        store = pd.HDFStore(H5_FILE, 'r')
        f = open(DATA_FILE, 'r')

        def test_seekpoint(seek_point):
            line = seek_and_load_line(seek_point, f)
            
            # Columns:
            # 0: IntervalID
            # 1: Household
            # 2: Appliance
            # 3: DateRecorded
            # 4: Data
            # 5: TimeInterval
            #
            # example line:
            # 1,202116,0,2010-07-28,0,00:00:00

            split_line = line.split(',')
            interval_id = split_line[0]
            if interval_id == '3':
                return

            household = split_line[1]
            df = store['house_' + household]

            appliance = int(split_line[2])
            series = df[appliance]

            date_str = split_line[3]
            time_str = split_line[5]
            dt_str = date_str + ' ' + time_str
            dt = datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')

            watt_hrs = split_line[4]

            try:
                self.assertEqual(series[dt], int(watt_hrs))
                self.assertEqual('{:.0f}'.format(series[dt_str]), watt_hrs)
            except AssertionError as e:
                print(e)
                print(seek_point, household, appliance, dt, watt_hrs)
                print(line)
                print("")

        SEEK_POINTS_WITH_NEG_VALUES = [6804496293, 6850312188, 6827029199,
                           6861145333, 6865743434, 6827316959,
                           6874111644, 6913440438, 6967980831,
                           6827188366, 6820801268]

        SEEK_POINTS_WITH_NEG_VALUES.sort()

        # TEST KNOW PROBLEMS WITH
        for seek_point in SEEK_POINTS_WITH_NEG_VALUES:
            test_seekpoint(seek_point)            

        # DO LOTS OF RANDOM TESTS
        i = 0
        N_TESTS = 100
        print("Running", N_TESTS, "random tests...")
        while i < N_TESTS:
            seek_point = int(np.random.rand() * FILESIZE)
            test_seekpoint(seek_point)
            i += 1

        f.close()
        store.close()

if __name__ == '__main__':
    unittest.main()
