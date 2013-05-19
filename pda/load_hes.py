from __future__ import print_function, division
import pandas as pd
import numpy as np
import sys
import datetime

"""This script imports the data in the 8.6GB 'appliance_group_data.csv'
file and creates a compressed HDF5 file.

Usage
-----

> houses = load_list_of_houses('</directory/to/ipsos - public.csv>')
> load_dataset('</directory/to/appliance_group_data.csv', houses)


Notes for re-writing this script
--------------------------------

This script is VERY SLOW (takes about 24hrs on my i5 with an SSD)
largely because it re-reads every line of the CSV file for every
house.

If I ever have to re-import the data then I should re-write this
script to work like this:

1) Go through the whole file once to create a dict which maps
   (house_id, appliance_id) -> (start_seek_point, end_seek_point)
   (ignoring interval_id==3)
2) Then loop through each house id and appliance id and use the map
   to only load the appropriate parts of the CSV file.

"""

DATETIME_FMT = '%Y-%m-%d %H:%M:%S'

def str_to_datetime(date_str, time_str):
    datetime_str = date_str + ' ' + time_str
    py_datetime = datetime.datetime.strptime(datetime_str, DATETIME_FMT)
    return np.datetime64(py_datetime)

def load_house(filename, house):
    # Columns (comma separated):
    # 0: IntervalID   e.g. '1'
    # 1: Household    e.g. '202166'
    # 2: Appliance    e.g. '0'
    # 3: DateRecorded e.g. '2010-07-28'
    # 4: Data         e.g. '0'
    # 5: TimeInterval e.g. '02:12:00'
    
    f = open(filename, 'r')
    line = f.readline() # ignore header in file
    appliance_data = {}
    
    while True:
        line = f.readline().strip()
        if not line:
            break

        split_line = line.split(',')

        house_id_from_file = split_line[1]
        if house_id_from_file != house:
            continue

        interval_id = split_line[0]
        if interval_id == '3':
            continue  # ignore "1 year, 10 mins" data

        appliance = int(split_line[2])
        date_str = split_line[3]
        watts = int(split_line[4])
        time_str = split_line[5]

        dt = str_to_datetime(date_str, time_str)
        try:
            appliance_data[appliance][0].append(dt)
            appliance_data[appliance][1].append(watts)
        except KeyError:
            appliance_data[appliance] = ([dt], [watts])

    f.close()

    dict_of_series = {}
    for appliance, data in appliance_data.iteritems():
        rng = pd.DatetimeIndex(data[0], freq='2T')
        # would have liked to use PeriodIndex but can't seem
        # to use PeriodIndex with 2-minute freq.
        series = pd.Series(data[1], index=rng, dtype=np.int16)
        # Don't use unsigned ints because "appliances" 251-255
        # record temperature values and sometimes go negative.
        dict_of_series[appliance] = series

    df = pd.DataFrame.from_dict(dict_of_series)
    return df


def load_list_of_houses(filename='/data/HES/CSV data/ipsos - public.csv'):
    """Returns a list of strings."""
    f = open(filename, 'r')
    lines = f.readlines()[1:]
    f.close()
    houses = [l.split(',')[68] for l in lines]
    return houses


# '/data/HES/CSV data/appliance_group_data.csv'
def load_dataset(filename='/data/HES/CSV data/test.csv', houses=['202116']):

    store = pd.HDFStore('HES.h5', 'w', complevel=9, complib='blosc')
    
    try:
        for house in houses:
            print('Loading house ', house, '... ', sep='', end='')
            sys.stdout.flush()
            df = load_house(filename, house)
            if df.empty:
                print('empty.')
            else:
                print(len(df), 'rows read. Done.')
                store['house_' + house] = df
            store.flush()
    finally:
        store.close()
