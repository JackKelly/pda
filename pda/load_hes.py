from __future__ import print_function, division
import pandas as pd
import numpy as np
import tables as tb
import datetime

# houses = load list of house IDs
# appliances_in_house = dict mapping house ID to list of appliance IDs

DATETIME_FMT = '%Y-%m-%d %H:%M:%S'

def str_to_datetime(date_str, time_str):
    datetime_str = date_str + ' ' + time_str
    py_datetime = datetime.datetime.strptime(datetime_str, DATETIME_FMT)
    return np.datetime64(py_datetime)

def load_house(filename, house):
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
            appliance_data[np.uint16(appliance)] = ([dt], [watts])

    f.close()

    dict_of_series = {}
    for appliance, data in appliance_data.iteritems():
        rng = pd.DatetimeIndex(data[0], freq='2T')
        # would have liked to use PeriodIndex but can't seem
        # to use PeriodIndex with 2-minute freq.
        series = pd.Series(data[1], index=rng, dtype=np.uint16)
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
            df = load_house(filename, house)
            if df.empty:
                print('empty.')
            else:
                print(len(df), 'rows read. Done.')
                store['house_' + house] = df
            store.flush()
    finally:
        store.close()
