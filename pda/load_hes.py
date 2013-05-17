from __future__ import print_function, division
import pandas as pd
import tables as tb
import datetime

# houses = load list of house IDs
# appliances_in_house = dict mapping house ID to list of appliance IDs

DATETIME_FMT = '%Y-%m-%d %H:%M:%S'

def str_to_datetime(date_str, time_str):
    datetime_str = date_str + ' ' + time_str
    return datetime.datetime.strptime(datetime_str, DATETIME_FMT)

def load_house(filename, house):
    f = open(filename, 'r')
    line = f.readline() # ignore header in file
    line = f.readline().strip()
    dict_of_series = {}
    data_list = []
    index_list = []
    prev_appliance = 9999
    
    def save_data(dl, il):
        # Save data
        rng = pd.DatetimeIndex(il, freq='2T')
        # would have liked to use PeriodIndex but can't seem
        # to use PeriodIndex with 2-minute freq.
        series = pd.Series(dl, index=rng)
        try:
            dict_of_series[prev_appliance].append(series)
        except KeyError:
            dict_of_series[prev_appliance] = series
        

    while True:
        line = f.readline().strip()
        if not line:
            break
        split_line = line.split(',')

        interval_id = split_line[0]
        if interval_id == '3':
            continue  # ignore "1 year, 10 mins" data

        house_id_from_file = int(split_line[1])
        if house_id_from_file != house:
            continue
        
        appliance = int(split_line[2])
        if appliance != prev_appliance:
            if data_list:
                save_data(data_list, index_list)
                data_list = []
                index_list = []
            prev_appliance = appliance

        date_str = split_line[3]
        watts = int(split_line[4])
        time_str = split_line[5]

        data_list.append(watts)
        index_list.append(str_to_datetime(date_str, time_str))

    f.close()

    if data_list:
        save_data(data_list, index_list)
    df = pd.DataFrame.from_dict(dict_of_series)
    return df

# '/data/HES/CSV data/appliance_group_data.csv' 
def load_dataset(filename='/data/HES/CSV data/test.csv'):
    houses = [202116]

    for house in houses:
        df = load_house(filename, house)

    return df
