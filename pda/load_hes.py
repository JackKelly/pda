from __future__ import print_function, division
import pandas as pd
import tables as tb
import datetime
import time

# Notes:
# Inverval IDs:
# 1 = 1month, 2 mins
# 2 = 1 year, 2 mins
# 3 = 1 year, 10 mins (IGNORE THIS???)

# houses = load list of house IDs
# appliances_in_house = dict mapping house ID to list of appliance IDs

DATETIME_FMT = '%Y-%m-%d %H:%M:%S'

start_time = time.time()

def str_to_datetime(date_str, time_str):
    datetime_str = date_str + ' ' + time_str
    return datetime.datetime.strptime(datetime_str, DATETIME_FMT)

f = open('/data/HES/CSV data/appliance_group_data.csv', 'r')

houses = [202116]

for house in houses:
    # TODO: f.seek_to_beginning()
    line = f.readline()
    dict_of_series = {}
    data_list = []
    index_list = []
    prev_appliance = 9999
    i = 0
    
    def save_data(dl, il):
#        import ipdb; ipdb.set_trace()
        # Save data
        rng = pd.DatetimeIndex(il, freq='2T')
        series = pd.Series(dl, index=rng)
        try:
            dict_of_series[prev_appliance].append(series)
        except KeyError:
            dict_of_series[prev_appliance] = series
        

    while line and i < 5000000:
#        i += 1
        line = f.readline()
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
        time_str = split_line[5].strip()

        data_list.append(watts)
        index_list.append(str_to_datetime(date_str, time_str))

    if data_list:
        save_data(data_list, index_list)
    df = pd.DataFrame.from_dict(dict_of_series)
    print(df)
        
end_time = time.time()
print("Total seconds =", end_time-start_time)
