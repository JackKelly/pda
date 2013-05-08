#!/usr/bin/python
from __future__ import print_function, division
import matplotlib.pyplot as plt
import pda.metoffice as metoffice
import pda.dataset as ds
import pda.stats
import numpy as np

print("Opening metoffice data...")
weather = metoffice.open_daily_xls('/data/metoffice/Heathrow_DailyData.xls')

print("Opening power data...")
# 25 = lighting circuit # (R^2 = 0.343)
# 8 = kitchen lights (R^2 = 0.194)
# 2 = boiler (versus radiation R^2 = 0.052, 
#             versus mean_temp R^2 = 0.298,
#             versus max_temp  R^2 = 0.432,
#             versus min_temp  R^2 = 0.212)
# 3 = solar (R^2 = 0.798)


# DATA_DIR = '/data/mine/vadeec/jack-merged/'
DATA_DIR  = '/data/mine/vadeec/jack/137'

print("Loading dataset...")
dataset = ds.load_dataset(DATA_DIR)

print("Calculating on durations per day for each channel...")
for i in range(len(dataset)):
    name = dataset[i].name
    print("Loading on durations for", name)
    dataset[i].on_durations = dataset[i].on_duration_per_day(tz_convert='UTC')
    print("Got {} days of data from on_duration_per_day for {}."
          .format(dataset[i].on_durations.size, name))

# Create a matrix for storing correlation results in
results = np.zeros([len(dataset), len(weather.columns)])

fig = plt.figure()
ax = fig.add_subplot(111)

ON_DURATION_THRESHOLD = 0.1 # hours
MIN_DAYS_PER_CHAN = 1
for i_d in range(len(dataset)):
    for i_w in range(len(weather.columns)):
        print(dataset[i_d].name, weather.columns[i_w])
        on_durations = dataset[i_d].on_durations
        on_durations = on_durations[on_durations > ON_DURATION_THRESHOLD]
        if on_durations.size > MIN_DAYS_PER_CHAN:
            try:
                ax, results[i_d][i_w] = pda.stats.correlate(weather.ix[:,i_w], 
                                                            on_durations, ax)
            except IndexError as e:
                print(str(e))
                print(dataset[i_d].name, "has insufficient data")

# TODO:
# display a heatmap
# only display top N appliances in heatmap (take the max of each row?)
# don't require axes in correlate()

print(results)
# print("Plotting...")


# ax, r_squared = pda.stats.correlate(weather.max_temp, on[on > threshold], ax)

# print("R^2={:.3f}".format(r_squared))
plt.show()

print("Done")
