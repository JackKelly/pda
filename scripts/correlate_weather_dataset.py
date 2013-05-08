#!/usr/bin/python
from __future__ import print_function, division
import matplotlib.pyplot as plt
import pda.metoffice as metoffice
import pda.dataset as ds
import pda.stats
import numpy as np
from scipy.stats import linregress

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


DATA_DIR = '/data/mine/vadeec/jack-merged/'
# DATA_DIR  = '/data/mine/vadeec/jack/137'

print("Loading dataset...")
dataset = ds.load_dataset(DATA_DIR)

print("Calculating on durations per day for each channel...")
names = []
for i in range(len(dataset)):
    name = dataset[i].name
    print("Loading on durations for", name)
    dataset[i].on_durations = dataset[i].on_duration_per_day(tz_convert='UTC')
    print("Got {} days of data from on_duration_per_day for {}."
          .format(dataset[i].on_durations.size, name))
    names.append(name)

# Create a matrix for storing correlation results in
results = np.zeros([len(dataset), len(weather.columns)])

ON_DURATION_THRESHOLD = 0.1 # hours
MIN_DAYS_PER_CHAN = 10
for i_d in range(len(dataset)):
    for i_w in range(len(weather.columns)):
        print(dataset[i_d].name, weather.columns[i_w])
        on_durations = dataset[i_d].on_durations
        on_durations = on_durations[on_durations > ON_DURATION_THRESHOLD]
        if on_durations.size > MIN_DAYS_PER_CHAN:
            try:
                x_aligned, y_aligned = pda.stats.align(weather.ix[:,i_w], 
                                                       on_durations)
            except IndexError as e:
                print(str(e))
                print(dataset[i_d].name, "has insufficient data")
            else:
                # calculate linear regression
                slope, intercept, r_value, p_value, std_err = linregress(x_aligned.values,
                                                                         y_aligned.values)
                results[i_d][i_w] = r_value**2

# TODO:
# display a heatmap
# only display top N appliances in heatmap (take the max of each row?)
# don't require axes in correlate()

print(results)
print("Plotting...")

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)

from matplotlib.ticker import MultipleLocator

ax.imshow(results, interpolation='nearest')

ax.tick_params(labelsize=8, label2On=True)

# X axis
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.set_xticklabels(weather.columns, rotation=90)
ax.xaxis.set_ticks_position('both')

# Y axis
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.set_yticklabels(names)

# General
plt.show()

print("Done")
