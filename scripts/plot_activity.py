#!/usr/bin/python
from __future__ import print_function, division
import matplotlib.pyplot as plt
import pda.dataset as ds
from pda.channel import indicies_of_periods
import numpy as np
from matplotlib.ticker import MultipleLocator
import pandas as pd

# TODO:
# * Save dataframe (maybe create dataset class)
# * PWR_ON_THRESHOLD per channel
#
# * Maybe create a new script for the following:
# * Current plot is pretty hard to interpret.  Instead:
#   - Y axis just represents channels.  Just a single 24hr period.
#   - use several pixels per channel to make them nice and fat.
#   - heatmap.  Increment pixel whenever a channel is on for that period.

PWR_ON_THRESHOLD = 4 # watts
MIN_DAYS_PER_CHAN = 10
DATA_DIR = '/data/mine/vadeec/jack-merged/'
#DATA_DIR  = '/data/mine/vadeec/jack/137'

print("Loading dataset...")
dataset = ds.load_dataset(DATA_DIR)

# create pd.DataFrame of all channels
print("Creating DataFrame...")
chans = []
for channel in dataset:
    chans.append((channel.name, channel.series))
df = pd.DataFrame.from_items(chans)

print("Creating bitmap...")
day_range, day_boundaries = indicies_of_periods(df.index, 'D')
N_DAYS = day_range.size - 1
N_CHANNELS = df.columns.size
MINS_PER_DAY = 24 * 60
WIDTH = MINS_PER_DAY # 1 pixel per minute
HEIGHT = N_DAYS * N_CHANNELS
bitmap = np.zeros((HEIGHT, WIDTH), dtype=np.float)

for day_i in range(N_DAYS):
    day = day_range[day_i]
    try:
        start_index, end_index = day_boundaries[day]
    except KeyError:
        # No data available for this day
        continue

    data_for_day = df[start_index:end_index]
    data_for_day_minutely = data_for_day.resample('T', how='max').to_period()

    midnight = day.asfreq('T', how='start')
    first_minute = data_for_day_minutely.index[0] - midnight
    last_minute = data_for_day_minutely.index[-1] - midnight

    for chan_i in range(N_CHANNELS):
        on = data_for_day_minutely.ix[:,chan_i] > PWR_ON_THRESHOLD
        y = HEIGHT - chan_i - (day_i * N_CHANNELS) - 1
        try:
            bitmap[y][first_minute:last_minute+1] = on
        except ValueError as e:
            print(e, day, "likely because there's a gap in the day's data")
            break

# import ipdb; ipdb.set_trace()

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1)
im = ax.imshow(bitmap, interpolation='nearest')
plt.show()
