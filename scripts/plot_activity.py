#!/usr/bin/python
from __future__ import print_function, division
import matplotlib.pyplot as plt
import pda.dataset as ds
from pda.channel import indicies_of_periods
import numpy as np
from matplotlib.ticker import MultipleLocator
import pandas as pd
from pandas.tseries.offsets import DateOffset

THRESHOLD = 0.005 # remove any days will less kwh or hours on per day
MIN_DAYS_PER_CHAN = 10
#DATA_DIR = '/data/mine/vadeec/jack-merged/'
DATA_DIR  = '/data/mine/vadeec/jack/137'

print("Loading dataset...")
dataset = ds.load_dataset(DATA_DIR)

# create pd.DataFrame of all channels
print("Creating DataFrame...")
dct = {}
for channel in dataset:
    dct[channel.name] = channel.series
df = pd.DataFrame(dct)

# TODO:
# then I can select time ranges once for entire dataframe

period_range, period_boundaries = indicies_of_periods(df.index, 'D')
N_DAYS = period_range.size - 1
N_CHANNELS = len(dataset)
MINS_PER_DAY = 24 * 60
WIDTH = MINS_PER_DAY # 1 pixel per minute
HEIGHT = N_DAYS * N_CHANNELS
data = np.zeros((HEIGHT, WIDTH), dtype=np.float)

import ipdb; ipdb.set_trace()
for chan_i in range(N_CHANNELS): # "chan_i" means "channel index"
    print("Loading", dataset[chan_i].name)
#    period_range, period_boundaries = dataset[chan_i].days()
    for day_i in range(N_DAYS):
        day_start_index, day_end_index = period_boundaries[day_i]
        if day_start_index is None:
            continue
        data_for_day = dataset[chan_i].series[day_start_index:day_end_index]
        day = period_range[day_i]
        for minute_i in range(MINS_PER_DAY):
            t = day + DateOffset(minutes=minute_i)
            data_for_minute = dataset[chan_i].series[t.strftime('%Y-%m-%d %H:%M')]
            if any(data_for_minute > dataset[chan_i].on_power_threshold):
                y = HEIGHT - chan_i - (day_i * N_CHANNELS) - 1
                try:
                    data[y][minute_i] = 1
                except:
                    print(minute_i, y)
                    raise
