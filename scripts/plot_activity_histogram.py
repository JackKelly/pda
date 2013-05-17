#!/bin/python
from __future__ import print_function, division
from pda.channel import Channel
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator
import numpy as np
import datetime

START_DATE = None # datetime.datetime(year=2013, month=3, day=1)
END_DATE = None # datetime.datetime(year=2013, month=3, day=1)
BIN_SIZE = 'T'  # H (hourly) or T (minutely)
c = Channel('/data/mine/vadeec/jack-merged', 3)

fig = plt.figure()
ax = fig.add_subplot(111)

def format_time(x, pos=None):
    if BIN_SIZE == 'T': #minutely
        hours = x // 60
    else:
        hours = x
    return '{:d}'.format(int(hours))

if START_DATE:
    c.series = c.series[c.series.index >= START_DATE]
if END_DATE:
    c.series = c.series[c.series.index <= END_DATE]
distribution = c.activity_distribution(bin_size=BIN_SIZE)

x = np.arange(distribution.size)

COLOR = 'b'
ax.bar(x, distribution.values, facecolor=COLOR, edgecolor=COLOR)
ax.set_xlim([0, distribution.size])
ax.xaxis.set_major_locator(MultipleLocator(distribution.size / 12))

ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_time))

date_format = '%d-%m-%Y'
timespan = ' from ' + c.series.index[0].strftime(date_format)
timespan += ' to ' + c.series.index[-1].strftime(date_format)
ax.set_title('Daily activity histogram for ' + c.name.replace('_', ' ') + timespan)

ax.set_xlabel('hour')
ax.set_ylabel('count')

plt.show()

