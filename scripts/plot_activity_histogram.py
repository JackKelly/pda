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
BIN_SIZE = 'T' # D (daily) or H (hourly) or T (minutely)
TIMESPAN = 'D' # D (daily) or W (weekly)
c = Channel('/data/mine/vadeec/jack-merged', 3)

fig = plt.figure()
ax = fig.add_subplot(111)

def format_time(x, pos=None):
    if BIN_SIZE == 'T': #minutely
        hours = x // 60
    else:
        hours = x
    return '{:d}'.format(int(hours))

c = c.crop(START_DATE, END_DATE)

distribution = c.activity_distribution(bin_size=BIN_SIZE, timespan=TIMESPAN)

x = np.arange(distribution.size)

COLOR = 'b'
ax.bar(x, distribution.values, facecolor=COLOR, edgecolor=COLOR)

if TIMESPAN == 'D':
    BINS_PER_SUBTIMESPAN = 0 if BIN_SIZE=='T' else 1
    ax.xaxis.set_major_locator(MultipleLocator(distribution.size / 12))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_time))
elif TIMESPAN == 'W': # weekly
    BINS_PER_SUBTIMESPAN = 24 if BIN_SIZE=='H' else 1
    ax.xaxis.set_major_locator(MultipleLocator(BINS_PER_SUBTIMESPAN))
    ax.set_xticklabels(['M', 'T', 'W', 'T', 'F', 'S', 'S'])

ax.set_xlim([0, distribution.size])
ax.set_xticks([tick + (0.4*BINS_PER_SUBTIMESPAN) for tick in ax.get_xticks()])
ax.set_xlim([0, distribution.size])

date_format = '%d-%m-%Y'
title = "Daily" if TIMESPAN=='D' else "Weekly"
title += ' activity histogram for ' + c.name.replace('_', ' ')
title += ' from ' + c.series.index[0].strftime(date_format)
title += ' to ' + c.series.index[-1].strftime(date_format)
ax.set_title(title)
ax.set_xlabel('hour' if TIMESPAN == 'D' else 'day')
ax.set_ylabel('count')

plt.show()

