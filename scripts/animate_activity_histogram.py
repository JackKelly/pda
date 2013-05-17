#!/bin/python
from __future__ import print_function, division
from pda.channel import Channel
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import animation
import numpy as np
import datetime

BIN_SIZE = 'H'  # H (hourly) or T (minutely)
c = Channel('/data/mine/vadeec/jack-merged', 2)
START_PERIOD = c.series.index[0].to_period('W')

width = 1440 if BIN_SIZE=='T' else 24

fig = plt.figure()
ax = fig.add_subplot(111)
COLOR = 'b'
x = np.arange(width)
y = np.zeros(width)
rects = ax.bar(x, y, facecolor=COLOR, edgecolor=COLOR)
ax.set_xlim([0, width])
ax.set_ylim([0, 10])
ax.xaxis.set_major_locator(ticker.MultipleLocator(width / 12))

def format_time(x, pos=None):
    if BIN_SIZE == 'T': #minutely
        hours = x // 60
    else:
        hours = x
    return '{:d}'.format(int(hours))

ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_time))

date_format = '%d-%m-%Y'
ax.set_xlabel('hour')
ax.set_ylabel('count')

entire_dataset = c.series.copy()

def animate(i):
    period = START_PERIOD + i
    left_indicies = np.where(entire_dataset.index >= period.start_time)[0]
    right_indicies = np.where(entire_dataset.index < period.end_time)[0]
    c.series = entire_dataset[np.intersect1d(left_indicies, right_indicies)]

    timespan = ' from ' + c.series.index[0].strftime(date_format)
    timespan += ' to ' + c.series.index[-1].strftime(date_format)
    ax.set_title('Daily activity histogram for ' + c.name.replace('_', ' ') + timespan)

    distribution = c.activity_distribution(bin_size=BIN_SIZE)
    for bar_i in range(len(rects)):
        rects[bar_i].set_height(distribution.values[bar_i])

    return rects

anim = animation.FuncAnimation(fig, animate, frames=27, interval=40, repeat=True, repeat_delay=40)

plt.show()

