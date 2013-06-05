#!/bin/python
from __future__ import print_function, division
from pda.channel import Channel
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator
import numpy as np
import os

START_DATE = None # datetime.datetime(year=2013, month=3, day=1)
END_DATE = None # datetime.datetime(year=2013, month=3, day=1)
BIN_SIZE = 'T' # D (daily) or H (hourly) or T (minutely)
TIMESPAN = 'D' # D (daily) or W (weekly)
LATEX_PDF_OUTPUT_FILENAME = ('~/Dropbox/MyWork/imperial/PhD/writing/papers/'
                             'tetc2013/figures/'
                             'daily_usage_histogram_solar_thermal.pdf') # string or None
BAR_COLOR = 'k'
SPINE_COLOR = 'grey'

c = Channel('/data/mine/vadeec/jack-merged', 3)

if LATEX_PDF_OUTPUT_FILENAME is not None:
    LATEX_PDF_OUTPUT_FILENAME = os.path.expanduser(LATEX_PDF_OUTPUT_FILENAME)
    import setupPlottingForLaTeX as spfl
    spfl.setup()

fig = plt.figure()
plt.clf()
ax = fig.add_subplot(111)

def format_time(x, pos=None):
    if BIN_SIZE == 'T': #minutely
        hours = x // 60
    else:
        hours = x
    return '${:d}$'.format(int(hours))

c = c.crop(START_DATE, END_DATE)

distribution = c.activity_distribution(bin_size=BIN_SIZE, timespan=TIMESPAN)

x = np.arange(distribution.size)

ax.bar(x, distribution.values, facecolor=BAR_COLOR, edgecolor=BAR_COLOR)

if TIMESPAN == 'D':
    BINS_PER_SUBTIMESPAN = 0 if BIN_SIZE=='T' else 1
    ax.xaxis.set_major_locator(MultipleLocator(distribution.size / 4))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_time))
elif TIMESPAN == 'W': # weekly
    BINS_PER_SUBTIMESPAN = 24 if BIN_SIZE=='H' else 1
    ax.xaxis.set_major_locator(MultipleLocator(BINS_PER_SUBTIMESPAN))
    ax.set_xticklabels(['M', 'T', 'W', 'T', 'F', 'S', 'S'])

ax.set_xlim([0, distribution.size])
ax.set_xticks([tick + (0.4*BINS_PER_SUBTIMESPAN) for tick in ax.get_xticks()])
ax.set_xlim([0, distribution.size])

plt.locator_params(axis='y', nbins=2)

if LATEX_PDF_OUTPUT_FILENAME is None:
    # Add title
    date_format = '%d-%m-%Y'
    title = "Daily" if TIMESPAN=='D' else "Weekly"
    title += ' activity histogram for ' + c.name.replace('_', ' ')
    title += ' from ' + c.series.index[0].strftime(date_format)
    title += ' to ' + c.series.index[-1].strftime(date_format)
    ax.set_title(title)

ax.set_xlabel('hour' if TIMESPAN == 'D' else 'day')
ax.set_ylabel('count')

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

for spine in ['left', 'bottom']:
    ax.spines[spine].set_color(SPINE_COLOR)
    ax.spines[spine].set_linewidth(0.5)

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

for axis in [ax.xaxis, ax.yaxis]:
    axis.set_tick_params(direction='out', color=SPINE_COLOR)

plt.tight_layout()
plt.show()
plt.savefig(LATEX_PDF_OUTPUT_FILENAME)
