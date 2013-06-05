#!/bin/python
from __future__ import print_function, division
from pda.channel import Channel
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator
import numpy as np
import os
import setupPlottingForLaTeX as spfl
import math

CHAN_IDS = [2,3,7,17,9,19,25,8,10,11,13,42,14,45,16]
START_DATE = None # datetime.datetime(year=2013, month=3, day=1)
END_DATE = None # datetime.datetime(year=2013, month=3, day=1)
BIN_SIZE = 'T' # D (daily) or H (hourly) or T (minutely)
TIMESPAN = 'D' # D (daily) or W (weekly)
LATEX_PDF_OUTPUT_FILENAME = ('~/Dropbox/MyWork/imperial/PhD/writing/papers/'
                             'tetc2013/figures/'
                             'daily_usage_histograms.pdf') # string or None
BAR_COLOR = 'k'
SPINE_COLOR = 'grey'

if LATEX_PDF_OUTPUT_FILENAME is not None:
    LATEX_PDF_OUTPUT_FILENAME = os.path.expanduser(LATEX_PDF_OUTPUT_FILENAME)
    spfl.setup(fig_height=8)

fig = plt.figure()
plt.clf()

def format_time(x, pos=None):
    if BIN_SIZE == 'T': #minutely
        hours = x // 60
    else:
        hours = x
    return '${:d}$'.format(int(hours))

n_subplots = len(CHAN_IDS)
for chan_id in CHAN_IDS:
    # Get channel data
    c = Channel('/data/mine/vadeec/jack-merged', chan_id)
    c = c.crop(START_DATE, END_DATE)
    distribution = c.activity_distribution(bin_size=BIN_SIZE, timespan=TIMESPAN)
    x = np.arange(distribution.size)

    # plot
    subplot_index = CHAN_IDS.index(chan_id) + 1
    ax = fig.add_subplot(n_subplots, 1, subplot_index)
    ax.bar(x, distribution.values, facecolor=BAR_COLOR, edgecolor=BAR_COLOR)

    # Process X tick marks and labels0
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

    # y ticks
    # plt.locator_params(axis='y', nbins=2, tight=True)
    yticks = ax.get_yticks()
    ax.set_yticks([yticks[0], yticks[-2]])

    spfl.format_axes(ax)

    if subplot_index < n_subplots:
        ax.xaxis.set_ticks_position('none')
        ax.set_xticklabels([])
    else:
        ax.set_xlabel('hour' if TIMESPAN == 'D' else 'day')

    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Draw tick marks and labels at the very top of the figure
    # if subplot_index == 1:
    #     ax.spines['top'].set_visible(True)
    #     ax.spines['top'].set_color(spfl.SPINE_COLOR)
    #     ax.spines['top'].set_linewidth(0.5)
    #     ax.xaxis.set_ticks_position('both')
    #     ax.tick_params(axis='x', labelbottom=False, labeltop=True)

    if subplot_index == math.ceil(n_subplots / 2):
        ax.set_ylabel('frequency')

    # Add title
#    date_format = '%d-%m-%Y'
#    title = "Daily" if TIMESPAN=='D' else "Weekly"
#    title += ' activity histogram for ' + c.name.replace('_', ' ')
#    title += ' from ' + c.series.index[0].strftime(date_format)
#    title += ' to ' + c.series.index[-1].strftime(date_format)
#    ax.set_title(title)

    ax.set_title(c.get_long_name(), x=0.05, y=0.87, ha='left')

    ax.xaxis.grid(color='gray')

plt.subplots_adjust(hspace=0.4)
plt.show()
plt.savefig(LATEX_PDF_OUTPUT_FILENAME)
