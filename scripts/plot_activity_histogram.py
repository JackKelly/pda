#!/bin/python
from __future__ import print_function, division
from pda.channel import Channel, DD
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator
import numpy as np
import os
import datetime
import setupPlottingForLaTeX as spfl

BAR_COLOR = 'k'
SPINE_COLOR = 'grey'

# FIGURE_PRESET options:
#   'daily usage histogram'
#   'daily usage histogram for poster'
#   'weekly usage histogram'
#   'boiler seasons'
FIGURE_PRESET = 'weekly usage histogram'

if FIGURE_PRESET.endswith('for poster'):
    FIGURE_PATH = os.path.expanduser('~/Dropbox/MyWork/imperial/PhD/writing'
                                     '/posters/UKERCposter2013/')
else:
    FIGURE_PATH = os.path.expanduser('~/Dropbox/MyWork/imperial/PhD/writing'
                                     '/papers/tetc2013/figures/')

FIGURE_SUFFIX = '.pdf'
DATA_DIR = '/data/mine/vadeec/merged/house1'

if FIGURE_PRESET in ['daily usage histogram', 'daily usage histogram for poster']:
    START_DATE = None # datetime.datetime(year=2013, month=3, day=1)
    END_DATE = None # datetime.datetime(year=2013, month=3, day=1)
    BIN_SIZE = 'T' # D (daily) or H (hourly) or T (minutely)
    TIMESPAN = 'D' # D (daily) or W (weekly)
    spfl.setup(fig_height=8)
    GRID = True
    XTICKS_ON = False
    LATEX_PDF_OUTPUT_FILENAME = os.path.join(FIGURE_PATH,
                                             'daily_usage_histograms'+FIGURE_SUFFIX)
    if FIGURE_PRESET == 'daily usage histogram':
        CHAN_IDS = [2,3,7,17,9,19,25,8,10,11,13,42,14,45,16]
        TITLE_Y = 0.87
    else:
        CHAN_IDS = [2,3,7,25,10,42,14]
        TITLE_Y = 0.8
elif FIGURE_PRESET == 'weekly usage histogram':
    START_DATE = None # datetime.datetime(year=2013, month=3, day=1)
    END_DATE = None # datetime.datetime(year=2013, month=3, day=1)
    BIN_SIZE = 'D' # D (daily) or H (hourly) or T (minutely)
    TIMESPAN = 'W' # D (daily) or W (weekly)
    CHAN_IDS = [14,22]
    spfl.setup()
    GRID = False
    TITLE_Y = 0.75
    XTICKS_ON = True
    LATEX_PDF_OUTPUT_FILENAME = os.path.join(FIGURE_PATH,
                                             'weekly_usage_histograms'+FIGURE_SUFFIX)
else:
    CHAN_IDS = []

CHANS = []
for chan_id in CHAN_IDS:
    # Get channel data
    print("Loading channel", chan_id)
    c = Channel(DATA_DIR, chan_id)
    c = c.crop(START_DATE, END_DATE)
    CHANS.append(c)

if FIGURE_PRESET == 'boiler seasons':
    BIN_SIZE = 'T' # D (daily) or H (hourly) or T (minutely)
    TIMESPAN = 'D' # D (daily) or W (weekly)
    spfl.setup()
    GRID = False
    TITLE_Y = 0.7
    XTICKS_ON = True
    LATEX_PDF_OUTPUT_FILENAME = os.path.join(FIGURE_PATH,
                                             'seasonal_variation'+FIGURE_SUFFIX)
    print("Loading winter boiler data...")
    winter_boiler = Channel(DATA_DIR, 2)
    winter_boiler = winter_boiler.crop(datetime.datetime(year=2013, month=2, day=1),
                                       datetime.datetime(year=2013, month=3, day=1))
    winter_boiler.name = 'boiler February 2013'

    print("Loading summer boiler data...")
    summer_boiler = Channel(DATA_DIR, 2)
    summer_boiler = summer_boiler.crop(datetime.datetime(year=2013, month=6, day=1),
                                       datetime.datetime(year=2013, month=7, day=1))
    summer_boiler.name = 'boiler June 2013'

    CHANS.append(winter_boiler)
    CHANS.append(summer_boiler)

#-------------------------------------------

fig = plt.figure()
plt.clf()

def format_time(x, pos=None):
    if BIN_SIZE == 'T': #minutely
        hours = x // 60
    else:
        hours = x
    return '${:d}$'.format(int(hours))

n_subplots = len(CHANS)
for c in CHANS:
    distribution = c.activity_distribution(bin_size=BIN_SIZE, timespan=TIMESPAN)
    x = np.arange(distribution.size)

    # plot
    subplot_index = CHANS.index(c) + 1
    ax = fig.add_subplot(n_subplots, 1, subplot_index)
    ax.set_axis_bgcolor('#eeeeee')
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
    yticks = ax.get_yticks()
    ax.set_yticks([yticks[0], yticks[-2]])
    ylim = ax.get_ylim()
    ax.set_ylim([ylim[0], ylim[1]*1.2])

    spfl.format_axes(ax)

    if subplot_index < n_subplots:
        ax.set_xticklabels([])
        if not XTICKS_ON:
            ax.xaxis.set_ticks_position('none')
    else:
        ax.set_xlabel('hour' if TIMESPAN == 'D' else 'day')

    ax.spines['left'].set_visible(False)

    ax.set_title(c.get_long_name(), x=0.5, y=TITLE_Y, ha='center')

    if GRID:
        ax.xaxis.grid(color='gray')

fig.text(0.02, 0.5, 'frequency', rotation=90, va='center', ha='left', size=8)
plt.subplots_adjust(hspace=0.2)
plt.savefig(LATEX_PDF_OUTPUT_FILENAME)
# plt.show()
print("Done!")
