#!/bin/python
from __future__ import print_function, division
from pda.channel import Channel
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator, MaxNLocator
import numpy as np
import os
import datetime
import setupPlottingForLaTeX as spfl

BAR_COLOR = 'k'
SPINE_COLOR = 'grey'

FIGURE_PATH = os.path.expanduser('~/Dropbox/MyWork/imperial/PhD/writing'
                                 '/papers/tetc2013/figures/')
LATEX_PDF_OUTPUT_FILENAME = os.path.join(FIGURE_PATH, 'power_histograms.pdf')

CHAN_IDS = [24,5,12,22,7,9,8,11,42,14,16,4]
spfl.setup(columns=2)
TITLE_Y = 0.95
CHANS = []
MINIMUM_BIN_COUNT = 100

for chan_id in CHAN_IDS:
    # Get channel data
    print("loading channel", chan_id)
    c = Channel('/data/mine/vadeec/jack-merged', chan_id)
    CHANS.append(c)

#-------------------------------------------

fig = plt.figure()

n_subplots = len(CHANS)
for c in CHANS:
    c.series = c.series[c.series > 3]
    c.series = c.series[c.series < 5000]

    # First get unconstrained histogram from which we will 
    # automatically find a sensible range
    hist, bin_edges = np.histogram(c.series, bins=100)
    above_threshold = np.where(hist > MINIMUM_BIN_COUNT)[0]

    if len(above_threshold) < 1:
        print(c.name, c.chan, " does not have enough data above threshold")
        print(bin_edges)
        print(hist)
        continue

    min_power = int(round(bin_edges[above_threshold[0]]))
    max_power = int(round(bin_edges[above_threshold[-1]+1]))

    # Now get histogram just for the auto-selected range
    subplot_index = CHANS.index(c) + 1
    ax = fig.add_subplot(n_subplots/3, 3, subplot_index)
    n, bins, patches = ax.hist(c.series.values, color=BAR_COLOR,
                               range=(min_power, max_power), 
                               bins=(max_power-min_power)/2)

    # format plot
    yticks = ax.get_yticks()
    ax.set_yticks([])
    spfl.format_axes(ax)
    ax.spines['left'].set_visible(False)
    ax.set_ylim([0, np.max(n)])
    xlim = ax.get_xlim()
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.set_title(c.get_long_name(), x=0.5, y=TITLE_Y, ha='center')

plt.subplots_adjust(hspace=2, wspace=0.3)
plt.savefig(LATEX_PDF_OUTPUT_FILENAME)
plt.show()
