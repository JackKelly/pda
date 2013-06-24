#!/bin/python
from __future__ import print_function, division
from pda.channel import Channel
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import os, datetime
import setupPlottingForLaTeX as spfl

BAR_COLOR = 'gray'

DATA_DIR = '/data/mine/vadeec/merged/house1'
FIGURE_PATH = os.path.expanduser('~/Dropbox/MyWork/imperial/PhD/writing'
                                 '/papers/tetc2013/figures/')
LATEX_PDF_OUTPUT_FILENAME = os.path.join(FIGURE_PATH, 'on_durations.pdf')

CHAN_IDS = [24,5,12,22,7,9,8,11,42,14,16,4]
spfl.setup(columns=2)
TITLE_Y = 0.7
MINIMUM_BIN_COUNT = 15

chans = []

for chan_id in CHAN_IDS:
    # Get channel data
    print("loading channel", chan_id)
    c = Channel(DATA_DIR, chan_id)
    chans.append(c)

#-------------------------------------------

fig = plt.figure()

n_subplots = len(chans)
for c in chans:
    subplot_index = chans.index(c) + 1
    on_durations = c.durations('on', min_state_duration=1000)
    on_durations = on_durations[on_durations > 20]

    # First get unconstrained histogram from which we will 
    # automatically find a sensible range
    hist, bin_edges = np.histogram(on_durations, bins=20)
    above_threshold = np.where(hist > MINIMUM_BIN_COUNT)[0]

    if len(above_threshold) < 1:
        print(c.name, c.chan, " does not have enough data above threshold")
        print(bin_edges)
        print(hist)
        continue

    min_duration = int(round(bin_edges[above_threshold[0]]))
    max_duration = int(round(bin_edges[above_threshold[-1]+1]))

    # Now get histogram just for the auto-selected range
    ax = fig.add_subplot(n_subplots/3, 3, subplot_index)

    # Draw histogram for normalised values
    n, bins, patches = ax.hist(on_durations, 
                               facecolor=BAR_COLOR,
                               edgecolor=BAR_COLOR,
                               range=(min_duration, max_duration), 
                               bins=10)
#                               bins=int(round((max_duration-min_duration)/20)))

    # format plot
    ax.set_axis_bgcolor('#eeeeee')
    yticks = ax.get_yticks()
    ax.set_yticks([])
    spfl.format_axes(ax)
    ax.spines['left'].set_visible(False)
    # if c.name == 'washing_machine':
    #     ax.set_ylim([0, np.max(n)*0.2])
    #     title_x = 0.5
    # elif c.name in ['bedroom_ds_lamp', 'kitchen_lights']:
    #     ax.set_ylim([0, np.max(n)*1.65])
    #     title_x = 0.5        
    # elif c.name ==  'lcd_office':
    #     ax.set_ylim([0, np.max(n)*1.5])
    #     title_x = 0.5        
    # elif c.name == 'htpc':
    #     ax.set_ylim([0, np.max(n)*1.2])
    #     title_x = 0.4
    # elif c.name == 'breadmaker':
    #     ax.set_ylim([0, np.max(n)*0.5])
    #     title_x = 0.5
    # elif c.name == 'laptop':
    #     ax.set_ylim([0, np.max(n)*0.9])
    #     title_x = 0.5
    # elif c.name == 'toaster':
    #     ax.set_ylim([0, np.max(n)*1.1])
    #     title_x = 0.4
    # elif c.name == 'hoover':
    #     ax.set_ylim([0, np.max(n)*1.0])
    #     title_x = 0.5
    # else:
    #     ax.set_ylim([0, np.max(n)*1.2])
    #     title_x = 0.5

    xlim = ax.get_xlim()
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.set_title(c.get_long_name(), x=0.5, y=TITLE_Y, ha='center')

plt.subplots_adjust(hspace=0.5, wspace=0.3)
plt.savefig(LATEX_PDF_OUTPUT_FILENAME)
plt.show()
