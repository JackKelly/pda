#!/bin/python
from __future__ import print_function, division
from pda.channel import Channel
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import os, datetime
import setupPlottingForLaTeX as spfl

NORMALISED_BAR_COLOR = 'gray'
UNNORMALISED_LINE_COLOR = 'k'

DATA_DIR = '/data/mine/vadeec/merged/house1'
FIGURE_PATH = os.path.expanduser('~/Dropbox/MyWork/imperial/PhD/writing'
                                 '/papers/tetc2013/figures/')
LATEX_PDF_OUTPUT_FILENAME = os.path.join(FIGURE_PATH, 'power_histograms.pdf')
voltage = Channel()
voltage.load_high_freq_mains(os.path.join(DATA_DIR, 'mains.dat'), 'volts')

CHAN_IDS = [24,5,12,22,7,9,8,11,42,14,16,4]
spfl.setup(columns=2)
TITLE_Y = 0.7
MINIMUM_BIN_COUNT = 100

chans = []
normalised = []

for chan_id in CHAN_IDS:
    # Get channel data
    print("loading channel", chan_id)
    c = Channel(DATA_DIR, chan_id)
    chans.append(c)
    normalised.append(c.normalise_power(voltage.series))


#-------------------------------------------

fig = plt.figure()

n_subplots = len(chans)
for c, c_normalised in zip(chans, normalised):
    subplot_index = chans.index(c) + 1

    # hard-coded tweaks for individual appliances
    if c.name == 'kitchen_lights':
        # Ignore 50W halogens; only use data after all lights replaced by LEDs
        c_normalised = c_normalised.crop(datetime.datetime(year=2013, month=4, day=27))
        
    c_normalised.series = c_normalised.series[c_normalised.series > 3]
    c_normalised.series = c_normalised.series[c_normalised.series < 5000]
    c = c.crop(c_normalised.series.index[0], c_normalised.series.index[-1])
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
    ax = fig.add_subplot(n_subplots/3, 3, subplot_index)

    # Draw histogram for normalised values
    n, bins, patches = ax.hist(c_normalised.series.values, 
                               facecolor=NORMALISED_BAR_COLOR,
                               edgecolor=NORMALISED_BAR_COLOR,
                               range=(min_power, max_power), 
                               bins=int(round((max_power-min_power)/2)))

    # Draw histogram for unnormalised values
    ax.hist(c.series.values,
            histtype='step',
            color=UNNORMALISED_LINE_COLOR,
            alpha=0.5,
            range=(min_power, max_power), 
            bins=int(round((max_power-min_power)/2)))

    # format plot
    ax.set_axis_bgcolor('#eeeeee')
    yticks = ax.get_yticks()
    ax.set_yticks([])
    spfl.format_axes(ax)
    ax.spines['left'].set_visible(False)
    if c.name == 'washing_machine':
        ax.set_ylim([0, np.max(n)*0.2])
        title_x = 0.5
    elif c.name in ['bedroom_ds_lamp', 'kitchen_lights']:
        ax.set_ylim([0, np.max(n)*1.65])
        title_x = 0.5        
    elif c.name ==  'lcd_office':
        ax.set_ylim([0, np.max(n)*1.5])
        title_x = 0.5        
    elif c.name == 'htpc':
        ax.set_ylim([0, np.max(n)*1.2])
        title_x = 0.4
    elif c.name == 'breadmaker':
        ax.set_ylim([0, np.max(n)*0.5])
        title_x = 0.5
    elif c.name == 'laptop':
        ax.set_ylim([0, np.max(n)*0.9])
        title_x = 0.5
    elif c.name == 'toaster':
        ax.set_ylim([0, np.max(n)*1.1])
        title_x = 0.4
    elif c.name == 'hoover':
        ax.set_ylim([0, np.max(n)*1.0])
        title_x = 0.5
    else:
        ax.set_ylim([0, np.max(n)*1.2])
        title_x = 0.5
    xlim = ax.get_xlim()
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.set_title(c.get_long_name(), x=title_x, y=TITLE_Y, ha='center')

plt.subplots_adjust(hspace=0.5, wspace=0.3)
plt.savefig(LATEX_PDF_OUTPUT_FILENAME)
plt.show()
