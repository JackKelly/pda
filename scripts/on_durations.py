#!/bin/python
from __future__ import print_function, division
from pda.channel import Channel
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
import os
import pda.setupPlottingForLaTeX as spfl

BAR_COLOR = 'gray'

DATA_DIR = '/data/mine/vadeec/merged/house1'
FIGURE_PATH = os.path.expanduser('~/Dropbox/MyWork/imperial/PhD/writing'
                                 '/papers/tetc2013/figures/')
LATEX_PDF_OUTPUT_FILENAME = os.path.join(FIGURE_PATH, 'on_durations.pdf')

CHAN_IDS = [6,5,12,22,39,9,8,11,42,13,16,4]
spfl.setup(columns=2)
TITLE_Y = 0.7

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
    ignore_n_off_samples = {'breadmaker': 600, 
                            'washing_machine': 10,
                            'dishwasher': 10}
    on_durations = c.durations('on', 
                               ignore_n_off_samples=ignore_n_off_samples.get(c.name))

    ax = fig.add_subplot(n_subplots/3, 3, subplot_index)

    rng = None
    n, bins, patches = ax.hist(on_durations/60, 
                               facecolor=BAR_COLOR,
                               edgecolor=BAR_COLOR,
                               range=rng,
                               bins=40)

    # format plot
    ax.set_axis_bgcolor('#eeeeee')
    yticks = ax.get_yticks()
    ax.set_yticks([])
    spfl.format_axes(ax)
    ax.spines['left'].set_visible(False)
    if c.name == 'toaster':
#         ax.set_ylim([0, np.max(n)*1.1])
         title_x = 0.3
    else:
#        ax.set_ylim([0, np.max(n)*1.2])
        title_x = 0.5

    ax.xaxis.set_major_locator(MaxNLocator(5, prune=None, integer=True))

    def fmt_xticks(mins, pos):
        mins = int(mins)
        hours = mins // 60
        mins = mins % 60
        return '{:02d}:{:02d}'.format(hours,mins)

    formatter = FuncFormatter(fmt_xticks)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_title(c.get_long_name(), x=title_x, y=TITLE_Y, ha='center')

plt.subplots_adjust(hspace=0.5, wspace=0.3)
plt.savefig(LATEX_PDF_OUTPUT_FILENAME)
plt.show()
