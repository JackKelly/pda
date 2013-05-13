#!/usr/bin/python
from __future__ import print_function, division
import matplotlib.pyplot as plt
import pda.dataset as ds
import pda.stats
import numpy as np
from scipy.stats import linregress
from matplotlib.ticker import MultipleLocator

THRESHOLD = 0.005 # remove any days will less kwh or hours on per day
MIN_DAYS_PER_CHAN = 10
DATA_DIR = '/data/mine/vadeec/jack-merged/'
# DATA_DIR  = '/data/mine/vadeec/jack/137'

print("Loading dataset...")
dataset = ds.load_dataset(DATA_DIR)

days = pd.date_range(dataset[0].series.index[0], dataset[0].series.index[-1],
                      freq='D', normalize=True)


print("Plotting...")

def plot_heatmap(data, ax):
    im = ax.imshow(data, interpolation='nearest')
    ax.tick_params(labelsize=8) # , label2On=True)

    # X axis
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.set_xticklabels(weather.columns, rotation=90)
    ax.xaxis.set_ticks_position('both')

    # Y axis
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.set_yticklabels(names)
    return im

fig = plt.figure(figsize=(10,10))

ax1 = fig.add_subplot(1,2,1)
ax1.set_title('Appliance duration vs weather')
im1 = plot_heatmap(r_sq_hours, ax1)
fig.colorbar(im1)

ax2 = fig.add_subplot(1,2,2)
ax2.set_title('Appliance energy consumption vs weather')
im2 = plot_heatmap(r_sq_kwh, ax2)
fig.colorbar(im2)

plt.show()
print("Done")

# Trigger emacs to run this script using the "compile" command
# ;;; Local Variables: ***
# ;;; compile-command: "python correlate_weather_dataset.py" ***
# ;;; end: ***
