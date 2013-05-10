#!/usr/bin/python
from __future__ import print_function, division
import matplotlib.pyplot as plt
import pda.metoffice as metoffice
import pda.dataset as ds
import pda.stats
import numpy as np
from scipy.stats import linregress
from matplotlib.ticker import MultipleLocator

THRESHOLD = 0.005 # remove any days will less kwh or hours on per day
MIN_DAYS_PER_CHAN = 10
DATA_DIR = '/data/mine/vadeec/jack-merged/'
# DATA_DIR  = '/data/mine/vadeec/jack/137'

print("Opening metoffice data...")
weather = metoffice.open_daily_xls('/data/metoffice/Heathrow_DailyData.xls')

print("Loading dataset...")
dataset = ds.load_dataset(DATA_DIR)

def correlate(power_metric, i_w, name):
    pm_filtered = power_metric[power_metric > THRESHOLD]
    if pm_filtered.size > MIN_DAYS_PER_CHAN:
        try:
            x_aligned, y_aligned = pda.stats.align(weather.ix[:,i_w], 
                                                   pm_filtered)
        except IndexError as e:
            print(str(e))
            print(name, "has insufficient data")
        else:
            # calculate linear regression
            slope, intercept, r_value, p_value, std_err = linregress(x_aligned.values,
                                                                     y_aligned.values)
            return r_value**2


# Create a matrix for storing coefficient of determination values (R^2)
r_sq_hours = np.zeros([len(dataset), len(weather.columns)])
r_sq_kwh = np.zeros([len(dataset), len(weather.columns)])
names = []
for i_d in range(len(dataset)):
    name = dataset[i_d].name
    names.append(name)
    print("Loading on durations for", name)
    usage_per_day = dataset[i_d].usage_per_day(tz_convert='UTC')
    print("Got {} days of data from on_duration_per_day for {}."
          .format(usage_per_day.index.size, name))

    for i_w in range(len(weather.columns)):
        print(dataset[i_d].name, weather.columns[i_w])
        hours_on = usage_per_day.hours_on
        r_sq_hours[i_d][i_w] = correlate(hours_on, i_w, name)
        r_sq_kwh[i_d][i_w] = correlate(usage_per_day.kwh, i_w, name)            

print(r_sq_hours)
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
