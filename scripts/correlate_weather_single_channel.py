#!/usr/bin/python
from __future__ import print_function, division
import matplotlib.pyplot as plt
from pda.channel import Channel
import pda.metoffice as metoffice
import pda.stats
from scipy.stats import linregress
import setupPlottingForLaTeX as spfl
import os

ON_DURATION_THRESHOLD = 0.1 # hours

FIGURE_PATH = os.path.expanduser('~/Dropbox/MyWork/imperial/PhD/writing'
                                 '/papers/tetc2013/figures/')

# Load metoffice data
print("Opening metoffice data...")
weather = metoffice.open_daily_xls('/data/metoffice/Heathrow_DailyData.xls')

# Setup figure
spfl.setup(columns=2)
fig = plt.figure()

def correlate(chan_id, weather_variable, subplot_index, annotate_y):
    # 25 = lighting circuit # (R^2 = 0.443)
    # 8 = kitchen lights (R^2 = 0.194)
    # 2 = boiler (versus radiation R^2 = 0.052, 
    #             versus mean_temp R^2 = 0.298,
    #             versus max_temp  R^2 = 0.432,
    #             versus min_temp  R^2 = 0.212)
    # 3 = solar (R^2 = 0.798)
    # 12 = fridge vs min_temp R^2 = 0.255 (with on_power_threshold = 20)
    
    print("Opening channel data...")
    channel = Channel('/data/mine/vadeec/jack-merged/', chan_id)

    print("Calculating...")
    channel.on_power_threshold = 20
    hours_on = channel.usage_per_period('D', tz_convert='UTC').hours_on
    hours_on = hours_on[hours_on > ON_DURATION_THRESHOLD]
    hours_on.description = 'hours on'
    print("Got {} days of data from usage_per_period.".format(hours_on.size))

    print("Plotting...")

    x_aligned, y_aligned = pda.stats.align(weather_variable, hours_on)
    print(x_aligned.description)
    slope, intercept, r_value, p_value, std_err = linregress(x_aligned.values,
                                                             y_aligned.values)
    ax = fig.add_subplot(2,2,subplot_index)
    ax = spfl.format_axes(ax)
    ax = pda.stats.plot_regression_line(ax, x_aligned, y_aligned, slope,
                                        intercept, r_value, 
                                        annotate_y=annotate_y)
    print("R^2={:.3f}".format(r_value**2))
    ax.set_title('Correlation between ' + channel.get_long_name() + ' and ' + 
                 metoffice.get_long_name(weather_variable.name))


correlations_to_plot = [
                       (25, weather.radiation, 0.6), # 25 = lighting circuit
                       ( 2, weather.max_temp, 0.6), # 2 = boiler
                       ( 3, weather.radiation, 0.1), # 3 = solar thermal pump
                       (12, weather.min_temp, 0.1) # 12 = fridge
                       ]

for i, pair in enumerate(correlations_to_plot):
    correlate(pair[0], pair[1], i+1, pair[2])

plt.savefig(os.path.join(FIGURE_PATH, 'weather_correlations.pdf'))
plt.show()
print("Done")
