#!/usr/bin/python
from __future__ import print_function, division
import matplotlib.pyplot as plt
from pda.channel import Channel
import pda.metoffice as metoffice
import pda.stats
from scipy.stats import linregress

ON_DURATION_THRESHOLD = 0.1 # hours

print("Opening metoffice data...")
weather = metoffice.open_daily_xls('/data/metoffice/Heathrow_DailyData.xls')

print("Opening power data...")
# 25 = lighting circuit # (R^2 = 0.443)
# 8 = kitchen lights (R^2 = 0.194)
# 2 = boiler (versus radiation R^2 = 0.052, 
#             versus mean_temp R^2 = 0.298,
#             versus max_temp  R^2 = 0.432,
#             versus min_temp  R^2 = 0.212)
# 3 = solar (R^2 = 0.798)
# 12 = fridge vs min_temp R^2 = 0.255 (with on_power_threshold = 20)

power = Channel('/data/mine/vadeec/jack-merged/', 25)

print("Calculating...")
# power.on_power_threshold = 3
on = power.on_duration_per_day(tz_convert='UTC')
print("Got {} days of data from on_duration_per_day.".format(on.size))

print("Plotting...")

x_aligned, y_aligned = pda.stats.align(weather.radiation, on[on > ON_DURATION_THRESHOLD])
print(x_aligned.description)
slope, intercept, r_value, p_value, std_err = linregress(x_aligned.values,
                                                         y_aligned.values)
fig = plt.figure()
ax = fig.add_subplot(111)
ax = pda.stats.plot_regression_line(ax, x_aligned, y_aligned, slope, intercept, r_value)
print("R^2={:.3f}".format(r_value**2))

plt.show()
print("Done")
