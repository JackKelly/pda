#!/usr/bin/python
from __future__ import print_function, division
import matplotlib.pyplot as plt
from pda.channel import Channel
import pda.metoffice as metoffice
import pda.stats

print("Opening metoffice data...")
weather = metoffice.open_daily_xls('/data/metoffice/Heathrow_DailyData.xls')

print("Opening power data...")
# 25 = lighting circuit # (R^2 = 0.343)
# 8 = kitchen lights (R^2 = 0.194)
# 2 = boiler (versus radiation R^2 = , 
#             versus mean_temp R^2 = 0.298,
#             versus max_temp  R^2 = 0.432)
# 3 = solar (R^2 = 0.798)
power = Channel('/data/mine/vadeec/jack-merged/channel_2.dat')

print("Calculating...")
on = power.on_duration_per_day(tz_convert='UTC', pwr_threshold=20)

print("Plotting...")

fig = plt.figure()
ax = fig.add_subplot(111)

pda.stats.correlate(weather.min_temp, on.on_duration, ax, 
                    'solar total global radiation $KJ/m^{2}$',
                    'hours of electric light usage')

plt.show()

print("Done")
