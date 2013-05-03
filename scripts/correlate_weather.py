#!/usr/bin/python
from __future__ import print_function, division
import matplotlib.pyplot as plt
from pda.channel import Channel
import pda.metoffice as metoffice

print("Opening metoffice data...")
weather = metoffice.open_daily_xls('/data/metoffice/Heathrow_DailyData.xls')

print("Opening power data...")
lights = Channel('/data/mine/vadeec/jack-merged/channel_8.dat')
# lights = Channel('/data/mine/vadeec/jack/137/channel_8.dat')

print("Calculating...")
on = lights.on_duration_per_day(tz_convert='UTC')

on_dur_aligned, sun_aligned = on.on_duration.align(weather.sunshine.dropna(), 
                                                   join='inner')

print("Plotting...")

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(sun_aligned, on_dur_aligned, 'o', alpha=0.3)
# ax.set_xlabel('hours of sunshine')
# ax.set_ylabel('hours of electric light usage')

print("Done")

plt.show()
