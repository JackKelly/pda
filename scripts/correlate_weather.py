from __future__ import print_function, division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pda.channel import Channel
import pda.metoffice as metoffice

weather = metoffice.open_daily_xls('/data/metoffice/Heathrow_DailyData.xls')
lights = Channel('/data/mine/dat/from_atom/137/channel_8.dat')
lights_per_day = lights.on_duration_per_day(tz_convert='UTC')

x = weather.sunshine[lights_per_day.index]
y = lights_per_day.values.astype('timedelta64[s]')

plt.plot(x, y)
plt.show()