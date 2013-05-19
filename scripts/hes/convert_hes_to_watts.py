from __future__ import print_function, division
import pandas as pd
import numpy as np

"""
The raw HES data is stored as 0.1 watt hours for the power data and
0.1 degrees C for the temperature data.  "Appliances" 251-255 (inclusive)
are temperature measurements.  Also converts to correct timezone.
"""

src_store = pd.HDFStore('/data/HES/h5/HES.h5', 'r')
dst_store = pd.HDFStore('/data/HES/h5/HESfloat.h5', 'w', 
                        complevel=9, complib='blosc')

LAST_PWR_COLUMN = 250
NANOSECONDS_PER_TENTH_OF_AN_HOUR = 1E9 * 60 * 6

for house_id in src_store.keys():
    house_id = house_id[1:]
    print(house_id)
    df = src_store[house_id]

    df = df.tz_localize('UTC').tz_convert('Europe/London')

    tenth_hours_delta = (np.diff(df.index.values).astype(int) /
                         NANOSECONDS_PER_TENTH_OF_AN_HOUR)

    df = df.ix[1:,:] # chop off first data reading because hours_delta is gap between readings
    tenth_hrs_delta_series = pd.Series(tenth_hours_delta, index=df.index)
    
    df.ix[:,:LAST_PWR_COLUMN] /= tenth_hrs_delta_series
    df.ix[:,LAST_PWR_COLUMN:] /= 10.0

    dst_store[house_id] = df
    
src_store.close()
dst_store.close()
