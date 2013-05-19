#!/usr/bin/python

"""I initially imported the data as uint16.  But this broke the negative 
temperature values. So this script was used once to convert the uint16 
data to int16.
"""

from __future__ import print_function, division
import numpy as np
import pandas as pd

SRC_H5_FILE = '/home/jack/workspace/python/pda/pda/HES.h5'
DST_H5_FILE = '/home/jack/workspace/python/pda/pda/HESnew.h5'

src_store = pd.HDFStore(SRC_H5_FILE, 'r')
dst_store = pd.HDFStore(DST_H5_FILE, 'w', complevel=9, complib='blosc')

for house_id in src_store.keys():
    house_id = house_id[1:]
    print(house_id)
    df = src_store[house_id]
    for appliance_id in df:
        s = df[appliance_id]
        _, s = s.align(s.dropna().astype(np.int16))
        df[appliance_id] = s

    dst_store[house_id] = df

src_store.close()
dst_store.close()
