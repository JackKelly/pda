from __future__ import print_function, division
import pandas as pd
import numpy as np

src_store = pd.HDFStore('/data/HES/h5/HESfloat.h5', 'r')

for house_id in src_store.keys():
    house_id = house_id[1:]
    try:
        df = src_store[house_id]['2011-03-28']
    except KeyError:
        print("skipping", house_id)
        continue

    time_delta = (np.diff(df.index.values
                          .astype('datetime64[s]')))
    
    non_standard_tds = np.where(time_delta != 120)[0]
    if any(non_standard_tds):
        print(house_id, non_standard_tds)
        print(time_delta[non_standard_tds])
    
src_store.close()

