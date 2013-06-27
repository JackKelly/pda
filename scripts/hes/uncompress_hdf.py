from __future__ import print_function, division
import pandas as pd
import sys, os

DIR = '/home/dk3810/Dropbox/Data/HES/h5'
src = pd.HDFStore(os.path.join(DIR, 'HESfloat.h5'), 'r')
dst = pd.HDFStore(os.path.join(DIR, 'HESfloat_uncompressed.h5'), 'w')

try:
    for house_id in src.keys():
        house_id = house_id[1:]
        print('Loading house ', house_id, '... ', sep='', end='')
        sys.stdout.flush()
        dst[house_id] = src[house_id]
        dst.flush()
        print('done')
finally:
    dst.close()
    src.close()
