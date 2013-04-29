import pandas as pd
import numpy as np
import datetime as dt

class Channel(object):
    def __init__(self, filename, timezone='Europe/London'):
        date_parser = lambda x: dt.datetime.fromtimestamp(x)
        df = pd.read_csv(filename, sep=' ', parse_dates=True,
                         index_col=0, names=['timestamp', 'power'],
                         dtype={'timestamp':np.float64, 'power':np.float64},
                         date_parser=date_parser)
        self.series = df.icol(0).tz_localize('Europe/London')
