from __future__ import print_function, division
import pandas as pd
import numpy as np
import datetime as dt

class Channel(object):
    """A single channel of data.
    
    Attributes:
        series (pd.series): the power data in Watts.
        max_sample_period (np.timedelta64, s): The maximum time allowed
            between samples. If data is missing for longer than
            max_sample_period then the appliance is assumed to be off.
    """
    
    def __init__(self, filename, timezone='Europe/London',
                 max_sample_period=20 # seconds
                 ):
        
        date_parser = lambda x: dt.datetime.fromtimestamp(x)
        df = pd.read_csv(filename, sep=' ', parse_dates=True,
                         index_col=0, names=['timestamp', 'power'],
                         dtype={'timestamp':np.float64, 'power':np.float64},
                         date_parser=date_parser)
        self.series = df.icol(0).tz_localize(timezone)
        self.max_sample_period = np.timedelta64(max_sample_period, 's')

    def on_duration_per_day(self, pwr_threshold=5):
        """
        Args:
            pwr_threshold (float or int): Optional. Threshold which defines the
                distinction between "on" and "off".  Watts.    
            
        Returns:
            pd.DataFrame.  One row per day.  Columns:
                on_duration (np.timedelta64[ns])
                sample_size (np.int64)
        """

        # Construct the index for the output.  Each item is a Datetime
        # at midnight.  Get rid of the first & last items so rng represents
        # just the full-days for which we have date.
        rng = pd.date_range(self.series.index[0], self.series.index[-1],
                            freq='D', normalize=True)[1:-1]
        on_durations = pd.Series(   index=rng, dtype=np.timedelta64)
        sample_sizes = pd.Series(0, index=rng, dtype=np.int64)

        for day in rng:
            data_for_day = self.series[day.strftime('%Y-%m-%d')]
            i_above_threshold = np.where(data_for_day[:-1] >= pwr_threshold)[0]
            timedeltas = (data_for_day.index[i_above_threshold+1].values -
                          data_for_day.index[i_above_threshold].values)
            timedeltas[timedeltas > self.max_sample_period] = self.max_sample_period
            on_durations[day] = timedeltas.sum()
            sample_sizes[day] = data_for_day.size

        return pd.DataFrame({'on_duration':on_durations,
                             'sample_size':sample_sizes})
