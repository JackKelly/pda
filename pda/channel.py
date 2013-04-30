from __future__ import print_function, division
import pandas as pd
import numpy as np
import datetime as dt

SECS_PER_DAY = 60*60*24 # seconds per day

class Channel(object):
    """A single channel of data.
    
    Attributes:
        series (pd.series): the power data in Watts.
        max_sample_period (np.timedelta64, s): The maximum time allowed
            between samples. If data is missing for longer than
            max_sample_period then the appliance is assumed to be off.
        sample_period (np.timedelta64, s)
    """
    
    DEFAULT_TIMEZONE = 'Europe/London'
    
    def __init__(self, filename=None, timezone=DEFAULT_TIMEZONE,
                 sample_period=None, # seconds
                 max_sample_period=20 # seconds
                 ):
        self.max_sample_period = np.timedelta64(max_sample_period, 's')
        if filename is not None:
            self.load(filename, timezone)
            if sample_period is None:
                self.sample_period = (np.diff(self.series.index.values).min()
                                      .astype('timedelta64[s]'))
            else:
                self.sample_period = np.timedelta64(sample_period, 's') 
        
    def load(self, filename, timezone=DEFAULT_TIMEZONE):
        date_parser = lambda x: dt.datetime.fromtimestamp(x)
        df = pd.read_csv(filename, sep=' ', parse_dates=True,
                         index_col=0, names=['timestamp', 'power'],
                         dtype={'timestamp':np.float64, 'power':np.float64},
                         date_parser=date_parser)
        self.series = df.icol(0).tz_localize(timezone)

    def on_duration_per_day(self, pwr_threshold=5, acceptable_dropout_rate=0.2,
                            tz_convert=None):
        """
        Args:
            pwr_threshold (float or int): Optional. Threshold which defines the
                distinction between "on" and "off".  Watts.
            acceptable_dropout_rate (float). Must be >= 0 and <= 1.  Remove any
                row which has a greater sampling dropout rate.
            tz_convert (str): (optional) Use 'midnight' in this timezone.
            
        Returns:
            pd.DataFrame.  One row per day.  Columns:
                on_duration (np.timedelta64[ns])
                sample_size (np.int64)
        """
        
        assert(0 <= acceptable_dropout_rate <= 1)

        if tz_convert is not None:
            series = self.series.tz_convert(tz_convert)
        else:
            series = self.series

        # Construct the index for the output.  Each item is a Datetime
        # at midnight.  Get rid of the first & last items so rng represents
        # just the full-days for which we have date.
        rng = pd.date_range(series.index[0], series.index[-1],
                            freq='D', normalize=True)
        on_durations = pd.Series(   index=rng, dtype=np.timedelta64)
        sample_sizes = pd.Series(0, index=rng, dtype=np.int64)

        min_samples_per_day = ((SECS_PER_DAY / self.sample_period.astype(np.int64)) * 
                               (1-acceptable_dropout_rate))

        for day in rng:
            data_for_day = series[day.strftime('%Y-%m-%d')]
            if data_for_day.size < min_samples_per_day:
                continue
            i_above_threshold = np.where(data_for_day[:-1] >= pwr_threshold)[0]
            timedeltas = (data_for_day.index[i_above_threshold+1].values -
                          data_for_day.index[i_above_threshold].values)
            timedeltas[timedeltas > self.max_sample_period] = self.max_sample_period
            on_durations[day] = timedeltas.sum()
            sample_sizes[day] = data_for_day.size

        df = pd.DataFrame({'on_duration':on_durations,
                           'sample_size':sample_sizes})

        return df.dropna()
