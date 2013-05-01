"""
REQUIREMENTS:
  pandas >= 0.11.0
"""

from __future__ import print_function, division
import pandas as pd
import numpy as np
import datetime as dt
import scipy.stats as stats

SECS_PER_HOUR = 3600
SECS_PER_DAY = 86400

class Channel(object):
    """A single channel of data.
    
    Attributes:
        series (pd.series): the power data in Watts.
        max_sample_period (float): The maximum time allowed
            between samples. If data is missing for longer than
            max_sample_period then the appliance is assumed to be off.
        sample_period (float)
    """
    
    DEFAULT_TIMEZONE = 'Europe/London'
    
    def __init__(self, filename=None, timezone=DEFAULT_TIMEZONE,
                 sample_period=None, # seconds
                 max_sample_period=20 # seconds
                 ):
        self.max_sample_period = max_sample_period
        self.sample_period = sample_period
        if filename is not None:
            self.load(filename, timezone)
        
    def load(self, filename, timezone=DEFAULT_TIMEZONE):
        date_parser = lambda x: dt.datetime.fromtimestamp(x)
        df = pd.read_csv(filename, sep=' ', parse_dates=True,
                         index_col=0, names=['timestamp', 'power'],
                         dtype={'timestamp':np.float64, 'power':np.float64},
                         date_parser=date_parser)
        self.series = df.icol(0).tz_localize(timezone)
        if self.sample_period is None:
            fwd_diff = np.diff(self.series.index.values).astype(np.float)
            mode_fwd_diff = int(round(stats.mode(fwd_diff)[0][0]))
            self.sample_period = mode_fwd_diff / 1E9

    def dropout_rate(self):
        duration = self.series.index[-1] - self.series.index[0]        
        n_expected_samples = duration.total_seconds() / self.sample_period
        return 1 - (self.series.size / n_expected_samples)

    def __str__(self):
        s = ""
        s += "     start date = {}\n".format(self.series.index[0])
        s += "       end date = {}\n".format(self.series.index[-1])
        s += "       duration = {}\n".format(self.series.index[-1] - self.series.index[0])
        s += "  sample period = {:>7.1f}s\n".format(self.sample_period)
        s += "total # samples = {:>5d}\n".format(self.series.size)
        s += "   dropout rate = {:>8.1%}\n".format(self.dropout_rate())
        s += "      max power = {:>7.1f}W\n".format(self.series.max())
        s += "      min power = {:>7.1f}W\n".format(self.series.min())
        s += "     mode power = {:>7.1f}W\n".format(stats.mode(self.series)[0][0])
        return s
            
    def on_duration_per_day(self, pwr_threshold=5, acceptable_dropout_rate=0.2,
                            tz_convert=None):
        """
        Args:
            pwr_threshold (float or int): Optional. Threshold which defines the
                distinction between "on" and "off".  Watts.
            acceptable_dropout_rate (float). Must be >= 0 and <= 1.  Remove any
                row which has a worse dropout rate.
            tz_convert (str): (optional) Use 'midnight' in this timezone.
            
        Returns:
            pd.DataFrame.  One row per day.  Columns:
                on_duration (np.float): hours
                sample_size (np.int64): number of samples per day
        """
        
        assert(0 <= acceptable_dropout_rate <= 1)

        if tz_convert is not None:
            series = self.series.tz_convert(tz_convert)
        else:
            series = self.series

        # Construct the index for the output.  Each item is a Datetime
        # at midnight.
        rng = pd.date_range(series.index[0], series.index[-1],
                            freq='D', normalize=True)
        print(rng)
        on_durations = pd.Series(   index=rng, dtype=np.float)
        sample_sizes = pd.Series(0, index=rng, dtype=np.int64)

        min_samples_per_day = ((SECS_PER_DAY / self.sample_period) * 
                               (1-acceptable_dropout_rate))

        max_sample_period = np.timedelta64(self.max_sample_period, 's')

        for day in rng:
            try:
                # TODO: I think the line below is really slow.  Would probably
                # be faster to make use of the fact that the data
                # is in order, somehow...
                data_for_day = series[day.strftime('%Y-%m-%d')]
            except IndexError:
                continue
            if data_for_day.size < min_samples_per_day:
                continue
            i_above_threshold = np.where(data_for_day[:-1] >= pwr_threshold)[0]
            timedeltas = (data_for_day.index[i_above_threshold+1].values -
                          data_for_day.index[i_above_threshold].values)
            timedeltas[timedeltas > max_sample_period] = max_sample_period
            on_durations[day] = timedeltas.sum().astype('timedelta64[s]').astype(np.int64) / SECS_PER_HOUR
            sample_sizes[day] = data_for_day.size

        df = pd.DataFrame({'on_duration':on_durations,
                           'sample_size':sample_sizes})

        return df.dropna()
