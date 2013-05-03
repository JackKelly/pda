from __future__ import print_function, division
import pandas as pd
import numpy as np
import datetime as dt
import scipy.stats as stats

"""
REQUIREMENTS:
  pandas >= 0.11.0
"""

SECS_PER_HOUR = 3600
SECS_PER_DAY = 86400
DEFAULT_TIMEZONE = 'Europe/London'

class Channel(object):
    """
    A single channel of data.
    
    Attributes:
        series (pd.series): the power data in Watts.
        max_sample_period (float): The maximum time allowed
            between samples. If data is missing for longer than
            max_sample_period then the appliance is assumed to be off.
        sample_period (float)
    """
    
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
        """Calculate the dropout rate based on self.sample_period."""
        duration = self.series.index[-1] - self.series.index[0]        
        n_expected_samples = duration.total_seconds() / self.sample_period
        return 1 - (self.series.size / n_expected_samples)

    def __str__(self):
        s = ""
        s += "     start date = {}\n".format(self.series.index[0])
        s += "       end date = {}\n".format(self.series.index[-1])
        s += "       duration = {}\n".format(self.series.index[-1] - 
                                             self.series.index[0])
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

        max_samples_per_day = SECS_PER_DAY / self.sample_period
        min_samples_per_day = max_samples_per_day * (1-acceptable_dropout_rate)
        max_sample_period = np.timedelta64(self.max_sample_period, 's')
        max_samples_per_2days = max_samples_per_day * 2

        unprocessed_data = series.copy()
        for day_i in range(rng.size-1):
            # The simplest way to get data for just a single day is to use
            # data_for_day = series[day.strftime('%Y-%m-%d')]
            # but this takes about 300ms per call on my machine.
            # So we take advantage of several features of the data to achieve
            # a 300x speedup:
            # 1. We use the fact that the data is sorted by date, hence 
            #    we can chomp through it in order.  The variable
            #    unprocessed_data stores the data still to be processed.
            # 2. max_samples_per_day sets an upper bound on the number of
            #    datapoints per day.  The code is conservative and uses 
            #    max_samples_per_2days. We only search through a small subset
            #    of the available data.
            indicies_for_day = np.where(unprocessed_data.index[:max_samples_per_2days] 
                                        < rng[day_i+1])[0]
            day = rng[day_i]
            if indicies_for_day.size == 0:
                print("No data available for   ", day.strftime('%Y-%m-%d'))
                continue
            data_for_day = unprocessed_data[indicies_for_day]
            unprocessed_data = unprocessed_data[indicies_for_day[-1]+1:]
            if data_for_day.size < min_samples_per_day:
                print("Insufficient samples for", day.strftime('%Y-%m-%d'),
                      "; samples =", data_for_day.size,
                      "dropout_rate = {:.2%}".format(1 - (data_for_day.size / 
                                                          max_samples_per_day)))
                print("                 start =", data_for_day.index[0])
                print("                   end =", data_for_day.index[-1])
                continue
            i_above_threshold = np.where(data_for_day[:-1] >= pwr_threshold)[0]
            timedeltas = (data_for_day.index[i_above_threshold+1].values -
                          data_for_day.index[i_above_threshold].values)
            timedeltas[timedeltas > max_sample_period] = max_sample_period
            on_durations[day] = (timedeltas.sum().astype('timedelta64[s]')
                                 .astype(np.int64) / SECS_PER_HOUR)
            sample_sizes[day] = data_for_day.size

        df = pd.DataFrame({'on_duration':on_durations,
                           'sample_size':sample_sizes})

        return df.dropna()

