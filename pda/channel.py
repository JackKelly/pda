from __future__ import print_function, division
import pandas as pd
import numpy as np
import scipy.stats as stats
import load_pwr_data
import os

"""
REQUIREMENTS:
  pandas >= 0.11.0
  pytables (sudo apt-get install python-tables)
"""

SECS_PER_HOUR = 3600
SECS_PER_DAY = 86400
DEFAULT_TIMEZONE = 'Europe/London'
ACCEPTABLE_DROPOUT_RATE_IF_SOMETIMES_UNPLUGGED = 0.9
ACCEPTABLE_DROPOUT_RATE_IF_NOT_UNPLUGGED = 0.2

# Threshold between "on" and "off" in watts. Will be overridden if
# data_dir contains a "custom_on_power_thresholds.dat" file
DEFAULT_ON_POWER_THRESHOLD = 3


def load_labels(data_dir):
    """Loads data from labels.dat file.

    Args:
        data_dir (str)

    Returns:
        A dict mapping channel numbers (ints) to appliance names (str)
    """
    filename = os.path.join(data_dir, 'labels.dat')
    with open(filename) as labels_file:
        lines = labels_file.readlines()
    
    labels = {}
    for line in lines:
        line = line.split(' ')
        labels[int(line[0])] = line[1].strip() # TODO add error handling if line[0] not an int

    return labels


def load_sometimes_unplugged(data_dir):
    """Loads data_dir/sometimes_unplugged.dat file and returns a
    list of strings.  Returns an empty list if file doesn't exist.
    """
    su_filename = os.path.join(data_dir, 'sometimes_unplugged.dat')
    try:
        file = open(su_filename)
    except IOError:
        return []
    else:
        lines = file.readlines()
        return [line.strip() for line in lines if line.strip()]


class Channel(object):
    """
    A single channel of data.
    
    Attributes:
        series (pd.Series): the power data in Watts.
        max_sample_period (float): The maximum time allowed
            between samples. If data is missing for longer than
            max_sample_period then the appliance is assumed to be off.
        sample_period (float)
        acceptable_dropout_period (float): [0, 1]
        on_power_threshold (float): watts
        name (str)
        data_dir (str)
        chan (int): channel number
    """
    
    def __init__(self, 
                 data_dir=None, # str
                 chan=None, # int
                 timezone=DEFAULT_TIMEZONE, # str
                 sample_period=None, # seconds
                 max_sample_period=20, # seconds
                 ):
        self.data_dir = data_dir
        self.chan = chan
        self.sample_period = sample_period
        self.max_sample_period = max_sample_period

        self.name = ""
        self.series = None
        self.acceptable_dropout_rate = ACCEPTABLE_DROPOUT_RATE_IF_NOT_UNPLUGGED
        self.on_power_threshold = DEFAULT_ON_POWER_THRESHOLD

        if data_dir is not None and chan is not None:
            self.load(data_dir, chan, timezone)

    def get_filename(self, data_dir=None, suffix='dat'):
        data_dir = data_dir if data_dir else self.data_dir
        return os.path.join(data_dir, 'channel_{:d}.{:s}'.format(self.chan, 
                                                                 suffix))
        
    def load(self, 
             data_dir, # str
             chan, # int
             timezone=DEFAULT_TIMEZONE # str
             ):
        """Load power data.  If an HDF5 (.h5) file exists for this channel
        and if that .h5 file is newer than the corresponding .dat file then
        the .h5 file will be loaded.  Else the .dat file will be loaded and
        and .h5 will be created.
        """

        self.data_dir = data_dir
        self.chan = chan
        self.load_metadata()
        
        dat_filename = self.get_filename()
        hdf5_filename = self.get_filename(suffix='h5')
        if (os.path.exists(hdf5_filename) and 
            os.path.getmtime(hdf5_filename) > os.path.getmtime(dat_filename)):
            store = pd.HDFStore(hdf5_filename)
            self.series = store['series']
            store.close()
        else:
            self.series = load_pwr_data.load(dat_filename, tz=timezone)
            self.save()

        if self.sample_period is None:
            # Find the sample period by finding the stats.mode of the 
            # forward difference.  Only use the first 100 samples (for speed).
            fwd_diff = np.diff(self.series.index.values[:100]).astype(np.float)
            mode_fwd_diff = int(round(stats.mode(fwd_diff)[0][0]))
            self.sample_period = mode_fwd_diff / 1E9

    def save(self, data_dir=None):
        """Saves self.series to data_dir/channel_<chan>.h5

        Args:
            data_dir (str): optional.  If provided then save hdf5 file to this
            data directory.  If not provided then use self.data_dir.
        """
        hdf5_filename = self.get_filename(data_dir, suffix='h5')
        store = pd.HDFStore(hdf5_filename)
        store['series'] = self.series
        store.close()

    def load_metadata(self):
        # load labels file
        labels = load_labels(self.data_dir)
        self.name = labels.get(self.chan, "")

        # load sometimes_unplugged file
        if self.name in load_sometimes_unplugged(self.data_dir):
            self.acceptable_dropout_rate = ACCEPTABLE_DROPOUT_RATE_IF_SOMETIMES_UNPLUGGED 

        # load custom on power thresholds
        opt_filename = os.path.join(self.data_dir, 
                                    'custom_on_power_thresholds.dat')
        try:
            file = open(opt_filename)
        except IOError:
            pass
        else:
            lines = file.readlines()
            for line in lines:
                line = line.split(' ')
                if line[0] == self.name:
                    self.on_power_threshold = float(line[1].strip())

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

    def usage_per_day(self, tz_convert=None, verbose=False):
        """
        Args:
            tz_convert (str): (optional) Use 'midnight' in this timezone.

        Object member variables which you may want to modify before calling
        this function:
            on_power_threshold (float or int): Threshold which defines the
                distinction between "on" and "off".  Watts.
            acceptable_dropout_rate (float). Must be >= 0 and <= 1.  Remove any
                row which has a worse dropout rate.

        Returns:
            pd.DataFrame.  One row per day.  data is (np.float)
                Series:
                     hours_on
                     kwh
        """
        
        assert(0 <= self.acceptable_dropout_rate <= 1)

        series = (self.series if tz_convert is None else
                  self.series.tz_convert(tz_convert))

        # Construct the index for the output.  Each item is a Datetime
        # at midnight.
        index = pd.date_range(series.index[0], series.index[-1],
                              freq='D', normalize=True)
        hours_on = pd.Series(index=index, dtype=np.float, 
                             name=self.name+' hours on')
        kwh = pd.Series(index=index, dtype=np.float, 
                        name=self.name+' kWh')

        max_samples_per_day = SECS_PER_DAY / self.sample_period
        min_samples_per_day = max_samples_per_day * (1-self.acceptable_dropout_rate)
        max_sample_period = np.timedelta64(self.max_sample_period, 's')
        max_samples_per_2days = max_samples_per_day * 2

        unprocessed_data = series.copy()
        for day_i in range(index.size-1):
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
                                        < index[day_i+1])[0]
            day = index[day_i]
            if indicies_for_day.size == 0:
                if verbose:
                    print("No data available for   ", day.strftime('%Y-%m-%d'))
                continue
            data_for_day = unprocessed_data[indicies_for_day]
            unprocessed_data = unprocessed_data[indicies_for_day[-1]+1:]
            if data_for_day.size < min_samples_per_day:
                if verbose:
                    print("Insufficient samples for", day.strftime('%Y-%m-%d'),
                          "; samples =", data_for_day.size,
                          "dropout_rate = {:.2%}".format(1 - (data_for_day.size / 
                                                              max_samples_per_day)))
                    print("                 start =", data_for_day.index[0])
                    print("                   end =", data_for_day.index[-1])
                continue
            i_above_threshold = np.where(data_for_day[:-1] >= 
                                         self.on_power_threshold)[0]
            td_above_thresh = (data_for_day.index[i_above_threshold+1].values -
                               data_for_day.index[i_above_threshold].values)
            td_above_thresh[td_above_thresh > max_sample_period] = max_sample_period
            hours_on[day] = (td_above_thresh.sum().astype('timedelta64[s]')
                                 .astype(np.int64) / SECS_PER_HOUR)

            # Calculate kWh per day
            td = np.diff(data_for_day.index.values.astype(np.int64)) / 1E9
            td_limited = np.where(td > self.max_sample_period,
                                  self.max_sample_period, td)
            watt_seconds = (td_limited * data_for_day.values[:-1]).sum()
            kwh[day] = watt_seconds / 3600000

        return pd.DataFrame({'hours_on': hours_on.dropna(),
                             'kwh': kwh.dropna()})

    def kwh(self):
        dt_limited = np.where(self._dt>self.max_sample_period, 
                              self.max_sample_period, self._dt)
        watt_seconds = (dt_limited * self.data['watts'][:-1]).sum()
        return watt_seconds / 3600000
