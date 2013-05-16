from __future__ import print_function, division
import pandas as pd
import numpy as np
import scipy.stats as stats
import pda.load_pwr_data as load_pwr_data
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

# Seconds for each period length.
SECS_PER_FREQ = {'T':60, 'H': 3600, 'D': 86400, 'M': 2678400, 'A': 31536000}


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


def get_sample_period(series):
    """Find the sample period by finding the stats.mode of the 
    forward difference.  Only use the first 100 samples (for speed).
    Returns period in seconds (float).
    """
    fwd_diff = np.diff(series.index.values[:100]).astype(np.float)
    mode_fwd_diff = stats.mode(fwd_diff)[0][0]
    return mode_fwd_diff / 1E9


def indicies_of_periods(datetime_index, freq):
    """
    Args:
        datetime_index (pd.tseries.index.DatetimeIndex)

        freq (str): one of the following:
            'A' for yearly
            'M' for monthly
            'D' for daily
            'H' for hourly
            'T' for minutely

    Returns: period_range, period_boundaries:

        period_range (pd.tseries.index.PeriodIndex): in the 
            local time of datetime_index.

        period_boundaries (dict):
            keys = pd.Period (in the local time of datetime_index)
            values = 2-tuples of ints: (start index, end index for period)
    """
    try:
        TZ = datetime_index[0].tz.zone
    except AttributeError:
        TZ = None

    # Generate a PeriodIndex object
    period_range = pd.period_range(datetime_index[0], datetime_index[-1], 
                                   freq=freq)

    # Declare and initialise some constants and variables used
    # during the loop...

    # Find the minimum sample period.
    # For the sake of speed, only use the first 100 samples.
    FWD_DIFF = np.diff(datetime_index.values[:100]).astype(np.float)
    MIN_SAMPLE_PERIOD = FWD_DIFF.min() / 1E9
    MAX_SAMPLES_PER_PERIOD = SECS_PER_FREQ[freq] / MIN_SAMPLE_PERIOD
    MAX_SAMPLES_PER_2_PERIODS = MAX_SAMPLES_PER_PERIOD * 2
    n_rows_processed = 0
    period_boundaries = {}
    for period in period_range:
        # The simplest way to get data for just a single period is to use
        # data_for_day = datetime_index[period.strftime('%Y-%m-%d')]
        # but this takes about 300ms per call on my machine.
        # So we take advantage of several features of the data to achieve
        # a 300x speedup:
        # 1. We use the fact that the data is sorted in order, hence 
        #    we can chomp through it in order.
        # 2. MAX_SAMPLES_PER_PERIOD sets an upper bound on the number of
        #    datapoints per period.  The code is conservative and uses 
        #    MAX_SAMPLES_PER_2_PERIODS. We only search through a small subset
        #    of the available data.
        end_index = n_rows_processed+MAX_SAMPLES_PER_2_PERIODS
        rows_to_process = datetime_index[n_rows_processed:end_index]

        end_time = period.end_time
        # Convert to correct timezone.  Can't use end_time.tz_convert(TZ)
        # because this chucks out loads of warnings:
        # "Warning: discarding nonzero nanoseconds".  
        # To silence these warnings we re-create the
        # tz_localize function (see pandas/tslib.pyx) to allow us to pass
        # warn=False to to_pydatetime().
        end_time = pd.Timestamp(end_time.to_pydatetime(warn=False), tz=TZ)
        indicies_for_period = np.where(rows_to_process < end_time)[0]
        if indicies_for_period.size > 0:
            first_i_for_period = indicies_for_period[0] + n_rows_processed
            last_i_for_period = indicies_for_period[-1] + n_rows_processed + 1
            period_boundaries[period] = (first_i_for_period, last_i_for_period)
            n_rows_processed += last_i_for_period - first_i_for_period

    return period_range, period_boundaries


class Channel(object):
    """
    A single channel of data.
    
    Attributes:
        series (pd.Series): the power data in Watts.
        max_sample_period (float): The maximum time allowed
            between samples. If data is missing for longer than
            max_sample_period then the appliance is assumed to be off.
        sample_period (float): seconds
        acceptable_dropout_rate (float): [0, 1] (0 means no dropouts allowed)
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
            self.series = self.series.sort_index() # MIT REDD data isn't always in order
            self.save()

        if self.sample_period is None:
            self.sample_period = get_sample_period(self.series)

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

    def usage_per_period(self, freq, tz_convert=None, verbose=False):
        """
        Args:
            freq (str): see indicies_of_periods() for acceptable values.

            tz_convert (str): (optional) e.g. 'UTC' or 'Europe/London'

        Object member variables which you may want to modify before calling
        this function:

            on_power_threshold (float or int): Threshold which defines the
                distinction between "on" and "off".  Watts.

            acceptable_dropout_rate (float). Must be >= 0 and <= 1.  Remove any
                row which has a worse dropout rate.

        Returns:
            pd.DataFrame.  One row per period.  Index is PeriodIndex (UTC).
                Series:
                     hours_on
                     kwh
        """
        
        assert(0 <= self.acceptable_dropout_rate <= 1)

        series = (self.series if tz_convert is None else
                  self.series.tz_convert(tz_convert))

        period_range, period_boundaries = indicies_of_periods(series.index,freq)
        hours_on = pd.Series(index=period_range, dtype=np.float, 
                             name=self.name+' hours on')
        kwh = pd.Series(index=period_range, dtype=np.float, 
                        name=self.name+' kWh')

        MAX_SAMPLES_PER_PERIOD = SECS_PER_FREQ[freq] / self.sample_period
        MIN_SAMPLES_PER_PERIOD = (MAX_SAMPLES_PER_PERIOD *
                                  (1-self.acceptable_dropout_rate))

        for period in period_range:
            try:
                period_start_i, period_end_i = period_boundaries[period]
            except KeyError:
                if verbose:
                    print("No data available for   ",
                          period.strftime('%Y-%m-%d'))
                continue

            data_for_period = series[period_start_i:period_end_i]
            if data_for_period.size < MIN_SAMPLES_PER_PERIOD:
                if verbose:
                    dropout_rate = (1 - (data_for_period.size / 
                                         MAX_SAMPLES_PER_PERIOD))
                    print("Insufficient samples for",
                          period.strftime('%Y-%m-%d'),
                          "; samples =", data_for_period.size,
                          "dropout_rate = {:.2%}".format(dropout_rate))
                    print("                 start =", data_for_period.index[0])
                    print("                   end =", data_for_period.index[-1])
                continue
            
            hours_on[period] = self.hours_on(data_for_period)
            kwh[period] = self.kwh(data_for_period)

        return pd.DataFrame({'hours_on': hours_on,
                             'kwh': kwh})

    def hours_on(self, series=None):
        """Returns a float representing the number of hours this channel
        has been above threshold.

        Args:
           series (pd.Series): optional.  Defaults to self.series
        """
        MAX_SAMPLE_PERIOD = np.timedelta64(self.max_sample_period, 's')
        series = self.series if series is None else series
        i_above_threshold = np.where(series[:-1] >= 
                                     self.on_power_threshold)[0]
        td_above_thresh = (series.index[i_above_threshold+1].values -
                           series.index[i_above_threshold].values)
        td_above_thresh[td_above_thresh > MAX_SAMPLE_PERIOD] = MAX_SAMPLE_PERIOD

        secs_on = td_above_thresh.sum().astype('timedelta64[s]').astype(np.int64)
        return secs_on / SECS_PER_HOUR

    def kwh(self, series=None):
        """Returns a float representing the number of kilowatt hours (kWh) this 
        channel consumed.

        Args:
           series (pd.Series): optional.  Defaults to self.series
        """
        series = self.series if series is None else series
        td = np.diff(series.index.values.astype(np.int64)) / 1E9
        td_limited = np.where(td > self.max_sample_period,
                              self.max_sample_period, td)
        watt_seconds = (td_limited * series.values[:-1]).sum()
        return watt_seconds / 3600000

