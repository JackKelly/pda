from __future__ import print_function, division
import pandas as pd
import numpy as np
import scipy.stats as stats
import pda.load_pwr_data as load_pwr_data
import os, copy, datetime, sys
import matplotlib.dates as mdates

"""
Contains the Channel class (for representing a single channel or appliance)
along with some helper functions.

REQUIREMENTS:
  pandas >= 0.11.0
  pytables (sudo apt-get install python-tables)
"""

DD = '/data/mine/vadeec/merged/house_1'
SECS_PER_HOUR = 3600
SECS_PER_DAY = 86400
MINS_PER_DAY = 1440
DEFAULT_TIMEZONE = 'Europe/London'
ACCEPTABLE_DROPOUT_RATE_IF_SOMETIMES_UNPLUGGED = 0.9
ACCEPTABLE_DROPOUT_RATE_IF_NOT_UNPLUGGED = 0.2

# Threshold between "on" and "off" in watts. Will be overridden if
# data_dir contains a "custom_on_power_thresholds.dat" file
DEFAULT_ON_POWER_THRESHOLD = 3


def _secs_per_period_alias(alias):
    """Seconds for each period length."""
    dr = pd.date_range('00:00', periods=2, freq=alias)
    return (dr[-1] - dr[0]).total_seconds()


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


def _load_sometimes_unplugged(data_dir):
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


def _get_sample_period(series):
    """Find the sample period by finding the stats.mode of the 
    forward difference.  Only use the first 100 samples (for speed).
    Returns period in seconds (float).
    """
    fwd_diff = np.diff(series.index.values[:100]).astype(np.float)
    mode_fwd_diff = stats.mode(fwd_diff)[0][0]
    return mode_fwd_diff / 1E9


def _indicies_of_periods(datetime_index, freq):
    """
    Args:
        datetime_index (pd.tseries.index.DatetimeIndex)

        freq (str): one of the following:
            'A' for yearly
            'M' for monthly
            'D' for daily
            'H' for hourly
            'T' for minutely

    Returns: date_range, boundaries:

        date_range (pd.tseries.index.DateTimeIndex)

        boundaries (list):
            2-tuples of ints: (start index, end index for period)
    """
    date_range = pd.date_range(datetime_index[0], datetime_index[-1], freq=freq)

    # Declare and initialise some constants and variables used
    # during the loop...

    # Find the minimum sample period.
    # For the sake of speed, only use the first 100 samples.
    FWD_DIFF = np.diff(datetime_index.values[:100]).astype(np.float)
    MIN_SAMPLE_PERIOD = FWD_DIFF.min() / 1E9
    MAX_SAMPLES_PER_PERIOD = _secs_per_period_alias(freq) / MIN_SAMPLE_PERIOD
    MAX_SAMPLES_PER_2_PERIODS = MAX_SAMPLES_PER_PERIOD * 2
    n_rows_processed = 0
    boundaries = []
    for end_time in date_range[1:]:
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

        indicies_for_period = np.where(rows_to_process < end_time)[0]
        if indicies_for_period.size > 0:
            first_i_for_period = indicies_for_period[0] + n_rows_processed
            last_i_for_period = indicies_for_period[-1] + n_rows_processed + 1
            boundaries.append((first_i_for_period, last_i_for_period))
            n_rows_processed += last_i_for_period - first_i_for_period

    date_range = date_range[:-1] # so as not to exceed size of boundaries
    return date_range, boundaries


def _has_subsecond_resolution(series):
    """Returns true if series.index contains sub-second resolution.

    Only searches the first 1000 entries of series.index.
    """
    us = np.array([ts.microsecond for ts in series.index[:1000]])
    return np.any(us > 0)


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
                 series=None, # pd.Series
                 name="", # str
                 acceptable_dropout_rate = ACCEPTABLE_DROPOUT_RATE_IF_NOT_UNPLUGGED,
                 on_power_threshold = DEFAULT_ON_POWER_THRESHOLD,
                 start_date=None, # str
                 end_date=None # str
                 ):
        self.data_dir = data_dir
        if isinstance(chan, basestring):
            self.chan = self._get_chan_id_from_label(chan)
        else:
            self.chan = chan
        self.sample_period = sample_period
        self.max_sample_period = max_sample_period
        self.name = name
        self.series = None if series is None else series.dropna()
        self.acceptable_dropout_rate = acceptable_dropout_rate
        self.on_power_threshold = on_power_threshold

        # Load REDD / Jack's data
        if self.data_dir is not None and self.chan is not None:
            self._load(self.data_dir, self.chan, timezone, start_date, end_date)
        else:
            self._update_sample_period()

    def _get_chan_id_from_label(self, label):
        labels = load_labels(self.data_dir)
        inv_labels = {v:k for k,v in labels.items()}
        try:
            return int(inv_labels[label])
        except KeyError:
            print(label, 'not found in labels.dat.  Valid labels=',
                  file=sys.stderr)
            for chan, label in labels.items():
                print('{}: {}'.format(chan, label))
            raise

    def _update_sample_period(self):
        if (self.sample_period is None and 
            self.series is not None and self.series.size):
            self.sample_period = _get_sample_period(self.series)

    def _get_filename(self, data_dir=None, prefix='', suffix='dat'):
        data_dir = data_dir if data_dir else self.data_dir
        filename = prefix + 'channel_{:d}.{:s}'.format(self.chan, suffix)
        return os.path.join(data_dir, filename)

    def _load(self, 
              data_dir, # str
              chan, # int
              timezone=DEFAULT_TIMEZONE, # str
              start_date=None, end_date=None, # str
              force_reload=False
             ):
        """Load power data.  If an HDF5 (.h5) file exists for this channel
        and if that .h5 file is newer than the corresponding .dat file then
        the .h5 file will be loaded.  Else the .dat file will be loaded and
        and .h5 will be created.
        """

        self.data_dir = data_dir
        self.chan = chan
        self._load_metadata()
        
        dat_filename = self._get_filename()
        hdf5_filename = self._get_filename(suffix='h5')
        load_dat_func = lambda: self._load_pwr_data(dat_filename, timezone)
        self._load_cropped_hdf5(hdf5_filename, dat_filename, start_date, 
                                end_date, force_reload, load_dat_func)

    def _load_pwr_data(self, dat_filename, timezone):
        self.series = load_pwr_data.load(dat_filename, tz=timezone)
        self.series = self.series.sort_index() # MIT REDD data isn't always in order
        self._update_sample_period()

    def load_high_freq_mains(self, filename, param='active',
                             timezone=DEFAULT_TIMEZONE):
        """
        Args:
           filename (str): including full path and suffix.
           param (str): active | apparent | volts
           timezone (str): Optional.  Defaults to DEFAULT_TIMEZONE.
        """

        self.data_dir = os.path.dirname(filename)
        hdf5_filename = os.path.splitext(filename)[0] + '.h5'
        if (os.path.exists(hdf5_filename) and 
            os.path.getmtime(hdf5_filename) > os.path.getmtime(filename)):
            store = pd.HDFStore(hdf5_filename, 'r')
            df = store['df']
        else:
            date_parser = lambda x: datetime.datetime.utcfromtimestamp(x)
            df = pd.read_csv(filename, sep=' ', header=None, index_col=0,
                             parse_dates=True, date_parser=date_parser, 
                             names=['active','apparent','volts'])
            df = df.tz_localize('UTC').tz_convert(timezone)
            df = df.astype(np.float32)
            store = pd.HDFStore(hdf5_filename, 'w', complevel=9, complib='blosc')
            store['df'] = df
        store.close()

        self.series = df[param]
        self.name = param
        self._update_sample_period()

    def load_wattsup(self, filename, start_time=0,
                          timezone=DEFAULT_TIMEZONE):
        """
        Args:
            filename (str): including full path and suffix.
            start_time (str or datetime): Optional    
        """

        self.data_dir = os.path.dirname(filename)
        data = np.genfromtxt(filename)
        rng = pd.date_range(start_time, periods=len(data), 
                            freq='S', tz=timezone)
        self.series = pd.Series(data, index=rng)
        self.sample_period = 1

    def load_wattsup_raw(self, filename, start_time=0,
                         timezone=DEFAULT_TIMEZONE):
        """
        Loads output file from wattsup.  Also normalises using voltage.
        Args:
            filename (str): including full path and suffix.
            start_time (str or datetime): Optional    
        """

        self.data_dir = os.path.dirname(filename)
        names = [u'Time', u'Watts', u'Volts', u'Amps', u'WattHrs',
                 u'Cost', u'Avg Kwh', u'Mo Cost', u'Max Wts', u'Max Vlt', 
                 u'Max Amp', u'Min Wts', u'Min Vlt', u'Min Amp',
                 u'Pwr Fct', u'Dty Cyc', u'Pwr Cyc', 'junk1', 'junk2']
        df = pd.read_csv(filename, sep='\t', header=None, skiprows=[0,1], 
                         skipfooter=1, dtype=np.float32, names=names)
        rng = pd.date_range(start_time, periods=len(df), 
                            freq='S', tz=timezone)
        df.index = rng
        self.series = df.Watts
        self.sample_period = 1
        self = self.normalise_power(voltage=df.Volts)

    def _save_hdf5(self, data_dir=None, hdf5_filename=None):
        """Saves self.series to HDF5 file.

        Args:
            hdf5_filename (str): optional. Full filename including path 
                and suffix.  Defaults to data_dir/channel_<chan>.h5
            data_dir (str): optional.  If provided then save hdf5 file to this
                data directory.  If not provided then use self.data_dir.
        """
        if hdf5_filename is None:
            hdf5_filename = self._get_filename(data_dir, suffix='h5')
        store = pd.HDFStore(hdf5_filename, 'w', complevel=9, complib='blosc')
        if self.series is not None and self.series.size:
            store['series'] = self.series
        store.close()

    def _load_hdf5(self, hdf5_filename):
        """Loads HDF5 file to self.series.

        Args:
            hdf5_filename (str): optional. Full filename including path 
                and suffix.
        """
        store = pd.HDFStore(hdf5_filename, 'r')
        try:
            self.series = store['series']
        except KeyError as e:
            self.series = None
            if str(e) == '\'No object named series in the file\'':
                print('no data in HDF5 file, probably just inactive. ', end='')
            else:
                raise
        store.close()
        self._update_sample_period()

    def _load_cropped_hdf5(self, hdf5_filename, dat_filename,
                           start_date, end_date, force_reload, load_dat_func):
        # Now handle loading of cropped data
        def date_to_str(date):
            return pd.datetools.parse(date).strftime('%s') if date else "_"

        hdf5_cropped_data_fname = None
        if start_date or end_date:
            date_range_str = "-"
            date_range_str += date_to_str(start_date)
            date_range_str += "-to-"
            date_range_str += date_to_str(end_date)
            hdf5_cropped_data_fname = (os.path.splitext(hdf5_filename)[0] + 
                                       date_range_str + '.h5')

        # Load the data
        if (not force_reload and hdf5_cropped_data_fname and 
            os.path.exists(hdf5_cropped_data_fname) and 
            os.path.getmtime(hdf5_cropped_data_fname) > os.path.getmtime(dat_filename)):
            # Load the cached, cropped and normalised HDF5 file
            self._load_hdf5(hdf5_cropped_data_fname)
        elif (not force_reload and os.path.exists(hdf5_filename) and 
              os.path.getmtime(hdf5_filename) > os.path.getmtime(dat_filename)):
            # Load the cached (but not cropped) HDF5 file
            self._load_hdf5(hdf5_filename)
            if start_date or end_date:
                self.series = self.crop(start_date, end_date).series
                self._save_hdf5(hdf5_filename=hdf5_cropped_data_fname)
        else:
            # Load the raw dat file
            load_dat_func()
            self._save_hdf5(hdf5_filename=hdf5_filename)
            if start_date or end_date:
                self.series = self.crop(start_date, end_date).series
                self._save_hdf5(hdf5_filename=hdf5_cropped_data_fname)

    def _load_metadata(self):
        # load labels file
        labels = load_labels(self.data_dir)
        self.name = labels.get(self.chan, "")

        # load sometimes_unplugged file
        if self.name in _load_sometimes_unplugged(self.data_dir):
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

    def get_long_name(self):
        long_labels = {'tv': 'TV',
                       'htpc': 'home theatre PC',
                       'lcd office': 'office LCD screen',
                       'livingroom s lamp': 'livingroom standing lamp',
                       'childs ds lamp': 'reading lamp in child\'s room',
                       'bedroom ds lamp': 'bedroom dimmable standing lamp',
                       'kitchen lights': 'dimmable kitchen ceiling lights',}
        short_label = self.name.replace('_', ' ')
        try:
            return long_labels[short_label]
        except KeyError:
            return short_label

    def normalise_power(self, voltage=None, v_norm=None,
                        use_subsecond_data=None):
        """Uses Hart's formula to calculate:

            "admittance in the guise of 'normalized power':
        
            P_{Norm}(t) = 230^2 x Y(t) = (230 / V(t))^2 x P(t)

            This is just the admittance adjusted by a constant scale
            factor, resulting in the power normalized to 120 V, i.e.,
            what the power would be if the utility provided a steady
            120 V and the load obeyed a linear model. It is a far more
            consistent signature than power... All of our prototype
            NALMs use step changes in the normalized power as the
            signature."

        (equation 4, page 8 of Hart 1992)

        Does not alter self.  Instead returns a normalised copy.

        Args:
            voltage (pd.Series)
            v_norm (pd.Series).  Need to provide one of v_norm or voltage.
            use_subsecond_data (bool): Optional.  Decides whether or not
                to discard subsecond data on the voltage timeseries. Default
                is to use subsecond data only if it is available on both self
                and voltage.

        Returns:
            p_norm (Channel)
        """
        
        assert(voltage is not None or v_norm is not None)
        p_norm = copy.copy(self)
        p_norm.name += '_normalised'

        if use_subsecond_data is None:
            use_subsecond_data = (_has_subsecond_resolution(self.series) and 
                                  _has_subsecond_resolution(v_norm if voltage is None
                                                           else voltage))
        if v_norm is None:
            v_norm = (242 / voltage)**2

        if not use_subsecond_data:
            # Discard sub-second data
            v_norm_tz = v_norm.index.tz
            v_norm = v_norm.to_period('S').to_timestamp()
            v_norm = v_norm.tz_localize(v_norm_tz)

        p_norm.series *= v_norm
        p_norm.series = p_norm.series.dropna()

        return p_norm

    def load_normalised(self, data_dir=None, high_freq_basename='mains.dat', 
                        chan=None, high_freq_param=None, force_reload=False,
                        start_date=None, end_date=None,
                        timezone=DEFAULT_TIMEZONE):
        """
        Loads normalised power.

        Caches normalised power to disk as HDF5 file.

        Args:
            data_dir (str): Required.
            high_freq_basename (str): Defaults to 'mains.dat'
            chan (int): Optional.
            high_freq_param (str): Optional. active | apparent
            force_reload (boolean): Default=False. If True then ignore cached H5
            timezone: Defaults to DEFAULT_TIMEZONE
            start_date, end_date (str)

        Examples:
            To load normalised active power from SCPM data file:
            load_normalised(directory=DD, high_freq_basename='mains.dat', 
                            high_freq_param='active')

            To load normalised power from channel 5:
            load_normalised(directory=DD, high_freq_basename='mains.dat', 
                            chan=5)
        """
        # First, set dat_filename and hdf5_filename:
        self.data_dir = data_dir
        self.chan = chan
        high_freq_filename = os.path.join(self.data_dir, high_freq_basename)
        if chan is None:
            dat_filename = high_freq_filename
            self.name = ('normalised_' + high_freq_param + '_' +
                         os.path.splitext(high_freq_basename)[0])
            hdf5_filename = os.path.join(self.data_dir, self.name + '.h5')
        else:
            dat_filename = self._get_filename()
            hdf5_filename = self._get_filename(prefix='normalised_', suffix='h5')

        load_dat_func = lambda: self._load_dat_and_normalise(high_freq_filename,
                                                             high_freq_param, 
                                                             timezone)

        self._load_cropped_hdf5(hdf5_filename, dat_filename, 
                                start_date, end_date, force_reload,
                                load_dat_func)

    def _load_dat_and_normalise(self, high_freq_filename, high_freq_param, 
                                timezone):
        if self.chan is None:
            self.load_high_freq_mains(high_freq_filename, 
                                      param=high_freq_param, timezone=timezone)
        else:
            self.load(self.data_dir, self.chan, timezone=timezone)

        v = Channel()
        v.load_high_freq_mains(high_freq_filename, 'volts', timezone=timezone)
        self = self.normalise_power(voltage=v.series)

    def __str__(self):
        s = ""
        s += "           name = {}\n".format(self.name)
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

    def crop(self, start_date=None, end_date=None):
        """Does not modify self.  Instead returns a cropped channel.
        Args:
            start_date, end_date: strings like '2013/6/15' or datetime objects
        """
        cropped_chan = copy.copy(self)
        if start_date:
            cropped_chan.series = cropped_chan.series[cropped_chan.series.index 
                                                      >= start_date]
        if end_date:
            cropped_chan.series = cropped_chan.series[cropped_chan.series.index
                                                      <= end_date]
        return cropped_chan

    def usage_per_period(self, freq, tz_convert=None, verbose=False):
        """
        Args:
            freq (str): see _indicies_of_periods() for acceptable values.

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

        date_range, boundaries = _indicies_of_periods(series.index,freq)
        period_range = date_range.as_period(freq=freq)
        hours_on = pd.Series(index=period_range, dtype=np.float, 
                             name=self.name+' hours on')
        kwh = pd.Series(index=period_range, dtype=np.float, 
                        name=self.name+' kWh')

        MAX_SAMPLES_PER_PERIOD = _secs_per_period_alias(freq) / self.sample_period
        MIN_SAMPLES_PER_PERIOD = (MAX_SAMPLES_PER_PERIOD *
                                  (1-self.acceptable_dropout_rate))

        for period_i, period in enumerate(period_range):
            try:
                period_start_i, period_end_i = boundaries[period_i]
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

    def activity_distribution(self, bin_size='T', timespan='D'):
        """Returns a distribution describing when this appliance was turned
        on over repeating timespans.  For example, if you want to see
        which times of day this appliance was used then use 
        bin_size='T' (minutely) or bin_size='H' (hourly) and
        timespan='D' (daily).

        Args:
            bin_size (str): offset alias
            timespan (str): offset alias

        For valid offset aliases, see:
        http://pandas.pydata.org/pandas-docs/dev/timeseries.html#offset-aliases

        Returns:
            pd.Series. One row for each bin in a timespan.
            The values count the number of times this appliance has been on at
            that particular time of the timespan.
            Times are handled in local time.
        """
        
        # Create a pd.Series with PeriodIndex with period length of 1 minute.
        binned_data = self.series.resample(bin_size, how='max').to_period()
        binned_data = binned_data > self.on_power_threshold

        timespans, boundaries = _indicies_of_periods(binned_data.index.to_timestamp(),
                                                     timespan)
        timespans = timespans.to_period(freq=timespan)

        first_timespan = timespans[0]
        bins = pd.period_range(first_timespan.start_time, 
                               first_timespan.end_time,
                               freq=bin_size)
        distribution = pd.Series(0, index=bins)
        
        bins_per_timespan = int(round(_secs_per_period_alias(timespan) /
                                      _secs_per_period_alias(bin_size)))

        for span_i, span in enumerate(timespans):
            try:
                start_index, end_index = boundaries[span_i]
                data_for_timespan = binned_data[start_index:end_index]
            except IndexError:
                print("No data for", span)
                continue

            bins_since_first_timespan = (first_timespan - span) * bins_per_timespan
            data_shifted = data_for_timespan.shift(bins_since_first_timespan, 
                                                   bin_size)
            distribution = distribution.add(data_shifted, fill_value = 0)

        return distribution

    def on(self):
        """Returns pd.Series with Boolean values indicating whether the
        appliance is on (True) or off (False).  Adds an 'off' entry if data
        is lost for more than self.max_sample_period.
        """
        when_on = self.series >= self.on_power_threshold
        # add an 'off' entry whenever data is lost for > self.max_sample_period
        time_delta = np.diff(self.series.index.values)
        max_sample_period = np.timedelta64(self.max_sample_period, 's')
        dropout_dates = self.series.index[:-1][time_delta > max_sample_period]
        insert_offs = pd.Series(False, 
                                index=dropout_dates +
                                      pd.DateOffset(seconds=self.max_sample_period))
        when_on = when_on.append(insert_offs)
        when_on = when_on.sort_index()
        return when_on

    def on_off_events(self, ignore_n_off_samples=None):
        """
        Detects on/off switch events.

        Returns a pd.Series with np.int8 values.
               1 == turn-on event.
              -1 == turn-off event.
        
        Example (pseudo-code):
            self.series = [0, 0, 100, 100, 100, 0]
            c.on_off_events()
            2:  1
            5: -1

        Args:
            ignore_n_off_samples (int): Optional.  Ignore this number of 
            off samples.  For example, if the input is [0, 100, 0, 100, 100, 0, 0]
            then the single zero with 100 either side could be ignored if
            ignore_n_off_samples = 1, hence only one on-event and one off-event
            would be reported.

        """
        on = self.on()
        
        if ignore_n_off_samples is not None:
            on_smoothed = pd.rolling_max(on, window=ignore_n_off_samples+1) 
            on_smoothed.iloc[:ignore_n_off_samples] = on.iloc[:ignore_n_off_samples].values
            on = on_smoothed.dropna()
        
        on = on.astype(np.int8)
        events = on[1:] - on.shift(1)[1:]

        if ignore_n_off_samples is not None:
            i_off_events = np.where(events == -1)[0] # indicies of off-events
            for i in range(ignore_n_off_samples):
                events.iloc[i_off_events-i] = 0
            events.iloc[i_off_events-ignore_n_off_samples] = -1

        events = events[events != 0]
        return events

    def durations(self, on_or_off, ignore_n_off_samples=None):
        """Returns an array describing length of every on or off durations (in
        seconds).

        Args:
            on_or_off (str): "on" or "off"
            ignore_n_off_samples (int): Optional.  Ignore this number of 
                off-samples. See on_off_events().  Only makes sense to use this
                when on_or_off == 'on'.

        """
        events = self.on_off_events(ignore_n_off_samples=ignore_n_off_samples)
        delta_time_array = np.diff(events.index.values).astype(int) / 1E9
        delta_time = pd.Series(delta_time_array, index=events.index[:-1])
        diff_for_mode = 1 if on_or_off == 'on' else -1
        events_for_mode = events == diff_for_mode
        durations = delta_time[events_for_mode]
        if ignore_n_off_samples is not None:
            durations = durations[durations > self.sample_period*ignore_n_off_samples]

        return durations

    def plot(self, ax, label=None, date_format='%d/%m/%y %H:%M:%S', **kwargs):
        ax.xaxis.axis_date(tz=self.series.index.tzinfo)
        ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        label = label if label else self.name
        _to_ordinalf_np_vectorized = np.vectorize(mdates._to_ordinalf)
        x = _to_ordinalf_np_vectorized(self.series.index.to_pydatetime())
        ax.plot(x, self.series, label=label, **kwargs)
        ax.set_ylabel('watts')
