import pandas as pd
import numpy as np
import datetime as dt

class Channel(object):
    """A single channel of data.
    
    Attributes:
        series (pd.series): the power data in Watts.
        max_sample_period (dt.timedelta): The maximum time allowed
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
        self.max_sample_period = dt.timedelta(seconds=max_sample_period)

    def on_duration_per_day(self, pwr_threshold=2):
        """
        Args:
            pwr_threshold (float or int): Optional. Threshold which defines the
                distinction between "on" and "off".  Watts.    
            
        Returns:
            pd.dataframe.  One row per day.
        """
        
        # Go through each pair of consecutive samples.
        #   Is the first sample above threshold?  If yes:
        #      Is the period < max_sample_period?  If yes:
        #         Add this period to the total
        
        # Construct the index for the output.  Each item is a Datetime
        # at midnight.  Get rid of the very first item so rng represents
        # just the full-days for which we have date.
        rng = pd.date_range(self.series.index[0], self.series.index[-1],
                            freq='D', normalize=True)[1:]
        
        for day in rng:
            on_time = dt.timedelta(0)
            data_for_day = self.series[day.strftime('%Y-%m-%d')]
            for i in range(data_for_day.size-2):
                if data_for_day[i] >= pwr_threshold:
                    period = data_for_day.index[i+1] - data_for_day.index[i]
                    if period > self.max_sample_period:
                        period = self.max_sample_period
                    on_time += period    
            print(day, on_time)