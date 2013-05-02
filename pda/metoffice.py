from __future__ import print_function, division
import pandas as pd
import numpy as np

def open_daily_xls(filename, sheet='HEATHROW'):
    xls = pd.ExcelFile(filename)
    df = xls.parse(sheet, skiprows=4, skip_footer=4, na_values=['n/a'], 
                   parse_dates=True, index_col=0, parse_cols=10)

    columns = {
    u'Daily Maximum Temperature (0900-0900) (\xb0C)': 'max_temp',
    u'Daily Minimum Temperature (0900-0900) (\xb0C)': 'min_temp',
    u'Daily Mean Temperature from Hourly Data (0900-0900) (\xb0C)': 'mean_temp',
    u'Daily Mean Windspeed (0100-2400) (knots)': 'mean_windspeed',
    u'Daily Maximum Gust (0100-2400) (knots)': 'max_gust',
    u'Daily Total Rainfall (0900-0900)(mm)': 'rainfall',
    u'Daily Total Global Radiation (KJ/m2)': 'radiation',
    u'Daily Total Sunshine (0100-2400) (hrs)': 'sunshine',
    u'Daily Minimum Grass Temperature (0900-0900) (\xb0C)': 'max_grass_temp',
    u'Daily Minimum Concrete Temperature (0900-0900) (\xb0C)': 'max_concrete_temp'}
    
    df = df.rename(columns=columns)
    df = df.tz_localize('UTC')
    df = df.asfreq('D')
    
    # the rainfall column contains "tr" and "n/a".  
    # "tr" means "trace" (less than 0.05mm).  Replace "tr" with 0.04
    df['rainfall'][df['rainfall'] == 'tr'] = 0.04
    df['rainfall'] = df['rainfall'].astype(np.float)

    return df