from __future__ import print_function, division
import pandas as pd

def open_xls(filename, sheet='HEATHROW'):
    xls = pd.ExcelFile(filename)
    xls.parse(sheet)
    # http://pandas.pydata.org/pandas-docs/dev/io.html#excel-files
