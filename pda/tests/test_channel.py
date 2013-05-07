#!/usr/bin/python
from __future__ import print_function, division
import unittest, os, inspect
from pda.channel import Channel, SECS_PER_HOUR
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Taken from http://stackoverflow.com/a/6098238/732596
FILE_PATH = os.path.dirname(inspect.getfile(inspect.currentframe()))
SMALL_TEST_DATA_PATH = os.path.join(FILE_PATH, 'test_data', 'small')
LARGE_TEST_DATA_PATH = os.path.join(FILE_PATH, 'test_data', 'large')

class TestChannel(unittest.TestCase):
    def setUp(self):
        self.channel = Channel(os.path.join(LARGE_TEST_DATA_PATH, 'channel_8.dat'))
        
    def test_init(self):
        self.assertIsNotNone(self.channel)
         
    # def test_plot(self):
    #     print(self.channel.series.tail())
    #     print(self.channel.series.index)
    #     self.channel.series.plot()
    #     plt.show()

    def test_on_duration_per_day(self):
        c = Channel()
        idx = pd.date_range('2013-04-01', '2013-04-05', freq='6S')
        target_on_duration = 86394 / SECS_PER_HOUR
        c.series = pd.Series(100, index=idx)
        c.sample_period = 6
        df = c.on_duration_per_day()
        self.assertEqual(df['on_duration'][0], target_on_duration)
        self.assertEqual(df['sample_size'][0], 86400/6)
        
        # c.series.plot()
        # plt.show()


if __name__ == '__main__':
    unittest.main()
