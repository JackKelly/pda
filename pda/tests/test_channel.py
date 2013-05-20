#!/usr/bin/python
from __future__ import print_function, division
import unittest, os, inspect
from pda.channel import *
import pandas as pd
import correct_answers

# Taken from http://stackoverflow.com/a/6098238/732596
FILE_PATH = os.path.dirname(inspect.getfile(inspect.currentframe()))
SMALL_TEST_DATA_PATH = os.path.join(FILE_PATH, 'test_data', 'small')
LARGE_TEST_DATA_PATH = os.path.join(FILE_PATH, 'test_data', 'large')

class TestChannel(unittest.TestCase):
    def setUp(self):
        self.channel = Channel(LARGE_TEST_DATA_PATH, 8)
        
    def test_load_labels(self):
        labels = load_labels(SMALL_TEST_DATA_PATH)
        self.assertEqual(labels, correct_answers.labels)

    def test_load_sometimes_unplugged(self):
        su = load_sometimes_unplugged(SMALL_TEST_DATA_PATH)
        self.assertEqual(su,
                         ['laptop',
                          'kettle',
                          'toaster',
                          'lcd_office',
                          'hifi_office',
                          'livingroom_s_lamp',
                          'soldering_iron',
                          'gigE_&_USBhub',
                          'hoover',
                          'iPad_charger',
                          'utilityrm_lamp',
                          'hair_dryer',
                          'straighteners',
                          'iron',
                          'childs_ds_lamp',
                          'office_lamp3',
                          'office_pc',
                          'gigE_switch'])

    def test_init(self):
        self.assertIsNotNone(self.channel)
         
    # def test_plot(self):
    #     import matplotlib.pyplot as plt
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
        usage = c.usage_per_period('D')
        self.assertEqual(usage.hours_on[0], target_on_duration)
        self.assertAlmostEqual(usage.kwh[0], 2.4, places=3)
        
        # c.series.plot()
        # plt.show()

    def test_load_metadata(self):
        c = Channel(SMALL_TEST_DATA_PATH, 2)
        self.assertEqual(c.name, 'boiler')
        self.assertEqual(c.on_power_threshold, 50)
        self.assertEqual(c.acceptable_dropout_rate, 
                         ACCEPTABLE_DROPOUT_RATE_IF_NOT_UNPLUGGED)

        c = Channel(SMALL_TEST_DATA_PATH, 4)
        self.assertEqual(c.name, 'laptop')
        self.assertEqual(c.on_power_threshold, DEFAULT_ON_POWER_THRESHOLD)
        self.assertEqual(c.acceptable_dropout_rate, 
                         ACCEPTABLE_DROPOUT_RATE_IF_SOMETIMES_UNPLUGGED)

    def test_secs_per_period_alias(self):
        SECS_PER_FREQ = {'T':60, 'H': 3600, 'D': 86400, 'M': 2678400, 
                         'A': 31536000}
        for alias, secs in SECS_PER_FREQ.iteritems():
            self.assertEqual(secs_per_period_alias(alias), secs)

    def test_on_off_events(self):
        s = pd.Series([0, 0, 100, 100, 100, 0])
        c = Channel(series=s)
        events = c.on_off_events()
        self.assertEqual(events[2], 1)
        self.assertEqual(events[5], -1)

if __name__ == '__main__':
    unittest.main()
