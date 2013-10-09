#!/usr/bin/python
from __future__ import print_function, division
import unittest, os, inspect
from pda.channel import *
from pda.channel import _load_sometimes_unplugged, _secs_per_period_alias
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
        su = _load_sometimes_unplugged(SMALL_TEST_DATA_PATH)
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
        SECS_PER_FREQ = {'T':60, 'H': 3600, 'D': 86400, 'M': 2592000, 
                         'A': 31536000}
        for alias, secs in SECS_PER_FREQ.iteritems():
            try:
                self.assertEqual(_secs_per_period_alias(alias), secs)
            except:
                print(alias, secs)
                raise

    def test_on(self):
        pwr = [0, 0, 100, 100, 100, 100, 0, 0, 0, 100, 100, 0, 0, 100, 100]
        rng = pd.date_range('1/1/2013', periods=len(pwr), freq='6S')
        pwr2 = [0, 0, 0, 0]
        rng2 = pd.date_range(rng[-1]+100, periods=len(pwr2), freq='6S')
        series2 = pd.Series(pwr + pwr2, index = rng + rng2)
        c2 = Channel(series=series2)
        on = c2.on()
        self.assertSequenceEqual(list(on.values), 
                                 [False, False, True, True, True, True, 
                                  False, False, False, True, True, False, 
                                  False, True, True, False, False, False,
                                  False, False])

    def test_on_off_events(self):
        series = pd.Series([0, 0, 100, 100, 100, 0])
        c = Channel(series=series)
        events = c.on_off_events()
        self.assertEqual(events[2], 1)
        self.assertEqual(events[5], -1)

    def test_durations(self):
        pwr = [0, 0, 100, 100, 100, 100, 0, 0, 0, 100, 100, 0, 100, 100]
        rng = pd.date_range('1/1/2013', periods=len(pwr), freq='6S')
        series = pd.Series(pwr, index=rng)
        c = Channel(series=series)

        # Check on-durations
        on_durations = c.durations(on_or_off='on')
        self.assertEqual(len(on_durations), 2)
        self.assertEqual(on_durations[0], 6*4)
        self.assertEqual(on_durations[1], 6*2)

        # Check off-durations
        off_durations = c.durations(on_or_off='off')
        self.assertEqual(len(off_durations), 2)
        self.assertEqual(off_durations[0], 6*3)
        self.assertEqual(off_durations[1], 6*1)

        # Now check that we do the correct thing when there are gaps
        pwr2 = [0, 0, 0, 0]
        rng2 = pd.date_range(rng[-1]+100, periods=len(pwr2), freq='6S')
        series2 = pd.Series(pwr + pwr2, index = rng + rng2)
        c2 = Channel(series=series2)

        # Check on-durations
        on_durations = c2.durations(on_or_off='on')
        self.assertEqual(len(on_durations), 3)
        self.assertEqual(on_durations[0], 6*4)
        self.assertEqual(on_durations[1], 6*2)
        self.assertEqual(on_durations[2], c2.max_sample_period+c2.sample_period)

        # Check off-durations
        off_durations = c2.durations(on_or_off='off')
        self.assertEqual(len(off_durations), 2)
        self.assertEqual(off_durations[0], 6*3)
        self.assertEqual(off_durations[1], 6*1)

        # Check merging events
        on_durations = c2.durations('on', 1)
        self.assertEqual(len(on_durations), 2)
        self.assertEqual(on_durations[0], 6*4)
        self.assertEqual(on_durations[1], 42)

    def test_diff_ignoring_long_outages(self):
        index = [np.datetime64(s, 's') for s in [0,1,2,3,10,11,12]]
        series = pd.Series([0,100,150,100,1000,800,0], index=index)
        c = Channel(series=series)
        diff = c.diff_ignoring_long_outages()
        np.testing.assert_array_equal(diff.values, [100, 50, -50, -200, -800])

if __name__ == '__main__':
    unittest.main()
