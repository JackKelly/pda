#!/usr/bin/python
from __future__ import print_function, division
import unittest, inspect, os
from os.path import join
from pda.channel import Channel, SECS_PER_HOUR
import pda.dataset as ds

# Taken from http://stackoverflow.com/a/6098238/732596
FILE_PATH = os.path.dirname(inspect.getfile(inspect.currentframe()))
SMALL_TEST_DATA_PATH = os.path.join(FILE_PATH, 'test_data', 'small')
LARGE_TEST_DATA_PATH = os.path.join(FILE_PATH, 'test_data', 'large')

correct_labels = {1: 'aggregate', 2: 'boiler', 3: 'solar',
                  4: 'laptop', 5: 'washing_machine', 
                  6: 'dishwasher', 7: 'tv', 8: 'kitchen_lights',
                  9: 'htpc', 10: 'kettle', 11: 'toaster', 
                  12: 'fridge', 13: 'microwave', 14: 'lcd_office',
                  15: 'hifi_office', 16: 'breadmaker', 
                  17: 'amp_livingroom', 18: 'adsl_router', 
                  19: 'coffee'}


class TestChannel(unittest.TestCase):
         
    def test_load_labels(self):
        labels = ds.load_labels(join(SMALL_TEST_DATA_PATH, 'labels.dat'))
        self.assertEqual(labels, correct_labels)

    def test_load_dataset(self):
        dataset = ds.load_dataset(SMALL_TEST_DATA_PATH)
        i = 1
        for channel in dataset:
            self.assertEqual(channel.name, correct_labels[i])
            i += 1

if __name__ == '__main__':
    unittest.main()
