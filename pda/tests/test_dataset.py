#!/usr/bin/python
from __future__ import print_function, division
import unittest, inspect, os
import pda.dataset as ds
import correct_answers

# Taken from http://stackoverflow.com/a/6098238/732596
FILE_PATH = os.path.dirname(inspect.getfile(inspect.currentframe()))
SMALL_TEST_DATA_PATH = os.path.join(FILE_PATH, 'test_data', 'small')
LARGE_TEST_DATA_PATH = os.path.join(FILE_PATH, 'test_data', 'large')

class TestDataset(unittest.TestCase):

    def test_load_dataset(self):
        dataset = ds.load_dataset(SMALL_TEST_DATA_PATH)
        i = 1
        for channel in dataset:
            self.assertEqual(channel.name, correct_answers.labels[i])
            i += 1

if __name__ == '__main__':
    unittest.main()
