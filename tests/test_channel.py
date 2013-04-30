from __future__ import print_function, division
import unittest, os, inspect
from pda.channel import Channel
import matplotlib.pyplot as plt

# Taken from http://stackoverflow.com/a/6098238/732596
FILE_PATH = os.path.dirname(inspect.getfile(inspect.currentframe()))
TEST_DATA_PATH = os.path.join(FILE_PATH, 'test_data')

class TestChannel(unittest.TestCase):
    def setUp(self):
        self.channel = Channel(os.path.join(TEST_DATA_PATH, 'channel_1.dat'))
        # self.channel = Channel('/data/mine/vadeec/jack-merged/channel_1.dat')
        
    def test_init(self):
        self.assertIsNotNone(self.channel)
         
    def test_plot(self):
        print(self.channel.series.tail())
        print(self.channel.series.index)
        self.channel.series.plot()
        plt.show()

if __name__ == "__main__":
    unittest.main()
