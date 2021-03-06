from __future__ import print_function, division
import load_redd
import time

class TimeIt(object):
    def __init__(self):
        self.times = []

    def time(self, label=''):
        self.times.append((label, time.time()))

    def __str__(self):
        s = ''
        if len(self.times) < 2:
            return s

        prev_t = self.times[0][1]
        for label, t in self.times[1:]:
            s += '{:>30s} = {:.3f}s\n'.format(label, t - prev_t)
            prev_t = t
        s += '{:>30s} = {:.3f}s\n'.format('TOTAL', 
                                          self.times[-1][1] - self.times[0][1])
        return s

t = TimeIt()

t.time()
d = load_redd.load('/data/mine/vadeec/jack-merged/channel_3.dat')
t.time('load redd')

print(t)
print(d)
