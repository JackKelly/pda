import pda.dataset as ds
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DATASET_DIR = '/data/mine/vadeec/jack-merged'
TOP_N = 20 # how many clusters to display?

dataset = ds.load_dataset(DATASET_DIR)

period_range = pd.period_range(dataset[0].series.index[0], 
                              dataset[0].series.index[-1], freq='D')

freqs = ds.cluster_appliances(dataset, 
                              ignore_chans=[1, 2, 3, 5, 12, 16, 18, 25, 32, 38, 43],
                              period_range=period_range[5:])

# remove "clusters" of a single appliance and remove repeats
filtered_freqs = []
for f in freqs:
    if len(f[0]) > 1:
        if f not in filtered_freqs:
            filtered_freqs.append(f)

# trim
filtered_freqs = filtered_freqs[:TOP_N]

# extract set counts
counts = [f[1] for f in filtered_freqs]

# plotting
fig = plt.figure()
ax = fig.add_subplot(111)
pos = np.arange(TOP_N, 0, -1) + 0.5 # the bar centres on the y axis
ax.barh(pos, counts, align='center', color='grey')

# labels
chan_labels = ds.load_labels(DATASET_DIR)
plot_labels = []
for s, count in filtered_freqs:
    freq_label_list = [chan_labels[chan] for chan in s]
    freq_label = ' '
    freq_label += ', '.join(freq_label_list)
    plot_labels.append(freq_label)

for i in range(TOP_N):
    ax.text(0, pos[i], plot_labels[i], va='center')

ax.set_yticks([])
# ax.set_yticks(pos)
# ax.set_yticklabels(plot_labels)
ax.set_ylim([0.5, TOP_N+1.5])
# xlim = ax.get_xlim()
# ax.set_xticks(np.arange(xlim[1]+1))
ax.set_xlabel('Number of times set of appliances turn on within 10 minutes of each other')
ax.set_title('Sets of appliances which turn on within 10 minutes of each other')
fig.tight_layout()
plt.show()

"""
TODO:

New design:

------------------------+-----------------------------\
kitchen lights, kettle, : kitchen lights, microwave   |
                        :--------------------------+--/
toaster, microwave      : kitchen lights, toaster  |
------------------------+--------------------------/

"""
