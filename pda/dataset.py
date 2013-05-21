from __future__ import print_function, division
from pda.channel import Channel, load_labels
import sys
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt
from itertools import cycle

def load_dataset(data_dir):
    """Loads an entire dataset directory.

    Args:
        data_dir (str)

    Returns:
        list of Channels
    """

    channels = []
    labels = load_labels(data_dir)
    print("Found", len(labels), "entries in labels.dat")
    for chan, label in labels.iteritems():
        print("Attempting to load chan", chan, label, "...", end=" ")
        sys.stdout.flush()
        try:
            channels.append(Channel(data_dir, chan))
        except IOError:
            print("FAILED!")
        else:
            print("success.")

    return channels


def cluster_appliances(dataset):
    """Args:
           dataset (list of pda.channel.Channels)
    
    Relevant docs:
    http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html
    """

    merged_events = pd.Series()
    for c in dataset:
        c = c.crop('2013-05-01', '2013-05-02')
        events = c.on_off_events()
        events = events[events == 1]
        merged_events = merged_events.append(events)

    merged_events.sort_index()

    # distance.pdist() requires a 2D array so convert
    # datetimeIndex to a 2D array
    x = merged_events.index.astype(int) / 1E9
    x2d = np.zeros((len(x), 2))
    x2d[:,0] = x

    # Calculate square distance vector
    D = distance.squareform(distance.pdist(x2d))

    # Run cluster algorithm
    # eps is the maximum distance between samples.  In our case,
    # it is in units of seconds.
    db = DBSCAN(eps=60*60, min_samples=2, metric="precomputed").fit(D)
    core_samples = db.core_sample_indices_
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print('Estimated number of clusters: {:d}'.format(n_clusters_))
    print("Silhouette Coefficient: {:0.3f}".format(
           metrics.silhouette_score(D, labels, metric='precomputed')))

    # PLOT
    fig = plt.figure()
    ax = fig.add_subplot(111)

    colors = cycle('bgrcmybgrcmybgrcmybgrcmy')
    for k, col in zip(set(labels), colors):
        if k == -1:
            # Black used for noise.
            col = 'k'
            markersize = 6
        class_members = [index[0] for index in np.argwhere(labels == k)]
        cluster_core_samples = [index for index in core_samples
                                if labels[index] == k]
        for index in class_members:
            plot_x = merged_events.index[index]
            if index in core_samples and k != -1:
                markersize = 14
            else:
                markersize = 6
            ax.plot(plot_x, 1, 'o', markerfacecolor=col,
                    markeredgecolor='k', markersize=markersize)

    plt.show()

#    import ipdb; ipdb.set_trace()
