from __future__ import print_function, division
from pda.channel import Channel, load_labels
import sys
import numpy as np
from scipy.spatial import distance
from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt

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

Notes:
This is ML so need to make sure we're not overfitting.  Split dataset.

Using K-Means: Unfortunately there is no general theoretical
solution to find the optimal number of clusters for any given data
set. A simple approach is to compare the results of multiple runs with
different k classes and choose the best one according to a given
criterion (for instance the Schwarz Criterion - see Moore's slides[1]),
but we need to be careful because increasing k results in smaller
error function values by definition, but also an increasing risk of
overfitting.)


[1]: http://home.deib.polimi.it/matteucc/Clustering/tutorial_html/kmeans.html#moore

http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#example-cluster-plot-dbscan-py
    """

    merged_events = pd.Series()
    for c in dataset:
        c = c.crop('2013-05-01', '2013-05-02')
        events = c.on_off_events()
        events = events[events == 1]
        merged_events = merged_events.append(events)

    merged_events.sort_index()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(merged_events.index, merged_events.values, 'o')
    plt.show()

    x = merged_events.index.astype(int)
    x2d = np.zeros((len(x), 2))
    x2d[:,0] = x

    D = distance.squareform(distance.pdist(x2d))
    S = 1 - (D / np.max(D)) # similarities

    db = DBSCAN(eps=0.95, min_samples=1).fit(S)
    core_samples = db.core_sample_indices_
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

#    import ipdb; ipdb.set_trace()

    print('Estimated number of clusters: {:d}'.format(n_clusters_))
    # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels)
    # print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels)
    # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels)
    # print("Adjusted Rand Index: %0.3f" % \
    #     metrics.adjusted_rand_score(labels_true, labels)
    # print("Adjusted Mutual Information: %0.3f" % \
    #     metrics.adjusted_mutual_info_score(labels_true, labels)
    # print("Silhouette Coefficient: %0.3f" %
    #        metrics.silhouette_score(D, labels, metric='precomputed'))
