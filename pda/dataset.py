from __future__ import print_function, division
from pda.channel import Channel, load_labels
import sys

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


def cluster_appliances(df):
    """Args:
       df (pd.DataFrame)

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
    """

    
