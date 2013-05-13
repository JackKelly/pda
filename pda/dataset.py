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

