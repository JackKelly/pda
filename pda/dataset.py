from pda.channel import Channel, load_labels

def load_dataset(data_dir):
    """Loads an entire dataset directory.

    Args:
        data_dir (str)

    Returns:
        list of Channels
    """

    channels = []
    labels = load_labels(data_dir)
    for chan, label in labels.iteritems():
        try:
            channels.append(Channel(data_dir, chan))
        except IOError:
            pass
        else:
            print("loaded chan {} {}".format(chan, label))

    return channels

