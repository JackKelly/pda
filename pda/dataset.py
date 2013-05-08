from pda.channel import Channel
from os.path import join


def load_labels(filename):
    """
    Loads data from labels.dat file.

    Args:
        filename (str): labels filename, including full path.

    Returns:
        A dict mapping channel numbers (ints) to appliance names (str)
    """
    with open(filename) as labels_file:
        lines = labels_file.readlines()
    
    labels = {}
    for line in lines:
        line = line.partition(' ')
        labels[int(line[0])] = line[2].strip() # TODO add error handling if line[0] not an int
        
    print("Loaded {} lines from labels.dat".format(len(labels)))
    return labels


def load_dataset(data_dir):
    """Loads an entire dataset directory.

    Args:
        data_dir (str)

    Returns:
        list of Channels
    """

    channels = []
    labels = load_labels(join(data_dir, 'labels.dat'))
    for chan, label in labels.iteritems():
        chan_filename = join(data_dir, 'channel_{:d}.dat'.format(chan))
        channels.append(Channel(chan_filename, name=label))

    return channels
