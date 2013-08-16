from __future__ import print_function, division
from pda.channel import Channel, load_labels, DD
import sys
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from itertools import cycle

"""
Functions for loading an entire data directory into a list of
Channels and then manipulating those datasets.

I'm using the term "dataset" to mean a list of Channels.
"""

def load_dataset(data_dir=DD, ignore_chans=None, only_load_chans=None,
                 start_date=None, end_date=None):
    """Loads an entire dataset directory.

    Args:
        data_dir (str)
        ignore_chans (list of ints or label strings): optional.  
            Don't load these channels.
        only_load_chans (list of ints or label strings): optional.

    Returns:
        list of Channels
    """

    if ignore_chans is not None:
        assert(isinstance(ignore_chans, list))

    channels = []
    labels = load_labels(data_dir)
    print("Found", len(labels), "entries in labels.dat")
    for chan, label in labels.iteritems():
        if ignore_chans is not None:
            if chan in ignore_chans or label in ignore_chans:
                print("Ignoring chan", chan, label)
                continue

        if only_load_chans is not None:
            if chan not in only_load_chans and label not in only_load_chans:
                print("Ignoring chan", chan, label)
                continue

        print("Attempting to load chan", chan, label, "...", end=" ")
        sys.stdout.flush()
        try:
            c= Channel(data_dir, chan, start_date=start_date, end_date=end_date)
        except IOError:
            print("FAILED!")
        else:
            channels.append(c)
            print("success.")

    return channels


def dataset_to_dataframe(dataset):
    d = {}
    for ds in dataset:
        d[ds.name] = ds.series
    return pd.DataFrame(d)


def crop_dataset(dataset, start_date, end_date):
    cropped_dataset = []
    for i in range(len(dataset)):
        c = dataset[i].crop(start_date, end_date)
        cropped_dataset.append(c)
    return cropped_dataset


def plot_each_channel_activity(ax, dataset, add_colorbar=False):
    df = dataset_to_dataframe(dataset)
    df = df.resample('10S', how='max')
    img = df.values
    img[np.isnan(img)] = 0

    # Convert each channel's power consumption in watts
    # to a value between 0 and 1 for imshow.  'Autoscale'
    # each channel such that 1 corresponds to the maximum power
    # for that appliance.  Can't just use the max power for the
    # channel because some appliances briefly use much higher powers
    # than the vast majority of the time.
    for i in range(img.shape[1]):
        maximum = np.percentile(img[:,i], 99.9)
        if maximum > 3000:
            maximum = 3000
        img[:,i] = img[:,i] / maximum
        img[:,i][img[:,i] > 1] = 1

    img[np.isnan(img)] = 0
    img = np.transpose(img)
    im = ax.imshow(img, aspect='auto', interpolation='none', origin='lower',
                   extent=(mdates.date2num(df.index[0]),
                           mdates.date2num(df.index[-1]), 
                           0, df.columns.size))
    if add_colorbar:
        plt.colorbar(im)

    ax.set_yticks(np.arange(0.5, len(df.columns)+0.5))

    def formatter(x, pos):
        x = int(x)
        return df.columns[x]

    ax.yaxis.set_major_formatter(FuncFormatter(formatter))
    for item in ax.get_yticklabels():
        item.set_fontsize(6)
    ax.set_title('Appliance ground truth')
    return ax


def init_aggregate_and_appliance_dataset_figure(
        start_date, end_date, n_subplots=2, 
        aggregate_type='one second', plot_both_aggregate_signals=False, 
        data_dir=DD, plot_appliance_ground_truth=True, ignore_chans=None,
        **kwargs):
    """Initialise a basic figure with multiple subplots.  Plot aggregate
    data.  Optionally plot appliance ground truth dataset.

    Args:
        start_date, end_date (str): Required.  e.g. '2013/6/4 18:00'
        n_subplots (int): Required.  Must be >= 1.  Includes aggregate and 
            appliance ground truth plots.
        aggregate_type (str): 'one second' or 'current cost'.  The flavour of 
            aggregate data to load, plot and return.
        plot_both_aggregate_signals (bool): Default==False. Plot both flavours
            of aggregate data?  Has no effect on which flavour is returned.
        data_dir (str): Default=DD
        plot_appliance_ground_truth (bool): Default==True
        ignore_chans (list of strings or ints): Defaults to a standard list of 
            channels to ignore.
        **kwargs: passed to ax.plot
    Returns:
        subplots (list of axes), 
        chan (pda.Channel)

    """
    if plot_appliance_ground_truth:
        assert(n_subplots >= 2)
    else:
        assert(n_subplots >= 1)

    # Initialise figure and subplots
    fig = plt.figure()
    fig.canvas.set_window_title(start_date + ' - ' + end_date)
    subplots = [fig.add_subplot(n_subplots, 1, 1)]
    for i in range(2, n_subplots+1):
        subplots.append(fig.add_subplot(n_subplots, 1, i, sharex=subplots[0]))

    # Load and plot aggregate channel(s)
    if aggregate_type=='one second' or plot_both_aggregate_signals:
        print('Loading high freq mains...')
        one_sec = Channel()
        one_sec.load_normalised(data_dir, high_freq_param='active', 
                                start_date=start_date, end_date=end_date)
        one_sec.plot(subplots[0], color='k', **kwargs)

    if aggregate_type=='current cost' or plot_both_aggregate_signals:
        print('Loading Current Cost aggregate...')
        cc = Channel(data_dir, 'aggregate', 
                     start_date=start_date, end_date=end_date) # cc = Current cost
        cc.plot(subplots[0], color='r', **kwargs)

    subplots[0].set_title('Aggregate. 1s active power, normalised.')
    subplots[0].legend()
    chan = one_sec if aggregate_type=='one second' else cc

    if plot_appliance_ground_truth:
        print('Loading appliance ground truth dataset...')
        if ignore_chans is None:
            ignore_chans=['aggregate', 'amp_livingroom', 'adsl_router',
                          'livingroom_s_lamp', 'gigE_&_USBhub',
                          'livingroom_s_lamp2', 'iPad_charger', 
                          'subwoofer_livingroom', 'livingroom_lamp_tv',
                          'DAB_radio_livingroom', 'kitchen_lamp2',
                          'kitchen_phone&stereo', 'utilityrm_lamp', 
                          'samsung_charger', 'kitchen_radio', 
                          'bedroom_chargers', 'data_logger_pc', 
                          'childs_table_lamp', 'baby_monitor_tx',
                          'battery_charger', 'office_lamp1', 'office_lamp2',
                          'office_lamp3', 'gigE_switch']
        ds = load_dataset(data_dir, ignore_chans=ignore_chans, 
                          start_date=start_date, end_date=end_date)
        print("Removing inactive channels...")
        ds = remove_inactive_channels(ds)
        print("Plotting dataset ground truth...")
        plot_each_channel_activity(subplots[1], ds)

    return subplots, chan


def remove_inactive_channels(ds):
    filtered_ds = []
    for c in ds:
        if (c.series is not None and c.series.size > 0 and 
            c.series.values.max() >= c.on_power_threshold):
            filtered_ds.append(c)
    return filtered_ds


def cluster_appliances_period(dataset, period, ignore_chans=[], plot=False):
    """
    Args:
       dataset (list of pda.channel.Channels)
       period (pd.Period)
       ignore_chans (list of ints)

    Returns:
       list of sets of ints.  Each set stores the channel.chan (int)
           of each channel in that set.
    
    Relevant docs:
    http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html
    """

    merged_events = pd.Series()
    for c in dataset:
        if c.chan in ignore_chans:
            continue
        cropped_c = c.crop(period.start_time, period.end_time)
        events = cropped_c.on_off_events()
        events = events[events == 1] # select turn-on events
        events[:] = c.chan # so we can decipher which chan IDs are in each cluster
        merged_events = merged_events.append(events)

    merged_events = merged_events.sort_index()

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
    db = DBSCAN(eps=60*10, min_samples=2, metric="precomputed").fit(D)
    core_samples = db.core_sample_indices_
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print('Estimated number of clusters: {:d}'.format(n_clusters_))

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    colors = cycle('bgrcmybgrcmybgrcmybgrcmy')
    chans_in_each_cluster = []
    for k, col in zip(set(labels), colors):
        if k == -1:
            # Black used for noise.
            col = 'k'
            markersize = 6
        class_members = [index[0] for index in np.argwhere(labels == k)]

        if k != -1:
            chans_in_each_cluster.append(set(merged_events.ix[class_members]))

        if plot:
            for index in class_members:
                plot_x = merged_events.index[index]
                if index in core_samples and k != -1:
                    markersize = 14
                else:
                    markersize = 6

                ax.plot(plot_x, merged_events.ix[index], 'o', markerfacecolor=col,
                        markeredgecolor='k', markersize=markersize)

    if plot:
        plt.show()
        ylim = ax.get_ylim()
        ax.set_ylim( [ylim[0]-1, ylim[1]+1] )
        ax.set_title(str(period))
        ax.set_xlabel('time')
        ax.set_ylabel('channel number')
    return chans_in_each_cluster


def cluster_appliances(dataset, ignore_chans=[], period_range=None):
    if period_range is None:
        period_range = pd.period_range(dataset[0].series.index[0], 
                                       dataset[0].series.index[-1], freq='D')
    
    chans_in_each_cluster = []
    for period in period_range:
        print(period)
        chans_in_each_cluster.extend(cluster_appliances_period(dataset, period, ignore_chans))

    freqs = []
    # now find frequently occurring sets
    for s in chans_in_each_cluster:
        freqs.append((s, chans_in_each_cluster.count(s)))

    # Sort by count; highest count first
    freqs.sort(key=lambda x: x[1], reverse=True)
    return freqs


