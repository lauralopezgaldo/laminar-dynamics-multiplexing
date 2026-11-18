"""
This file contains functions linked to data preprocessing and manipulation.
"""

import re
import numpy as np
from scipy.io import loadmat
import mat73


def cut_mua_by_markers(mua_xr, events_alignment, t_extra):
    """
    Cut the MUA data by the markers of the task.
    :param mua_xr: DataArray with the MUA data
    :param events_alignment: list with the names of the events to cut the data
    :param t_extra: list with the extra time to add to the events in seconds
    ----
    :return: mua_shorter: DataArray with the MUA data cut by the events
             n_times_shorter: int with the number of time-points in the new MUA
    """
    # get the events
    events_onset = mua_xr.task_events_onset
    events_names = re.split('-', mua_xr.task_events_labels)

    # get the start and end of the trial
    t_start = events_onset[events_names.index(events_alignment[0])] + t_extra[0]  # first event
    t_end = events_onset[events_names.index(events_alignment[1])] + t_extra[1]  # second event

    # cut the mua in the t_start t_mvt
    mua_shorter = mua_xr.sel(times=slice(t_start, t_end))
    n_times_shorter = len(mua_shorter.times)

    return mua_shorter, n_times_shorter


def open_matlab_behaviour(filename):
    """
    Open a matlab behaviour structure from OrganizeBehaviour and save it in a python dictionary. One
    SESSION corresponds to one behavioural file - independently of the number of probes.

    :param filename: name of the matlab structure containing the behavioural information
    ----
    :return: behaviour: dictionary containing the same fields as the behaviour structure from matlab
    """

    # import the matlab file
    mat_behaviour = loadmat(filename)

    # get the sub-structure name
    data_str_name = list(mat_behaviour.keys())[-1]

    # get the fields of the structure and use them as keys for the new dictionary
    fields = []
    for key in mat_behaviour[data_str_name].dtype.fields.keys():
        fields.append(key)

    data = mat_behaviour[data_str_name][0][0]  # get the data inside the structure
    behaviour = {field: data[i_field][0] for i_field, field in enumerate(fields)}

    return behaviour


def concatenate_probes(filename_probe1, filename_probe2, loc_1, loc_2):
    """
    Concatenate LFPs coming from different probes but recorded in the same SESSION.

    :param filename_probe1: name of the file containing LFP in probe 1
    :param filename_probe2: name of the file containing LFP in probe 2
    :param loc_1: location of probe 1
    :param loc_2: location of probe 2
    ----
    :return: channel_names: list including the names of the channels 'area-ch_number'
    :return: concat_lfp: concatenated LFP array with (n_channels both probes, n_times)
    """

    # get the LFPs from mat files
    probe1 = open_matlab_lfp(filename_probe1)
    probe2 = open_matlab_lfp(filename_probe2)
    concat_lfp = np.concatenate((probe1, probe2))  # concatenate in the channel dimension

    # keep track of the number of channels in each probe and label them
    channels_probe1 = [loc_1 + '-' + str(ch) for ch in range(1, probe1.shape[0] + 1)]
    channels_probe2 = [loc_2 + '-' + str(ch) for ch in range(1, probe2.shape[0] + 1)]

    channel_names = np.append(channels_probe1, channels_probe2)  # join together all channel names

    return channel_names, concat_lfp


def open_matlab_lfp(filename):
    """
    Open matlab files containing the output of SortChanLFP in one SESSION and PROBE.

    :param filename: name of the file containing LFP
    ----
    :return: array_lfp: numpy array with LFP in shape (n_channels, n_times)
    """

    # import the matlab file
    mat_lfp = loadmat(filename)
    data_str_name = list(mat_lfp.keys())[-1]  # sub-structure name

    # store the LFPs in an array
    array_lfp = mat_lfp[data_str_name]  # array containing channels x time-points

    return array_lfp


def open_matlab_analog(filename):
    """
    Open a matlab structure containing eye and hand information. Store it as a python dictionary.

    :param filename: name of the matlab structure containing the analog signals
    ----
    :return: analog_dictionary: python dictionary containing the structure fields
    """
    # import the matlab structure
    mat_structure = mat73.loadmat(filename)

    # get the sub-structure name
    data_str_name = list(mat_structure.keys())[-1]

    # access the structure to get the dictionary
    analog_dictionary = mat_structure[data_str_name]

    return analog_dictionary


def cut_in_epochs(lfp, t_markers=None, t_min=500, t_start=None, t_end=None):
    """
    Cut the continuous LFP filtered signal into epochs. By default, one epoch corresponds to one
    full trial, aligned in the SEL cue appearing. For better precision, we should use 'TimLED'
    times, stored in the behaviour dictionary. It is also possible to align in times from the
    analog behaviour dictionary, e.g. hand movement onset.

    :param lfp: continuous filtered LFP signal, array with (n_channels, n_times)
    :param t_markers: list of time markers determining the start of the event to align to.
    :param t_min: time to include before the start of the trial. Default 500ms before touch.
    :param t_start: time marker setting the start of the epoch [int]. If just a window inside the
    trial, this is the half-window size.
    :param t_end: time marker setting the end of the epoch [int]. If just a window inside the trial,
    this is the half-window size.
    ----
    :return: lfp_epochs: a 3D array with the filtered lfp cut in epochs and all trials with the same
    number of time-points (n_trials, n_channels, n_times)
    """
    # start an empty list
    lfp_epochs = []

    for t_marker in t_markers:
        # start t_min (500ms) before the t_start (touch) and aligned to t_marker (SEL) - default!
        t_1 = t_marker + t_start - t_min
        # finish at t_end (target) - default!
        t_2 = t_marker + t_end
        # append into a list
        lfp_epochs.append(lfp[:, t_1: t_2])

    lfp_epochs = np.array(lfp_epochs)  # convert list into np array

    return lfp_epochs


def get_depth_from_layer(layer_mask, electrode_spacing):
    """
    Get the depth of each contact based on the layer mask. The first contact before the cortex will
    be depth 0 and then it will grow in steps of -s.
    :param layer_mask: array with layer names and -1 for contacts not belonging to the cortex. It is
    of size (n_channels,)
    :param electrode_spacing: electrode spacing [mm]
    ----
    :return: contact_depth: array with the depth of every contact, considering th electrode spacing
    and the 0 in the WM, i.e. last contact which is not in the cortex.
    """
    n_channels = len(layer_mask)  # get the number of channels
    contact_depth = [1] * n_channels  # initialize the array with the depths

    # check if there is any contact in the dura
    if '-1' in layer_mask:
        zero_idx = np.where(layer_mask == '-1')[0].max()  # check last contact on the dura mattter
        contact_depth[zero_idx] = 0

        # start the depths from the dura matter, i.e. DM = 0 and then grow in steps of -s
        depths = np.arange(len(contact_depth[zero_idx::])) * electrode_spacing
        contact_depth[zero_idx::] = -depths
    else:
        # all contacts are in the cortex
        depths = np.arange(len(contact_depth)) * electrode_spacing
        contact_depth = -depths

    return np.array(contact_depth)


def block_from_trial(trial_types):
    """
    Label the blocks of the trials in each session. Every time there's a change in trial type,
    there's a change in block. This function generates an array with the number of the block to
    which each trial belongs to.
    :param trial_types: list or array including the trial type for all trials in one session.
    ----
    :return: block: array of the same length as trial_types which represents the block instead of
    the trial type.

    """
    # define for each trial in which block it is
    idx_ttype_change = np.where(np.diff(trial_types))[0] + 1  # get when the trial type changes
    idx_ttype_change = np.insert(idx_ttype_change, 0, 0)  # add 0
    idx_ttype_change = np.append(idx_ttype_change, len(trial_types))  # add length of trial type

    # get the amount of trials in each block
    trial_per_block = np.diff(idx_ttype_change)

    block = []
    for value, idx in enumerate(trial_per_block):
        block += [value + 1] * idx  # start blocks at 1
    block = np.array(block).T

    return block


def get_layer_mask(df_layers, n_channels):
    """
    Get the layer mask for each probe using the information in the dataframe. Return a list with
    equal length to the n_channels which contains '-1' if the channel is not in the cortex or the
    string with the layer name otherwise.
    :param df_layers: dataframe entry with laminar information. String containing the layer names,
    separated by commas and the channel number of start and end of each layer.
    :param n_channels:  number of channels of that probe.
    ----
    :return: layers_mask: array with layer names of the same length as n_channels
    """
    layers_loc = re.split(', ', df_layers)  # split individual layers
    layers_mask = ['-1'] * n_channels  # create a list with all -1 of the shape of the output

    for layer in layers_loc:
        layer_name, layer_loc = re.split(':', layer)
        min_ch, max_ch = re.split('-', layer_loc)
        layers_mask[int(min_ch) - 1:int(max_ch)] = [layer_name] * (int(max_ch) - int(min_ch) + 1)

    return np.array(layers_mask)
