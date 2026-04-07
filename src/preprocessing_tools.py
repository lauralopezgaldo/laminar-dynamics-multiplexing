"""
This file contains functions linked to data preprocessing and manipulation.
"""

import re


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
