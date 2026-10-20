"""
This script takes the recorded hand and eye behavior of each monkey and plots a summary of it.
"""
# import packages
import os
import mne
import h5py
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from frites import io
from matplotlib.colors import to_rgb


def plot_eye_behavior(monkey_name, session_list):
    """
    Plot the eye behavior for single trials and across trials as a proportion
    :param monkey_name: name of the monkey to plot
    :param session_list: all sites existing
    :return:
    """
    current_path = os.getcwd()

    # Set the paths
    if current_path.startswith('C:'):
        server = 'W:'  # local w VPN
    else:
        server = '/envau/work'  # niolon

    # path for saving figures
    path_figures = server + '/comco/lopez.l/Electrophysiology/' \
                            'ephy_laminar_MUA/Results/Full_trial/plots_revisions/behavior/'

    # 1. Initialize the list to store all behavioral sessions
    concat_behavior = []

    # 2. Iterate on all sessions and open the relevant ones
    for session in session_list:
        try:
            # path of the data
            path_data = server + '/comco/lopez.l/Electrophysiology/' \
                                 'ephy_laminar_MUA/Results/Preprocessed_data/' + session + '/'

            # Load only the sessions corresponding to the desired monkey
            if session.startswith(monkey_name):
                # load the data and annotations
                eye_behavior = xr.load_dataarray(
                    os.path.join(path_data, f'{session}-eye_behavior.nc'))

                if eye_behavior.sum() == 0:
                    print(f'{session} ALL ZEROS')
                else:
                    concat_behavior.append(eye_behavior)
                    io.logger.info(f"Concatenated behavior for {session}")

        except Exception as e:
            io.logger.error(f"No behavior for site {session}: {e}")
            continue

    # 3. Group the results in an individual x-array
    grouped_sessions = xr.concat(concat_behavior, dim='trials')

    # 4. Organize the behavior based on trial type
    # sort them by trial type
    sorted_sessions = grouped_sessions.sortby('trial_type')
    # number of trials per group
    trial_id, n_trials_group = np.unique(grouped_sessions.trial_type,
                                         return_counts=True)

    # 5. Make each binary entry a color-code rather than 1/0
    # One color per trial type
    trial_colors = ['tab:blue', 'tab:green', 'pink']

    # Build an RGB image so we can plot
    rgb = np.ones((*sorted_sessions.shape, 3))  # background = white

    start = 0
    for color, count in zip(trial_colors, n_trials_group):
        end = start + count
        color_rgb = np.array(to_rgb(color))

        # Color samples where the signal is 1
        block = sorted_sessions.values[start:end]
        rgb[start:end][block == 1] = color_rgb

        # Display NaNs as black
        rgb[start:end][np.isnan(block)] = to_rgb('k')
        start = end

    # 6. Create the plot - both the average and the single trials (vertically stack)
    # Create two vertically stacked subplots
    fig, (ax_avg, ax_raster) = plt.subplots(
        2, 1,
        figsize=(12, 10),
        sharex='col',
        gridspec_kw={'height_ratios': [1, 5], 'hspace': 0.05}
    )

    # 6.1. Group by trial-type to get the averages
    grouped_mean = grouped_sessions.groupby('trial_type').mean()

    for i, trial_type in enumerate(grouped_mean.trial_type.values):
        ax_avg.plot(
            grouped_sessions.times,
            grouped_mean.sel(trial_type=trial_type),
            color=trial_colors[i],
            label=int(trial_type)
        )

    ax_avg.set_ylim(0, 1)
    ax_avg.legend(frameon=False)

    # 6.2. Display the single trials as a colormap
    ax_raster.imshow(rgb,
                     aspect='auto',
                     interpolation='nearest',
                     extent=[
                         sorted_sessions.times.values[0],
                         sorted_sessions.times.values[-1],
                         sorted_sessions.values.shape[0], 0
                     ]
                     )

    # 6.3. Include the temporal markers
    marker_times = [0, 1.3, 2.6, 3.9, 5.2]
    marker_times_lines = [0, 0.3, 1.3, 1.6, 2.6, 2.9, 3.9, 4.2, 5.2]
    marker_labels = ['Sel', 'SC1', 'SC2', 'SC3', 'Go']

    [ax_raster.axvline(line, lw=2, color='firebrick') for line in marker_times_lines]
    ax_raster.set_xticks(marker_times, marker_labels)

    # 6.4. Finalize the figure and save it
    sns.despine(fig)
    fig.suptitle(f'Monkey {monkey_name}', y=0.9)

    # save the figure
    fig.savefig(os.path.join(path_figures, f'Grouped_sessions_monkey_{monkey_name}_eye.svg'))

    return fig


def plot_eye_position(monkey_name, session_list):
    """
    Plot the eye position, for each trial type, when the eye is inside the working area for
    unambiguous trials, separating eye on center and eye on target.
    :param monkey_name: name of the monkey to plot
    :param session_list: all sites existing
    :return:
    """
    # Set the paths
    current_path = os.getcwd()
    if current_path.startswith('C:'):
        server = 'W:'  # local w VPN
    else:
        server = '/envau/work'  # niolon

    # path for saving figures
    path_figures = server + '/comco/lopez.l/Electrophysiology/' \
                            'ephy_laminar_MUA/Results/Full_trial/plots_revisions/behavior/'

    # 1. Initialize the list to store all behavioral sessions
    concat_behavior = []

    # 2. Iterate on all sessions and open the relevant ones
    for session in session_list:
        try:
            # path of the data
            path_data = server + '/comco/lopez.l/Electrophysiology/' \
                                 'ephy_laminar_MUA/Results/Preprocessed_data/' + session + '/'

            # Load only the sessions corresponding to the desired monkey
            if session.startswith(monkey_name):
                # load the data and annotations
                eye_behavior = xr.load_dataarray(os.path.join(path_data,
                                                              f'{session}-eye_behavior.nc'))
                eye_on_target = xr.load_dataarray(
                    os.path.join(path_data, f'{session}-eye_on_target.nc'))
                eye_on_center = xr.load_dataarray(
                    os.path.join(path_data, f'{session}-eye_on_center.nc'))

                if eye_behavior.sum() == 0:
                    print(f'{session} ALL ZEROS')
                else:
                    # create one x-array with all the features of the eye
                    eye_features = xr.concat(
                        [eye_behavior, eye_on_center, eye_on_target],
                        dim=pd.Index(
                            ['eye_behavior', 'eye_on_center', 'eye_on_target'],
                            name='eye_feature'
                        )
                    )
                    concat_behavior.append(eye_features)
                    io.logger.info(f"Concatenated behavior for {session}")

        except Exception as e:
            io.logger.error(f"No behavior for site {session}: {e}")
            continue

    # 3. Group the results in an individual x-array
    grouped_sessions = xr.concat(concat_behavior, dim='trials')

    # 5. Make each binary entry a color-code rather than 1/0
    # One color per trial type
    trial_colors = ['tab:blue', 'tab:green', 'pink']

    # 6. Create the plot - both the average and the single trials (vertically stack)
    # Create three vertically stacked subplots
    fig, ax = plt.subplots(
        3, 1,
        figsize=(12, 10),
        sharex='col',
        gridspec_kw={'hspace': 0.05})

    # 6.1. Group by trial-type to get the averages
    grouped_mean = grouped_sessions.groupby('trial_type').mean()

    for i, trial_type in enumerate(grouped_mean.trial_type.values):
        ax[i].plot(
            grouped_sessions.times,
            grouped_mean.sel(trial_type=trial_type, eye_feature='eye_behavior'),
            color=trial_colors[i],
            label='gaze inside WA'
        )
        ax[i].fill_between(
            grouped_sessions.times,
            0, grouped_mean.sel(trial_type=trial_type, eye_feature='eye_on_center'),
            color='slategray', alpha=.7,
            label='gaze on center'
        )
        ax[i].fill_between(
            grouped_sessions.times,
            grouped_mean.sel(trial_type=trial_type, eye_feature='eye_on_center'),
            grouped_mean.sel(trial_type=trial_type, eye_feature='eye_on_center') +
            grouped_mean.sel(trial_type=trial_type, eye_feature='eye_on_target'),
            color=trial_colors[i], alpha=0.7,
            label='gaze on target'
        )

    [ax[i].set_ylim(0, 1) for i in range(3)]
    [ax[i].legend(frameon=False) for i in range(3)]

    # 6.3. Include the temporal markers
    marker_times = [0, 1.3, 2.6, 3.9, 5.2]
    marker_times_lines = [0, 0.3, 1.3, 1.6, 2.6, 2.9, 3.9, 4.2, 5.2]
    marker_labels = ['Sel', 'SC1', 'SC2', 'SC3', 'Go']

    [ax[i].axvline(line, lw=0.5, color='k', ls='--')
        for line in marker_times_lines for i in range(3)]
    [ax[i].set_xticks(marker_times, marker_labels) for i in range(3)]

    # 6.4. Finalize the figure and save it
    sns.despine(fig)
    fig.suptitle(f'Monkey {monkey_name}', y=0.9)

    # save the figure
    fig.savefig(os.path.join(path_figures,
                             f'Grouped_sessions_monkey_{monkey_name}_eye_position.svg'))

    return fig


def plot_hand_behavior(monkey_name, session_list):
    """
    Plot the eye behavior for single trials and across trials as a percentile
    :param monkey_name: name of the monkey to plot
    :param session_list: all sites existing
    :return:
    """
    current_path = os.getcwd()

    # Set the paths
    if current_path.startswith('C:'):
        server = 'W:'  # local w VPN
    else:
        server = '/envau/work'  # niolon

    # path for saving figures
    path_figures = server + '/comco/lopez.l/Electrophysiology/' \
                            'ephy_laminar_MUA/Results/Full_trial/plots_revisions/behavior/'

    # 1. Initialize the list to store all behavioral sessions
    concat_behavior = []

    # 2. Iterate on all sessions and open the relevant ones
    for session in session_list:
        try:
            # path of the data
            path_data = server + '/comco/lopez.l/Electrophysiology/' \
                                 'ephy_laminar_MUA/Results/Preprocessed_data/' + session + '/'

            # Load only the sessions corresponding to the desired monkey
            if session.startswith(monkey_name):
                # load the data and annotations
                hand_velocity = xr.load_dataarray(os.path.join(path_data,
                                                               f'{session}-hand_velocity.nc'))

                if hand_velocity.sum() == 0:
                    print(f'{session} ALL ZEROS')
                else:
                    concat_behavior.append(hand_velocity)
                    io.logger.info(f"Concatenated behavior for {session}")

        except Exception as e:
            io.logger.error(f"No behavior for site {session}: {e}")
            continue

    # 3. Group the results in an individual x-array
    grouped_sessions = xr.concat(concat_behavior, dim='trials')

    # One color per trial type
    trial_colors = ['tab:blue', 'tab:green', 'pink']

    # 6. Create the plot - only for the group average
    # Create two vertically stacked subplots
    fig, ax = plt.subplots(
        1, 1,
        figsize=(12, 4))

    # 6.1. Group by trial-type to get the averages
    grouped_mean = grouped_sessions.groupby('trial_type').mean()

    for i, trial_type in enumerate(grouped_mean.trial_type.values):
        ax.plot(
            grouped_sessions.times,
            grouped_mean.sel(trial_type=trial_type),
            color=trial_colors[i],
            label=int(trial_type)
        )

    if monkey_name == 'M':
        ax.set_ylim([0.1, 0.5])
    elif monkey_name == 't':
        ax.set_ylim([0.1, 0.6])

    ax.legend(frameon=False)

    # 6.3. Include the temporal markers
    marker_times = [0, 1.3, 2.6, 3.9, 5.2]
    marker_times_lines = [0, 0.3, 1.3, 1.6, 2.6, 2.9, 3.9, 4.2, 5.2]
    marker_labels = ['Sel', 'SC1', 'SC2', 'SC3', 'Go']

    [ax.axvline(line, lw=0.5, color='k', ls='--') for line in marker_times_lines]
    ax.set_xticks(marker_times, marker_labels)

    # 6.4. Finalize the figure and save it
    sns.despine(fig)
    fig.suptitle(f'Monkey {monkey_name} - Hand velocity')

    # save the figure
    fig.savefig(os.path.join(path_figures,
                             f'Grouped_sessions_monkey_{monkey_name}_hand.svg'))

    return fig


if __name__ == "__main__":
    # define the sessions to run the analysis
    all_laminar = ['Mo180411001', 'Mo180412002', 'Mo180626003', 'Mo180627003', 'Mo180619002',
                   'Mo180622002', 'Mo180704003', 'Mo180418002', 'Mo180419003', 'Mo180426004',
                   'Mo180601001',
                   'Mo180523002', 'Mo180705002', 'Mo180706002', 'Mo180710002',
                   'Mo180711004', 'Mo180712006',
                   'Mo180615002', 'Mo180615005',
                   't150123001', 't150128001', 't150204001', 't150205004',
                   't150303002', 't150319003',
                   't150320002', 't150327002', 't150327003', 't150423002',
                   't150716001', 'Mo180615002-Mo180615005', 't150327002-t150327003']

    # select the monkey
    monkey_id = 't'  # or 'M' or 't' for each of the monkeys

    ###############################
    # 1. EYE BEHAVIOR ########
    plot_eye_behavior(monkey_name=monkey_id, session_list=all_laminar)

    ###############################
    # 2. HAND BEHAVIOR ########
    plot_hand_behavior(monkey_name=monkey_id, session_list=all_laminar)

    ###############################
    # 3. EYE DESCRIPTION
    plot_eye_position(monkey_name=monkey_id, session_list=all_laminar)
