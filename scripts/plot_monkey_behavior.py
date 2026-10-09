"""
This script takes the recorded hand and eye behavior of each monkey and plots a summary of it.
"""
# import packages
import os
import mne
import h5py
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from frites import io
from matplotlib.colors import to_rgb

if __name__ == "__main__":
    # define the sessions to run the analysis
    all_laminar = ['Mo180411001', 'Mo180412002', 'Mo180626003', 'Mo180627003', 'Mo180619002',
                   'Mo180622002', 'Mo180704003', 'Mo180418002', 'Mo180419003', 'Mo180426004',
                   'Mo180601001',
                   'Mo180523002', 'Mo180705002', 'Mo180706002', 'Mo180710002',
                   'Mo180711004', 'Mo180712006', 't150303002', 't150319003', 't150423002',
                   't150320002',
                   't150123001', 't150128001', 't150204001',
                   't150205004',
                   't150716001',
                   'Mo180615002', 'Mo180615005', 't150327002', 't150327003']

    # 'Mo180615002-Mo180615005', 't150327002-t150327003']

    # check where are we running the code
    current_path = os.getcwd()

    # select the monkey
    monkey_id = 'M'  # or 'M' or 't' for each of the monkeys

    concat_behavior = []

    if current_path.startswith('C:'):
        server = 'W:'  # local w VPN
    else:
        server = '/envau/work'  # niolon

    # path for saving figures
    path_figures = server + '/comco/lopez.l/Electrophysiology/' \
                            'ephy_laminar_MUA/Results/Full_trial/plots_revisions/behavior/'

    # get all sessions per monkey together
    for session in all_laminar:
        try:
            # path of the data
            path_data = server + '/comco/lopez.l/Electrophysiology/' \
                                 'ephy_laminar_MUA/Results/Preprocessed_data/' + session + '/'

            file_name_annot = [file for file in os.listdir(path_data) if 'annot' in file][0]



            # eye_on_target = xr.load_dataarray(os.path.join(path_data, f'{site}-eye_on_target.nc'))
            # eye_on_center = xr.load_dataarray(os.path.join(path_data, f'{site}-eye_on_center.nc'))
            # hand_velocity = xr.load_dataarray(os.path.join(path_data, f'{site}-hand_velocity.nc'))

            # annotations = mne.read_annotations(
            #     os.path.join(path_data, file_name_annot))  # annotations
            #
            # # get the SEL marker and the MVT one
            # t_sel = annotations.onset[annotations.description == 'SEL'][0]
            # t_mvt = annotations.onset[annotations.description == 'MVT'][0]
            # t_GO = annotations.onset[annotations.description == 'GO'][0]

            if session.startswith(monkey_id):
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

    # plot the results of all sessions
    grouped_sessions = xr.concat(concat_behavior, dim='trials')

    # 1. Plot the single trial eye behavior (in-out)
    # sort them by trial type
    sorted_sessions = grouped_sessions.sortby('trial_type')
    # number of trials per group
    trial_id, n_trials_group = np.unique(grouped_sessions.trial_type,
                                         return_counts=True)

    # One color per trial type
    trial_colors = ['tab:blue', 'tab:green', 'pink']
    # Build an RGB image
    rgb = np.ones((*sorted_sessions.shape, 3))  # background = white

    start = 0
    for color, count in zip(trial_colors, n_trials_group):
        end = start + count
        color_rgb = np.array(to_rgb(color))

        # Color samples where the signal is 1
        block = sorted_sessions.values[start:end]
        rgb[start:end][block == 1] = color_rgb

        # Display NaNs as light gray
        rgb[start:end][np.isnan(block)] = to_rgb('k')
        start = end

    # Plot
    # Create two vertically stacked plots
    fig, (ax_avg, ax_raster) = plt.subplots(
        2, 1,
        figsize=(12, 10),
        sharex='col',
        gridspec_kw={'height_ratios': [1, 5], 'hspace': 0.05}
    )

    # Plot the averages
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

    # Plot the single trials
    ax_raster.imshow(rgb,
                     aspect='auto',
                     interpolation='nearest',
                     extent=[
                         sorted_sessions.times.values[0],
                         sorted_sessions.times.values[-1],
                         sorted_sessions.values.shape[0], 0
                     ]
                     )

    # average markers
    marker_times = [0, 1.3, 2.6, 3.9, 5.2]
    marker_times_lines = [0, 0.3, 1.3, 1.6, 2.6, 2.9, 3.9, 4.2, 5.2]
    marker_labels = ['Sel', 'SC1', 'SC2', 'SC3', 'Go']

    [ax_raster.axvline(line, lw=2, color='firebrick') for line in marker_times_lines]
    ax_raster.set_xticks(marker_times, marker_labels)

    # set title
    sns.despine(fig)
    fig.suptitle(f'Monkey {monkey_id}', y=0.9)

    # save the figure
    fig.savefig(os.path.join(path_figures, f'Grouped_sessions_monkey_{monkey_id}.svg'))
