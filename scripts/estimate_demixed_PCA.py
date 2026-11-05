"""
Estimate the different PCs using the de-mixed-PCA method to compare the results with the LDA
analysis.
"""
import os
import warnings
import re
import h5py
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from frites import io
from dPCA import dPCA
import seaborn as sns
from scipy.signal import savgol_filter

# to remove the spam of pandas FutureWarning with iteritems
warnings.simplefilter(action='ignore', category=FutureWarning)


def cut_mua_by_markers_fixed_samples(mua_xr, event_start, t_extra=(-1, +5.5),
                                     time_resolution=0.001):
    """
    Cut the MUA data by the markers of the task.
    :param mua_xr: DataArray with the MUA data
    :param event_start: string with the name of the event to start the cut
    :param t_extra: list with the extra time to add to the events in seconds
    :param time_resolution: float with the time resolution of the MUA data in seconds
    (default is 1ms)
    ----
    :return: mua_shorter: DataArray with the MUA data cut by the events
             n_times_shorter: int with the number of time-points in the new MUA
    """
    # get the events
    events_onset = mua_xr.task_events_onset
    events_names = re.split('-', mua_xr.task_events_labels)

    # get the time of the event
    t_start = events_onset[events_names.index(event_start)]

    # get the time extra (positive and negative)
    t_before, t_after = t_extra

    # find the index where time is closest to 0
    start_val = mua_xr.times.sel(times=t_start, method="nearest").item()
    start_index = mua_xr.times.to_index().get_loc(start_val)

    # this signal is not at 1ms resolution - so we need to find the closest index
    samples_min = np.floor(-t_before / time_resolution).astype(int)  # x seconds before the event
    samples_max = np.floor(t_after / time_resolution).astype(int)  # x seconds after the event

    # create the new times so that then the time dimension will be exactly the same
    new_times = np.linspace(t_before, t_after, samples_min + samples_max)
    n_times_shorter = len(new_times)

    # cut the data
    mua_shorter = mua_xr.isel(times=slice(start_index - samples_min, start_index + samples_max))
    mua_shorter = mua_shorter.assign_coords(times=new_times)

    return mua_shorter, n_times_shorter


def split_train_test(mua_xr, condition, seed=0, percent_train=0.5):
    """
    Split the MUA data into train and test sets.
    :param mua_xr:
    :param condition: condition to split the data
    :param seed:
    :param percent_train:
    :return:
    """
    # Get unique classes
    classes = np.unique(mua_xr[condition].values)

    # Determine the minimum number of trials available across all classes
    min_trials_per_class = min([
        np.sum(mua_xr[condition].values == c)
        for c in classes
    ])

    # Decide how many trials per class to keep in training
    n_keep_per_class = int(np.floor(min_trials_per_class * percent_train))

    # Seed for reproducibility
    np.random.seed(seed)

    # Collect train indices
    idx_trials_to_keep = []

    for c in classes:
        class_indices = np.where(mua_xr[condition].values == c)[0]
        selected = np.random.choice(class_indices, n_keep_per_class, replace=False)
        idx_trials_to_keep.extend(selected)

    # Convert to array and sort
    idx_trials_to_keep = np.sort(np.array(idx_trials_to_keep))

    # Create exclusion mask
    n_trials = mua_xr[condition].size
    mask = np.ones(n_trials, dtype=bool)
    mask[idx_trials_to_keep] = False
    idx_trials_to_exclude = np.where(mask)[0]

    # Split data
    mua_xr_train = mua_xr.isel(trial_type=idx_trials_to_keep)
    mua_xr_test = mua_xr.isel(trial_type=idx_trials_to_exclude)

    return mua_xr_train, mua_xr_test


if __name__ == "__main__":
    # 1) Define the sessions to run the analysis - and paths
    SITES = ['Mo180412002']

    # all possible probes
    PROBES = [1]

    # set the paths
    current_path = os.getcwd()

    if current_path.startswith('C:'):
        server = 'W:'  # local w VPN
    else:
        server = '/envau/work'  # niolon

    # path of the figures
    PATH_EXCEL = server + \
        '/comco/lopez.l/Electrophysiology/ephy_laminar_MUA/Results/Full_trial/' \
        'plots_similarity_corrected/'

    # load the excel file from the PATH_Figures
    excel_file = os.path.join(PATH_EXCEL, f'Mua_check_channels.xlsx')

    # load the excel file with important metadata
    df_mua_info = pd.read_excel(excel_file)

    # 2) Loop over the sites
    for SESSION in SITES:
        # set the path of the data and get the name of the files
        PATH = server + \
            '/comco/lopez.l/Electrophysiology/ephy_laminar_MUA/Results/Preprocessed_data/' + \
            SESSION + '/'

        # iterate on the probes
        for PROBE in PROBES:
            # Open in this case the single trials aligned to SEL because the SC2 ones are not full
            file_name_mua = [i for i in os.listdir(PATH) if os.path.isfile(os.path.join(PATH, i))
                             and
                             f'{SESSION}' in i and '.nc' in i and 'MUA-' in i and 'Sel' in i and
                             'SC' not in i and f'probe_{PROBE}' in i]
            if len(file_name_mua) == 0:
                io.logger.info(f'No MUA file found for probe {PROBE}')
                continue
            else:
                file_name_mua = file_name_mua[0]

            # 3) Open the file
            mua_site = xr.load_dataarray(os.path.join(PATH, file_name_mua))
            io.logger.info(f'Loaded file: {file_name_mua}')

            # 4) Cut the MUA from -1.5 seconds before SEL until 5.7 seconds after SEL (full trial)
            mua_site_shorter, n_times = \
                cut_mua_by_markers_fixed_samples(mua_xr=mua_site, event_start='SEL',
                                                 t_extra=(-0.5, + 5.7), time_resolution=0.001)

            # 5) Select the channels of interest: remove bad channels
            df_mua_site = df_mua_info[np.logical_and(df_mua_info['session'] == SESSION,
                                                     df_mua_info['probe'] == PROBE)]

            bad_channel = df_mua_site.bad_channels.values[0]

            # If bad_channel is not NaN and not empty
            if pd.notna(bad_channel):
                # If it’s a string of multiple channels (e.g. 'PMd-25, PMd-26'), split into list
                if isinstance(bad_channel, str):
                    bad_channels = [ch.strip() for ch in bad_channel.split(',')]
                else:
                    bad_channels = [bad_channel]  # single channel as list

                # Set selected bad channels to NaN
                for ch in bad_channels:
                    mua_site_shorter.loc[dict(channels=ch)] = np.nan

                # Remove any channel that has any NaN
                has_nan = mua_site_shorter.isnull().any(dim=('trial_type', 'times'))
                mua_site_shorter = mua_site_shorter.sel(channels=~has_nan)

            # 6) Prepare the data for dPCA
            # Get a new coordinate that combines the movement direction and the trial type
            mua_site_shorter = mua_site_shorter.assign_coords(
                condition=('trial_type', pd.MultiIndex.from_arrays(
                    [mua_site_shorter.trial_type.values, mua_site_shorter.mvt_dir.values],
                    names=['trial_type', 'mvt_dir']
                )))

            # Split the data in train and test
            mua_train, mua_test = split_train_test(mua_xr=mua_site_shorter,
                                                   condition='trial_type',
                                                   seed=0, percent_train=0.7)

            # Group-by this new condition and average, to get (n_conditions, n_channels, times)
            mua_train_condition_avg = mua_train.groupby('condition').mean('trial_type')
            mua_test_condition_avg = mua_test.groupby('condition').mean('trial_type')

            # Unstack and Re-arrange the order of the variables
            mua_train_condition_avg = mua_train_condition_avg.unstack('condition')
            mua_test_condition_avg = mua_test_condition_avg.unstack('condition')

            mua_train_condition_avg = \
                mua_train_condition_avg.transpose('channels', 'times', 'trial_type', 'mvt_dir')
            mua_test_condition_avg = \
                mua_test_condition_avg.transpose('channels', 'times', 'trial_type', 'mvt_dir')

            # Down-sample the times
            mua_avg_train_ds = mua_train_condition_avg.coarsen(times=10, boundary='trim').mean()
            mua_avg_test_ds = mua_test_condition_avg.coarsen(times=10, boundary='trim').mean()

            # Get a simple array
            X_train = mua_avg_train_ds.values.copy()
            X_test = mua_avg_test_ds.values.copy()

            # Demean each channel across all times and conditions
            X_train -= X_train.mean(axis=(1, 2, 3), keepdims=True)
            X_test -= X_test.mean(axis=(1, 2, 3), keepdims=True)

            # 7) Compute dPCA
            # Initialize
            dpca = dPCA.dPCA(labels='tsd', n_components=10, regularizer=None)

            # Fit
            dpca.fit(X_train)

            X_transformed = dpca.transform(X_test)

            # 8) Plot the results
            # Direction
            times = mua_avg_test_ds.times.values
            # Direction
            fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex='col')

            colors_directions = ['#393D3F', '#A882DD', '#F4D35E', '#EE964B']
            # Savitzky-Golay parameters
            window_length = 21  # samples (110 ms at 100 Hz)
            polyorder = 3
            events_labels = re.split('-', mua_site_shorter.task_events_labels)

            for i in range(3):
                component = X_transformed['td'][i]
                component = component.mean(axis=1)
                for d in range(4):
                    # Smooth temporal trajectory
                    smoothed = savgol_filter(
                        component[:, d],
                        window_length=window_length,
                        polyorder=polyorder
                    )
                    axes[i].plot(
                        times,
                        smoothed,
                        label=f'Direction {d + 1}',
                        color=colors_directions[d], lw=2)
                axes[i].set_ylabel(f'dPC {i + 1}')
                axes[i].legend()
                axes[i].axhline(0, color='grey', linewidth=0.5)
                [axes[i].axvline(line, lw=0.5, c='gray', ls='--')
                 for line in mua_site_shorter.task_events_onset[2:]]
                [axes[i].axvline(line+.3, lw=0.5, c='gray', ls='--')
                 for line in mua_site_shorter.task_events_onset[2:6]]
                axes[i].set_xticks(mua_site_shorter.task_events_onset[2:], events_labels[2:])

            axes[-1].set_xlabel('Time (s)')
            fig.suptitle('dPCA: Time × Movement Direction')
            plt.tight_layout()
            sns.despine(fig)

            # Trial type
            fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex='col')
            colors_ttype = ['tab:blue', 'tab:green', 'pink']

            # Savitzky-Golay parameters
            window_length = 21  # 210 ms at 100 Hz
            polyorder = 3

            for i in range(3):

                # Shape: (time, trial_type, mvt_dir)
                component = X_transformed['ts'][i]

                # Average over movement directions
                component = component.mean(axis=2)

                for s in range(3):
                    # Smooth temporal trajectory
                    smoothed = savgol_filter(
                        component[:, s],
                        window_length=window_length,
                        polyorder=polyorder
                    )

                    axes[i].plot(
                        times,
                        smoothed,
                        label=f'Trial type {s + 1}',
                        color=colors_ttype[s]
                    )

                axes[i].set_ylabel(f'dPC {i + 1}')
                axes[i].legend()
                axes[i].axhline(0, color='grey', linewidth=0.5)
                [axes[i].axvline(line, lw=0.5, c='gray', ls='--')
                 for line in mua_site_shorter.task_events_onset[2:]]
                [axes[i].axvline(line+.3, lw=0.5, c='gray', ls='--')
                 for line in mua_site_shorter.task_events_onset[2:6]]
                axes[i].set_xticks(mua_site_shorter.task_events_onset[2:], events_labels[2:])

            axes[-1].set_xlabel('Time (s)')
            fig.suptitle('dPCA: Time × Trial Type')

            plt.tight_layout()
            sns.despine(fig)

            # Three-way interaction: one figure per trial type
            colors_directions = ['#393D3F', '#A882DD', '#F4D35E', '#EE964B']
            window_length = 21
            polyorder = 3
            for s in range(3):
                fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex='col')
                for i in range(3):
                    # Shape: (time, trial_type, mvt_dir)
                    component = X_transformed['tsd'][i]
                    for d in range(4):
                        smoothed = savgol_filter(
                            component[:, s, d],
                            window_length=window_length,
                            polyorder=polyorder
                        )
                        axes[i].plot(
                            times,
                            smoothed,
                            label=f'Direction {d+1}',
                            color=colors_directions[d]
                        )
                    axes[i].set_ylabel(f'dPC {i+1}')
                    axes[i].legend()
                    axes[i].axhline(0, color='grey', linewidth=0.5)
                    [axes[i].axvline(line, lw=0.5, c='gray', ls='--')
                     for line in mua_site_shorter.task_events_onset[2:]]
                    [axes[i].axvline(line + .3, lw=0.5, c='gray', ls='--')
                     for line in mua_site_shorter.task_events_onset[2:6]]
                    axes[i].set_xticks(mua_site_shorter.task_events_onset[2:], events_labels[2:])
                axes[-1].set_xlabel('Time (s)')
                fig.suptitle(f'dPCA: Time × Trial Type × Direction | Trial type {s+1}')
                plt.tight_layout()
                sns.despine(fig)

            # Plot the variance explained as a stack-plot
            marginalizations = ['t', 's', 'd', 'ts', 'td', 'tsd']

            colors = {
                't': '#999999',
                's': '#5da3d9',
                'd': '#ed6b6b',
                'ts': '#2878B5',
                'td': '#C62828',
                'tsd': '#9C4A9C'
            }

            labels = {
                't': 'Condition-independent',
                's': 'Trial type',
                'd': 'Direction',
                'ts': 'Trial type × time',
                'td': 'Direction × time',
                'tsd': 'Trial type × direction × time'
            }

            # Automatically determine the number of components
            n_components = min(
                len(dpca.explained_variance_ratio_[key])
                for key in marginalizations
            )

            # Extract explained variance (%)
            values = {
                key: np.array(dpca.explained_variance_ratio_[key])[:n_components] * 100
                for key in marginalizations
            }

            # Plot
            sns.set_context('poster')
            fig, ax = plt.subplots(figsize=(12, 5))

            x = np.arange(n_components)
            bottom = np.zeros(n_components)

            for key in marginalizations:
                ax.bar(
                    x,
                    values[key],
                    bottom=bottom,
                    color=colors[key],
                    edgecolor='black',
                    linewidth=0.5,
                    label=labels[key]
                )
                bottom += values[key]

            ax.set_xlabel('PC')
            ax.set_ylabel('Explained variance (%)')
            ax.set_title('dPCA: Explained variance per component')
            ax.set_xticks(x)
            ax.set_xticklabels(np.arange(1, n_components + 1))
            ax.legend(frameon=False)
            ax.spines[['top', 'right']].set_visible(False)

            plt.tight_layout()
            plt.show()

            # Condition-independent temporal activity
            fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex='col')

            # Savitzky-Golay parameters
            window_length = 21  # 210 ms at 100 Hz
            polyorder = 3

            for i in range(4):

                # Shape: (time, trial_type, mvt_dir)
                component = X_transformed['t'][i]

                # Average over trial type and movement direction
                component = component.mean(axis=(1, 2))

                # Smooth temporal trajectory
                smoothed = savgol_filter(
                    component,
                    window_length=window_length,
                    polyorder=polyorder
                )

                axes[i].plot(
                    times,
                    smoothed,
                    color='grey',
                    linewidth=1.5,
                    label='Condition-independent'
                )

                axes[i].set_ylabel(f'dPC {i + 1}')
                axes[i].legend()
                axes[i].axhline(0, color='grey', linewidth=0.5)

                [axes[i].axvline(line, lw=0.5, c='gray', ls='--')
                 for line in mua_site_shorter.task_events_onset[2:]]

                [axes[i].axvline(line + .3, lw=0.5, c='gray', ls='--')
                 for line in mua_site_shorter.task_events_onset[2:6]]

                axes[i].set_xticks(
                    mua_site_shorter.task_events_onset[2:],
                    events_labels[2:]
                )

            axes[-1].set_xlabel('Time (s)')
            fig.suptitle('dPCA: Condition-independent temporal activity')
            plt.tight_layout()