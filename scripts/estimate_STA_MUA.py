import os
import warnings
import re
import h5py
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.io import loadmat
from frites import io
from src.preprocessing_tools import open_matlab_behaviour
import seaborn as sns

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
    new_times = np.arange(samples_min + samples_max) * time_resolution + t_before
    n_times_shorter = len(new_times)

    # cut the data
    mua_shorter = mua_xr.isel(times=slice(start_index - samples_min, start_index + samples_max))
    mua_shorter = mua_shorter.assign_coords(times=new_times)

    return mua_shorter, n_times_shorter


def plot_sta_laminar(sta, channels, window_length, unit_id, n_spikes=None,
                     zscore=False, save_path=None):
    """
    Plot the laminar spike-triggered average of the MUA.
    :param sta: array (n_channels, 2 * window_length)
    :param channels: labels of the channels (same order as the rows of sta)
    :param window_length: half window in samples (ms if MUA is at 1 kHz)
    :param unit_id: string with the unit identifier (for the title)
    :param n_spikes: number of spikes used (optional, shown in the title)
    :param zscore: if True, z-score each channel; otherwise subtract its mean
    :param save_path: where to save the figure (optional)
    """
    sta = np.asarray(sta, dtype=float)
    lags = np.arange(-window_length, window_length)  # sample 'window_length' = spike time

    # MUA is a non-negative rate with a non-zero mean: remove each channel's mean so
    # the colormap shows modulation around the spike, not the baseline rate
    sta_plot = sta - np.nanmean(sta, axis=1, keepdims=True)
    if zscore:
        sta_plot = sta_plot / np.nanstd(sta, axis=1, keepdims=True)
    cbar_label = 'STA (z-score)' if zscore else 'STA - channel mean (a.u.)'

    vmax = np.nanmax(np.abs(sta_plot))  # symmetric color scale around 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6), sharey=False,
                                   gridspec_kw={'width_ratios': [1.2, 1]})

    # --- 1) heatmap: channels (depth) x lag ---
    im = ax1.imshow(sta_plot, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                    extent=[lags[0] - 0.5, lags[-1] + 0.5, len(channels) - 0.5, -0.5])
    ax1.axvline(0, color='k', ls='--', lw=1)
    ax1.set_yticks(np.arange(len(channels)))
    ax1.set_yticklabels(channels, fontsize=7)
    ax1.set_xlabel('Lag from spike (ms)')
    ax1.set_ylabel('Channel (superficial to deep)')
    ax1.set_title('STA heatmap')
    fig.colorbar(im, ax=ax1, label=cbar_label, fraction=0.046, pad=0.04)

    # --- 2) stacked traces ---
    offset = 1.2 * np.nanmax(np.abs(sta_plot))
    for i, trace in enumerate(sta_plot):
        ax2.plot(lags, trace - i * offset, color='k', lw=0.8)
    ax2.axvline(0, color='r', ls='--', lw=1)
    ax2.set_yticks(-np.arange(len(channels)) * offset)
    ax2.set_yticklabels(channels, fontsize=7)
    ax2.set_xlabel('Lag from spike (ms)')
    ax2.set_title('STA traces')
    ax2.spines[['top', 'right']].set_visible(False)

    title = f'MUA STA - unit {unit_id}'
    if n_spikes is not None:
        title += f' (n spikes = {n_spikes})'
    fig.suptitle(title)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, (ax1, ax2)


def get_time_stamps_df(spike_times, behavior_site, trial_num_lfp):
    """
    Build a dataframe with the time-stamps of the neurons between SEL and MVT of each trial. Include
    the trial number, the time in the trial and the time index. In other words, unstack the
    continuous time of spiking into trials and time-in-trial build a dataframe with the timestamp
    of the spike and time/trial information
    :param spike_times: array of spike times (n_spikes,)
    :param behavior_site: dataframe including the behavioral markers for each trial in that session
    :param trial_num_lfp: sub-selection of trials (trial numbers) present in the LFP epoch data
    ----
    :return: spike_times_df: dataframe with (n_spikes, features) in which features include the time
    and trial index for each spike
    """
    spikes = []

    for ts in spike_times:
        spike_dict = {'timestamp': ts}

        # check that the times are between SEL and MVT
        trial_markers = behavior_site[(ts > behavior_site.SelTim) & (ts < behavior_site.MVTTim)]

        # if not, don't include it
        if len(trial_markers) == 0:
            continue

        # check that the trial belongs to the trials from the LFP
        trial_i = int(trial_markers.TrialNum.iloc[0])
        if trial_i not in trial_num_lfp:
            continue

        # get the trial number, the time in the trial and the time_index from SEL
        spike_dict['trial_num'] = trial_i
        spike_dict['time_in_trial'] = (ts - trial_markers.SelTim.iloc[0]) / 1000
        spike_dict['time_idx_from_SEL'] = int(ts - trial_markers.SelTim.iloc[0])

        spikes.append(spike_dict)

    return pd.DataFrame(spikes)


def compute_sta(spike_times_df, continuous_signal, trial_nums_cont, window_length):
    """
    Computes the Spike-triggered average of any continuous signal, given the spikes and the
    continuous signal we want to take snippets of
    :param spike_times_df: dataframe with spike information (spike time in trials and time indexes)
    :param continuous_signal: array of the shape (n_trials, n_times)
    :param trial_nums_cont: array with the original trial number of the continuous signal
    :param window_length: size of the window to extract around the spike
    ----
    :return: sta: array of the shape (window_length) which contains the spike-triggered average of
    the continuous signal across all events
    """
    # initialize the list for the snippets
    spike_triggered_snippets = []

    # shape of the continuous signal
    n_tri, n_t = continuous_signal.shape

    # iterate in all the rows of the dataframe (all spikes)
    for ts in spike_times_df.itertuples(index=False):
        # get the info for each spike (row)
        trial_ts = ts.trial_num
        t_idx_ts = ts.time_idx_from_SEL

        # break if the time is out of the span of the MUA
        # 1) the time from sel should be bigger than the window (the cont signal starts at SEL)
        # 2) the time should be smaller than the total time and the window
        if np.logical_or((t_idx_ts < window_length), (t_idx_ts + window_length > n_t)):
            continue

        # get the index of the trial for the continuous data (based on orig trial number)
        trial_idx_cont = np.where(trial_ts == trial_nums_cont)[0][0]

        # get the LFP around
        spike_triggered_snippets.append(continuous_signal[trial_idx_cont,
                                        t_idx_ts - window_length:t_idx_ts + window_length])

    # Spike-triggered average (across all snippets)
    sta = np.mean(spike_triggered_snippets, axis=0)

    return sta


if __name__ == "__main__":
    # define the sessions to run the analysis
    SESSIONS = ['Mo180412002']
    monkey = 'Mourad'

    # all possible probes
    PROBES = [1]

    # get the paths
    current_path = os.getcwd()

    if current_path.startswith('C:'):
        server = 'W:'  # local w VPN
    else:
        server = '/envau/work'  # niolon

    # path to store the figures
    PATH_FIGURES = server + \
        '/comco/lopez.l/Electrophysiology/ephy_laminar_MUA/Results/Full_trial/' \
        'plots_revisions/spatial_correlations/'

    PATH_EXCEL = server + \
        '/comco/lopez.l/Electrophysiology/ephy_laminar_MUA/Results/Full_trial/' \
        'plots_similarity_corrected/'

    # load the excel file from the PATH_Figures
    excel_file = os.path.join(PATH_EXCEL, f'Mua_check_channels.xlsx')

    # load the excel file with important metadata
    df_mua_info = pd.read_excel(excel_file)

    # loop over the sites
    for SESSION in SESSIONS:
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

            # ---------------------------
            # LOAD MULTI-UNIT ACTIVITY OF THE SITE
            # ---------------------------

            # open the file
            mua_site = xr.load_dataarray(os.path.join(PATH, file_name_mua))
            io.logger.info(f'Loaded file: {file_name_mua}')

            # 1) Cut the MUA from SEL until 5.7 seconds after SEL (full trial)
            mua_site_shorter, n_times = \
                cut_mua_by_markers_fixed_samples(mua_xr=mua_site, event_start='SEL',
                                                 t_extra=(0, + 5.7), time_resolution=0.001)

            # 2) Select the channels of interest: remove bad channels
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

            # ---------------------------
            # LOAD SINGLE UNITS FOR THAT SITE
            # ---------------------------
            path_spikes = \
                f"{server}/comco/nandi.n//Timescales_Paper/data/SUA/{monkey}/matData/{SESSION}"
            file_names_sua = [filename for filename in os.listdir(path_spikes) if
                              filename.endswith(".mat")]

            # Path for raw data
            path_raw = server + '/comco/lopez.l/Electrophysiology/ephy_laminar_MUA/Raw_data/'
            file_name_bhv = os.path.join(path_raw, "behaviour-" + SESSION + ".mat")

            # Load the behavioral markers for that
            behavior = open_matlab_behaviour(file_name_bhv)

            # check to which trial the timestamps belong
            behavior_df = pd.DataFrame({'SelTim': behavior['SelTim'],
                                        'TouchTim': behavior['TouchTim'],
                                        'MVTTim': behavior['MVTTim'],
                                        'TrialNum': behavior['OrigCorrTrials']})

            for file_neuron_i in file_names_sua:
                # get the identifier of the neuron
                _, _, neuron_contact, neuron_id = re.split('_', re.split('.mat', file_neuron_i)[0])

                unit_id = f'{neuron_contact}_{neuron_id}'

                # Spiking activity
                dict_spiking = loadmat(os.path.join(path_spikes, file_neuron_i))

                # get the timestamps of the neuron
                time_stamps = np.squeeze(dict_spiking['ts'])  # to remove the double array

                # 5) Unstack the continuous time of spiking into trials and time-in-trial (df)
                time_stamps_df = \
                    get_time_stamps_df(spike_times=time_stamps,
                                       behavior_site=behavior_df,
                                       trial_num_lfp=mua_site_shorter['t_number_orig'].values)

                # 6) Compute the STA for each neuron and channel
                t_window = 50  # ms
                n_channels = len(mua_site_shorter.channels)

                # Compute the STA-MUA across all channels
                sta_mua_laminar = []

                for channel in range(n_channels):
                    # compute the STA
                    sta_channels = compute_sta(spike_times_df=time_stamps_df,
                                               continuous_signal=mua_site_shorter.
                                               isel(channels=channel).values,
                                               trial_nums_cont=mua_site_shorter['t_number_orig']
                                               .values,
                                               window_length=t_window)
                    # append them
                    sta_mua_laminar.append(sta_channels)

                # convert to array
                sta_mua_laminar = np.array(sta_mua_laminar)

                fig, _ = plot_sta_laminar(sta_mua_laminar,
                                          channels=mua_site_shorter.channels.values,
                                          window_length=t_window,
                                          unit_id=unit_id,
                                          n_spikes=len(time_stamps_df),
                                          save_path=os.path.join(PATH_FIGURES,
                                                                 f'STA_MUA_{SESSION}_{unit_id}'
                                                                 f'.png'))
                plt.close(fig)
