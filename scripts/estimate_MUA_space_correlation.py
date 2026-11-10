import os
import warnings
import re
import h5py
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from frites import io
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
    new_times = np.linspace(t_before, t_after, samples_min + samples_max)
    n_times_shorter = len(new_times)

    # cut the data
    mua_shorter = mua_xr.isel(times=slice(start_index - samples_min, start_index + samples_max))
    mua_shorter = mua_shorter.assign_coords(times=new_times)

    return mua_shorter, n_times_shorter


def compute_spatial_correlations_xr(data, depths, session):
    data_concat = data.stack(
        samples=("trial_type", "times")
    ).transpose("samples", "channels")

    corr_matrix = np.corrcoef(data_concat.values.T)

    depths = np.asarray(depths)
    num_channels = corr_matrix.shape[0]

    # Unique channel pairs (exclude self-correlations)
    i, j = np.triu_indices(num_channels, k=1)

    return pd.DataFrame({
        "session": session,
        "channel_i": i,
        "channel_j": j,
        "depth_i": depths[i],
        "depth_j": depths[j],
        "distance": np.abs(depths[i] - depths[j]),
        "correlation": corr_matrix[i, j],
    })


def compute_spatial_correlations(data, depths, site):
    """
    Compute the spatial correlations across channels in each probe
    :param data: array of the shape (n_trials, n_channels, n_times)
    :param depths: vector of depths (n_channels)
    :param site: name of the session-probe
    :return:
    """
    # Concatenate trials and time into a single dimension
    # Shape: (trials * times, channels)
    data_concat = data.transpose(0, 2, 1).reshape(-1, data.shape[1])

    # Correlation matrix between channels
    corr_matrix = np.corrcoef(data_concat, rowvar=False)

    depths = np.asarray(depths)
    num_channels = corr_matrix.shape[0]

    # Unique channel pairs (exclude self-correlations)
    i, j = np.triu_indices(num_channels, k=1)

    return pd.DataFrame({
        "site": site, "channel_i": i, "channel_j": j,
        "depth_i": depths[i], "depth_j": depths[j],
        "distance": np.abs(depths[i] - depths[j]), "correlation": corr_matrix[i, j]})


def shuffle_time(data, rng=None):
    """
    Shuffle time samples independently for each trial and channel.

    Parameters
    ----------
    data : np.ndarray
        Shape (trials, channels, times).
    rng : int or None
        Random seed for reproducibility.

    Returns
    -------
    surrogate : np.ndarray
        Same shape as data, with time samples shuffled.
    """
    rng = np.random.default_rng(rng)

    data = np.asarray(data)
    surrogate = np.empty_like(data)

    for trial in range(data.shape[0]):
        for ch in range(data.shape[1]):
            surrogate[trial, ch] = data[
                trial, ch, rng.permutation(data.shape[2])
            ]

    return surrogate


def shuffle_trials(data, conditions, rng=None):
    """
    Shuffle trials independently for each channel, within each condition.

    Parameters
    ----------
    data : np.array
        Shape (trials, channels, times).
    conditions : np.array
        Shape (trials,). Condition label for each trial.
    rng : int or None
        Random seed for reproducibility.

    Returns
    -------
    surrogate : np.ndarray
        Same shape as data, with trials shuffled independently
        for each channel within each condition.
    """
    rng = np.random.default_rng(rng)

    if len(conditions) != data.shape[0]:
        raise ValueError("Number of condition labels must match number of trials.")

    surrogate = np.empty_like(data)

    for ch in range(data.shape[1]):
        for cond in np.unique(conditions):

            # Identify trials belonging to this condition
            idx = np.where(conditions == cond)[0]

            # Shuffle trial indices within this condition
            shuffled_idx = rng.permutation(idx)

            surrogate[idx, ch, :] = data[shuffled_idx, ch, :]

    return surrogate


if __name__ == "__main__":
    # --------------------------------------------------
    # 1) Define sessions and probes
    # --------------------------------------------------
    ALL_LAMINAR_NEW = [
        'Mo180411001', 'Mo180412002', 'Mo180626003', 'Mo180627003',
        'Mo180619002', 'Mo180622002', 'Mo180704003', 'Mo180418002',
        'Mo180419003', 'Mo180426004', 'Mo180601001', 'Mo180523002',
        'Mo180705002', 'Mo180706002', 'Mo180710002', 'Mo180711004',
        'Mo180712006', 't150303002', 't150319003', 't150423002',
        't150320002', 't150123001', 't150128001', 't150204001',
        't150205004', 't150716001', 'Mo180615002-Mo180615005',
        't150327002-t150327003'
    ]

    PROBES = [1, 2]

    N_SHUFFLES = 50

    # Reproducibility
    rng = np.random.default_rng(42)

    # --------------------------------------------------
    # 2) Define paths
    # --------------------------------------------------
    current_path = os.getcwd()

    if current_path.startswith('C:'):
        server = 'W:'
    else:
        server = '/envau/work'

    PATH_FIGURES = server + \
        '/comco/lopez.l/Electrophysiology/ephy_laminar_MUA/Results/Full_trial/' \
        'plots_revisions/spatial_correlations/'

    PATH_EXCEL = server + \
        '/comco/lopez.l/Electrophysiology/ephy_laminar_MUA/Results/Full_trial/' \
        'plots_similarity_corrected/'

    # --------------------------------------------------
    # 3) Load metadata
    # --------------------------------------------------
    excel_file = os.path.join(PATH_EXCEL, 'Mua_check_channels.xlsx')

    df_mua_info = pd.read_excel(excel_file)

    # --------------------------------------------------
    # 4) Initialize result containers
    # --------------------------------------------------
    all_correlations = []
    all_sh_correlations = []

    # --------------------------------------------------
    # 5) Loop over sessions and probes
    # --------------------------------------------------
    for SESSION in ALL_LAMINAR_NEW:
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

            # 1) Load MUA data
            mua_site = xr.load_dataarray(
                os.path.join(PATH, file_name_mua)
            )

            print(f'Loaded file: {file_name_mua}')

            # 2) Cut the MUA from -1.5 seconds before SEL until 5.7 seconds after SEL (full trial)
            mua_site_shorter, n_times = \
                cut_mua_by_markers_fixed_samples(mua_xr=mua_site, event_start='SEL',
                                                 t_extra=(-1.1, + 5.7), time_resolution=0.001)

            # 3) Select the channels of interest: remove bad channels
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

            # 4) Compute the correlation
            site_name = f'{SESSION}-{PROBE}'
            corr_site = compute_spatial_correlations(data=mua_site_shorter.values,
                                                     depths=mua_site_shorter.depths.values,
                                                     site=site_name)
            all_correlations.append(corr_site)

            # 5) Compute the shuffled correlations
            for sh in range(N_SHUFFLES):
                mua_shuffled_trials = shuffle_trials(
                    data=mua_site_shorter.values,
                    conditions=mua_site_shorter.trial_type.values, rng=rng)

                corr_site_sh = compute_spatial_correlations(
                    data=mua_shuffled_trials,
                    depths=mua_site_shorter.depths.values,
                    site=site_name)

                corr_site_sh['shuffle_id'] = sh

                all_sh_correlations.append(corr_site_sh)

            io.logger.info(f'Completed {site_name}')

            # Release large arrays
            del mua_site, mua_site_shorter

    # --------------------------------------------------
    # 8) Concatenate results
    # --------------------------------------------------
    if len(all_correlations) == 0:
        raise RuntimeError('No spatial correlations were computed.')

    all_correlations_pd = pd.concat(
        all_correlations,
        ignore_index=True
    )

    all_sh_correlations_pd = pd.concat(
        all_sh_correlations,
        ignore_index=True
    )

    # --------------------------------------------------
    # 9) Save results as CSV
    # --------------------------------------------------
    observed_path = os.path.join(
        PATH_FIGURES,
        'spatial_correlations_observed.csv'
    )

    shuffled_path = os.path.join(
        PATH_FIGURES,
        'spatial_correlations_shuffled.csv'
    )

    all_correlations_pd.to_csv(observed_path, index=False)
    all_sh_correlations_pd.to_csv(shuffled_path, index=False)

    # 3) Compute the correlation of the signal
    # mua_concat = mua_site_shorter.stack(
    #     samples=("trial_type", "times")).transpose("samples", "channels")
    #
    # ch_ch_corr = np.corrcoef(mua_concat.T)
    #
    # n_channels = ch_ch_corr.shape[0]
    #
    # distances = []
    # correlations = []
    # mean_corr = []
    #
    # for d in range(1, n_channels):
    #     corr_d = np.diag(ch_ch_corr, k=d)
    #
    #     distances.extend([d] * len(corr_d))
    #     correlations.extend(corr_d)
    #     mean_corr.append(corr_d.mean())
    #
    # fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    #
    # # Individual channel pairs
    # ax.scatter(
    #     distances, correlations,
    #     alpha=0.35, s=15, label="Channel pairs"
    # )
    #
    # # Mean correlation at each distance
    # ax.plot(
    #     range(1, n_channels), mean_corr,
    #     color="black", marker="o", markersize=6,
    #     linewidth=2, label="Mean correlation"
    # )
    #
    # ax.set_xlabel("Channel distance (channel positions)")
    # ax.set_ylabel("Pearson correlation")
    # ax.set_title(f"Channel correlation vs. distance | {SESSION}-{PROBE}")
    # ax.set_xticks(range(1, n_channels))
    # ax.legend()
    # fig.tight_layout()
    # sns.despine(fig)
    #
    # fig.savefig(os.path.join(PATH_FIGURES,
    #                          f'Correlation_vs_distance{SESSION}-{PROBE}.png'))
    #
    # plt.close('all')

