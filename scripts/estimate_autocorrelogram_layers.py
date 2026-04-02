"""
This script takes the index of layers computed for the LDA vectors and checks whether these time
signals are or not periodic. Finding the average across bootstraps and then the 95CI of the
surrogate data, checks whether the first peak is above the limit. If periodic, we could extract
the amplitude and the phase of the time signal.
"""

import os
import warnings
import numpy as np
import pandas as pd
import re
import h5py
import xarray as xr
import matplotlib.pyplot as plt
from frites import io
import seaborn as sns
from scipy.signal import find_peaks


# to remove the spam of pandas FutureWarning with iteritems
warnings.simplefilter(action='ignore', category=FutureWarning)


def block_shuffle_fixed_length(signal, block_size_n):
    """
    Shuffle a 1D signal in blocks of fixed size to preserve
    local autocorrelation but break long-range structure.
    """
    sig = np.asarray(signal).copy()
    L = len(sig)

    # pad only at the end, and only enough to complete the last block
    pad = (-L) % block_size_n
    if pad > 0:
        sig = np.concatenate([sig, np.zeros(pad)])  # zeros are fine at the end

    # reshape into blocks of fixed size
    blocks = sig.reshape(-1, block_size_n)

    # shuffle the blocks
    np.random.shuffle(blocks)

    # reconstruct
    shuffled = blocks.reshape(-1)

    # trim to original length
    shuffled = shuffled[:L]

    return shuffled


def compute_normalized_autocorrelation(signal, n_max_delay):
    """
    Get the time-varying signal, compute the z-scored version, the autocorrelation, and get the
    maximum and minimum tau to crop it.
    :param signal:
    :return: n_max_delay
    """
    # zscore the signal
    z_t = (signal - signal.mean()) / np.std(signal)

    # correlate with itself
    ac_full = np.correlate(z_t, z_t, mode='full')

    # extract only lags from -n_max_tau to +n_max_tau
    mid = len(ac_full) // 2

    ac = ac_full[mid - n_max_delay: mid + n_max_delay]
    ac_norm = ac / len(z_t)

    return ac_norm


if __name__ == "__main__":
    # define the session and probe to run the analysis
    SESSIONS = ['Mo180411001', 'Mo180412002', 'Mo180626003', 'Mo180627003', 'Mo180619002',
                'Mo180622002', 'Mo180704003', 'Mo180418002', 'Mo180419003', 'Mo180426004',
                'Mo180601001',
                'Mo180523002', 'Mo180705002', 'Mo180706002', 'Mo180710002',
                'Mo180711004', 'Mo180712006', 't150303002', 't150319003', 't150423002',
                't150320002', 't150123001', 't150128001', 't150204001', 't150205004', 't150716001',
                'Mo180615002-Mo180615005', 't150327002-t150327003']

    # all possible probes
    probes = [1, 2]
    # define the method to run
    method = 'LDA'  # 'PCA' or 'LDA'
    behavior = 'trial_type'  # 'trial_type' or 'mvt_dir'

    # set the path of the data and get the name of the files
    # check where are we running the code
    current_path = os.getcwd()

    # set the path of the data and get the name of the files
    if current_path.startswith('C:'):
        server = 'W:'  # local w VPN
    else:
        server = '/envau/work'  # niolon

    PATH_FIGURES = server + \
        '/comco/lopez.l/Electrophysiology/ephy_laminar_MUA/Results/Full_trial/' \
        'plots_periodicity_layers/'

    # loop over the sessions
    for SESSION in SESSIONS:
        PATH_DATA = server + \
            '/comco/lopez.l/Electrophysiology/ephy_laminar_MUA/Results/Full_trial/' \
            'data_paper/' + SESSION + '/'
        # iterate on the probes
        for probe in probes:
            # Open the evolution of the LDA computed for one cross-validation fold
            file_name_mua = [i for i in os.listdir(PATH_DATA) if
                             os.path.isfile(os.path.join(PATH_DATA, i))
                             and f'{SESSION}' in i and '.nc' in i and
                             'SC' not in i and f'probe_{probe}' in i and behavior in i and 'cv'
                             not in i and method in i]
            if len(file_name_mua) == 0:
                io.logger.info(f'No file found for probe {probe}')
                continue
            else:
                file_name_mua = file_name_mua[0]

            # open the file
            lda_evolution_site = xr.load_dataset(os.path.join(PATH_DATA, file_name_mua))

            # get the events
            events_onset = lda_evolution_site.task_events_onset
            events_names = re.split('-', lda_evolution_site.task_events_labels)

            # get the first LD
            LD1 = lda_evolution_site.weights_evolution.sel(dimensions='dim-1')
            n_times = len(LD1.times)

            # get the theoretical times of SC1 and Go
            t_SEL = events_onset[events_names.index('SEL')]
            t_SC1 = events_onset[events_names.index('SC1')]

            # get the index of SEL
            idx_SC1 = np.argmin(abs(LD1.times - t_SC1).values)

            # get the absolute value
            LD = abs(LD1)

            # 2) Get the auto-correlogram for the layers
            # get an index for the layers
            norm_layers =\
                LD.groupby('layers').mean() / \
                LD.groupby('layers').mean().sum(dim='layers').values.reshape(
                    -1, 1)

            # get the index of layer relevance
            idx_layers =\
                norm_layers[:, 0:2].mean(dim='layers') / norm_layers[:, 2:4].mean(dim='layers')

            # get just the array of values
            x = idx_layers.values.astype(float)

            # define the parameters to compute the correlation
            dt = 0.025  # 25ms
            max_tau = 2  # s
            n_max_tau = int(max_tau/dt)  # amount of dots in the matrix to take

            # 1) Generate the real distribution
            # get the amount of times we can check the data on
            n_times_short = n_times - n_max_tau - idx_SC1
            n_boot = 10000  # number of bootstraps with replacement
            AC_time = np.zeros((n_boot, 2 * n_max_tau))

            # get a bootstrapped version of the time vector
            all_times = np.random.randint(0, n_times_short, size=n_boot)

            for i, i_time in enumerate(all_times):
                # get a slice of t-tau, t +tau
                x_t = x[idx_SC1 + i_time - n_max_tau: idx_SC1 + i_time + n_max_tau]

                # get the autocorrelation
                ac_norm_t = compute_normalized_autocorrelation(signal=x_t, n_max_delay=n_max_tau)

                # store
                AC_time[i] = ac_norm_t

            # build an array of times
            time_array = np.linspace(-max_tau, max_tau, 2*n_max_tau)
            AC_global = AC_time.mean(axis=0)
            AC_global_ub, AC_global_lb = \
                np.percentile(AC_time, 95, axis=0), np.percentile(AC_time, 5, axis=0)

            # 2) Generate the null distribution
            n_surrogates = 1000  # number of bootstraps with replacement
            AC_null = np.zeros((n_times_short, 2 * n_max_tau, n_surrogates))

            for i in range(n_times_short):
                # get a slice of t-tau, t +tau
                x_t = x[idx_SC1 + i - n_max_tau: idx_SC1 + i + n_max_tau]
                for s in range(n_surrogates):
                    block_size_t = np.random.uniform(0.3, .8)  # seconds
                    block_size = int(block_size_t / dt)  # convert to samples
                    block_size = max(1, block_size)  # safety guard
                    x_shuff = block_shuffle_fixed_length(x_t, block_size)
                    ac_norm_t_shuff = \
                        compute_normalized_autocorrelation(signal=x_shuff, n_max_delay=n_max_tau)
                    AC_null[i, :, s] = ac_norm_t_shuff

            AC_null_global = AC_null.mean(axis=0)

            AC_null_ub, AC_null_lb = \
                np.percentile(AC_null_global, 95, axis=-1), \
                np.percentile(AC_null_global, 5, axis=-1)

            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            ax.plot(time_array, AC_global, '.-', color='gray')
            ax.fill_between(time_array, AC_global_lb, AC_global_ub, color='gray', alpha=.3)
            ax.fill_between(time_array, AC_null_lb, AC_null_ub, color='red', alpha=.3)
            [ax.axvline(peak, linestyle='--', color='lightgray')
             for peak in time_array[find_peaks(AC_global)[0]]]
            ax.set_title(f'{SESSION}-{probe} - Autocorrelation of layers index - shuffle in '
                         f'blocks uniformly sampled from 0.3-0.8s')
            sns.despine(fig)
            fig.savefig(os.path.join(PATH_FIGURES,
                                     f'{SESSION}-{probe}-autocorr_layers-maxlag_{max_tau}.svg'),
                        dpi=300, bbox_inches='tight')
