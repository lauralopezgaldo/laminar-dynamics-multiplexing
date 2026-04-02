"""
This script takes the already computed cosine similarity matrix across bootstraps and checks the
periodicity. For that it extracts chucks of data -tau, t, +tau from the diagonal and computes the
autocorrelation as a function of tau. This way, when averaging along all the times in the diagonal
we can check the periodicity. It compares the amplitude of the first peak with the shuffled estimate
(by shuffling the signal in blocks to preserve the autocorrelation but destroy the periodicity).
If it is above the 95th of the chance, it gets the time value corresponding to the peak to generate
a distribution of periods.
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
from frites.stats.stats_nonparam import confidence_interval
import seaborn as sns
from scipy.signal import find_peaks

# to remove the spam of pandas FutureWarning with iteritems
warnings.simplefilter(action='ignore', category=FutureWarning)


def compute_cosine_similarity(vector_a, vector_b):
    """
    It computes the cosine distance between two vectors. In practice, it computes the dot product
    of both vectors and normalizes it by the product of the norms of the vectors.
    If the vectors are perpendicular, the cosine distance is 0. If they are parallel, the cosine
    distance is 1.
    :param vector_a: shape (n_samples, n_features)
    :param vector_b: shape (n_samples, n_features)
    :return:
    """
    # compute the dot product between the two vectors at every time-point
    numerator = np.dot(vector_a, vector_b.T)
    denominator = np.outer(np.linalg.norm(vector_a, axis=1), np.linalg.norm(vector_b, axis=1))
    dot_product_norm = numerator / denominator
    return dot_product_norm


def block_shuffle(signal, block_size_n):
    # split into blocks
    n_blocks = len(signal) // block_size_n
    blocks = np.array_split(signal[:n_blocks*block_size_n], n_blocks)
    # shuffle blocks
    np.random.shuffle(blocks)
    # flatten back
    return np.concatenate(blocks)


def block_shuffle_fixed_length(signal, block_size_n):
    """
    Shuffle a 1D signal in blocks of fixed size to preserve
    local autocorrelation but break long-range structure.
    """
    sig = np.asarray(signal)
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

    # store across all sites
    cosine_periodicity = []

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
        'plots_periodicity_LDA_svg/'

    # loop over the sessions
    for SESSION in SESSIONS:
        PATH_DATA = server + \
            '/comco/lopez.l/Electrophysiology/ephy_laminar_MUA/Results/Full_trial/' \
            'data_paper/' + SESSION + '/'
        # iterate on the probes
        for probe in probes:
            # Open the similarity matrix for a behavior across cv
            file_name_mua = [i for i in os.listdir(PATH_DATA) if
                             os.path.isfile(os.path.join(PATH_DATA, i))
                             and f'{SESSION}' in i and '.nc' in i and
                             'SC' not in i and f'probe_{probe}' in i and behavior in i and 'cv'
                             in i and method in i and 'similarity' in i and 'mvt_dir' not in i]
            if len(file_name_mua) == 0:
                io.logger.info(f'No file found for probe {probe}')
                continue
            else:
                file_name_mua = file_name_mua[0]

            # open the file
            similarity_mat = xr.load_dataarray(os.path.join(PATH_DATA, file_name_mua))  # (bt, T, T)
            n_times = len(similarity_mat.times)

            # get the events
            events_onset = similarity_mat.task_events_onset
            events_names = re.split('-', similarity_mat.task_events_labels)

            # get the theoretical times of SC1 and Go
            t_SEL = events_onset[events_names.index('SEL')]
            t_SC1 = events_onset[events_names.index('SC1')]

            # get the index of SEL
            idx_SC1 = np.argmin(abs(similarity_mat.times - t_SC1).values)

            # define the parameters to compute the correlation
            dt = 0.025  # 25ms
            max_tau = 2  # s
            n_max_tau = int(max_tau/dt)  # amount of dots in the matrix to take

            # get the amount of times we can check the data on
            n_times_short = n_times - n_max_tau - idx_SC1

            # get the average across bootstraps
            similarity_cv = similarity_mat.mean(dim='bootstraps').values

            # 1) Compute the auto correlation of the signal
            # initialize
            AC_time = np.zeros((n_times_short, 2 * n_max_tau))
            n_surrogates = 1000  # number of phase-randomized surrogates
            AC_null = np.zeros((n_times_short, 2 * n_max_tau, n_surrogates))

            # iterate on the matrix
            for i in range(n_times_short):
                # get the slice of t - tau, t + tau
                x_t = similarity_cv[idx_SC1 + i, idx_SC1 + i - n_max_tau: idx_SC1 + i + n_max_tau]

                # z-score the signal
                z_t = (x_t - x_t.mean())/np.std(x_t)

                # correlate with itself
                ac_full = np.correlate(z_t, z_t, mode='full')

                # extract only lags from -n_max_tau to +n_max_tau
                mid = len(ac_full) // 2

                ac = ac_full[mid - n_max_tau: mid + n_max_tau]
                ac_norm = ac / len(z_t)

                # store
                AC_time[i] = ac_norm

                # --- block-shuffled surrogates ---
                for s in range(n_surrogates):
                    # change the size of the block in every shuffle iteration
                    block_size_t = np.random.uniform(0.6, 1.0)  # seconds
                    block_size = int(block_size_t / dt)  # convert to samples
                    block_size = max(1, block_size)  # safety guard

                    # compute the shuffle blocks
                    x_shuff = block_shuffle_fixed_length(x_t, block_size)
                    z_shuff = (x_shuff - x_shuff.mean()) / np.std(x_shuff)
                    ac_full_shuff = np.correlate(z_shuff, z_shuff, mode='full')
                    AC_null[i, :, s] = ac_full_shuff[mid - n_max_tau: mid + n_max_tau] / len(
                        z_shuff)

            time_array = np.linspace(-max_tau, max_tau, 2*n_max_tau)
            AC_global = AC_time.mean(axis=0)
            AC_null_global = AC_null.mean(axis=0)  # shape: (2*n_max_tau, n_shuffles)

            AC_null_global_ub = np.percentile(AC_null_global, 95, axis=-1)
            AC_null_global_lb = np.percentile(AC_null_global, 5, axis=-1)

            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            ax.plot(time_array, AC_global, '.-', lw=.5, c='k')
            # ax.plot(time_array, AC_null_global_ub, '.-', color='red')
            ax.fill_between(time_array, AC_null_global_lb, AC_null_global_ub, color='#C00000',
                            alpha=.3,  edgecolor='none')
            [ax.axvline(peak, linestyle='--', color='lightgray', linewidth=.35)
             for peak in time_array[find_peaks(AC_global)[0]]]
            ax.set_title(f'{SESSION}-{probe} - Autocorrelation of similarity matrix - shuffle in '
                         f'blocks uniformly sampled from 0.6-1s')
            sns.despine(fig)
            fig.savefig(os.path.join(PATH_FIGURES,
                                     f'{SESSION}-{probe}-autocorr_similarity-maxlag_{max_tau}.svg'),
                        dpi=300, bbox_inches='tight')

            plt.close('all')

            # store the lags and the amplitudes
            idx_peak = find_peaks(AC_global)[0][0]

            # check significance
            if AC_global[idx_peak] > AC_null_global_ub[idx_peak]:
                lag = time_array[idx_peak]
                idx_per = AC_global[idx_peak]

                # store in a dictionary
                site_info = {'site': f'{SESSION}-{probe}', 'monkey': SESSION[0],
                             'period': lag, 'idx_per': idx_per}

                # concat
                cosine_periodicity.append(site_info)

    cosine_periodicity_df = pd.DataFrame(cosine_periodicity)
    # save the dataframe
    cosine_periodicity_df.to_csv(os.path.join(PATH_FIGURES, 'cosine_sim_periodicity.csv'),
                                 index=False)

    # plot the distribution
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    sns.histplot(abs(cosine_periodicity_df.period), ax=ax, color="gray", edgecolor="white")
    ax.set_xlabel('Estimated period [s]')
    ax.set_title(f'Distribution across significant periodicity sites '
                 f'({len(cosine_periodicity_df)})')
    ax.set_xlim([0.5, 1.5])
    sns.despine(fig)
    fig.savefig(os.path.join(PATH_FIGURES, 'Cosine_similarity_periodicity_distribution.svg'),
                dpi=300)
