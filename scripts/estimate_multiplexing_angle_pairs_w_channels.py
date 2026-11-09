"""
Compute the angle between the multiplexing LDs (trial type and movement) and store all the data
together.
"""
import os
import warnings
import numpy as np
import re
import h5py
import pandas as pd
import xarray as xr
from scipy.linalg import subspace_angles
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from frites import io


def get_lda_weight_vector(mua_xr, event_name, trials_train, behavioral_parameter,
                          events_onset_mua, events_names_mua):
    """
    Get the LDA weight vector for the specific trial type and event.
    :return:
    """
    # get the time of the event
    t_cue_event = events_onset_mua[events_names_mua.index(event_name)]

    # get the start and end of the space
    if event_name == 'GO':
        start_space = t_cue_event - .15  # before the cue
        end_space = t_cue_event + 0  # start of the cue
    else:
        start_space = t_cue_event + .15  # half way on the cue
        end_space = t_cue_event + .3  # end of the cue

    # get the mua in the time-window, for the specific trial_type_group
    mua_train_windows = \
        mua_xr.sel(times=slice(start_space, end_space),
                   trial_type=mua_xr.trial_type
                   .isin(trials_train)).coarsen(times=50, boundary="trim").mean()

    # stack the trials
    mua_train = mua_train_windows.stack(samples=('trial_type', 'times'))

    # compute the LDA dimensions
    clf = LDA()
    labels = mua_train[behavioral_parameter].values
    clf.fit(mua_train.values.T, labels)

    # get the weights
    n_dims = clf.scalings_.shape[1]
    dimension_labels = [f'dim-{i}' for i in range(1, n_dims + 1)]
    weights = clf.scalings_

    # create the x_array for the weights
    weights_xr = xr.DataArray(weights, dims=('layers', 'dimensions'),
                              coords={'dimensions': dimension_labels,
                                      'layers': mua_xr.layers.values},
                              attrs=mua_xr.attrs)
    # add some info to the xarray
    weights_xr['event'] = f'{event_name}-{behavioral_parameter}'

    return weights_xr


if __name__ == "__main__":
    # --------------------------------------------------
    # 2. REAL DATA - estimate the angles
    # --------------------------------------------------
    # 1) Describe paths and method
    method = 'LDA'  # 'PCA' or 'LDA'

    current_path = os.getcwd()
    if current_path.startswith('C:'):
        server = 'W:'  # local w VPN
    else:
        server = '/envau/work'  # niolon

    PATH_EXCEL = server + \
        '/comco/lopez.l/Electrophysiology/ephy_laminar_MUA/Results/Full_trial/' \
        'plots_similarity_corrected/'

    PATH_FIGURES = server + \
        '/comco/lopez.l/Electrophysiology/ephy_laminar_MUA/Results/Full_trial/' \
        'plots_revisions/angles/'

    # 2) load the excel file from the excel path
    excel_file = os.path.join(PATH_EXCEL, 'Mua_check_channels.xlsx')
    df_mua_info = pd.read_excel(excel_file)

    # 3) define the session and probe to run the analysis
    SESSIONS = ['Mo180411001', 'Mo180412002', 'Mo180626003', 'Mo180627003', 'Mo180619002',
                'Mo180622002', 'Mo180704003', 'Mo180418002', 'Mo180419003', 'Mo180426004',
                'Mo180601001',
                'Mo180523002', 'Mo180705002', 'Mo180706002', 'Mo180710002',
                'Mo180711004', 'Mo180712006', 't150303002', 't150319003', 't150423002',
                't150320002', 't150123001', 't150128001', 't150204001', 't150205004', 't150716001',
                'Mo180615002-Mo180615005', 't150327002-t150327003']
    # all possible probes
    probes = [1, 2]

    # 4) Initialize to store the angles across sites
    all_angles_summary = []

    # 5) iterate on the sessions and probes
    for SESSION in SESSIONS:
        PATH = server + \
               '/comco/lopez.l/Electrophysiology/ephy_laminar_MUA/Results/Preprocessed_data/' + \
               SESSION + '/'

        for probe in probes:
            # Open in this case the single trials aligned to SEL because the SC2 ones are not full
            file_name_mua = [i for i in os.listdir(PATH) if os.path.isfile(os.path.join(PATH, i))
                             and
                             f'{SESSION}' in i and '.nc' in i and 'MUA-' in i and 'Sel' in i and
                             'SC' not in i and f'probe_{probe}' in i]
            if len(file_name_mua) == 0:
                io.logger.info(f'No MUA file found for {SESSION} probe {probe}')
                continue
            else:
                file_name_mua = file_name_mua[0]

            # 6) open the file
            mua_site = xr.load_dataarray(os.path.join(PATH, file_name_mua))
            io.logger.info(f'MUA for {SESSION}-{probe} loaded successfully.')

            # 7) Remove the bad channels
            df_mua_site = df_mua_info[np.logical_and(df_mua_info['session'] == SESSION,
                                                     df_mua_info['probe'] == probe)]

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
                    mua_site.loc[dict(channels=ch)] = np.nan

                # Remove any channel that has any NaN
                has_nan = mua_site.isnull().any(dim=('trial_type', 'times'))
                mua_site = mua_site.sel(channels=~has_nan)

            # get the number of channels per site
            n_channels_site = len(mua_site.channels)

            # 8) Get the events where to check the LDA subspace
            # get the events
            events_onset = mua_site.task_events_onset
            events_names = re.split('-', mua_site.task_events_labels)

            # concatenate all the weights
            weights_scs = []

            # 9) Get the vectors associated with the CUE
            behavior_train = 'trial_type'
            events_cue = ['SC1', 'SC2', 'SC3']
            trial_type_group_train = [1, 2, 3]

            for event in events_cue:
                weights_lda = \
                    get_lda_weight_vector(mua_xr=mua_site, event_name=event,
                                          trials_train=trial_type_group_train,
                                          behavioral_parameter=behavior_train,
                                          events_onset_mua=events_onset,
                                          events_names_mua=events_names)

                weights_scs.append(weights_lda)

            # 10) Get the vectors associated with the DIRECTION at SCs
            behavior_train = 'mvt_dir'
            events_mvt_cues = ['SC1', 'SC2', 'SC3']

            for event in events_mvt_cues:
                # trial type for train - only the match
                trial_type_group_train = int(event[-1])

                weights_lda = \
                    get_lda_weight_vector(mua_xr=mua_site, event_name=event,
                                          trials_train=trial_type_group_train,
                                          behavioral_parameter=behavior_train,
                                          events_onset_mua=events_onset,
                                          events_names_mua=events_names)

                weights_scs.append(weights_lda)

            # concatenate all the weights
            weights_scs_xr = xr.concat(weights_scs, dim='event')

            # 11) Compute the angle pairs - with no repetitions
            for event_i, event_j in combinations(weights_scs_xr.event.values, 2):
                subspace_i = weights_scs_xr.sel(event=event_i)[:, 0:2]
                subspace_j = weights_scs_xr.sel(event=event_j)[:, 0:2]

                # Principal angles
                angles_rad = subspace_angles(subspace_i, subspace_j)
                angles_deg = np.degrees(angles_rad)

                mean_angle = angles_deg.mean()

                # Recover behavioral parameters
                behavior_i = str(event_i).split('-')[-1]
                behavior_j = str(event_j).split('-')[-1]

                pair_type = (
                    'same_behavior'
                    if behavior_i == behavior_j
                    else 'different_behavior'
                )

                # Store one row per pair
                all_angles_summary.append({
                    'session': SESSION, 'probe': probe, 'n_channels': n_channels_site,
                    'event_i': str(event_i), 'event_j': str(event_j), 'behavior_i': behavior_i,
                    'behavior_j': behavior_j, 'pair_type': pair_type, 'angle_deg': mean_angle})

    # 12) Convert to dataframe
    df_angles = pd.DataFrame(all_angles_summary)
    df_angles.to_csv(os.path.join(PATH_FIGURES, 'subspace_angles_summary.csv'), index=False)
