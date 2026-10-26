import os
import warnings
import numpy as np
import re
import h5py
import xarray as xr
from scipy.linalg import svd
from scipy.linalg import subspace_angles
from matplotlib import gridspec
from frites.workflow import WfMi
from frites.dataset import DatasetEphy
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from frites import io
import seaborn as sns

# to remove the spam of pandas FutureWarning with iteritems
warnings.simplefilter(action='ignore', category=FutureWarning)


def plot_pca_weights_summary(components_vector, component_labels, num_dim, axis=None):
    """
    Plots the weights of the PCA components for each of the PCs.
    :param components_vector: vector with the weights of each channel in one PCA dimension
    :param component_labels: names of the components, either channel name, or depth or layer
    :param num_dim: number of the dimension
    :param axis: axis to plot
    :return:
    """
    if axis is None:
        fig, axis = plt.subplots(figsize=(3, 15))
    # reshape to make 2d
    components_2d = components_vector[:, np.newaxis]

    # create the grid
    sns.heatmap(components_2d, cmap='coolwarm', cbar=True, ax=axis, center=0,
                vmax=2, vmin=-2,
                yticklabels=component_labels,
                xticklabels=[f'LDA{num_dim}'], cbar_kws={"orientation": "vertical", "pad": 0.1})
    axis.tick_params(axis="both", labelsize=8)  # Change x and y tick label fontsize


def get_lda_weight_vector(mua_xr, event_name, trials_train, behavioral_parameter):
    """
    Get the LDA weight vector for the specific trial type and event.
    :return:
    """
    # get the time of the event
    t_cue_event = events_onset[events_names.index(event_name)]

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
    method = 'LDA'  # 'PCA' or 'LDA'

    current_path = os.getcwd()

    if current_path.startswith('C:'):
        server = 'W:'  # local w VPN
    else:
        server = '/envau/work'  # niolon

    PATH_FIGURES = server + \
        '/comco/lopez.l/Electrophysiology/ephy_laminar_MUA/Results/Full_trial/' \
        'plots_angles/'

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

    all_angles_summary = []

    # iterate on the sessions
    for SESSION in SESSIONS:
        PATH = server + \
               '/comco/lopez.l/Electrophysiology/ephy_laminar_MUA/Results/Preprocessed_data/' + \
               SESSION + '/'

        # iterate on the probes
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

            # open the file
            mua_site = xr.load_dataarray(os.path.join(PATH, file_name_mua))
            io.logger.info(f'MUA for {SESSION}-{probe} loaded successfully.')

            # get the events
            events_onset = mua_site.task_events_onset
            events_names = re.split('-', mua_site.task_events_labels)

            # concatenate all the weights
            weights_scs = []

            # 1) Get the vectors associated with the CUE
            behavior_train = 'trial_type'
            events_cue = ['SEL', 'SC1', 'SC2', 'SC3']
            trial_type_group_train = [1, 2, 3]

            for event in events_cue:
                weights_lda = \
                    get_lda_weight_vector(mua_xr=mua_site, event_name=event,
                                          trials_train=trial_type_group_train,
                                          behavioral_parameter=behavior_train)

                weights_scs.append(weights_lda)

            # 2) Get the vectors associated with the DIRECTION at SCs
            behavior_train = 'mvt_dir'
            events_mvt_cues = ['SC1', 'SC2', 'SC3']

            for event in events_mvt_cues:
                # trial type for train - only the match
                trial_type_group_train = int(event[-1])

                weights_lda = \
                    get_lda_weight_vector(mua_xr=mua_site, event_name=event,
                                          trials_train=trial_type_group_train,
                                          behavioral_parameter=behavior_train)

                weights_scs.append(weights_lda)

            # 3) Get the vectors associated with the DIRECTION at GO
            behavior_train = 'mvt_dir'
            events_mvt = ['GO']
            trial_type_group_train = [1, 2, 3]

            for event in events_mvt:
                weights_lda = \
                    get_lda_weight_vector(mua_xr=mua_site, event_name=event,
                                          trials_train=trial_type_group_train,
                                          behavioral_parameter=behavior_train)

                weights_scs.append(weights_lda)

            # concatenate all the weights
            weights_scs_xr = xr.concat(weights_scs, dim='event')

            # get the subspace angles
            all_angles = []
            for event_i in weights_scs_xr.event.values:
                for event_j in weights_scs_xr.event.values:
                    # Principal angles
                    angles_rad = subspace_angles(weights_scs_xr.sel(event=event_i)[:, 0:2],
                                                 weights_scs_xr.sel(event=event_j)[:, 0:2])
                    angles_deg = np.degrees(angles_rad)
                    all_angles.append(angles_deg.mean())

            all_angles = np.array(all_angles).reshape(len(weights_scs_xr.event.values),
                                                      len(weights_scs_xr.event.values))
            all_angles[np.diag_indices_from(all_angles)] = np.nan  # Set diagonal to zero

            all_angles_summary.append([all_angles])

    all_angles_summary = np.array(all_angles_summary).squeeze()
    # Define colormap: add gray for NaNs
    cmap = plt.cm.viridis
    cmap.set_bad(color='white')  # Color for NaNs

    plt.figure(figsize=(6, 5))
    im = plt.imshow(all_angles_summary.mean(axis=0), cmap=cmap)

    # Add colorbar
    plt.colorbar(im, label='Mean Principal Angle (°)')

    # Add labels
    labels = ['SEL', 'SC1_c', 'SC2_c', 'SC3_c', 'SC1_m', 'SC2_m', 'SC3_m', 'GO']
    plt.xticks(np.arange(8), labels)
    plt.yticks(np.arange(8), labels)
    plt.title(f'BOTH - subspace angles')
    plt.savefig(os.path.join(PATH_FIGURES, 'subspace_angles_both.png'),
                bbox_inches='tight', dpi=300)
