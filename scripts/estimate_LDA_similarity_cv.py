import os
import warnings
import numpy as np
import re
import h5py
# from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from frites import io
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

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

    # Find the index where time is closest to 0
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


def run_dimensionality_reduction_analysis_crossval(mua_xr_train, mua_xr_test, method_name, num_dims,
                                                   behavioral_condition='trial_type'):
    """
    Run the dimensionality reduction analysis on the MUA data.
    :param mua_xr_train: DataArray with the MUA data, cut in the specific window in which we want to
    train the model
    :param mua_xr_test: DataArray with the MUA data, cut in the specific window in which we want to
    test the model
    :param method_name: string with the name of the method to use for the dimensionality reduction
    either 'PCA' or 'LDA'
    :param num_dims: int with the number of dimensions to reduce the data to
    :param behavioral_condition: string with the name of the behavioral condition to use for the
    dimensionality reduction regression. Default is 'trial_type' (blue, green, pink)
    ----
    :return:
    """
    # 1) Define the labels for the dimensions, layers and flatten the single trials inside the
    # time window
    dimension_labels = [f'dim-{i}' for i in range(1, num_dims + 1)]

    # make sure that the behavioral condition is on the star dimensions
    mua_xr_train = mua_xr_train.swap_dims({'trial_type': behavioral_condition})
    mua_xr_test = mua_xr_test.swap_dims({'trial_type': behavioral_condition})

    # prepare the training data
    mua_flat_train = mua_xr_train.stack(samples=(behavioral_condition, 'times'))
    mua_flat_test = mua_xr_test.stack(samples=(behavioral_condition, 'times'))
    layers = mua_xr_train.layers.values

    # get the n_times to reshape back correctly
    n_times_epoch = len(mua_xr_test.times)
    n_trials_test = len(mua_xr_test.trial_type)

    # 2) Fit and transform the data - build the space from the single trials
    if method_name == 'PCA':
        clf = PCA(n_components=num_dims)

        # fit and project the PCA
        clf.fit(mua_flat_train.values.T)
        proj_single_trials_flat = \
            clf.transform(mua_flat_test.values.T)

    elif method_name == 'LDA':
        clf = LDA(n_components=num_dims)
        # get the labels matching the single trials
        labels = mua_flat_train[behavioral_condition].values

        # fit and project the LDA
        clf.fit(mua_flat_train.values.T, labels)

        # project the test data
        proj_single_trials_flat = clf.transform(mua_flat_test.values.T)

    else:
        raise ValueError('Method not recognized')

    # 3) Reshape the single trials and average them inside the time-window
    # reshape the single trials
    proj_single_trials = proj_single_trials_flat.reshape(n_trials_test, n_times_epoch, num_dims)

    # average the single trials inside the window
    proj_single_trials_avg = proj_single_trials.mean(axis=1)  # average over the time

    # 4) Correct the sign - if PCA - just by making it positive (the max value)
    mappers = np.zeros((len(layers), num_dims))
    for i_dim in range(num_dims):
        if method_name == 'PCA':
            # change the sign if the maximum absolute value entry is negative
            idx_max = np.argmax(np.abs(clf.components_[i_dim, :]))
            if clf.components_[i_dim, idx_max] < 0:
                proj_single_trials_avg[:, i_dim] = \
                    -proj_single_trials_avg[:, i_dim]
                mappers[:, i_dim] = -clf.components_[i_dim, :]
            else:
                mappers[:, i_dim] = clf.components_[i_dim, :]

        elif method_name == 'LDA':
            mappers[:, i_dim] = clf.scalings_[:, i_dim]

    # 5) Build the x_arrays with both data
    # Create the x_array with the projections
    proj_single_trials_avg_xr = xr.DataArray(proj_single_trials_avg, dims=('trial_type',
                                                                           'dimensions'),
                                             coords={'dimensions': dimension_labels,
                                                     'trial_type':
                                                         mua_xr_test['trial_type'].values},
                                             attrs=mua_xr_test.attrs)

    # add the movement direction to the xarray
    proj_single_trials_avg_xr = proj_single_trials_avg_xr.assign_coords(
        mvt_dir=('trial_type', mua_xr_test.mvt_dir.values),
        unamb_mask=('trial_type', mua_xr_test.unamb_mask.values),
        block=('trial_type', mua_xr_test.block.values),
        t_number=('trial_type', mua_xr_test.t_number.values)
    )

    # create the x_array for the weights
    weights_xr = xr.DataArray(mappers, dims=('layers', 'dimensions'),
                              coords={'dimensions': dimension_labels,
                                      'layers': layers},
                              attrs=mua_xr_test.attrs)

    return weights_xr, proj_single_trials_avg_xr


def run_dimensionality_reduction_analysis_crossval_fast(mua_xr_train, method_name,
                                                        num_dims,
                                                        behavioral_condition='trial_type'):
    """
    Run the dimensionality reduction analysis on the MUA data.
    :param mua_xr_train: DataArray with the MUA data, cut in the specific window in which we want to
    train the model
    :param method_name: string with the name of the method to use for the dimensionality reduction
    either 'PCA' or 'LDA'
    :param num_dims: int with the number of dimensions to reduce the data to
    :param behavioral_condition: string with the name of the behavioral condition to use for the
    dimensionality reduction regression. Default is 'trial_type' (blue, green, pink)
    ----
    :return:
    """
    # 1) Define the labels for the dimensions, layers and flatten the single trials inside the
    # time window
    dimension_labels = [f'dim-{i}' for i in range(1, num_dims + 1)]

    # make sure that the behavioral condition is on the star dimensions
    mua_xr_train = mua_xr_train.swap_dims({'trial_type': behavioral_condition})

    # prepare the training data
    mua_flat_train = mua_xr_train.stack(samples=(behavioral_condition, 'times'))
    layers = mua_xr_train.layers.values

    # 2) Fit and transform the data - build the space from the single trials
    if method_name == 'PCA':
        clf = PCA(n_components=num_dims)
        # fit and project the PCA
        clf.fit(mua_flat_train.values.T)

    elif method_name == 'LDA':
        clf = LDA(n_components=num_dims)
        # get the labels matching the single trials
        labels = mua_flat_train[behavioral_condition].values

        # fit and project the LDA
        clf.fit(mua_flat_train.values.T, labels)

    else:
        raise ValueError('Method not recognized')

    # 4) Correct the sign - if PCA - just by making it positive (the max value)
    mappers = np.zeros((len(layers), num_dims))
    for i_dim in range(num_dims):
        if method_name == 'PCA':
            # change the sign if the maximum absolute value entry is negative
            idx_max = np.argmax(np.abs(clf.components_[i_dim, :]))
            if clf.components_[i_dim, idx_max] < 0:
                mappers[:, i_dim] = -clf.components_[i_dim, :]
            else:
                mappers[:, i_dim] = clf.components_[i_dim, :]

        elif method_name == 'LDA':
            mappers[:, i_dim] = clf.scalings_[:, i_dim]

    return mappers


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


def plot_similarity_matrix(cosine_similarity_mat,
                           times, attributes, title, xlabel=None, ylabel=None):
    """
    Plot (and compute) the similarity matrix in a specific dimension. Just the correlation
    coefficient between the weights at each time point. The result is times x times.
    :param cosine_similarity_mat: the cosine similarity matrix
    :param times: the time points of the weights
    :param attributes: the attributes of the site
    :param title:
    :param xlabel:
    :param ylabel:

    :return:
    """
    figure, axis = plt.subplots(figsize=(7, 7))
    # get the timing of the events
    onset_events = attributes['task_events_onset']
    events_labels = re.split('-', attributes['task_events_labels'])

    similarity_matrix = abs(cosine_similarity_mat)
    v_max = np.percentile(similarity_matrix, 97.5)
    v_min = np.percentile(similarity_matrix, 2.5)

    # plot the weights in the upper one
    cax1 = axis.pcolormesh(times,
                           times,
                           similarity_matrix, vmax=v_max, vmin=v_min,
                           cmap='YlOrBr', shading='auto')
    # x ticks
    axis.set_xticks(onset_events[1:-1], events_labels[1:-1], fontsize=11)
    [axis.axvline(line, color='k', linestyle='--', linewidth=.5) for line in onset_events[1:-1]]
    [axis.axvline(line + .3, color='k', linestyle='--', linewidth=.5)
     for line in onset_events[2:-2]]
    # y ticks
    axis.set_yticks(onset_events[1:-1], events_labels[1:-1], fontsize=11)
    [axis.axhline(line, color='k', linestyle='--', linewidth=.5) for line in onset_events[1:-1]]
    [axis.axhline(line + .3, color='k', linestyle='--', linewidth=.5)
     for line in onset_events[2:-2]]
    axis.set_xlim([times[0] - .1,
                   times[-1] - .1])
    axis.set_ylim([times[0] - .1,
                   times[-1] - .1])
    cbar1 = figure.colorbar(cax1, ax=axis, orientation="horizontal", pad=0.15, shrink=.3)
    cbar1.set_label('cosine similarity', fontsize=12)
    axis.set_aspect('equal')
    axis.set_xlabel(xlabel, fontsize=12)
    axis.set_ylabel(ylabel, fontsize=12)
    axis.set_title(title, fontsize=14)
    return figure


if __name__ == "__main__":
    # define the sessions to run the analysis
    # ALL_LAMINAR = ['Mo180411001', 'Mo180412002', 'Mo180626003', 'Mo180627003', 'Mo180619002',
    #                'Mo180622002', 'Mo180704003', 'Mo180418002', 'Mo180419003', 'Mo180426004',
    #                'Mo180601001',
    #                'Mo180523002', 'Mo180705002', 'Mo180706002', 'Mo180710002',
    #                'Mo180711004', 'Mo180712006', 't150303002', 't150319003', 't150423002',
    #                't150430002', 't150320002',
    #                't140924003', 't140930001', 't141001001', 't141008001', 't141010003',
    #                't150122001', 't150123001', 't150128001', 't150204001', 't150205004',
    #                't150716001',
    #                'Mo180615002-Mo180615005', 't150327002-t150327003',
    #                't150520003-t150520005', 't150702002-t150702001']

    ALL_LAMINAR = ['Mo180411001', 'Mo180412002', 'Mo180626003', 'Mo180627003', 'Mo180619002',
                   'Mo180622002', 'Mo180704003', 'Mo180418002', 'Mo180419003', 'Mo180426004',
                   'Mo180601001',
                   'Mo180523002', 'Mo180705002', 'Mo180706002', 'Mo180710002',
                   'Mo180711004', 'Mo180712006', 't150303002', 't150319003', 't150423002',
                   't150320002',
                   't150123001', 't150128001', 't150204001',
                   't150205004',
                   't150716001',
                   'Mo180615002-Mo180615005', 't150327002-t150327003']

    # all possible probes
    probes = [1, 2]
    # define the method to run
    method = 'LDA'  # 'PCA' or 'LDA'
    behavior = 'mvt_dir'  # 'trial_type' or 'mvt_dir'
    trial_type = 3  # trial type to use to compute the mvt_dir behavior

    # specific parameters
    window_size = 0.2  # 150ms
    t_step = 0.025  # 25ms
    n_bts = 50  # number of bootstrap iterations

    # set the paths
    current_path = os.getcwd()

    if current_path.startswith('C:'):
        server = 'W:'  # local w VPN
    else:
        server = '/envau/work'  # niolon

    # path of the figures
    PATH_FIGURES = server + \
        '/comco/lopez.l/Electrophysiology/ephy_laminar_MUA/Results/Full_trial/' \
        'plots_new_sites/supplementary/'

    # path of the figures
    PATH_EXCEL = server + \
        '/comco/lopez.l/Electrophysiology/ephy_laminar_MUA/Results/Full_trial/' \
        'plots_similarity_corrected/'

    # load the excel file from the PATH_Figures
    excel_file = os.path.join(PATH_EXCEL, f'Mua_check_channels.xlsx')

    # load the excel file
    df_mua_info = pd.read_excel(excel_file)

    # loop over the sessions
    for SESSION in ALL_LAMINAR:
        # set the path of the data and get the name of the files
        PATH = server + \
            '/comco/lopez.l/Electrophysiology/ephy_laminar_MUA/Results/Preprocessed_data/' + \
            SESSION + '/'
        PATH_DATA = server + \
            '/comco/lopez.l/Electrophysiology/ephy_laminar_MUA/Results/Full_trial/' \
            'data_paper/' + SESSION + '/'

        # create the directory if it does not exist
        if not os.path.exists(PATH_DATA):
            os.makedirs(PATH_DATA)

        # iterate on the probes
        for probe in probes:
            # Open in this case the single trials aligned to SEL because the SC2 ones are not full
            file_name_mua = [i for i in os.listdir(PATH) if os.path.isfile(os.path.join(PATH, i))
                             and
                             f'{SESSION}' in i and '.nc' in i and 'MUA-' in i and 'Sel' in i and
                             'SC' not in i and f'probe_{probe}' in i]
            if len(file_name_mua) == 0:
                io.logger.info(f'No MUA file found for probe {probe}')
                continue
            else:
                file_name_mua = file_name_mua[0]

            # open the file
            mua_site = xr.load_dataarray(os.path.join(PATH, file_name_mua))
            io.logger.info(f'Loaded file: {file_name_mua}')

            # if there are not all trial types just skip the session if behavior is trial_type
            if (len(np.unique(mua_site.trial_type.values)) < 3) & (behavior == 'trial_type'):
                continue

            # 1) Cut the MUA from -1.5 seconds before SEL until 5.7 seconds after SEL (full trial)
            mua_site_shorter, n_times = \
                cut_mua_by_markers_fixed_samples(mua_xr=mua_site, event_start='SEL',
                                                 t_extra=(-1.1, + 5.7), time_resolution=0.001)

            # 2) Select the channels of interest: remove bad channels
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
                    mua_site_shorter.loc[dict(channels=ch)] = np.nan

                # Remove any channel that has any NaN
                has_nan = mua_site_shorter.isnull().any(dim=('trial_type', 'times'))
                mua_site_shorter = mua_site_shorter.sel(channels=~has_nan)

            # 3) Compute the similarity matrix for the LDA weights of the MUA (with a specific
            # behavioral variable)
            all_similarity = []

            # 4) Get the parameters for the time-windows
            # convert the time into samples based on the time-steps of the data
            dt = mua_site_shorter.times.values[1] - mua_site_shorter.times.values[0]
            n_times_window = int(np.ceil(window_size / dt))
            n_step = int(np.ceil(t_step / dt))
            # generate the starting indices for the time-windows
            idx_t_windows_start = np.arange(0, n_times - n_times_window, n_step)
            n_windows = len(idx_t_windows_start)

            n_dims = 1
            # iterate on the bootstrap iterations - to get different data splits
            for bt in range(n_bts):
                # store all the projected data
                all_projected_data = []
                all_weight_evolution = []

                # 2) Sub-select half of the trials to train and half to test
                if behavior == 'trial_type':
                    mua_train, _ = split_train_test(mua_xr=mua_site_shorter,
                                                    condition=behavior,
                                                    seed=bt, percent_train=0.5)
                elif behavior == 'mvt_dir':
                    mua_train, _ = \
                        split_train_test(mua_xr=mua_site_shorter
                                         .sel(trial_type=trial_type),
                                         condition=behavior,
                                         seed=bt, percent_train=0.7)
                else:
                    raise ValueError('behavior should be either trial_type or mvt_dir')

                # 4) Iterate on the possible time-windows
                all_middle_times = []  # to store the middle times of the windows
                all_weights = []  # to store the weights of the PCA/LDA

                for i_window, idx_t_window_start in enumerate(idx_t_windows_start):
                    # get the MUA in the time window
                    mua_epoch_train = mua_train \
                        .isel(times=np.arange(idx_t_window_start, idx_t_window_start +
                                              n_times_window))

                    # get all the times of the middle of the window
                    all_middle_times.append(
                        np.round(mua_epoch_train.times.values[int(n_times_window / 2)],
                                 2))

                    # Run the PCA/LDA
                    weights_epoch = \
                        run_dimensionality_reduction_analysis_crossval_fast(
                            mua_xr_train=mua_epoch_train,
                            method_name=method, num_dims=n_dims, behavioral_condition=behavior)

                    # append the results to a list
                    all_weights.append(weights_epoch)

                # squeeze the weights to remove the singleton dimensions
                weight_evolution = np.squeeze(all_weights, axis=-1)
                # convert times into an array
                all_middle_times = np.array(all_middle_times)

                # compute (and plot) the similarity matrix
                similarity = compute_cosine_similarity(weight_evolution, weight_evolution)
                # append all the values - but take the absolute value! otherwise we will destroy
                # the average
                all_similarity.append(abs(similarity))

            similarity_avg = np.mean(all_similarity, axis=0)
            figure_similarity = \
                plot_similarity_matrix(cosine_similarity_mat=similarity_avg,
                                       times=all_middle_times, attributes=mua_site_shorter.attrs,
                                       title=f'Similarity-{SESSION}-{probe}-{method}-{trial_type}',
                                       xlabel=None,
                                       ylabel=None)

            figure_similarity.savefig(os.path.join(PATH_FIGURES,
                                                   f'{SESSION}_probe_{probe}_similarity_{method}_'
                                                   f'{behavior}-trial_type-{trial_type}.png'),
                                      dpi=300, bbox_inches='tight')

            # save the similarity matrix
            similarity_xr = xr.DataArray(np.array(all_similarity),
                                         dims=['bootstraps', 'times', 'times'],
                                         coords={'bootstraps': range(n_bts),
                                                 'times': all_middle_times},
                                         attrs=mua_site_shorter.attrs)

            similarity_xr.to_netcdf(os.path.join(PATH_DATA,
                                                 f'{SESSION}_probe_{probe}_similarity_{method}_'
                                                 f'{behavior}_trial_type_{trial_type}_cv.nc'))

            plt.close('all')
            io.logger.info(f'Finished processing {SESSION} probe {probe} with {method} method!')
