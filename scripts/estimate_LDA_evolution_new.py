import os
import warnings
import numpy as np
import re
import h5py
import matplotlib.gridspec as gridspec
import pandas as pd
import xarray as xr
from frites.workflow import WfMi
from frites.dataset import DatasetEphy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from frites import io
from frites.stats.stats_nonparam import confidence_interval
import seaborn as sns
from scipy.signal import savgol_filter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from collections import Counter

# to remove the spam of pandas FutureWarning with iteritems
warnings.simplefilter(action='ignore', category=FutureWarning)


def cut_mua_by_markers(mua_xr, events_alignment, t_extra):
    """
    Cut the MUA data by the markers of the task.
    :param mua_xr: DataArray with the MUA data
    :param events_alignment: list with the names of the events to cut the data
    :param t_extra: list with the extra time to add to the events in seconds
    ----
    :return: mua_shorter: DataArray with the MUA data cut by the events
             n_times_shorter: int with the number of time-points in the new MUA
    """
    # get the events
    events_onset = mua_xr.task_events_onset
    events_names = re.split('-', mua_xr.task_events_labels)

    # get the start and end of the trial
    t_start = events_onset[events_names.index(events_alignment[0])] + t_extra[0]  # first event
    t_end = events_onset[events_names.index(events_alignment[1])] + t_extra[1]  # second event

    # cut the mua in the t_start t_mvt
    mua_shorter = mua_xr.sel(times=slice(t_start, t_end))
    n_times_shorter = len(mua_shorter.times)

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
    dot_product_norm = np.dot(vector_a, vector_b.T) / \
        (np.linalg.norm(vector_a, axis=1) * np.linalg.norm(vector_b, axis=1))
    return dot_product_norm


def plot_similarity_matrix(weights, cosine_similarity, title, xlabel=None, ylabel=None):
    """
    Plot (and compute) the similarity matrix in a specific dimension. Just the correlation
    coefficient between the weights at each time point. The result is times x times.
    :param weights:
    :param cosine_similarity:
    :param title:
    :param xlabel:
    :param ylabel:

    :return:
    """
    figure, axis = plt.subplots(figsize=(7, 7))
    # get the timing of the events
    onset_events = weights.task_events_onset
    events_labels = re.split('-', weights.task_events_labels)

    similarity_matrix = abs(cosine_similarity)
    v_max = np.percentile(similarity_matrix, 97.5)
    v_min = np.percentile(similarity_matrix, 2.5)

    # plot the weights in the upper one
    cax1 = axis.pcolormesh(weights.times.values,
                           weights.times.values,
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
    axis.set_xlim([weights.times.values[0] - .1,
                   weights.times.values[-1] - .1])
    axis.set_ylim([weights.times.values[0] - .1,
                   weights.times.values[-1] - .1])
    cbar1 = figure.colorbar(cax1, ax=axis, orientation="horizontal", pad=0.15, shrink=.3)
    cbar1.set_label('cosine similarity', fontsize=12)
    axis.set_aspect('equal')
    axis.set_xlabel(xlabel, fontsize=12)
    axis.set_ylabel(ylabel, fontsize=12)
    axis.set_title(title, fontsize=14)
    return figure


def plot_shannon_entropy(projected_mua_xr, axis, method=None, trials_color=None):
    """
    Plots the Shannon entropy of the PCA components in time.
    :param projected_mua_xr: projected_mua_xr with the projected data
    :param axis: axis to plot the entropy
    :param method: method used to project the data ('LDA', 'PCA'), it will define color.
    :param trials_color: color of the trials to plot
    :return:
    """
    if trials_color == 'b':
        color_line = 'tab:blue'
    elif trials_color == 'g':
        color_line = 'tab:green'
    elif trials_color == 'p':
        color_line = 'pink'
    elif trials_color == 'LDA':
        color_line = '#da635d'
    elif trials_color == 'PCA':
        color_line = '#b1938b'
    else:
        color_line = 'k'
    # get the timing of the events
    onset_events = projected_mua_xr.task_events_onset
    events_labels = re.split('-', projected_mua_xr.task_events_labels)
    axis.plot(projected_mua_xr.times, projected_mua_xr['entropy_dim-1'], color=color_line,
              linewidth=2.5, label=method, linestyle='--')
    [axis.axvline(line, color='lightgray', linestyle='--', lw=1) for line in onset_events[2:-1]]
    [axis.axvline(line + .3, color='lightgray', linestyle='--', lw=1) for line in
     onset_events[2:-2]]
    axis.set_xticks(onset_events[1:-1], events_labels[1:-1], fontsize=14)
    axis.set_xlim([projected_mua_xr.times[0] - .1, projected_mua_xr.times[-1] - .1])
    axis.set_ylabel('Shannon Entropy %', fontsize=12)
    axis.set_xlim([onset_events[1] - .1, onset_events[-1] - .1])
    axis.set_ylim([0, 1])


def plot_weight_evolution(coefficients_evolution, i_dimension):
    """
    :param coefficients_evolution:
    :param i_dimension:
    :param axis:
    :param figure:
    :return:
    """
    figure, axis = plt.subplots(figsize=(15, 7))
    # set the vmax and vmin in the colorbar
    v_min = -np.percentile(abs(coefficients_evolution.sel(dimensions=f'dim-{i_dimension+1}')), 98)
    v_max = -v_min
    # get the timing of the events
    onset_events = coefficients_evolution.task_events_onset
    events_labels = re.split('-', coefficients_evolution.task_events_labels)

    # plot the weights in the upper one
    cax1 = axis.pcolormesh(coefficients_evolution.times.values,
                           np.arange(coefficients_evolution.shape[1])[::-1],
                           coefficients_evolution.sel(dimensions=f'dim-{i_dimension+1}')[:, :].T,
                           cmap='coolwarm', shading='auto', vmax=v_max, vmin=v_min)
    axis.set_yticks(np.arange(coefficients_evolution.shape[1])[::-1],
                    coefficients_evolution.layers.values)
    axis.set_xticks(onset_events[1:-1], events_labels[1:-1], fontsize=16)
    [axis.axvline(line, color='white', linestyle='--') for line in onset_events[1:-1]]
    [axis.axvline(line + .3, color='white', linestyle='--') for line in onset_events[2:-2]]
    cbar1 = figure.colorbar(cax1, ax=axis, orientation="horizontal", pad=0.09, shrink=.3)
    cbar1.set_label('coefficient value', fontsize=12)
    axis.set_title(f'Weights evolution in dimension {i_dimension+1}', fontsize=14)
    sns.despine(figure)
    figure.tight_layout()

    return figure


def correct_consecutive_sign_flips(weights_xr, projected_data_xr, num_dims, behavioral_condition):
    """
    Flips the sign when the difference between two points is bigger than the difference between
    the same two points flipping the sign of the second. It computes this in every dimension, by
    using the average projection per condition.
    :param weights_xr:
    :param projected_data_xr:
    :param num_dims:
    :param behavioral_condition:
    ----
    :return:
    """
    flipped_projected_data = projected_data_xr.copy()
    flipped_weights = weights_xr.copy()
    for i_dim in range(1, num_dims + 1):
        # iterate on the times
        for i_time in range(len(flipped_projected_data.times) - 1):
            projections_per_condition = \
                flipped_projected_data.sel(
                    dimensions=f'dim-{i_dim}').groupby(behavioral_condition).mean(
                    dim='trial_type')

            # get the difference between two consecutive points in time per condition
            x_t = projections_per_condition.isel(times=i_time).values
            x_t_plus_1 = projections_per_condition.isel(times=i_time + 1).values

            # difference between Xt and Xt+1 vs Xt and -Xt+1
            diff = x_t - x_t_plus_1
            diff_shifted = x_t - (-x_t_plus_1)
            bool_diff = abs(diff) > abs(diff_shifted)

            # if all values of diff are bigger than diff_shifted
            if np.sum(bool_diff) > 1:
                # change sign
                flipped_projected_data[i_time + 1, :, i_dim - 1] = \
                    -flipped_projected_data[i_time + 1, :, i_dim - 1]
                flipped_weights[i_time + 1, :, i_dim - 1] = \
                    -flipped_weights[i_time + 1, :, i_dim - 1]

    return flipped_weights, flipped_projected_data


def get_lda_weight_vector(mua_xr, event_name, behavioral_parameter):
    """
    Get the LDA weight vector for the specific trial type and event.
    :return:
    """
    # get the events
    events_onset_ = mua_xr.task_events_onset
    events_names_ = re.split('-', mua_site.task_events_labels)

    # get the time of the event
    t_cue_event = events_onset_[events_names_.index(event_name)]

    # get the start and end of the space
    if event_name == 'GO':
        start_space = t_cue_event - .15  # before the cue
        end_space = t_cue_event + 0  # start of the cue
    else:
        start_space = t_cue_event + .15  # half way on the cue
        end_space = t_cue_event + .3  # end of the cue

    # get the mua in the time-window, for the specific trial_type_group
    mua_train_windows = \
        mua_xr.sel(times=slice(start_space, end_space)).coarsen(times=50, boundary="trim").mean()

    # stack the trials
    mua_train_event = mua_train_windows.stack(samples=('trial_type', 'times'))

    # compute the LDA dimensions
    clf_event = LDA()
    labels_ = mua_train_event[behavioral_parameter].values
    clf_event.fit(mua_train_event.values.T, labels_)

    # get the weights vector
    weights_vector = clf_event.scalings_

    return weights_vector


def correct_sign_with_reference(weights_xr, projected_data_xr, reference_vec):
    """
    Flips the sign of the weights if the dot product between the weights and the reference vector
    is negative. The reference vector is the average on specific time-points depending on the
    behavioral condition.
    :param weights_xr:
    :param projected_data_xr:
    :param reference_vec: The reference vector to align the weights with.
    :return:
    """
    num_times, num_channels, num_dims = weights_xr.shape
    # Normalize the reference vector
    reference_vector = reference_vec / np.linalg.norm(reference_vec, axis=0)

    # Initialize a copy of the weights to modify
    aligned_weights = weights_xr.values.copy()
    aligned_projections = projected_data_xr.values.copy()

    for i_dim in range(num_dims):
        # Loop over time points
        for t in range(num_times):
            current_vector = aligned_weights[t, :, i_dim]
            dot_product_ref = np.dot(current_vector, reference_vector[:, i_dim])
            dot_product_inv = np.dot(-current_vector, reference_vector[:, i_dim])
            
            # Flip if the dot product is negative
            if dot_product_ref < 0:
                aligned_weights[t, :, i_dim] *= -1
                aligned_projections[t, :, i_dim] *= -1

    # Convert the modified arrays back to xarray DataArrays
    aligned_weights = xr.DataArray(
        aligned_weights,
        coords=weights_xr.coords,
        dims=weights_xr.dims,
        attrs=weights_xr.attrs
    )
    aligned_projections = xr.DataArray(
        aligned_projections,
        coords=projected_data_xr.coords,
        dims=projected_data_xr.dims,
        attrs=projected_data_xr.attrs
    )
    return aligned_weights, aligned_projections


def plot_mua_per_trial_type_ci(mua_xr):
    """
    Plots the average MUA per trial type and the confidence interval on the single trials with a
    more transparent color.
    :param mua_xr: xarray with the MUA data
    :return:
    """
    figure, axis = plt.subplots(figsize=(15, 3))
    # get the timing of the events
    onset_events = mua_xr.task_events_onset
    events_labels = re.split('-', mua_xr.task_events_labels)

    colors = ['tab:blue', 'tab:green', 'pink']
    conditions = [1, 2, 3]

    for condition, color in zip(conditions, colors):
        if condition not in mua_xr.trial_type:
            pass
        else:
            # plot the PCs - average and CI on single trials
            axis.plot(mua_xr.times,
                      mua_xr.sel(trial_type=condition).mean(dim='trial_type'), linewidth=2,
                      color=color, label=f'{condition}')
            # get the confidence intervals for the smoothed data
            li, ui = \
                confidence_interval(mua_xr.sel(trial_type=condition),
                                    axis=1)[0]

            # plot the first component of the PCA
            axis.fill_between(mua_xr.times, li, ui, color=color, alpha=0.3)

    [axis.axvline(line, color='lightgray', linestyle='--', lw=1) for line in onset_events[1:-1]]
    [axis.axvline(line + .3, color='lightgray', linestyle='--', lw=1) for line in
     onset_events[2:-2]]
    axis.set_xticks(onset_events[1:-1], events_labels[1:-1], fontsize=14)
    axis.set_xlim([mua_xr.times[0] - .1, mua_xr.times[-1] - .1])
    axis.set_ylabel('Projected MUA', fontsize=14)
    sns.despine(figure)
    return figure


def plot_mua_per_mvt_dir_ci(mua_xr):
    """
    Plots the average MUA per movement dirction and the confidence interval on the single trials
    with a
    more transparent color.
    :param mua_xr: xarray with the MUA data
    :return:
    """
    figure, axis = plt.subplots(figsize=(15, 3))
    # get the timing of the events
    onset_events = mua_xr.task_events_onset
    events_labels = re.split('-', mua_xr.task_events_labels)

    colors = ['#393D3F', '#A882DD', '#F4D35E', '#EE964B']  # for the mvt directions
    conditions = [1, 2, 3, 4]

    for condition, color in zip(conditions, colors):
        if condition not in mua_xr.mvt_dir:
            pass
        else:
            # plot the PCs - average and CI on single trials
            axis.plot(mua_xr.times,
                      mua_xr.sel(trial_type=mua_xr.mvt_dir == condition).mean(dim='trial_type'),
                      linewidth=2,
                      color=color, label=f'{condition}')
            # get the confidence intervals for the smoothed data
            li, ui = \
                confidence_interval(mua_xr.sel(trial_type=mua_xr.mvt_dir == condition),
                                    axis=1)[0]

            # plot the first component of the PCA
            axis.fill_between(mua_xr.times, li, ui, color=color, alpha=0.3)

    [axis.axvline(line, color='lightgray', linestyle='--', lw=1) for line in onset_events[1:-1]]
    [axis.axvline(line + .3, color='lightgray', linestyle='--', lw=1) for line in
     onset_events[2:-2]]
    axis.set_xticks(onset_events[1:-1], events_labels[1:-1], fontsize=14)
    axis.set_xlim([mua_xr.times[0] - .1, mua_xr.times[-1] - .1])
    axis.set_ylabel('Projected MUA', fontsize=14)
    sns.despine(figure)
    return figure


def compute_shannon_entropy_xr(weights_xr, num_dims):
    """
    Shannon entropy measures the "spread" of the vector components. Normalize the vector and treat
    its squared components as probabilities:
    H = - sum(p_i * log(p_i))
    Low Entropy: Localized vector.
    High Entropy: Distributed vector.
    :param weights_xr: xarray with the weights of the PCA/LDA
    :param num_dims: scalar with the number of dimensions in the weights_xr
    ----
    :return: weights_xr_entropy: xarray with the entropy of the weights in the last dimension
    being entropy the value of the entropy for that vector, float. Bounded by 0 and log(N), being N
    the number of components in the vector.
    """
    weights_xr_entropy = weights_xr.copy()
    for i_dim in range(1, num_dims + 1):
        # get the values
        vector = weights_xr.sel(dimensions=f'dim-{i_dim}').values
        squared = vector**2
        num_features = vector.shape[1]
        # because now we have a vector of weights, we need to make it sum one in the dimension of
        # the channels/depths/layers
        probabilities = squared.T / np.sum(squared, axis=1)
        # probabilities = probabilities[probabilities > 0]  # Avoid log(0)
        max_entropy = np.log(num_features)
        entropy_abs = -np.sum(probabilities * np.log(probabilities), axis=0)
        entropy_rel = entropy_abs / max_entropy

        # save it in the xarray
        label_entropy = f'entropy_dim-{i_dim}'
        weights_xr_entropy = weights_xr_entropy.assign_coords(
            **{
                label_entropy: ('times', entropy_rel)
            })

    return weights_xr_entropy


def compute_mutual_information(projected_data_xr, num_dims, behavioral_condition='trial_type'):
    """
    Compute the mutual information between the MUA and the selected labels.
    :return:
    """
    # compute the MI
    projected_data_xr_mi = projected_data_xr.copy()

    for dim_i in range(1, num_dims+1):
        # mutual information type ('cd' = continuous / discret)
        mi_type = 'cd'

        # define the workflow
        wf = WfMi(mi_type=mi_type, inference='ffx', verbose=False)

        # define the dataset
        dataset = DatasetEphy([projected_data_xr.sel(dimensions=f'dim-{dim_i}')
                               .values.T[:, np.newaxis, :]],
                              y=projected_data_xr[behavioral_condition].values,
                              times=projected_data_xr.sel(dimensions=f'dim-{dim_i}').
                              times.values)

        # compute the mutual information
        stats = 'cluster'
        mi, p_values = wf.fit(dataset,
                              mcp=stats, n_perm=500, cluster_th=None,
                              cluster_alpha=0.05, n_jobs=1,
                              random_state=1)

        # save the values in the x_array
        # labels for the coordinates
        label_mi = f'mi_dim-{dim_i}'
        label_p_val = f'p_val_dim-{dim_i}'

        # add the mi and p_values to the projected data
        projected_data_xr_mi = projected_data_xr_mi.assign_coords(
            **{
                label_mi: ('times', mi.values[:, 0]),
                label_p_val: ('times', p_values.values[:, 0])
            })

    return projected_data_xr_mi


def plot_significant_mi(mua_xr, axis, behavioral_condition, trials_color=None):
    """
    Plots the significant MI with a black line. The points with non-significant MI will not appear.
    :param mua_xr: mutual information values
    :param axis: axis to plot the data
    :param behavioral_condition: condition to plot the data
    :param trials_color: color of the group of trials
    ---
    :return:
    """
    if trials_color == 'b':
        color_line = 'tab:blue'
        color_name = 'blue (SC1)'
    elif trials_color == 'g':
        color_line = 'tab:green'
        color_name = 'green (SC2)'
    elif trials_color == 'p':
        color_line = 'pink'
        color_name = 'pink (SC3)'
    elif trials_color == 'LDA':
        color_line = '#da635d'
        color_name = 'LDA'
    elif trials_color == 'PCA':
        color_line = '#b1938b'
        color_name = 'PCA'
    else:
        color_line = 'k'
        color_name = 'all'

    if behavioral_condition == 'trial_type':
        # there are 3 possible trial types
        max_mi = np.log2(3)
    elif behavioral_condition == 'mvt_dir':
        # there are 4 possible directions
        max_mi = np.log2(4)
    else:
        raise ValueError(f'Behavioral condition {behavioral_condition} not recognized.')

    # get the events
    # get the timing of the events
    onset_events = mua_xr.task_events_onset
    events_labels = re.split('-', mua_xr.task_events_labels)

    # significant mi
    significant_mi = np.where(mua_xr['p_val_dim-1'] < 0.05, mua_xr['mi_dim-1'], np.nan)

    axis.plot(mua_xr.times, mua_xr['mi_dim-1']/max_mi, color=color_line, alpha=.15,
              linestyle='-', linewidth=2)
    axis.plot(mua_xr.times, significant_mi/max_mi, color=color_line, alpha=1, linewidth=2,
              label=color_name)
    [axis.axvline(line, color='lightgray', linestyle='--', lw=1) for line in onset_events[2:-1]]
    [axis.axvline(line + .3, color='lightgray', linestyle='--', lw=1) for line in
     onset_events[2:-2]]
    axis.set_xticks(onset_events[1:-1], events_labels[1:-1], fontsize=14)
    axis.set_xlim([mua_xr.times[0] - .1, mua_xr.times[-1] - .1])
    axis.set_ylabel('normalized MI')
    axis.set_ylim(-.05, .65)
    axis.spines['bottom'].set_visible(False)


if __name__ == "__main__":
    # define the sessions to run the analysis
    ALL_LAMINAR = ['Mo180411001', 'Mo180412002', 'Mo180626003', 'Mo180627003', 'Mo180619002',
                   'Mo180622002', 'Mo180704003', 'Mo180418002', 'Mo180419003', 'Mo180426004',
                   'Mo180601001',
                   'Mo180523002', 'Mo180705002', 'Mo180706002', 'Mo180710002',
                   'Mo180711004', 'Mo180712006', 't150303002', 't150319003', 't150423002',
                   't150430002', 't150320002',
                   't140924003', 't140930001', 't141001001', 't141008001', 't141010003',
                   't150122001', 't150123001', 't150128001', 't150204001', 't150205004',
                   't150716001',
                   'Mo180615002-Mo180615005', 't150327002-t150327003',
                   't150520003-t150520005', 't150702002-t150702001']

    # all possible probes
    probes = [1, 2]
    # define the method to run
    method = 'LDA'  # 'PCA' or 'LDA'
    behavior = 'trial_type'  # 'trial_type' or 'mvt_dir'
    trial_type = 2  # trial type to use to compute the mvt_dir behavior
    trial_type_labels = ['blue', 'green', 'pink']

    # specific parameters
    window_size = 0.2  # 150ms
    t_step = 0.025  # 25ms

    # loop over the sessions
    for SESSION in ALL_LAMINAR:
        # set the path of the data and get the name of the files
        # check where are we running the code
        current_path = os.getcwd()

        if current_path.startswith('C:'):
            server = 'W:'  # local w VPN
        else:
            server = '/envau/work'  # niolon

        PATH = server + \
            '/comco/lopez.l/Electrophysiology/ephy_laminar_MUA/Results/Preprocessed_data/' + \
            SESSION + '/'
        PATH_DATA = server + \
            '/comco/lopez.l/Electrophysiology/ephy_laminar_MUA/Results/Full_trial/' \
            'data_paper/' + SESSION + '/'

        # create the directory if it does not exist
        if not os.path.exists(PATH_DATA):
            os.makedirs(PATH_DATA)

        PATH_FIGURES = server + \
            '/comco/lopez.l/Electrophysiology/ephy_laminar_MUA/Results/Full_trial/' \
            'plots_similarity_corrected/'

        # load the excel file from the PATH_Figures
        excel_file = os.path.join(PATH_FIGURES, f'Mua_check_channels.xlsx')

        # load the excel file
        df_mua_info = pd.read_excel(excel_file)

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
            # if there are not all trial types just skip the session
            if (len(np.unique(mua_site.trial_type.values)) < 3) & (behavior == 'trial_type'):
                continue

            # 1) Cut the MUA from TOUCH to MVT + 200 ms
            mua_site_shorter, n_times = \
                cut_mua_by_markers(mua_xr=mua_site, events_alignment=['touch', 'MVT'],
                                   t_extra=[0, 0.2])

            # 2) Select the channels of interest
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

            # all_similarity = []
            # for seed in range(15):
            seed = 0

            # store all the projected data
            all_projected_data = []
            all_weight_evolution = []

            # 2) Sub-select half of the trials to train and half to test
            if behavior == 'trial_type':
                mua_train, mua_test = split_train_test(mua_xr=mua_site_shorter,
                                                       condition=behavior,
                                                       seed=seed, percent_train=0.5)
            elif behavior == 'mvt_dir':
                mua_train, mua_test = split_train_test(mua_xr=mua_site_shorter
                                                       .sel(trial_type=trial_type),
                                                       condition=behavior,
                                                       seed=seed, percent_train=0.7)
            else:
                raise ValueError('behavior should be either trial_type or mvt_dir')

            # 3) Iterate on the time windows
            # convert the time into samples based on the time-steps of the data
            dt = mua_site.times.values[1] - mua_site.times.values[0]
            n_times_window = int(np.ceil(window_size / dt))
            n_step = int(np.ceil(t_step / dt))
            # generate the starting indices for the time-windows
            idx_t_windows_start = np.arange(0, n_times - n_times_window, n_step)
            n_windows = len(idx_t_windows_start)

            n_dims = 2

            # 4) Iterate on the possible time-windows
            all_middle_times = []  # to store the middle times of the windows
            all_weights = []  # to store the weights of the PCA/LDA
            all_projections = []  # to store the projections of the data

            for i_window, idx_t_window_start in enumerate(idx_t_windows_start):
                # get the MUA in the time window
                mua_epoch_train = mua_train \
                    .isel(times=np.arange(idx_t_window_start, idx_t_window_start +
                                          n_times_window))
                # get the MUA for testing
                mua_epoch_test = mua_test.isel(times=np.arange(idx_t_window_start,
                                                               idx_t_window_start +
                                                               n_times_window))

                # get all the times of the middle of the window
                all_middle_times.append(
                    np.round(mua_epoch_train.times.values[int(n_times_window / 2)],
                             2))

                # Run the PCA/LDA
                weights_epoch, projections_epoch = \
                    run_dimensionality_reduction_analysis_crossval(
                        mua_xr_train=mua_epoch_train, mua_xr_test=mua_epoch_test,
                        method_name=method, num_dims=n_dims, behavioral_condition=behavior)

                # append the results to a list
                all_weights.append(weights_epoch)
                all_projections.append(projections_epoch)

                # concatenate the x-array data on the time-dimension
                weight_evolution = xr.concat(all_weights, dim='times')
                weight_evolution['times'] = all_middle_times

                # complete the x-array with the depths and channels
                weight_evolution = weight_evolution. \
                    assign_coords(depths=('layers', mua_site_shorter.depths.values),
                                  channels=('layers', mua_site_shorter.channels.values))

                projection_evolution = xr.concat(all_projections, dim='times')
                projection_evolution['times'] = all_middle_times

                # 5) Plot the results
                # flip the signs - when using LDA
                # first correct the consecutive points that shift abruptly the sign
                if method == 'LDA':
                    new_weight_evolution, new_projection_evolution = \
                        correct_consecutive_sign_flips(weights_xr=weight_evolution,
                                                       projected_data_xr=projection_evolution,
                                                       num_dims=n_dims,
                                                       behavioral_condition=behavior)
                    # # get the reference vector
                    # if behavior == 'mvt_dir':
                    #     event_name_ref = 'GO'
                    # if behavior == 'trial_type':
                    #     event_name_ref = 'SEL'
                    # else:
                    #     raise ValueError('behavior should be either trial_type or mvt_dir')
                    #
                    # reference_vector_lda = get_lda_weight_vector(mua_xr=mua_train,
                    #                                              event_name=event_name_ref,
                    #                                              behavioral_parameter=behavior)
                    # new_weight_evolution, new_projection_evolution = \
                    #     correct_sign_with_reference(weights_xr=weight_evolution,
                    #                                 projected_data_xr=projection_evolution,
                    #                                 reference_vec=reference_vector_lda)
                else:
                    new_weight_evolution = weight_evolution
                    new_projection_evolution = projection_evolution

                # plot the similarity matrix
                similarity = compute_cosine_similarity(new_weight_evolution.sel(dimensions='dim-1'),
                                                       new_weight_evolution.sel(dimensions='dim-1'))


            # figure_similarity = \
            #     plot_similarity_matrix(weights=new_weight_evolution.sel(dimensions='dim-1'),
            #                            cosine_similarity=similarity,
            #                            title=f'Similarity-{SESSION}-{probe}-{method}',
            #                            xlabel=None,
            #                            ylabel=None)

            # figure_similarity.savefig(os.path.join(PATH_FIGURES,
            #                                        f'{SESSION}_probe_{probe}_similarity_{method}_'
            #                                        f'{behavior}.png'),
            #                           dpi=300, bbox_inches='tight')

            plt.close('all')
            io.logger.info(f'Finished processing {SESSION} probe {probe} with {method} method!')

            # # plot_weight_evolution(coefficients_evolution=new_weight_evolution, i_dimension=0)
            #

            #
            # # plot the evolution of the projections
            # if behavior == 'mvt_dir':
            #     figure_conditions_projection = \
            #         plot_mua_per_mvt_dir_ci(new_projection_evolution.sel(dimensions='dim-1'))
            #
            #     # figure_conditions_projection.savefig(os.path.join(
            #     #     PATH_FIGURES, f'{SESSION}_projections_evolution_mvt_'
            #     #                   f'{method}_ttype_{trial_type}.png'),
            #     #                                      dpi=300,
            #     #                                      bbox_inches='tight')
            # elif behavior == 'trial_type':
            #     figure_conditions_projection = \
            #         plot_mua_per_trial_type_ci(new_projection_evolution.sel(dimensions='dim-1'))
            #
            #     # figure_conditions_projection.savefig(os.path.join(
            #     # PATH_FIGURES,
            #     #                                                   f'{SESSION}_projections_evolution_'
            #     #                                                   f'ttype_'
            #     #                                                   f'{method}.png'), dpi=300,
            #     #                                      bbox_inches='tight')
            # compute the mutual information
            projection_evolution_mi = \
                compute_mutual_information(projected_data_xr=new_projection_evolution,
                                           num_dims=n_dims, behavioral_condition=behavior)
            # # plot the mutual information
            # figure_mi, axis_mi = plt.subplots(figsize=(15, 3))
            # plot_significant_mi(mua_xr=projection_evolution_mi,
            #                     axis=axis_mi,
            #                     behavioral_condition=behavior,
            #                     trials_color=method)
            # figure_mi.savefig(os.path.join(PATH_FIGURES,
            #                                f'{SESSION}_probe_{probe}_mutual_information.png'),
            #                   dpi=300, bbox_inches='tight')
            #
            # compute the Shannon entropy
            new_weight_evolution_full = \
                compute_shannon_entropy_xr(weights_xr=new_weight_evolution, num_dims=n_dims)

            projection_evolution_mi['method'] = method
            new_weight_evolution_full['method'] = method

            # 6) Create a dataset with both arrays and save the results
            dataset = xr.Dataset({'weights_evolution': new_weight_evolution_full,
                                  'projected_single_trials': projection_evolution_mi})
            # add attributes
            dataset.attrs = mua_site.attrs
            dataset.attrs['method'] = method
            dataset.attrs['window_size'] = window_size
            dataset.attrs['t_step'] = t_step

            # save the dataset
            dataset.to_netcdf(os.path.join(PATH_DATA,
                                           f'{SESSION}_probe_{probe}_{method}_{behavior}_win_200'
                                           f'.nc'))

            # close the figures
            plt.close('all')
            io.logger.info(f'Finished processing {SESSION} probe {probe} with {method} method!')
