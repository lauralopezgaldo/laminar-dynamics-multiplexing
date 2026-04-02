import os
import warnings
import numpy as np
import re
import h5py
import matplotlib.gridspec as gridspec
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
from matplotlib.colors import LinearSegmentedColormap

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
        color_line = 'slategray'
    # get the timing of the events
    onset_events = projected_mua_xr.task_events_onset
    events_labels = re.split('-', projected_mua_xr.task_events_labels)
    axis.plot(projected_mua_xr.times, projected_mua_xr['entropy_dim-1'], color=color_line,
              label=method, marker='o', markersize=2.5)
    axis.plot(projected_mua_xr.times, projected_mua_xr['entropy_dim-1'], color=color_line,
              linewidth=1, label=method)
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
    # v_min = -np.percentile(abs(coefficients_evolution.sel(dimensions=f'dim-{i_dimension+1}')), 98)
    # v_max = -v_min
    # fix it for figure 3
    v_min = -0.44
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


def plot_significant_mi_bar(mua_xr, dimension=1, cmap='viridis'):
    """
    Plots the significant MI with a black line and the non-significant with a dashed line.
    :param mua_xr: projection values with all the coordinates
    :param dimension: the dimension to plot
    :param cmap: colormap to use
    ---
    :return:
    """
    figure, axis = plt.subplots(figsize=(15, 3))
    mua_xr = mua_xr.sel(times=slice(-0.5, 5.7))
    # get the timing of the events
    events_times = mua_xr.task_events_onset
    events_labels = re.split('-', mua_xr.task_events_labels)

    cue_events = ['SEL', 'SC1', 'SC2', 'SC3']

    # Find indices where values match
    indices = [i for i, event in enumerate(events_labels) if event in cue_events]

    # plot the mutual information
    mi_significant = mua_xr[f'sig_mi_dim-{dimension}'].values

    # Define color values (e.g., cosine function)
    color_values = mi_significant

    cmap = axis.imshow(color_values.reshape(1, -1), aspect="auto",
                       extent=[mua_xr.times.min(), mua_xr.times.max(), -1, 1], cmap=cmap, alpha=1,
                       vmax=.4, vmin=0)

    # add colorbar
    cbar = plt.colorbar(cmap, ax=axis, orientation='horizontal', pad=0.5)
    cbar.set_label('MI [bits]')

    axis.set_yticks([])  # Remove y-ticks

    [axis.axvline(line, color='lightgray', linestyle='--') for line in
     events_times]
    [axis.axvline(line + .3, color='lightgray', linestyle='--') for line in
     events_times[indices]]
    axis.set_xticks(events_times, events_labels)
    axis.set_xlim(events_times[1], events_times[-1] + .2)


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

    if behavioral_condition == 'trial_type':
        # there are 3 possible trial types
        max_mi = np.log2(3)
    elif behavioral_condition == 'mvt_dir':
        # there are 4 possible directions
        max_mi = np.log2(4)
    else:
        raise ValueError(f'Behavioral condition {behavioral_condition} not recognized.')

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

        # get the significant mi
        masked_significant_mi = np.where(p_values < 0.05, mi/max_mi, np.nan)

        # save the values in the x_array
        # labels for the coordinates
        label_mi = f'mi_dim-{dim_i}'
        label_p_val = f'p_val_dim-{dim_i}'
        label_sig_mi = f'sig_mi_dim-{dim_i}'

        # add the mi and p_values to the projected data
        projected_data_xr_mi = projected_data_xr_mi.assign_coords(
            **{
                label_mi: ('times', mi.values[:, 0]/max_mi),
                label_p_val: ('times', p_values.values[:, 0]),
                label_sig_mi: ('times', masked_significant_mi[:, 0])
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
    # define the session and probe
    SESSION = 'Mo180412002'
    probe = 1

    # define the method to run
    methods = ['PCA', 'LDA']
    behavior = 'mvt_dir'  # 'trial_type' or 'mvt_dir'
    trial_type = 2  # trial type to use to compute the mvt_dir behavior
    trial_type_labels = ['blue', 'green', 'pink']

    # plot the entropy for the LDA
    figure_sen, axis_sen = plt.subplots(figsize=(15, 3))

    # specific parameters
    window_size = 0.2  # 150ms
    t_step = 0.025  # 25ms

    cmap_LDA = LinearSegmentedColormap.from_list("white_to_LDA", ['white', '#da635d'])
    cmap_PCA = LinearSegmentedColormap.from_list("white_to_PCA", ['white', '#b1938b'])

    PATH = 'data/'

    file_name_mua = [i for i in os.listdir(PATH) if os.path.isfile(os.path.join(PATH, i)) and
                     f'{SESSION}' in i and '.nc' in i and 'MUA-' in i and 'SC' not in i and
                     'Sel' in i and f'probe_{probe}' in i]

    # 0) load the data
    mua_site = xr.load_dataarray(os.path.join(PATH, file_name_mua[0]))

    # 1) Cut the MUA from TOUCH to MVT + 200 ms
    mua_site_shorter, n_times = \
        cut_mua_by_markers(mua_xr=mua_site, events_alignment=['touch', 'MVT'], t_extra=[0, 0.2])

    # store all the projected data
    all_projected_data = []
    all_weight_evolution = []

    # iterate on the methods
    for method in methods:
        # 2) Sub-select half of the trials to train and half to test
        if behavior == 'trial_type':
            mua_train, mua_test = split_train_test(mua_xr=mua_site_shorter,
                                                   condition=behavior,
                                                   seed=0, percent_train=0.5)
        elif behavior == 'mvt_dir':
            mua_train, mua_test = split_train_test(mua_xr=mua_site_shorter
                                                   .sel(trial_type=trial_type),
                                                   condition=behavior,
                                                   seed=0, percent_train=0.5)
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
                                                           idx_t_window_start + n_times_window))

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
            assign_coords(depths=('layers', mua_site.depths.values),
                          channels=('layers', mua_site.channels.values))

        projection_evolution = xr.concat(all_projections, dim='times')
        projection_evolution['times'] = all_middle_times

        # 5) Plot the results
        # flip the signs - when using LDA
        # first correct the consecutive points that shift abruptly the sign
        if method == 'LDA':
            new_weight_evolution, new_projection_evolution = \
                correct_consecutive_sign_flips(weights_xr=weight_evolution,
                                               projected_data_xr=projection_evolution,
                                               num_dims=n_dims, behavioral_condition=behavior)
        else:
            new_weight_evolution = weight_evolution
            new_projection_evolution = projection_evolution

        # plot the evolution of the weights
        figure_laminar_evolution = \
            plot_weight_evolution(coefficients_evolution=new_weight_evolution,
                                  i_dimension=0)

        # figure_laminar_evolution.savefig(os.path.join(PATH_FIGURES,
        #                                               f'{SESSION}_weights_evolution_{method}_'
        #                                               f'{behavior}'
        #                                               f'.png'),
        #                                  dpi=300, bbox_inches='tight')

        # plot the similarity matrix
        similarity = compute_cosine_similarity(new_weight_evolution.sel(dimensions='dim-1'),
                                               new_weight_evolution.sel(dimensions='dim-1'))

        figure_similarity = \
            plot_similarity_matrix(weights=new_weight_evolution.sel(dimensions='dim-1'),
                                   cosine_similarity=similarity,
                                   title=f'Similarity-{SESSION}-{method}', xlabel=None, ylabel=None)
        # figure_similarity.savefig(os.path.join(PATH_FIGURES,
        #                                        f'{SESSION}_similarity_{method}_'
        #                                        f'{behavior}.png'),
        #                           dpi=300, bbox_inches='tight')

        # plot the evolution of the projections
        if behavior == 'mvt_dir':
            figure_conditions_projection = \
                plot_mua_per_mvt_dir_ci(new_projection_evolution.sel(dimensions='dim-1'))

            # figure_conditions_projection.savefig(os.path.join(PATH_FIGURES,
            #                                                   f'{SESSION}_projections_evolution_'
            #                                                   f'mvt_'
            #                                                   f'{method}_ttype_{trial_type}.png'),
            #                                      dpi=300,
            #                                      bbox_inches='tight')
        elif behavior == 'trial_type':
            figure_conditions_projection = \
                plot_mua_per_trial_type_ci(new_projection_evolution.sel(dimensions='dim-1'))

            # figure_conditions_projection.savefig(os.path.join(PATH_FIGURES,
            #                                                   f'{SESSION}_projections_evolution_'
            #                                                   f'ttype_'
            #                                                   f'{method}.png'), dpi=300,
            #                                      bbox_inches='tight')
        # compute the mutual information
        projection_evolution_mi = \
            compute_mutual_information(projected_data_xr=new_projection_evolution,
                                       num_dims=n_dims, behavioral_condition=behavior)

        # compute the Shannon entropy
        new_weight_evolution_full = \
            compute_shannon_entropy_xr(weights_xr=new_weight_evolution, num_dims=n_dims)

        # plot the mutual information
        plot_significant_mi_bar(mua_xr=projection_evolution_mi, dimension=1, cmap='Greys')

        if method == 'LDA':
            if behavior == 'mvt_dir':
                plot_shannon_entropy(projected_mua_xr=new_weight_evolution_full, axis=axis_sen,
                                     method=method,
                                     trials_color='g')
            if behavior == 'trial_type':
                plot_shannon_entropy(projected_mua_xr=new_weight_evolution_full, axis=axis_sen,
                                     method=method,
                                     trials_color='k')
