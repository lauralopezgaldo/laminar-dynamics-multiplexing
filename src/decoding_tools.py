"""
This file contains functions linked to the different decoding algorithms used throughout the
scripts. Both for data preparation, for iterations across windows and reorganization of the data.
"""

import re
import numpy as np
import xarray as xr
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from frites.workflow import WfMi
from frites.dataset import DatasetEphy


def split_train_test(mua_xr, condition, seed=0, percent_train=0.5):
    """
    Split the MUA data into train and test sets.
    :param mua_xr: DataArray with the MUA data
    :param condition: condition to split the data
    :param seed: seed number for the class separation (fixed for reproducibility)
    :param percent_train: percent of the data used for training
    ----
    :return: mua_xr_train: DataArray with the MUA used for training the model
             mua_xr_test: DataArray with the MUA used for testing the model
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


def run_dimensionality_reduction_analysis_crossval(mua_train_arr, mua_test_arr, method_name,
                                                   num_dims,
                                                   behavioral_labels):
    """
    Run the dimensionality reduction analysis on the MUA data.
    :param mua_train_arr: Numpy array with the MUA data, cut in the specific window in which we
    want to train the model (n_channels, n_trials, n_times)
    :param mua_test_arr: Numpy array with the MUA data, cut in the specific window in which we
    want to test the model (n_channels, n_trials, n_times)
    :param method_name: string with the name of the method to use for the dimensionality reduction
    either 'PCA' or 'LDA'
    :param num_dims: int with the number of dimensions to reduce the data to
    :param behavioral_labels: array with the shape (n_trials) with the labels we are training the
    model on
    ----
    :return:
    """
    # 1) Organize the data to fit the models
    # prepare the training data
    mua_flat_train = mua_train_arr.reshape(mua_train_arr.shape[0], -1)
    mua_flat_test = mua_test_arr.reshape(mua_train_arr.shape[0], -1)

    # get the n_times to reshape back correctly
    n_channels, n_trials_test, n_times_epoch = mua_test_arr.shape

    # 2) Fit and transform the data - build the space from the single trials
    if method_name == 'PCA':
        clf = PCA(n_components=num_dims)

        # fit and project the PCA
        clf.fit(mua_flat_train.T)
        proj_single_trials_flat = \
            clf.transform(mua_flat_test.T)

    elif method_name == 'LDA':
        clf = LDA(n_components=num_dims)
        # get the labels matching the single trials
        labels = behavioral_labels.repeat(n_times_epoch)

        # fit and project the LDA
        clf.fit(mua_flat_train.T, labels)

        # project the test data
        proj_single_trials_flat = clf.transform(mua_flat_test.T)

    else:
        raise ValueError('Method not recognized')

    # 3) Reshape the single trials and average them inside the time-window
    # reshape the single trials
    proj_single_trials = proj_single_trials_flat.reshape(n_trials_test, n_times_epoch, num_dims)

    # average the single trials inside the window
    proj_single_trials_avg = proj_single_trials.mean(axis=1)  # average over the time dimension

    # 4) Correct the sign - if PCA - just by making it positive (the max value)
    mappers = np.zeros((n_channels, num_dims))
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

    return mappers, proj_single_trials_avg


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
                flipped_weights[:, i_time + 1, i_dim - 1] = \
                    -flipped_weights[:, i_time + 1, i_dim - 1]

    return flipped_weights, flipped_projected_data


def get_lda_weight_vector(mua_xr, event_name, behavioral_parameter):
    """
    Get the LDA weight vector for the specific trial type and event.
    :return:
    """
    # get the events
    events_onset_ = mua_xr.task_events_onset
    events_names_ = re.split('-', mua_xr.task_events_labels)

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
    num_channels, num_times, num_dims = weights_xr.shape
    # Normalize the reference vector
    reference_vector = reference_vec / np.linalg.norm(reference_vec, axis=0)

    # Initialize a copy of the weights to modify
    aligned_weights = weights_xr.values.copy().swapaxes(0, 1)
    aligned_projections = projected_data_xr.values.copy()

    for i_dim in range(num_dims):
        # Loop over time points
        for t in range(num_times):
            current_vector = aligned_weights[t, :, i_dim]
            dot_product_ref = np.dot(current_vector, reference_vector[:, i_dim])

            # Flip if the dot product is negative
            if dot_product_ref < 0:
                aligned_weights[t, :, i_dim] *= -1
                aligned_projections[t, :, i_dim] *= -1

    # Convert the modified arrays back to xarray DataArrays
    aligned_weights = xr.DataArray(
        aligned_weights.swapaxes(0, 1),
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
        vector = weights_xr.sel(dimensions=f'dim-{i_dim}').values.swapaxes(0, 1)
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
