"""
This file contains functions linked to the different decoding algorithms used throughout the
scripts. Both for data preparation, for iterations across windows and reorganization of the data.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


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
