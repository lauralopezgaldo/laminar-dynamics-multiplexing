"""
This file contains functions linked to the different decoding algorithms used throughout the
scripts. Both for data preparation, for iterations across windows and reorganization of the data.
"""

import numpy as np


def split_train_test(mua_xr, condition, seed=0, percent_train=0.5):
    """
    Split the MUA data into train and test sets.
    :param mua_xr: DataArray with the MUA data
    :param condition: condition to split the data
    :param seed: seed number for the class separation (fixed for reproducibility)
    :param percent_train: percent of the data used for training
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
