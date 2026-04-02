"""
This script takes the MUA in each site and computes the LDA dimension that best splits the four
movement directions at the moment of correct target shown. It recomputes the LDA for each of the
cues and checks again the measure. It uses the distance metric as the other script. It checks if
the space that best splits the directions during the SCs is capable of separating the directions
at the pre-GO.
"""

import os
import warnings
import numpy as np
import re
import h5py
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from frites import io
import seaborn as sns
import matplotlib.colors as mcolors
from scipy.signal import savgol_filter

# to remove the spam of pandas FutureWarning with iteritems
warnings.simplefilter(action='ignore', category=FutureWarning)


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


def split_train_test_array(single_trials, labels, seed=0, percent_train=0.5):
    """
    Split the MUA data into train and test sets.
    :param single_trials:
    :param labels:
    :param seed:
    :param percent_train:
    :return:
    """
    # Get unique classes
    classes = np.unique(labels)

    # Determine the minimum number of trials available across all classes
    min_trials_per_class = min([
        np.sum(labels == c)
        for c in classes
    ])

    # Decide how many trials per class to keep in training
    n_keep_per_class = int(np.floor(min_trials_per_class * percent_train))

    # Seed for reproducibility
    np.random.seed(seed)

    # Collect train indices
    idx_trials_to_keep = []

    for c in classes:
        class_indices = np.where(labels == c)[0]
        selected = np.random.choice(class_indices, n_keep_per_class, replace=False)
        idx_trials_to_keep.extend(selected)

    # Convert to array and sort
    idx_trials_to_keep = np.sort(np.array(idx_trials_to_keep))

    # Create exclusion mask
    n_trials = len(labels)
    mask = np.ones(n_trials, dtype=bool)
    mask[idx_trials_to_keep] = False
    idx_trials_to_exclude = np.where(mask)[0]

    # Split data
    mua_train_split = single_trials[idx_trials_to_keep]
    y_train_split = labels[idx_trials_to_keep]
    mua_test_split = single_trials[idx_trials_to_exclude]
    y_test_split = labels[idx_trials_to_exclude]

    return mua_train_split, y_train_split, mua_test_split, y_test_split


def train_model(data_train, train_time, trial_type_train, condition_train, model=LDA()):
    """
    Train the model with the training data. The function returns the trained model baised on the
    training split, and labels. In this case the labels are scrambled so the real link between
    the data and the labels is lost. This is used to get a baseline for the model.
    :param data_train: training data
    :param train_time: time window to train the model, tuple
    :param trial_type_train: trial type to train the model
    :param condition_train: condition to train the model
    :param model: model to train
    :return:
    """
    # get the train and test split
    x_train, y_train = \
        get_training_data(mua=data_train,
                          time_window=train_time,
                          trial_type_train=trial_type_train,
                          condition_train=condition_train,
                          super_sample=True)

    # initialize the model
    model = model
    # fit the model
    model.fit(x_train.values, y_train)
    return model


def get_training_data(mua, time_window, trial_type_train, condition_train, super_sample=True):
    """
    Get the data for training the LDA model: the MUA as samples x features and the labels.
    :param mua: xarray with the MUA data
    :param time_window: tuple with the start and end of the time window
    :param trial_type_train: group of trials used for the train
    :param condition_train: condition to split the data (train the model). If none, there will
    be no labels returned
    :param super_sample: boolean to "supersample" the data, if yes, we create 50ms windows and use
    the mean of the data in each window as a sample
    :return:
    """
    if super_sample:
        # get the mua in the time-window, for the specific trial_type_group, in windows of
        # 50 ms (to have a better representation of the data)
        mua_train_windows = \
            mua.sel(times=slice(time_window[0], time_window[1]),
                    trial_type=mua.trial_type.isin(trial_type_train)) \
            .coarsen(times=50, boundary="trim").mean()

        # stack the trials
        mua_train_flat = mua_train_windows.stack(samples=('trial_type', 'times')).T

    else:
        mua_train_flat = \
            mua.sel(times=slice(time_window[0], time_window[1]),
                    trial_type=mua.trial_type.isin(trial_type_train)).mean(dim='times')

    if condition_train is not None:
        # get the labels of the trials
        labels_train = mua_train_flat[condition_train].values
    else:
        labels_train = None

    return mua_train_flat, labels_train


def get_projected_data_model(data_test, test_time, trial_type_test, condition_test, model):
    """
    Get the projected data on the test set.
    :param data_test: test data
    :param test_time: time window to test the model, tuple
    :param trial_type_test: trial type to test the model
    :param condition_test: condition to test the model
    :param model: model to test - ALREADY TRAINED
    :return:
    """
    # get the data for the test
    x_test, y_test = \
        get_training_data(mua=data_test,
                          time_window=test_time,
                          trial_type_train=trial_type_test,
                          condition_train=condition_test,
                          super_sample=False)

    # predict the labels
    x_test_projected = model.transform(x_test.values)

    return x_test_projected


def get_time_window(mua, event_name, window_duration=.15, alignment='mid_cue'):
    """
    Get the start and end of the time window, aligned to a specific event.
    :param mua: xarray with the MUA data
    :param event_name: name of the event to align
    :param window_duration: duration of the window in seconds
    :param alignment: alignment of the window, can be 'mid_cue', 'start_cue', 'end_cue',
    'pre_cue'
    ----
    :return:
    """
    if alignment == 'start_cue':
        t_zero = 0
    elif alignment == 'mid_cue':
        t_zero = .15
    elif alignment == 'end_cue':
        t_zero = .3
    elif alignment == 'pre_cue':
        t_zero = -.2
    else:
        raise ValueError('alignment should be one of "mid_cue", "start_cue", "end_cue", "pre_cue"')

    # get the events
    events_onset = mua.task_events_onset
    events_names = re.split('-', mua.task_events_labels)

    # get the timing of the events - and the training space
    t_cue_event = events_onset[events_names.index(event_name)]
    start_space = t_cue_event + t_zero
    end_space = t_cue_event + t_zero + window_duration

    return start_space, end_space


def get_centroid_position(single_trials_2d, single_trial_labels):
    """
    Get the centroid position of the trials in the 2D space.
    :param single_trials_2d: array of the shape (n_trials, 2) with the trials in the 2D space
    :param single_trial_labels: array with the labels of the trials
    ----
    :return:
     centroid_position: dictionary with the centroid position for each label
    """
    # get the unique classes
    unique_classes = np.unique(single_trial_labels)

    # Compute centroids for all the classes
    centroids = {label: single_trials_2d[single_trial_labels == label].mean(axis=0)
                 for label in unique_classes}

    return centroids


def get_distance_to_centroid(trials_train, trials_test, labels_train, labels_test):
    """
    Get the distance to the 'train' centroid for each test trial. Compute the distance to the
    correct centroid (matching label) and the distance to the incorrect centroid (non-matching
    label).
    :param trials_train:
    :param trials_test:
    :param labels_train:
    :param labels_test:
    :return:
    """
    # get the centroid position
    centroids_train = \
        get_centroid_position(single_trials_2d=trials_train, single_trial_labels=labels_train)

    # get both the correct distances and the incorrect ones for each single trial
    correct_distances = []
    incorrect_distances = []

    for i in range(len(trials_test)):
        trial = trials_test[i]
        true_label = labels_test[i]

        # Distance to correct centroid
        correct_distance = np.linalg.norm(trial - centroids_train[true_label])

        # Distances to incorrect centroids
        other_labels = [label for label in centroids_train.keys() if label != true_label]
        distances_to_incorrect = [np.linalg.norm(trial - centroids_train[label]) for label in
                                  other_labels]
        avg_incorrect_distance = np.mean(distances_to_incorrect)

        correct_distances.append(correct_distance)
        incorrect_distances.append(avg_incorrect_distance)

    # Convert lists to numpy arrays for easier calculations
    correct_distances = np.array(correct_distances)
    incorrect_distances = np.array(incorrect_distances)

    # Compute the averages
    avg_correct_distance = np.mean(correct_distances)
    avg_incorrect_distance = np.mean(incorrect_distances)

    # compute the difference between the averages
    avg_distance_difference = avg_correct_distance - avg_incorrect_distance

    # Get the single-trial distance ratio
    distance_ratio_single_trial = correct_distances / incorrect_distances
    distance_ratio = np.mean(distance_ratio_single_trial < 1)

    return avg_distance_difference, distance_ratio


if __name__ == "__main__":
    # define the session and probe to run the analysis
    SESSIONS = ['Mo180411001', 'Mo180412002', 'Mo180626003', 'Mo180627003', 'Mo180619002',
                'Mo180622002', 'Mo180704003', 'Mo180418002', 'Mo180419003', 'Mo180426004',
                'Mo180601001',
                'Mo180523002', 'Mo180705002', 'Mo180706002', 'Mo180710002',
                'Mo180711004', 'Mo180712006', 't150303002', 't150319003', 't150423002',
                't150430002', 't150320002',
                't140924003', 't140930001', 't141001001', 't141008001', 't141010003',
                't150122001', 't150123001', 't150128001', 't150204001', 't150205004',
                't150716001',
                'Mo180615002-Mo180615005', 't150327002-t150327003',
                't150702002-t150702001']

    # all possible probes
    probes = [1, 2]

    # parameters for the analysis
    movement_directions = [1, 2, 3, 4]
    behavior_train = 'mvt_dir'
    events_train = ['SC1', 'SC2', 'SC3', 'GO']

    # set the colors
    colors = ['#393D3F', '#A882DD', '#F4D35E', '#EE964B']  # for the mvt directions
    colors_labels = ['UR', 'LR', 'LL', 'UL']

    current_path = os.getcwd()

    if current_path.startswith('C:'):
        server = 'W:'  # local w VPN
    else:
        server = '/envau/work'  # niolon

    PATH_EXCEL = server + \
        '/comco/lopez.l/Electrophysiology/ephy_laminar_MUA/Results/Full_trial/' \
        'plots_similarity_corrected/'

    # load the excel file from the PATH_Figures
    excel_file = os.path.join(PATH_EXCEL, f'Mua_check_channels.xlsx')

    # load the excel file
    df_mua_info = pd.read_excel(excel_file)

    # iterate on the sessions
    for SESSION in SESSIONS:
        PATH = server + \
            '/comco/lopez.l/Electrophysiology/ephy_laminar_MUA/Results/Preprocessed_data/' + \
            SESSION + '/'
        PATH_DATA = server + \
            '/comco/lopez.l/Electrophysiology/ephy_laminar_MUA/Results/Full_trial/' \
            'data/' + SESSION + '/'
        PATH_SAVE = server + \
            '/comco/lopez.l/Electrophysiology/ephy_laminar_MUA/Results/Full_trial/' \
            'data_paper/' + SESSION + '/'
        PATH_FIGURES = server + \
            '/comco/lopez.l/Electrophysiology/ephy_laminar_MUA/Results/Full_trial/' \
            'lda_SCs_distance_plots/'

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

            # 1) Get the MUA and define the time to build the LDA space
            # open the file
            mua_site = xr.load_dataarray(os.path.join(PATH, file_name_mua))

            # 1.1 - Remove the channels that are bad from the excel
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

            # 2) Iterate on the possible events
            cv_accuracy = []
            train_split_fraction = 0.7

            for event_train in events_train:
                if event_train == 'GO':
                    trial_type_correct = [1, 2, 3]

                    # 2.2 - Set the time-window to build the LDA model
                    start_space_train, end_space_train = \
                        get_time_window(mua=mua_site, event_name=event_train, window_duration=.2,
                                        alignment='pre_cue')

                else:
                    trial_type_correct = int(event_train[-1])

                    # 2.2 - Set the time-window to build the LDA model
                    start_space_train, end_space_train = \
                        get_time_window(mua=mua_site, event_name=event_train, window_duration=.3,
                                        alignment='mid_cue')

                for bt in range(100):
                    ##################
                    # BUILD THE SPACE AT SC
                    # 2.2 - Separate the MUA by train-test
                    mua_train, mua_test = \
                        split_train_test(mua_xr=mua_site.sel(trial_type=mua_site.trial_type
                                         .isin(trial_type_correct)),
                                         condition=behavior_train,
                                         seed=bt,
                                         percent_train=train_split_fraction)

                    # 2.3 - TRAIN the model: use MUA for train and the space defined before
                    clf = train_model(data_train=mua_train,
                                      train_time=(start_space_train, end_space_train),
                                      trial_type_train=trial_type_correct,
                                      condition_train=behavior_train,
                                      model=LDA())

                    # TEST AT SC - CORRECT TARGET DIRECTION
                    # 2.4 - PROJECT the test data onto the model: on the same time-window, for the
                    # train set and the test set
                    mua_train_projected = \
                        get_projected_data_model(data_test=mua_train,
                                                 test_time=(start_space_train, end_space_train),
                                                 trial_type_test=trial_type_correct,
                                                 condition_test=behavior_train, model=clf)
                    mua_test_projected = \
                        get_projected_data_model(data_test=mua_test,
                                                 test_time=(start_space_train, end_space_train),
                                                 trial_type_test=trial_type_correct,
                                                 condition_test=behavior_train, model=clf)

                    # 2.5 - Compute the distance to centroid for each trial type - real and shuffled
                    diff_avg_distances, accuracy = \
                        get_distance_to_centroid(trials_train=mua_train_projected,
                                                 trials_test=mua_test_projected,
                                                 labels_train=mua_train[behavior_train].values,
                                                 labels_test=mua_test[behavior_train].values)

                    # get the single trial accuracy
                    cv_accuracy.append({'event_train': event_train,
                                        'event_test': event_train,
                                        'bootstrap': bt,
                                        'accuracy': accuracy,
                                        'diff_distance': diff_avg_distances,
                                        'shuffled': False,
                                        'same_cue': True})

                    # shuffle the labels and compute the distance again
                    for seed_sh in range(10):
                        # shuffle the labels
                        np.random.seed(seed_sh)
                        new_labels_sh = np.copy(mua_test[behavior_train].values)
                        np.random.shuffle(new_labels_sh)
                        # compute the distance
                        diff_avg_distances_sh, accuracy_sh = \
                            get_distance_to_centroid(trials_train=mua_train_projected,
                                                     trials_test=mua_test_projected,
                                                     labels_train=mua_train[behavior_train].values,
                                                     labels_test=new_labels_sh)

                        cv_accuracy.append({'event_train': event_train,
                                            'event_test': event_train,
                                            'bootstrap': bt,
                                            'accuracy': accuracy_sh,
                                            'diff_distance': diff_avg_distances_sh,
                                            'shuffled': True,
                                            'same_cue': True})

                    # Test at PRE-GO on the test trials
                    # get the space at pre-GO
                    start_space_preGO, end_space_preGO = \
                        get_time_window(mua=mua_site, event_name='GO', window_duration=.2,
                                        alignment='pre_cue')

                    # get the mua to test
                    mua_test_pre_GO_projected = \
                        get_projected_data_model(data_test=mua_test,
                                                 test_time=(start_space_preGO, end_space_preGO),
                                                 trial_type_test=[1, 2, 3],
                                                 condition_test=behavior_train, model=clf)

                    # get the distance to the train positions
                    # 2.5 - Compute the distance to centroid for each trial type - real and shuffled
                    diff_avg_distances, accuracy = \
                        get_distance_to_centroid(trials_train=mua_train_projected,
                                                 trials_test=mua_test_pre_GO_projected,
                                                 labels_train=mua_train[behavior_train].values,
                                                 labels_test=mua_test[behavior_train].values)

                    # get the single trial accuracy
                    cv_accuracy.append({'event_train': event_train,
                                        'event_test': 'GO',
                                        'bootstrap': bt,
                                        'accuracy': accuracy,
                                        'diff_distance': diff_avg_distances,
                                        'shuffled': False,
                                        'same_cue': False})

                    # shuffle the labels and compute the distance again
                    for seed_sh in range(10):
                        # shuffle the labels
                        np.random.seed(seed_sh)
                        new_labels_sh = np.copy(mua_test[behavior_train].values)
                        np.random.shuffle(new_labels_sh)
                        # compute the distance
                        diff_avg_distances_sh, accuracy_sh = \
                            get_distance_to_centroid(trials_train=mua_train_projected,
                                                     trials_test=mua_test_pre_GO_projected,
                                                     labels_train=mua_train[behavior_train].values,
                                                     labels_test=new_labels_sh)

                        cv_accuracy.append({'event_train': event_train,
                                            'event_test': 'GO',
                                            'bootstrap': bt,
                                            'accuracy': accuracy_sh,
                                            'diff_distance': diff_avg_distances_sh,
                                            'shuffled': True,
                                            'same_cue': False})

            # convert the accuracy into a dataframe
            cv_accuracy_df = pd.DataFrame(cv_accuracy)

            # save the dataframe
            cv_accuracy_df.to_csv(os.path.join(PATH_SAVE,
                                               f'{SESSION}_{probe}_'
                                               f'cv_accuracy_{behavior_train}_LDA_preparatory'
                                               f'_dir.csv'),
                                  index=False)

            sns.set(context='talk', style='ticks')
            figure, axis = plt.subplots(1, 2, figsize=(10, 5))
            measure = 'diff_distance'
            x_order = ['SC2', 'GO']

            hue_order = [False, True]
            sns.boxplot(
                data=cv_accuracy_df[cv_accuracy_df.event_train == 'SC2'],
                x='event_test', y='diff_distance', hue='shuffled', whis=[5, 95],
                palette={False: '#d0ba98', True: '#b8b6b0'},
                dodge=True, order=x_order,
                hue_order=hue_order,
                showfliers=False,
                fill=True,
                zorder=2,
                widths=0.4, boxprops=dict(alpha=0.75, edgecolor='white'),
                medianprops=dict(color='black'),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'), ax=axis[0]
            )

            sns.stripplot(
                data=cv_accuracy_df[cv_accuracy_df.event_train == 'SC2'],
                x='event_test', y='diff_distance',
                hue='shuffled',
                dodge=True, order=x_order,
                hue_order=hue_order,
                jitter=True,
                palette={False: '#d0ba98', True: '#b8b6b0'},  # Avoid the ValueError
                alpha=.1, linewidth=0.1,
                marker='o', zorder=1, edgecolor='gray', ax=axis[0]
            )
            axis[0].legend_.remove()
            axis[0].invert_yaxis()
            sns.despine(offset=10, trim=True, ax=axis[0])
            axis[0].set_title(f'{measure} for dir'
                              f' (train SC2: {train_split_fraction})')

            x_order = ['GO', 'SC2']

            hue_order = [False, True]
            sns.boxplot(
                data=cv_accuracy_df[cv_accuracy_df.event_test == 'GO'],
                x='event_train', y='diff_distance', hue='shuffled', whis=[5, 95],
                palette={False: '#d0ba98', True: '#b8b6b0'},
                dodge=True, order=x_order,
                hue_order=hue_order,
                showfliers=False,
                fill=True,
                zorder=2,
                widths=0.4, boxprops=dict(alpha=0.75, edgecolor='white'),
                medianprops=dict(color='black'),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'), ax=axis[1]
            )

            sns.stripplot(
                data=cv_accuracy_df[cv_accuracy_df.event_test == 'GO'],
                x='event_train', y='diff_distance',
                hue='shuffled',
                dodge=True, order=x_order,
                hue_order=hue_order,
                jitter=True,
                palette={False: '#d0ba98', True: '#b8b6b0'},  # Avoid the ValueError
                alpha=.1, linewidth=0.1,
                marker='o', zorder=1, edgecolor='gray', ax=axis[1]
            )
            axis[1].legend_.remove()
            axis[1].invert_yaxis()
            sns.despine(offset=10, trim=True, ax=axis[1])
            axis[1].set_title(f'{measure} for dir'
                              f' (test preGO: {train_split_fraction})')
            figure.suptitle(f'{SESSION} - {probe}')
            figure.tight_layout()
            figure.savefig(f'{PATH_FIGURES}/{SESSION}_{probe}_{measure}_dir_train_'
                           f'{train_split_fraction}_SC2_and_preGO.png',
                           dpi=300, bbox_inches='tight')

            plt.close('all')
            io.logger.info(f'Finished plotting {measure} for {probe} - {SESSION}')
