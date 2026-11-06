import os
import warnings
import numpy as np
import re
import h5py
import xarray as xr
from matplotlib import gridspec
from frites.workflow import WfMi
from frites.dataset import DatasetEphy
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from frites import io
from frites.stats.stats_nonparam import confidence_interval
import seaborn as sns
import matplotlib.cm as cm
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


def plot_mua_per_trial_type_ci(mua, axis):
    """
    Plots the average MUA per trial type and the confidence interval on the single trials with a
    more transparent color.
    :param mua: xarray with the MUA data
    :param axis: axis to plot
    :param idx_pc: index of the PC
    :return:
    """
    colors = ['tab:blue', 'tab:green', '#d477b8']
    labels_sel = ['blue', 'green', 'pink']
    conditions_sel = [1, 2, 3]

    # cut the times
    mua = mua.sel(times=slice(-0.5, 5.3))

    # get the events
    events_times = mua.task_events_onset
    events_labels = re.split('-', mua.task_events_labels)

    cue_events = ['SEL', 'SC1', 'SC2', 'SC3']

    # Find indices where values match
    indices = [i for i, event in enumerate(events_labels) if event in cue_events]

    for condition, label, color in zip(conditions_sel, labels_sel, colors):
        if condition not in mua.trial_type:
            pass
        else:
            # plot the PCs - average and CI on single trials
            axis.plot(mua.times,
                      mua.sel(trial_type=condition).mean(dim='trial_type'), linewidth=2,
                      color=color, label=f'{label}')
            # get the confidence intervals for the smoothed data
            li, ui = \
                confidence_interval(mua.sel(trial_type=condition),
                                    axis=0)[0]

            # plot the first component of the PCA
            axis.fill_between(mua.times, li, ui, color=color, alpha=0.3)

    [axis.axvline(line, color='lightgray', linestyle='--') for line in
     events_times]
    [axis.axvline(line + .3, color='lightgray', linestyle='--') for line in
     events_times[indices]]
    axis.set_xticks(events_times, events_labels)
    axis.set_xlim(events_times[1], events_times[-1] + .2)
    axis.set_ylabel('Projected MUA [a.u.]')


def plot_average_2d_projections(mua, model, parameter, colors_groups, colormap_groups,
                                time_window, axis):
    """
    Plot the average trajectories of the MUA data in the 2D LDA space.
    :param mua: projected MUA data
    :param model: trained LDA model
    :param parameter: parameter to plot (behavioral variable used for the lda)
    :param colors_groups: colors for the different groups
    :param colormap_groups: colormap for the different groups
    :param time_window: time window to plot
    :param axis: axis to plot
    ---
    :return:
    """
    if axis is None:
        figure, axis = plt.subplots()

    # get the time window
    t_start, t_end = time_window

    # get cross-condition variance
    cross_cond_var = compute_cross_condition_variance(projected_mua=mua, parameter=parameter,
                                                      model=model)

    # get the time of maximum variance
    t_max_variance = cross_cond_var.sel(times=slice(time_window[0],
                                                    time_window[1])).idxmax(dim='times')

    # plot the trajectories of the averages
    for parameter_i, color_i in \
            zip(np.unique(mua[parameter]), colors_groups):
        axis.scatter(mua.sel(dimensions='dim-1',
                             times=slice(t_start - .2, t_end),
                             trial_type=mua[parameter]
                                        == parameter_i)
                        .mean(dim='trial_type'),
                     mua.sel(dimensions='dim-2',
                             times=slice(t_start - .2, t_end),
                             trial_type=mua[parameter]
                                        == parameter_i)
                        .mean(dim='trial_type'),
                     c=mua.times.sel(times=slice(t_start - .2,
                                     t_end)),
                     cmap=colormap_groups[color_i], s=1)

        axis.scatter(mua.sel(dimensions='dim-1',
                             trial_type=mua[parameter]
                                        == parameter_i, times=t_max_variance)
                     .mean(dim='trial_type'),
                     mua.sel(dimensions='dim-2',
                             trial_type=mua[parameter]
                                        == parameter_i, times=t_max_variance)
                     .mean(dim='trial_type'), s=500, marker='o',
                     c=color_i, alpha=1, label=f'{parameter_i}')


def compute_cross_condition_variance(projected_mua, parameter, model):
    """
    It computes the cross-condition variance for the first two dimensions of the model, given
    the projected data, the model (to normalize by the variance explain) and the parameter to
    check what is the trial-split.
    :param projected_mua: projected single trials, selected the time-window and the trial type
    :param parameter: string with the parameter to split the trials (e.g. 'trial_type' or
    'mvt_dir')
    :param model: trained model to get back the variance explained
    :return:
    """
    # get the cross-condition variance in the train
    cross_cond_var = projected_mua.groupby(parameter).mean().var(dim=parameter)

    # normalize the cross-condition variance
    cross_cond_var_norm = cross_cond_var / \
        cross_cond_var.max(dim='times')

    # weighted mean cross-condition variance for the first two dimensions
    weights_mean_ve = model.explained_variance_ratio_[0:2] / \
        (model.explained_variance_ratio_[0:2].sum())

    # compute the weighted average for the first two dimensions
    cross_cond_var_weighted = \
        (cross_cond_var_norm[:, 0:2] * weights_mean_ve).sum(dim='dimensions')

    return cross_cond_var_weighted


def cut_mua_by_markers(mua_xr, events, t_extra):
    """
    Cut the MUA data by the markers of the task.
    :param mua_xr: DataArray with the MUA data
    :param events: list with the names of the events to cut the data
    :param t_extra: list with the extra time to add to the events in seconds
    ----
    :return: mua_shorter: DataArray with the MUA data cut by the events
             n_times_shorter: int with the number of time-points in the new MUA
    """
    # get the events
    events_onset = mua_xr.task_events_onset
    events_names = re.split('-', mua_xr.task_events_labels)

    # get the start and end of the trial
    t_start = events_onset[events_names.index(events[0])] + t_extra[0]  # start with the first event
    t_end = events_onset[events_names.index(events[1])] + t_extra[1]  # end with the second event

    # cut the mua in the t_start t_mvt
    mua_shorter = mua_xr.sel(times=slice(t_start, t_end))
    n_times_shorter = len(mua_shorter.times)

    return mua_shorter, n_times_shorter


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
                'Mo180615002-Mo180615005', 'Mo180614002-Mo180614006', 't150327002-t150327003',
                't150702002-t150702001']
    SESSIONS = ['Mo180412002']
    probes = [1]

    for SESSION in SESSIONS:
        # define the method to run
        method = 'LDA'  # 'PCA' or 'LDA'
        trial_types = [1, 2, 3]  # corresponding to blue, green and pink
        trial_type_labels = ['blue', 'green', 'pink']

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
            'data/' + SESSION + '/'

        PATH_FIGURES = server + \
            '/comco/lopez.l/Electrophysiology/ephy_laminar_MUA/Results/Full_trial/' \
            'lda_SEL_plots/'

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

            # 1) Cut the MUA from TOUCH to MVT + 200 ms
            mua_site_shorter, n_times = \
                cut_mua_by_markers(mua_xr=mua_site, events=['touch', 'MVT'], t_extra=[0, 0.2])

            # 2) Set the time-window to build the LDAs
            # get the events
            events_onset = mua_site.task_events_onset
            events_names = re.split('-', mua_site.task_events_labels)

            # A) Find the space to train
            event = 'SEL'
            other_cues = [1, 2, 3]
            # trial_type_match = 1
            trial_type_group_train = [1, 2, 3]
            behavior_train = 'trial_type'

            t_cue_event = events_onset[events_names.index(event)]
            start_space = t_cue_event + .15  # half way on the cue
            end_space = t_cue_event + .3  # end of the cue

            # get the mua in the time-window, for the specific trial_type_group
            mua_train_windows = \
                mua_site_shorter.sel(times=slice(start_space, end_space)) \
                                .coarsen(times=50, boundary="trim").mean()

            # stack the trials
            mua_train = mua_train_windows.stack(samples=('trial_type', 'times'))

            # compute the LDA dimensions
            clf = LDA()
            labels_ = mua_train[behavior_train].values

            # train the model
            labels_lda = labels_
            clf.fit(mua_train.values.T, labels_lda)

            # get the weights
            n_dims = clf.scalings_.shape[1]
            dimension_labels = [f'dim-{i}' for i in range(1, n_dims + 1)]
            unique_labels = np.unique(labels_lda)
            weights = clf.scalings_

            # create the x_array for the weights
            weights_xr = xr.DataArray(weights, dims=('layers', 'dimensions'),
                                      coords={'dimensions': dimension_labels,
                                              'layers': mua_site_shorter.layers.values},
                                      attrs=mua_site_shorter.attrs)

            #######################
            # TEST # the behavior to test has been defined before already
            # transform the full-time single trial MUA
            mua_test = mua_site_shorter.stack(samples=('trial_type', 'times'))
            n_trials = len(mua_site_shorter.trial_type)
            mua_site_transformed_flat = clf.transform(mua_test.T)

            # unstack the data
            mua_site_transformed = mua_site_transformed_flat.reshape(n_trials, n_times, n_dims)

            # filter the data
            mua_transformed_filtered = savgol_filter(mua_site_transformed, window_length=251,
                                                     polyorder=5, axis=1)

            # create an x-array with the same dimensions as the original data
            proj_single_trials_smooth = \
                xr.DataArray(mua_transformed_filtered, dims=('trial_type', 'times',
                                                             'dimensions'),
                             coords={'trial_type': mua_site_shorter.trial_type,
                                     'times': mua_site_shorter.times,
                                     'dimensions': dimension_labels},
                             attrs=mua_site_shorter.attrs)

            # add the other coordinates
            proj_single_trials_smooth = proj_single_trials_smooth.assign_coords(
                {key: mua_site_shorter.coords[key] for key in mua_site_shorter.coords if
                 "trial_type" in mua_site_shorter.coords[key].dims})

            # plot the data in the 2D space
            # fig, ax = plt.subplots(1, figsize=(10, 10))
            # plot the correct space
            colors = ['tab:blue', 'tab:green', '#d477b8']  # for the trial type directions
            colors_labels = ['blue', 'green', 'pink']
            # Create a dictionary of colormaps
            custom_cmaps = {color: mcolors.LinearSegmentedColormap.from_list(f"custom_{i}",
                                                                             ["white", color])
                            for i, color in enumerate(colors)}

            # --------------------------------
            # 1) Plot first figure (subplot c)
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.set_title(f'Space built at spatial cues for trial_type')
            ax.set_xlim([-.75, .75])
            ax.set_ylim([-.75, .75])
            # ax.invert_xaxis()
            # ax.invert_yaxis()
            ax.set_axis_off()
            # plot the trajectories of the averages
            for t_type, color_i in \
                    zip(np.unique(mua_site_shorter.trial_type), colors):
                ax.scatter(proj_single_trials_smooth
                           .sel(dimensions='dim-1',
                                times=slice(start_space - .1, end_space),
                                trial_type=proj_single_trials_smooth.trial_type
                                           == t_type)
                           .mean(dim='trial_type'),
                           proj_single_trials_smooth
                           .sel(dimensions='dim-2',
                                times=slice(start_space - .1, end_space),
                                trial_type=proj_single_trials_smooth.trial_type
                                           == t_type)
                           .mean(dim='trial_type'),
                           c=proj_single_trials_smooth.times.sel(
                               times=slice(start_space - .1,
                                           end_space)),
                           cmap=custom_cmaps[color_i], s=15)

            # get the times for the max crosss-condition variance
            cross_cond_var = \
                compute_cross_condition_variance(projected_mua=proj_single_trials_smooth,
                                                 parameter=behavior_train,
                                                 model=clf)

            # get the time of maximum variance
            t_max_variance = cross_cond_var.sel(times=slice(start_space,
                                                            end_space)).idxmax(dim='times')

            # plot at every SC the average of all trials
            for ttype, color_i in zip(trial_types, colors):
                # new times depending on the cues
                t_cue_event_new = events_onset[events_names.index(f'SEL')]
                start_space_new = t_cue_event_new + .15  # half way on the cue
                end_space_new = t_cue_event_new + .3  # end of the cue

                ax.scatter(proj_single_trials_smooth
                           .sel(dimensions='dim-1',
                                times=t_max_variance)
                           .sel(trial_type=ttype)
                           .mean(dim='trial_type'),
                           proj_single_trials_smooth
                           .sel(dimensions='dim-2',
                                times=t_max_variance)
                           .sel(trial_type=ttype)
                           .mean(dim='trial_type'), s=700, marker='o',
                           c=color_i, alpha=1, label=f'SEL-{ttype}')

            # # plot at every SC the average of all trials
            for cue_test, color_i in zip(other_cues, colors):
                # new times depending on the cues
                t_cue_event_new = events_onset[events_names.index(f'SC{cue_test}')]
                start_space_new = t_cue_event_new + .15  # half way on the cue
                end_space_new = t_cue_event_new + .3  # end of the cue

                ax.scatter(proj_single_trials_smooth
                           .sel(dimensions='dim-1',
                                times=slice(start_space_new, end_space_new))
                           .mean(dim=('times', 'trial_type')),
                           proj_single_trials_smooth
                           .sel(dimensions='dim-2',
                                times=slice(start_space_new, end_space_new))
                           .mean(dim=('times', 'trial_type')), s=500, marker='*',
                           c=color_i, alpha=.75,
                           edgecolor=color_i, label=f'train-SC{cue_test}')

            fig.savefig(os.path.join(PATH_FIGURES, f'{SESSION}-{probe}-SEL-subplot_c.svg'))
            plt.close(fig)

            # --------------------------------
            # 2) Plot first figure (subplot d)
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.set_title(f'Space built at spatial cues for trial_type')
            ax.set_xlim([-.75, .75])
            ax.set_ylim([-.75, .75])
            # ax.invert_xaxis()
            # ax.invert_yaxis()
            ax.set_axis_off()
            # plot at every SC the average of all trials
            for ttype, color_i in zip(trial_types, colors):
                # new times depending on the cues
                t_cue_event_new = events_onset[events_names.index(f'SEL')]
                start_space_new = t_cue_event_new + .15  # half way on the cue
                end_space_new = t_cue_event_new + .3  # end of the cue

                ax.scatter(proj_single_trials_smooth
                           .sel(dimensions='dim-1',
                                times=t_max_variance)
                           .sel(trial_type=ttype)
                           .mean(dim='trial_type'),
                           proj_single_trials_smooth
                           .sel(dimensions='dim-2',
                                times=t_max_variance)
                           .sel(trial_type=ttype)
                           .mean(dim='trial_type'), s=700, marker='o',
                           c=color_i, alpha=.5, label=f'SEL-{ttype}')

            # plot at every SC the average of matching trials
            color_valid = '#F4A261'
            all_trials = []
            for cue_test, color_i in zip(other_cues, colors):
                # new times depending on the cues
                t_cue_event_new = events_onset[events_names.index(f'SC{cue_test}')]
                start_space_new = t_cue_event_new + .15  # half way on the cue
                end_space_new = t_cue_event_new + .3  # end of the cue

                # plot the trajectory for the valid trials
                ax.scatter(proj_single_trials_smooth
                           .sel(dimensions='dim-1',
                                times=slice(start_space_new - .1, end_space_new))
                           .sel(trial_type=cue_test)
                           .mean(dim='trial_type'),
                           proj_single_trials_smooth
                           .sel(dimensions='dim-2',
                                times=slice(start_space_new - .1, end_space_new))
                           .sel(trial_type=cue_test)
                           .mean(dim='trial_type'),
                           c=proj_single_trials_smooth.times.sel(
                               times=slice(start_space_new - .1,
                                           end_space_new)),
                           cmap=custom_cmaps[color_i], s=5)

                ax.scatter(proj_single_trials_smooth
                           .sel(dimensions='dim-1',
                                times=slice(start_space_new, end_space_new))
                           .sel(trial_type=cue_test)
                           .mean(dim=('times', 'trial_type')),
                           proj_single_trials_smooth
                           .sel(dimensions='dim-2',
                                times=slice(start_space_new, end_space_new))
                           .sel(trial_type=cue_test)
                           .mean(dim=('times', 'trial_type')), s=500, marker='^', edgecolor='white',
                           c=color_i, alpha=.85, label=f'train-SC{cue_test}')

                valid_tri = proj_single_trials_smooth.sel(times=slice(start_space_new,
                                                                      end_space_new))\
                    .sel(trial_type=cue_test).mean('times')

                valid_tri = valid_tri.assign_coords(valid=('trial_type', [1]*len(valid_tri)),
                                                    event=('trial_type', [cue_test]*len(valid_tri)))
                all_trials.append(valid_tri)

            # plot at every SC the average of non-matching trials
            color_invalid = '#6A0DAD'
            groups_test = [[2, 3], [1, 3], [1, 2]]
            for cue_test, color_i, group_trials in zip(other_cues, colors, groups_test):
                # new times depending on the cues
                t_cue_event_new = events_onset[events_names.index(f'SC{cue_test}')]
                start_space_new = t_cue_event_new + .15  # half way on the cue
                end_space_new = t_cue_event_new + .3  # end of the cue

                # plot the trajectory for the invalid trials
                ax.scatter(proj_single_trials_smooth
                           .sel(dimensions='dim-1',
                                times=slice(start_space_new - .1, end_space_new))
                           .sel(trial_type=proj_single_trials_smooth.trial_type.isin(group_trials))
                           .mean(dim='trial_type'),
                           proj_single_trials_smooth
                           .sel(dimensions='dim-2',
                                times=slice(start_space_new - .1, end_space_new))
                           .sel(trial_type=proj_single_trials_smooth.trial_type.isin(group_trials))
                           .mean(dim='trial_type'),
                           c=proj_single_trials_smooth.times.sel(
                               times=slice(start_space_new - .1,
                                           end_space_new)),
                           cmap=custom_cmaps[color_i], s=5)

                ax.scatter(proj_single_trials_smooth
                           .sel(dimensions='dim-1',
                                times=slice(start_space_new, end_space_new))
                           .sel(trial_type=proj_single_trials_smooth.trial_type.isin(group_trials))
                           .mean(dim=('times', 'trial_type')),
                           proj_single_trials_smooth
                           .sel(dimensions='dim-2',
                                times=slice(start_space_new, end_space_new))
                           .sel(trial_type=proj_single_trials_smooth.trial_type.isin(group_trials))
                           .mean(dim=('times', 'trial_type')), s=500, marker='v', edgecolor='white',
                           c=color_i, alpha=.85, label=f'non_matching_tri-SC{cue_test}')

                invalid_tri = proj_single_trials_smooth.sel(times=slice(start_space_new,
                                                                        end_space_new))\
                    .sel(trial_type=proj_single_trials_smooth.trial_type.isin(group_trials))\
                    .mean('times')

                invalid_tri = invalid_tri.assign_coords(valid=('trial_type', [0]*len(invalid_tri)),
                                                        event=('trial_type',
                                                               [cue_test]*len(invalid_tri)))
                all_trials.append(invalid_tri)

            # group all trials
            all_trials_xr = xr.concat(all_trials, dim='trial_type')
            all_trials_xr.name = 'projected_value'

            # plot the line that splits the valid and invalid trials using a new lda
            all_accuracy_validity = []
            all_accuracy_validity_sh = []
            for seed in range(50):
                clf_valid = LDA()
                # get the train test split
                projected_train, projected_test = split_train_test(mua_xr=all_trials_xr,
                                                                   condition='valid',
                                                                   seed=seed, percent_train=0.5)

                projected_test_balanced, _ = split_train_test(mua_xr=projected_test,
                                                              condition='valid',
                                                              seed=seed, percent_train=1)

                # fit the lda
                clf_valid.fit(projected_train.values, projected_train.valid.values)

                # project the single trials on the line
                projected_test_validity = clf_valid.transform(projected_test_balanced.values)
                # predict the class
                accuracy = clf_valid.score(projected_test_balanced.values,
                                           projected_test_balanced.valid.values)
                all_accuracy_validity.append(accuracy)

                # get the chance level
                for iii in range(10):
                    clf_valid_sh = LDA()
                    labels_new = np.copy(projected_train.valid.values)
                    np.random.seed(iii)
                    np.random.shuffle(labels_new)
                    # fit the lda
                    clf_valid_sh.fit(projected_train.values, labels_new)
                    # predict the class
                    predicted_validity = clf_valid_sh.predict(projected_test_balanced.values)
                    accuracy_sh = clf_valid_sh.score(projected_test_balanced.values,
                                                     projected_test_balanced.valid.values)
                    all_accuracy_validity_sh.append(accuracy_sh)

            # get the decision boundary
            # Get the LDA coefficients (normal vector to the separating hyperplane)
            w = clf_valid.coef_[0]  # shape: (n_features,)
            b = clf_valid.intercept_[0]  # scalar

            # Assume your data is 2D
            x_vals = np.linspace(-1, 1, 100)
            slope = -w[0] / w[1]
            intercept = -b / w[1]
            y_vals = slope * x_vals + intercept

            # plt.plot(x_vals, y_vals, label='LDA decision boundary', color='gray', linestyle='--',
            #          alpha=.3)

            # Normalize w for plotting (just for length control)
            w_norm = w / np.linalg.norm(w)

            # Choose an origin — could be the mean of your data
            origin = projected_train.values.mean(axis=0)

            # Create line in direction of w
            scale = 2  # how far the line should go
            line_points = np.array([origin - scale * w_norm, origin + scale * w_norm])

            plt.plot(line_points[:, 0], line_points[:, 1], label='LDA projection axis',
                     color='black', linestyle='-', linewidth=3, alpha=.3)

            # ax.legend(loc='upper right')
            fig.savefig(os.path.join(PATH_FIGURES, f'{SESSION}-probe-{probe}_'
                                                   f'LDA_{event}_validity.svg'), dpi=300)
            plt.close(fig)

            # create a dataframe from the xarray
            all_trials_df = all_trials_xr.to_dataframe().reset_index()
            all_trials_df['valid_label'] = all_trials_df['valid'].map({0: 'invalid', 1: 'valid'})

            palette = {'invalid': color_invalid, 'valid': color_valid}

            # plot distributions
            figure2, ax2 = plt.subplots(1, 2, figsize=(10, 5))
            for ax2_i, dim in zip(ax2, ['dim-1', 'dim-2']):
                sns.violinplot(x='event', y='projected_value',
                               data=all_trials_df[all_trials_df.dimensions == dim],
                               hue='valid_label', split=True, ax=ax2_i, palette=palette,
                               inner="quart")

                all_mi = []
                all_p_values = []
                for event_i in [1, 2, 3]:
                    mi_type = 'cd'
                    # define the workflow
                    wf = WfMi(mi_type=mi_type, inference='ffx', verbose=False)
                    # check the MI
                    dataset = \
                        DatasetEphy([all_trials_xr.sel(dimensions=dim,
                                                       trial_type=all_trials_xr.event ==
                                                       event_i).values.T[:,
                                     np.newaxis, np.newaxis]],
                                    y=all_trials_xr.sel(trial_type=all_trials_xr.event ==
                                                        event_i).valid.values, times=[1])
                    stats = 'maxstat'
                    mi, p_values = wf.fit(dataset,
                                          mcp=stats, n_perm=500, cluster_th=None,
                                          cluster_alpha=0.05, n_jobs=1,
                                          random_state=1)
                    all_mi.append(mi.values[0][0])
                    all_p_values.append(p_values.values[0][0])

                mi_p_val_text = [f'MI:{mi:.2f} (p={p_val:.3f})' for mi, p_val in zip(all_mi,
                                                                                     all_p_values)]
                ax2_i.set_title(f'{dim}\n {mi_p_val_text[0]} \n {mi_p_val_text[1]} \n '
                                f'{mi_p_val_text[2]}')

            sns.despine(figure2)
            figure2.tight_layout()
            figure2.savefig(os.path.join(PATH_FIGURES, f'{SESSION}-probe-{probe}_'
                                                       f'LDA_{event}_validity_per_SC_MI.png'),
                            dpi=300)
            io.logger.info(f'Figure saved for session {SESSION}')
