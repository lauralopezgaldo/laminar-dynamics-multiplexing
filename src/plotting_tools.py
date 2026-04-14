import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from frites.stats.stats_nonparam import confidence_interval


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
