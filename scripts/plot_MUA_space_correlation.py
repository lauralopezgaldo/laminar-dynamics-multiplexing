import os
import warnings
import re
import h5py
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from frites import io
import seaborn as sns

# to remove the spam of pandas FutureWarning with iteritems
warnings.simplefilter(action='ignore', category=FutureWarning)


if __name__ == "__main__":
    # 1) Define paths
    current_path = os.getcwd()

    if current_path.startswith('C:'):
        server = 'W:'
    else:
        server = '/envau/work'

    PATH_FIGURES = server + \
        '/comco/lopez.l/Electrophysiology/ephy_laminar_MUA/Results/Full_trial/' \
        'plots_revisions/spatial_correlations/'

    observed_path = os.path.join(
        PATH_FIGURES,
        'spatial_correlations_observed.csv'
    )

    shuffled_path = os.path.join(
        PATH_FIGURES,
        'spatial_correlations_shuffled.csv'
    )

    # 2) Load the data
    # observations
    df_observed = pd.read_csv(observed_path)
    # shuffles
    df_shuffled = pd.read_csv(shuffled_path)

    # Correct floating-point precision in distances
    df_observed['distance'] = df_observed['distance'].round(2)
    df_shuffled['distance'] = df_shuffled['distance'].round(2)

    # Remove missing values
    df_observed = df_observed.dropna(
        subset=['distance', 'correlation', 'site']
    )

    # 3) Calculate median correlation per site and distance
    df_site = (
        df_observed
        .groupby(['site', 'distance'])['correlation']
        .median()
        .reset_index()
    )

    # -----------------------------------------------
    # 4) Plot the distribution for each distance across all sites
    sns.set_context('poster')
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.boxplot(
        data=df_site,
        x='distance',
        y='correlation',
        color='lightgrey',
        width=0.6,
        showfliers=False,
        ax=ax
    )

    # Show only selected x-axis labels
    all_distances = np.sort(df_site['distance'].unique())

    tick_step = 2  # Show every 5th distance category

    tick_positions = np.arange(0, len(all_distances), tick_step)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels([
        f'{all_distances[i] * 1000:.0f}'
        for i in tick_positions
    ])

    ax.set_xlabel('Distance between contacts (µm)')
    ax.set_ylabel('Median correlation across channel pairs')
    ax.set_title('Observed spatial correlations across recording sites')

    sns.despine()
    plt.tight_layout()

    # -----------------------------------------------
    # 5) Plot identical but respecting the x-axis
    sns.set_context('poster')
    fig, ax = plt.subplots(figsize=(8, 5))

    all_distances = np.sort(df_site['distance'].unique())

    data_by_distance = [
        df_site.loc[
            df_site['distance'] == d, 'correlation'
        ].values
        for d in all_distances
    ]

    # Boxplots at their actual numerical positions
    ax.boxplot(
        data_by_distance,
        positions=all_distances,
        widths=0.035,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor='lightgrey', edgecolor='black'),
        medianprops=dict(color='black', linewidth=1.5),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black')
    )

    medians = [
        np.median(values) for values in data_by_distance
    ]

    ax.plot(
        all_distances,
        medians,
        color='firebrick',
        linewidth=1.5,
        marker='o',
        markersize=4,
        zorder=3,
        label='Median across sites'
    )

    # X-axis labels in µm (assuming distance is in mm)
    tick_interval = 0.200
    ticks = np.arange(
        0,
        all_distances.max() + tick_interval,
        tick_interval
    )

    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{x * 1000:.0f}' for x in ticks])

    ax.set_xlabel('Distance between contacts (µm)')
    ax.set_ylabel('Median correlation across channel pairs')
    ax.set_title('Observed spatial correlations across recording sites')

    ax.set_xlim(0, all_distances.max() + 0.1)

    sns.despine()
    plt.tight_layout()
    plt.show()

    # ---------------------------------------
    # 6) Plot the same but with the shuffles

    # Observed: median across channel pairs within each site and distance
    df_obs_site = (
        df_observed
            .groupby(['site', 'distance'])['correlation']
            .median()
            .reset_index()
    )

    # Shuffled: median across shuffle repetitions for each channel pair
    df_shuffle_pair = (
        df_shuffled
            .groupby(['site', 'channel_i', 'channel_j', 'distance'])['correlation']
            .median()
            .reset_index()
    )

    # Then median across channel pairs within each site and distance
    df_sh_site = (
        df_shuffle_pair
            .groupby(['site', 'distance'])['correlation']
            .median()
            .reset_index()
    )

    # Add condition labels
    df_obs_site['condition'] = 'Observed'
    df_sh_site['condition'] = 'Shuffled'

    df_plot = pd.concat(
        [df_obs_site, df_sh_site],
        ignore_index=True
    )

    sns.set_context('poster')
    fig, ax = plt.subplots(figsize=(8, 5))

    all_distances = np.sort(
        np.intersect1d(
            df_obs_site['distance'].unique(),
            df_sh_site['distance'].unique()
        )
    )

    data_observed = [
        df_obs_site.loc[df_obs_site['distance'] == d, 'correlation'].values
        for d in all_distances
    ]

    data_shuffled = [
        df_sh_site.loc[df_sh_site['distance'] == d, 'correlation'].values
        for d in all_distances
    ]

    # Shuffled boxplots (slightly offset)
    ax.boxplot(
        data_shuffled,
        positions=all_distances - 0.02,
        widths=0.035,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor='lightgrey', edgecolor='black'),
        medianprops=dict(color='black', linewidth=1.5),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black')
    )

    # Observed boxplots (slightly offset)
    ax.boxplot(
        data_observed,
        positions=all_distances + 0.02,
        widths=0.035,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor='steelblue', edgecolor='black'),
        medianprops=dict(color='black', linewidth=1.5),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black')
    )

    # Median lines
    median_shuffled = [
        np.median(values) for values in data_shuffled
    ]

    median_observed = [
        np.median(values) for values in data_observed
    ]

    ax.plot(
        all_distances - 0.02,
        median_shuffled,
        color='dimgray',
        linewidth=1.5,
        marker='o',
        markersize=4,
        zorder=3,
        label='Shuffled median'
    )

    ax.plot(
        all_distances + 0.02,
        median_observed,
        color='firebrick',
        linewidth=1.5,
        marker='o',
        markersize=4,
        zorder=3,
        label='Observed median'
    )

    # X-axis
    tick_interval = 0.200
    ticks = np.arange(
        0,
        all_distances.max() + tick_interval,
        tick_interval
    )

    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{x * 1000:.0f}' for x in ticks])

    ax.set_xlabel('Distance between contacts (µm)')
    ax.set_ylabel('Median correlation across channel pairs')
    ax.set_title('Observed vs shuffled spatial correlations')

    ax.set_xlim(0, all_distances.max() + 0.1)

    ax.legend(frameon=False)

    sns.despine()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------
    # 7) Calculate observed - chance for each site/distance

    # Merge observed and shuffled site-level medians
    df_excess = pd.merge(
        df_obs_site[['site', 'distance', 'correlation']],
        df_sh_site[['site', 'distance', 'correlation']],
        on=['site', 'distance'],
        suffixes=('_observed', '_shuffled')
    )

    # Calculate excess correlation
    df_excess['excess_correlation'] = (
        df_excess['correlation_observed'] -
        df_excess['correlation_shuffled']
    )

    sns.set_context('poster')
    fig, ax = plt.subplots(figsize=(8, 5))

    all_distances = np.sort(df_excess['distance'].unique())

    data_by_distance = [
        df_excess.loc[
            df_excess['distance'] == d, 'excess_correlation'
        ].values
        for d in all_distances
    ]

    # Boxplots at actual numerical positions
    ax.boxplot(
        data_by_distance,
        positions=all_distances,
        widths=0.035,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor='lightgrey', edgecolor='black'),
        medianprops=dict(color='black', linewidth=1.5),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black')
    )

    # Median across sites
    medians = [
        np.median(values) for values in data_by_distance
    ]

    ax.plot(
        all_distances,
        medians,
        color='firebrick',
        linewidth=1.5,
        marker='o',
        markersize=4,
        zorder=3,
        label='Median across sites'
    )

    # Zero reference line
    ax.axhline(
        0,
        color='black',
        linestyle='--',
        linewidth=1,
        alpha=0.7
    )

    # X-axis
    tick_interval = 0.200
    ticks = np.arange(
        0,
        all_distances.max() + tick_interval,
        tick_interval
    )

    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{x * 1000:.0f}' for x in ticks])

    ax.set_xlabel('Distance between contacts (µm)')
    ax.set_ylabel('Observed − shuffled correlation')
    ax.set_title('Excess spatial correlation across recording sites')

    ax.set_xlim(0, all_distances.max() + 0.1)

    ax.legend(frameon=False)

    sns.despine()
    plt.tight_layout()
