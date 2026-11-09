"""
It takes the plot of the 'random 2D angles' and plot each site depending on the number of
channels into that original plot, with the mean for within and across classes, to see whether
they are above and beyond the expected ones.
"""

import os
import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat


if __name__ == "__main__":
    # --------------------------------------------------
    # 1. SIMULATED DATA - already saved in a .mat matrix
    # --------------------------------------------------
    # 1) load the data from the 2D chance level
    path_local = 'C:/Users/laura/OneDrive/Documents/In2PB/MUA_paper/Submission_NatCom/Code/'
    data_simulated = loadmat(os.path.join(path_local + 'plane_angles_plot_data.mat'))

    # 2) get only the arithmetic mean
    idx = np.where(data_simulated['quantity_names'] == 'arithmetic_mean')[0][0]
    q05 = data_simulated['q05_deg'][:, idx].squeeze()
    q25 = data_simulated['q25_deg'][:, idx].squeeze()
    mean_angle = data_simulated['median_deg'][:, idx]  # median but ok
    q75 = data_simulated['q75_deg'][:, idx].squeeze()
    q95 = data_simulated['q95_deg'][:, idx].squeeze()
    dimensions = data_simulated['dimensions']

    # --------------------------------------------------
    # 2. REAL DATA - load the estimated multiplexing angles
    # --------------------------------------------------
    current_path = os.getcwd()
    if current_path.startswith('C:'):
        server = 'W:'  # local w VPN
    else:
        server = '/envau/work'  # niolon

    path_mua = server + \
        '/comco/lopez.l/Electrophysiology/ephy_laminar_MUA/Results/Full_trial/' \
        'plots_revisions/angles/'

    data_mua = pd.read_csv(os.path.join(path_mua, 'subspace_angles_summary.csv'))

    # --------------------------------------------------
    # 3. PLOTS
    # --------------------------------------------------
    # 1)  PLOT ALL THE DOTS - for each pair in each site
    max_dim = 35
    fig, ax = plt.subplots()
    ax.plot(dimensions[:max_dim], mean_angle[:max_dim], '-o',
            label='simulated angle between two n-dim planes', color='k')

    ax.set_xlabel('Number of dimensions (n)')
    ax.set_ylabel('Angle (º deg)')

    sns.despine(fig)

    # Add jitter only for visualization
    data_mua['n_channels_jitter'] = (
            data_mua['n_channels'] + np.random.uniform(-0.5, 0.5, len(data_mua))
    )
    sns.scatterplot(data=data_mua,
                    x='n_channels_jitter', y='angle_deg',
                    hue='pair_type',
                    palette={'same_behavior': '#227c9d',
                             'different_behavior': '#bd5b35'}, alpha=0.7, s=100, ax=ax)

    # 1 datapoint per site (for 'same' and for 'different')
    data_mua_avg = (
        data_mua.groupby(['session', 'probe', 'n_channels', 'pair_type'], as_index=False)
        ['angle_deg'].mean())

    # Add jitter only for visualization
    data_mua_avg['n_channels_jitter'] = (
            data_mua_avg['n_channels'] +
            np.random.uniform(-0.5, 0.5, len(data_mua_avg))
    )

    # 2)  PLOT ONE DOT PER each category
    sns.set_context('poster')
    fig, ax = plt.subplots(figsize=(8, 6))
    max_dim = 35
    x = dimensions[:max_dim]
    # Simulated data: median
    ax.plot(
        x, mean_angle[:max_dim],
        color='k', lw=1.5,
        label='Simulated median'
    )
    # Simulated data: interquartile range
    ax.fill_between(
        x,
        q25[:max_dim],
        q75[:max_dim],
        color='grey',
        alpha=0.25,
        label='Simulated 25th–75th percentile'
    )
    # Optional: 5th–95th percentile
    ax.fill_between(
        x,
        q05[:max_dim],
        q95[:max_dim],
        color='grey',
        alpha=0.10,
        label='Simulated 5th–95th percentile'
    )
    # Real data
    sns.scatterplot(
        data=data_mua_avg,
        x='n_channels_jitter',
        y='angle_deg',
        hue='pair_type',
        palette={
            'same_behavior': '#227c9d',
            'different_behavior': '#bd5b35'
        },
        alpha=0.75,
        s=100,
        ax=ax
    )

    ax.set_xlabel('Number of channels (D)')
    ax.set_ylabel('Angle (degrees)')
    ax.set_xlim(0, max_dim + 1)

    sns.despine(fig)
    plt.tight_layout()
    plt.show()

    # 3) SAME but with boxes
    # Simulated data: boxplots
    sns.set_context('poster')
    fig, ax = plt.subplots(figsize=(8, 6))
    max_dim = 35
    x = dimensions[:max_dim]
    # Simulated data: median
    ax.plot(
        x, mean_angle[:max_dim],
        color='k', lw=1.5,
        label='Simulated median'
    )
    # Simulated data: interquartile range
    stats = [
        dict(
            whislo=q05[i],
            q1=q25[i],
            med=mean_angle[i],
            q3=q75[i],
            whishi=q95[i],
            fliers=[]
        )
        for i in range(max_dim)
    ]

    ax.bxp(
        stats,
        positions=dimensions[:max_dim],
        widths=0.65,
        showfliers=False,
        patch_artist=True,
        boxprops={
            'facecolor': 'grey',
            'edgecolor': 'grey',
            'alpha': 0.35
        },
        medianprops={
            'color': 'black',
            'linewidth': 1.3
        },
        whiskerprops={
            'color': 'grey',
            'linewidth': 0.9
        },
        capprops={
            'color': 'grey',
            'linewidth': 0.9
        }
    )    # Real data
    sns.scatterplot(
        data=data_mua_avg,
        x='n_channels_jitter',
        y='angle_deg',
        hue='pair_type',
        palette={
            'same_behavior': '#227c9d',
            'different_behavior': '#bd5b35'
        },
        alpha=0.75,
        s=100,
        ax=ax
    )

    ax.set_xlabel('Number of channels (D)')
    ax.set_ylabel('Angle (degrees)')
    ax.set_xlim(0, max_dim + 1)

    sns.despine(fig)
    plt.tight_layout()
