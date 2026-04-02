"""
This script opens the LDA evolution for each site and computes the grassmanian trajectory.
Mainly for visualization purpose, nothing better than propagation scaffold.
"""

import os
import warnings
import numpy as np
import pandas as pd
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
from sklearn.manifold import TSNE
from scipy.signal import savgol_filter
from collections import Counter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# to remove the spam of pandas FutureWarning with iteritems
warnings.simplefilter(action='ignore', category=FutureWarning)


if __name__ == "__main__":
    # define all sites
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

    # set the path of the data and get the name of the files
    # check where are we running the code
    current_path = os.getcwd()

    # set the path of the data and get the name of the files
    if current_path.startswith('C:'):
        server = 'W:'  # local w VPN
    else:
        server = '/envau/work'  # niolon

    PATH_FIGURES = server + \
        '/comco/lopez.l/Electrophysiology/ephy_laminar_MUA/Results/Full_trial/' \
        'plots_grassmanian/'

    # loop over the sessions
    for SESSION in ALL_LAMINAR:
        PATH_DATA = server + \
            '/comco/lopez.l/Electrophysiology/ephy_laminar_MUA/Results/Full_trial/' \
            'data_paper/' + SESSION + '/'
        # iterate on the probes
        for probe in probes:
            # Open the evolution of the LDA computed for one cross-validation fold
            file_name_mua = [i for i in os.listdir(PATH_DATA) if
                             os.path.isfile(os.path.join(PATH_DATA, i))
                             and f'{SESSION}' in i and '.nc' in i and
                             'SC' not in i and f'probe_{probe}' in i and behavior in i and 'cv'
                             not in i and method in i]
            if len(file_name_mua) == 0:
                io.logger.info(f'No file found for probe {probe}')
                continue
            else:
                file_name_mua = file_name_mua[0]

            # open the file
            lda_evolution_site = xr.load_dataset(os.path.join(PATH_DATA, file_name_mua))

            # get the events
            events_onset = lda_evolution_site.task_events_onset
            events_names = re.split('-', lda_evolution_site.task_events_labels)

            # get the theoretical times of SC1 and Go
            t_SEL = events_onset[events_names.index('SEL')]
            t_GO = events_onset[events_names.index('GO')]

            # get the first LD
            LD1 = lda_evolution_site.weights_evolution.sel(dimensions='dim-1',
                                                           times=slice(t_SEL, t_GO))
            # get the absolute value
            LD = abs(LD1)

            # Step 1: Create the initial binary mask based on the 75th percentile
            # threshold = np.percentile(np.abs(LD1.values), 75)
            # LD = np.where((np.abs(LD1) > threshold), np.abs(LD1), 0)

            # get an index for the layers
            norm_layers =\
                LD.groupby('layers').mean() / \
                LD.groupby('layers').mean().sum(dim='layers').values.reshape(
                    -1, 1)

            idx_layers =\
                norm_layers[:, 0:2].mean(dim='layers') / norm_layers[:, 2:4].mean(dim='layers')

            # generate PCA of the LDA
            pca = PCA(n_components=3)
            LD_grass = pca.fit_transform(LD.values)

            # generate the TSNE
            # X_embedded = TSNE(n_components=2).fit_transform(LD)

            # smooth the trajectory using a Savitzky-Golay filter
            LD_grass_filt = savgol_filter(LD_grass, window_length=20, polyorder=2, axis=0)
            # LD_grass_filt = savgol_filter(LD_grass, window_length=30, polyorder=2, axis=0)
            # LD_grass_filt = savgol_filter(X_embedded, window_length=10, polyorder=2, axis=0)

            # create a 3d figure to plot the trajectory
            sns.set_style('ticks')
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(LD1.times, LD_grass_filt[:, 0], LD_grass_filt[:, 1],
                    alpha=.6, linewidth=2, color='k')
            cbar = ax.scatter3D(LD1.times, LD_grass_filt[:, 0], LD_grass_filt[:, 1],
                                alpha=.55, s=180,
                                c=idx_layers.values, cmap="icefire",
                                vmin=0.5, vmax=1.5)
            plt.colorbar(cbar)

            ax.set_xticks(events_onset[2:-1], events_names[2:-1], rotation=45)

            # LDA1 and LDA2 axis limits
            lda1_min, lda1_max = -.5, .5
            lda2_min, lda2_max = -.5, .5

            # Example time of the plane
            t_planes = events_onset[3:-2]
            for t_plane in t_planes:
                # Define the 4 corners of the plane at t = 0.3
                plane_verts = [[
                    [t_plane, lda1_min, lda2_min],
                    [t_plane, lda1_max, lda2_min],
                    [t_plane, lda1_max, lda2_max],
                    [t_plane, lda1_min, lda2_max]
                ]]

                # Plot the vertical plane
                plane = Poly3DCollection(plane_verts, color='gray', alpha=0.1, edgecolor='none')
                ax.add_collection3d(plane)

            ax.set_ylim(lda1_min, lda1_max)
            ax.set_zlim(lda2_min, lda2_max)

            fig.suptitle(f'{SESSION} - Probe {probe} - {method} - {behavior}', fontsize=16)
            fig.tight_layout()
            fig.savefig(os.path.join(PATH_FIGURES,
                                     f'{SESSION}_probe_{probe}_{method}_{behavior}.png'),
                        dpi=300, bbox_inches='tight')
