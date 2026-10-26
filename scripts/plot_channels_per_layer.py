import os
import warnings
import re
import h5py
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from frites import io
import seaborn as sns

# to remove the spam of pandas FutureWarning with iteritems
warnings.simplefilter(action='ignore', category=FutureWarning)


if __name__ == "__main__":
    # define the sessions to run the analysis
    ALL_LAMINAR = ['Mo180411001', 'Mo180412002', 'Mo180626003', 'Mo180627003', 'Mo180619002',
                   'Mo180622002', 'Mo180704003', 'Mo180418002', 'Mo180419003', 'Mo180426004',
                   'Mo180601001',
                   'Mo180523002', 'Mo180705002', 'Mo180706002', 'Mo180710002',
                   'Mo180711004', 'Mo180712006', 't150303002', 't150319003', 't150423002',
                   't150320002',
                   't150123001', 't150128001', 't150204001',
                   't150205004',
                   't150716001',
                   'Mo180615002-Mo180615005', 't150327002-t150327003']

    # all possible probes
    probes = [1, 2]

    # store the summary of channels for all sites
    layer_channel_summary = []

    # set the paths
    current_path = os.getcwd()

    if current_path.startswith('C:'):
        server = 'W:'  # local w VPN
    else:
        server = '/envau/work'  # niolon

    # path of the figures
    PATH_FIGURES = server + \
        '/comco/lopez.l/Electrophysiology/ephy_laminar_MUA/Results/Full_trial/' \
        'plots_revisions/angles/'

    # path of the figures
    PATH_EXCEL = server + \
        '/comco/lopez.l/Electrophysiology/ephy_laminar_MUA/Results/Full_trial/' \
        'plots_similarity_corrected/'

    # load the excel file from the PATH_Figures
    excel_file = os.path.join(PATH_EXCEL, f'Mua_check_channels.xlsx')

    # load the excel file
    df_mua_info = pd.read_excel(excel_file)

    # loop over the sessions
    for SESSION in ALL_LAMINAR:
        # set the path of the data and get the name of the files
        PATH = server + \
               '/comco/lopez.l/Electrophysiology/ephy_laminar_MUA/Results/Preprocessed_data/' + \
               SESSION + '/'

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

            # open the file
            mua_site = xr.load_dataarray(os.path.join(PATH, file_name_mua))
            io.logger.info(f'Loaded file: {file_name_mua}')
            # 1) Select the channels of interest: remove bad channels
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

            # 3) Store how many channels are in each of the layers
            # Returns two arrays: one with unique values, one with their counts
            values, counts = np.unique(mua_site["layers"].values, return_counts=True)

            # Zip them together into a dictionary to store all of them
            counts_dict = dict(zip(values, counts))

            counts_dict['site'] = f'{SESSION}-{probe}'

            layer_channel_summary.append(counts_dict)

        # make it a dataframe
        df = pd.DataFrame(layer_channel_summary)
        # save the results
        df.to_csv(os.path.join(PATH_FIGURES, "channels_per_layer_per_site.csv"), index=False)

        # plot the histograms
        fig, ax = plt.subplots(1, 4, figsize=(15, 3), sharey='row')
        bins = np.arange(1.5, 14.5, 1)  # Bin centers at integers 2, 3, ..., 14
        for layer, ax_i in zip(['L23-SUP', 'L23-DEEP', 'L5', 'L6'], ax):
            min_chan = np.min(df[layer])
            max_chan = np.max(df[layer])
            sns.histplot(data=df, x=layer, bins=bins, edgecolor='white', ax=ax_i)
            ax_i.set_xlim([1, 15])
            ax_i.set_xticks(np.arange(2, 15))  # Ticks centered on the bins
            ax_i.set_title(f'{layer}-{np.sum(~np.isnan(df[layer]))} sites')
            ax_i.set_xlabel('Number of channels')

        fig.suptitle('Number of channels per layer across sites (41 total sites)')
        fig.tight_layout()
        sns.despine(fig)
