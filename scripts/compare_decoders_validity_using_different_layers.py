"""
This script takes the decoders used for the different task variables and checks what happens to the
'accuracy' (i.e. whether they are still significant) when the laminar structure is modified.

In other words, we check if decoders leaving one layer out, or decoders using only one layer are
able to still uncover the information, or if the performance decreases below chance. We compare also
which layers have a bigger impact in the decodability, and which layers are able to decode 'on
their own'.
"""

import os
import warnings
import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from frites import io
import seaborn as sns
from statsmodels.stats.multitest import multipletests


def get_p_val_w_percentile(observations, shuffled_observations, stat=np.mean,
                           prediction='smaller'):
    """
    Calculate the p-value based on the percentile of shuffled observations.
    :param observations:
    :param shuffled_observations:
    :param stat:
    :param prediction: 'smaller' or 'larger', indicating the direction of the test
    :return:
    """
    # get the statistic of the observations
    obs_stat = stat(observations)

    # calculate the p-value
    if prediction == 'smaller':
        # One-sided p-value (real < shuffle is better)
        p_value = (1 + np.sum(shuffled_observations <= obs_stat)) / \
                  (1 + len(shuffled_observations))

    elif prediction == 'larger':
        # One-sided p-value (shuffle < real is better)
        p_value = (1 + np.sum(shuffled_observations >= obs_stat)) / \
                  (1 + len(shuffled_observations))
    else:
        raise ValueError("Prediction must be 'smaller' or 'larger'")

    return p_value


# to remove the spam of pandas FutureWarning with iteritems
warnings.simplefilter(action='ignore', category=FutureWarning)


if __name__ == "__main__":
    # list of sessions to check
    ALL_LAMINAR_NEW = ['Mo180411001', 'Mo180412002', 'Mo180626003', 'Mo180627003', 'Mo180619002',
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
    behavior = 'trial_type'
    cue = 'SC1'

    # store the data across sites
    all_summaries = []
    summaries_loo = []
    summaries_sl = []

    current_path = os.getcwd()

    if current_path.startswith('C:'):
        server = 'W:'  # local w VPN
    else:
        server = '/envau/work'  # niolon

    PATH_FIGURES = server + \
        '/comco/lopez.l/Electrophysiology/ephy_laminar_MUA/Results/Full_trial/' \
        'plots_revisions/summary_stats_for_different_layers_decoders/'

    # iterate on the sessions
    for SESSION in ALL_LAMINAR_NEW:
        PATH_DATA = server + \
                    '/comco/lopez.l/Electrophysiology/ephy_laminar_MUA/Results/Full_trial/' \
                    'data_paper/' + SESSION + '/'

        PATH_DATA_REVISIONS = server + \
            '/comco/lopez.l/Electrophysiology/ephy_laminar_MUA/Results/Full_trial/' \
            'data_revisions/' + SESSION + '/'

        # iterate on the probes
        for probe in probes:
            # Open in this case the single trials aligned to SEL because the SC2 ones are not full
            file_name_mua = [i for i in os.listdir(PATH_DATA) if
                             os.path.isfile(os.path.join(PATH_DATA, i))
                             and
                             f'{SESSION}_{probe}' in i and '.csv' in i and 'cv_accuracy' in i and
                             'LDA_SC' in i and behavior in i]
            if len(file_name_mua) == 0:
                io.logger.info(f'No MUA file found for probe {probe}')
                continue
            else:
                file_name_mua = file_name_mua[0]
                # file names for the loo (leave-one-out) and sl (single-layer)
                file_name_loo = [i for i in os.listdir(PATH_DATA_REVISIONS) if
                                 os.path.isfile(os.path.join(PATH_DATA_REVISIONS, i))
                                 and
                                 f'{SESSION}_{probe}' in i and '.csv' in i and 'cv_accuracy' in i
                                 and 'LDA_SC' in i and behavior in i and 'leave_one_out' in i][0]

                file_name_sl = [i for i in os.listdir(PATH_DATA_REVISIONS) if
                                os.path.isfile(os.path.join(PATH_DATA_REVISIONS, i))
                                and
                                f'{SESSION}_{probe}' in i and '.csv' in i and 'cv_accuracy' in i
                                and 'LDA_SC' in i and behavior in i and 'one_layer' in i][0]

            # load the dataframe
            df = pd.read_csv(os.path.join(PATH_DATA, file_name_mua))

            # check if there is a significant decoding at CUE
            p_val_sci = \
                get_p_val_w_percentile(observations=df[(df.event == cue) &
                                                       (df.same_cue == True) &
                                                       (df.shuffled == False)].diff_distance,
                                       shuffled_observations=df[(df.event == cue) &
                                                                (df.same_cue == True) &
                                                                (df.shuffled ==
                                                                 True)].diff_distance,
                                       stat=np.mean,
                                       prediction='smaller')

            if p_val_sci > 0.05:
                io.logger.info(f'{cue}-dir space not significant for {SESSION} probe {probe}')
                continue

            # run your code here per session
            summary_df = df.groupby(['same_cue', 'shuffled']).mean(numeric_only=True)

            # Add identifying info (e.g., session name)
            summary_df['session'] = f'{SESSION}-{probe}'

            # get the p-value for the generalization - so cue different to the CUE and
            # same cue false (so we did not train and test in the same)
            p_val_sci_gen = \
                get_p_val_w_percentile(observations=df[(df.event != cue) &
                                                       (df.same_cue == False) &
                                                       (df.shuffled == False)].diff_distance,
                                       shuffled_observations=df[(df.event != cue) &
                                                                (df.same_cue == False) &
                                                                (df.shuffled ==
                                                                 True)].diff_distance,
                                       stat=np.mean,
                                       prediction='smaller')

            # Add the hypothesis
            summary_df['p_val_sc'] = p_val_sci
            summary_df['p_val_sc_gen'] = p_val_sci_gen

            # Check what happens with the leave one layer out decoders
            # load the dataframe - leave one out
            df_loo = pd.read_csv(os.path.join(PATH_DATA_REVISIONS, file_name_loo))
            df_sl = pd.read_csv(os.path.join(PATH_DATA_REVISIONS, file_name_sl))

            # summarize the leave-one-out
            summary_loo = \
                df_loo.groupby(['same_cue', 'shuffled', 'layer_out']) \
                .mean(numeric_only=True)

            # Add identifying info (e.g., session name)
            summary_loo['session'] = f'{SESSION}-{probe}'

            # summarize the single layer
            summary_sl = \
                df_sl.groupby(['same_cue', 'shuffled', 'layer_used']) \
                .mean(numeric_only=True)

            # Add identifying info (e.g., session name)
            summary_sl['session'] = f'{SESSION}-{probe}'

            # re-compute the statistics for the new datasets
            for layer_i in pd.unique(df_loo.layer_out):
                # leave one out
                p_val_sci_loo = \
                    get_p_val_w_percentile(observations=df_loo[(df_loo.event == cue) &
                                                               (df_loo.same_cue == True) &
                                           (df_loo.shuffled == False) &
                                           (df_loo.layer_out == layer_i)].diff_distance,
                                           shuffled_observations=df_loo[(df_loo.event == cue)
                                                                        & (df_loo.same_cue == True)
                                                                        & (df_loo.shuffled == True)
                                                                        &
                                           (df_loo.layer_out == layer_i)].diff_distance,
                                           stat=np.mean,
                                           prediction='smaller')

                p_val_sci_loo_gen = \
                    get_p_val_w_percentile(observations=df_loo[(df_loo.event != cue) &
                                                               (df_loo.same_cue == False) &
                                           (df_loo.shuffled == False) &
                                           (df_loo.layer_out == layer_i)].diff_distance,
                                           shuffled_observations=df_loo[(df_loo.event != cue)
                                                                        &
                                                                        (df_loo.same_cue == False) &
                                                                        (df_loo.shuffled ==
                                                                        True) &
                                           (df_loo.layer_out == layer_i)].diff_distance,
                                           stat=np.mean,
                                           prediction='smaller')

                # Add the p-values
                summary_loo[f'p_val_{cue}_LOO_{layer_i}'] = p_val_sci_loo
                summary_loo[f'p_val_{cue}_LOO_{layer_i}_gen'] = p_val_sci_loo_gen

                # single layer
                p_val_sci_sl = \
                    get_p_val_w_percentile(observations=df_sl[(df_sl.event == cue) &
                                                           (df_sl.same_cue == True) &
                                           (df_sl.shuffled == False) &
                                           (df_sl.layer_used == layer_i)].diff_distance,
                                           shuffled_observations=df_sl[(df_sl.event == cue) &
                                                                    (df_sl.same_cue == True) &
                                                                    (df_sl.shuffled ==
                                                                     True) &
                                           (df_sl.layer_used == layer_i)].diff_distance,
                                           stat=np.mean,
                                           prediction='smaller')

                p_val_sci_sl_gen = \
                    get_p_val_w_percentile(observations=df_sl[(df_sl.event != cue) &
                                                              (df_sl.same_cue == False) &
                                           (df_sl.shuffled == False) &
                                           (df_sl.layer_used == layer_i)].diff_distance,
                                           shuffled_observations=df_sl[(df_sl.event != cue)
                                                                       & (df_sl.same_cue == False) &
                                                                       (df_sl.shuffled == True) &
                                           (df_sl.layer_used == layer_i)].diff_distance,
                                           stat=np.mean,
                                           prediction='smaller')

                # Add the p-values
                summary_sl[f'p_val_{cue}_sl_{layer_i}'] = p_val_sci_sl
                summary_sl[f'p_val_{cue}_sl_{layer_i}_gen'] = p_val_sci_sl_gen

            # Append to list
            all_summaries.append(summary_df)
            summaries_loo.append(summary_loo)
            summaries_sl.append(summary_sl)

    # LEAVE ONE OUT
    for monkey_name in ['Mo', 't']:
        # After loop, concatenate
        final_summary_df = pd.concat(all_summaries).reset_index()

        # change this to select the monkey
        final_summary_df = final_summary_df[final_summary_df['session'].str.startswith(monkey_name)]

        # number of sites
        num_sites = final_summary_df['session'].nunique()

        # correct the p_values
        rejected, p_vals_corrected, _, _ = multipletests(
            pvals=final_summary_df.groupby('session').p_val_sc_gen.mean().values,
            alpha=0.05,
            method='bonferroni')

        # keep the corrected p_values
        p_vals_h1_corrected = p_vals_corrected[:num_sites]
        p_vals_h0 = final_summary_df.groupby('session').p_val_sc.mean().values

        sns.set(style="ticks", context='poster')

        # Define colors
        palette = {False: '#5E5768', True: '#C8A676'}

        # Create subplots
        fig, axes = plt.subplots(5, 2, figsize=(10, 30), sharey='row')

        for i, same_cue in enumerate([True, False]):
            ax = axes[0, i]
            df_event = final_summary_df[(final_summary_df['same_cue'] == same_cue)]

            # Pivot so each session has both shuffled and not shuffled
            df_pivot = df_event.pivot(index='session', columns='shuffled', values='diff_distance')
            df_pivot = df_pivot.dropna()  # Remove incomplete pairs

            print(i)

            # add the corrected p_values
            if i == 0:
                df_pivot['p_val_corrected'] = p_vals_h0
                ax.set_title(f'{cue} ')
            elif i == 1:
                df_pivot['p_val_corrected'] = p_vals_h1_corrected
                ax.set_title(f'{cue}: {sum(rejected)}/{num_sites}')

            # Plot each session
            for session, row in df_pivot.iterrows():
                if row.p_val_corrected < 0.05:
                    # significant
                    ax.plot([0, 1], [row[False], row[True]], color='#5E5768', alpha=.75,
                            linewidth=1.5)
                    io.logger.info(f"{session}")

                else:
                    ax.plot([0, 1], [row[False], row[True]], color='#C8A676', alpha=.75,
                            linewidth=1.5)

                ax.scatter([-0, 1], [row[False], row[True]],
                           color=[palette[False], palette[True]],
                           alpha=0.8, edgecolor='k', linewidth=0.5, zorder=2)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Not Shuffled', 'Shuffled'])

            # ax.set_ylim([-1.25, 0.05])
            ax.set_ylim([-1.25, 0.15])
            ax.set_xlim([-0.2, 1.8])
            if i == 0:
                ax.set_ylabel('Difference distance')
                ax.set_yticks([-1.2, -0.6, 0])

            ax.set_ylabel('Difference distance')
            ax.set_yticks([-1.2, -0.6, 0])
            ax.invert_yaxis()  # Invert y-axis for better visualization

        # PLOT SEPARATING LAYER ANALYSIS
        for idx_layer, layer_i in enumerate(['L23-SUP', 'L23-DEEP', 'L5', 'L6']):
            # After loop, concatenate
            final_summary_df = pd.concat(summaries_loo).reset_index()

            # change this to select the monkey
            final_summary_df = final_summary_df[
                final_summary_df['session'].str.startswith(monkey_name)]

            # number of sites
            num_sites = final_summary_df['session'].nunique()

            # correct the p_values
            rejected, p_vals_corrected, _, _ = multipletests(
                pvals=final_summary_df.groupby('session')[f'p_val_{cue}_LOO_{layer_i}_gen']
                .mean().values,
                alpha=0.05,
                method='bonferroni')

            # keep the corrected p_values
            p_vals_h1_corrected = p_vals_corrected[:num_sites]
            p_vals_h0 = final_summary_df.groupby('session')[f'p_val_{cue}_LOO_{layer_i}']\
                .mean().values

            # correct for any possible Nan (coming from no layers present at that probe)
            p_vals_h1_corrected = p_vals_h1_corrected[~np.isnan(p_vals_h1_corrected)]
            p_vals_h0 = p_vals_h0[~np.isnan(p_vals_h0)]

            # correct the number of sites as well
            num_sites = len(p_vals_h0)

            sns.set(style="ticks", context='poster')

            # Define colors
            # palette = {False: '#d0ba98', True: '#b8b6b0'}
            palette = {False: '#5E5768', True: '#C8A676'}

            for i, same_cue in enumerate([True, False]):
                ax = axes[idx_layer+1, i]
                df_event = final_summary_df[(final_summary_df['same_cue'] == same_cue) &

                                            (final_summary_df['layer_out'] == layer_i)]

                # Pivot so each session has both shuffled and not shuffled
                df_pivot = df_event.pivot(index='session', columns='shuffled',
                                          values='diff_distance')
                df_pivot = df_pivot.dropna()  # Remove incomplete pairs

                # add the corrected p_values
                if i == 0:
                    df_pivot['p_val_corrected'] = p_vals_h0
                    ax.set_title(f'SC2 {np.sum(p_vals_h0<0.05)}/{num_sites}')
                elif i == 1:
                    df_pivot['p_val_corrected'] = p_vals_h1_corrected
                    ax.set_title(f'Gen - {sum(rejected)}/{num_sites}')

                # Plot each session
                for session, row in df_pivot.iterrows():
                    if row.p_val_corrected < 0.05:
                        # significant
                        ax.plot([0, 1], [row[False], row[True]], color='#5E5768', alpha=.75,
                                linewidth=1.5)
                        io.logger.info(f"{session}")

                    else:
                        ax.plot([0, 1], [row[False], row[True]], color='#C8A676', alpha=.75,
                                linewidth=1.5)

                    ax.scatter([-0, 1], [row[False], row[True]],
                               color=[palette[False], palette[True]],
                               alpha=0.8, edgecolor='k', linewidth=0.5, zorder=2)

                ax.set_xticks([0, 1])
                ax.set_xticklabels(['Not Shuffled', 'Shuffled'])

                ax.set_ylim([-1.25, 0.15])
                ax.set_xlim([-0.2, 1.8])

                ax.set_ylabel(f'LOO: {layer_i}')
                ax.set_yticks([-1.2, -0.6, 0])
                ax.invert_yaxis()  # Invert y-axis for better visualization

            sns.despine(trim=True)
            fig.tight_layout(pad=1)  # Reduce padding inside layout
            fig.subplots_adjust(wspace=0.05, hspace=0.3)  # Reduce space between subplots
            fig.savefig(
                os.path.join(PATH_FIGURES,
                             f'{monkey_name}_{cue}_diff_distance_layers_leave_one_out_'
                             f'{behavior}.png'),
                bbox_inches='tight', dpi=300)

    # SINGLE LAYER
    for monkey_name in ['Mo', 't']:
        # After loop, concatenate
        final_summary_df = pd.concat(all_summaries).reset_index()

        # change this to select the monkey
        final_summary_df = final_summary_df[final_summary_df['session'].str.startswith(monkey_name)]

        # number of sites
        num_sites = final_summary_df['session'].nunique()

        # correct the p_values
        rejected, p_vals_corrected, _, _ = multipletests(
            pvals=final_summary_df.groupby('session').p_val_sc_gen.mean().values,
            alpha=0.05,
            method='bonferroni')

        # keep the corrected p_values
        p_vals_h1_corrected = p_vals_corrected[:num_sites]
        p_vals_h0 = final_summary_df.groupby('session').p_val_sc.mean().values

        sns.set(style="ticks", context='poster')

        # Define colors
        palette = {False: '#5E5768', True: '#C8A676'}

        # Create subplots
        fig, axes = plt.subplots(5, 2, figsize=(10, 30), sharey='row')

        for i, same_cue in enumerate([True, False]):
            ax = axes[0, i]
            df_event = final_summary_df[(final_summary_df['same_cue'] == same_cue)]

            # Pivot so each session has both shuffled and not shuffled
            df_pivot = df_event.pivot(index='session', columns='shuffled', values='diff_distance')
            df_pivot = df_pivot.dropna()  # Remove incomplete pairs

            # add the corrected p_values
            if i == 0:
                df_pivot['p_val_corrected'] = p_vals_h0
                ax.set_title(f'{cue} ')
            elif i == 1:
                df_pivot['p_val_corrected'] = p_vals_h1_corrected
                ax.set_title(f'{cue}: {sum(rejected)}/{num_sites}')

            # Plot each session
            for session, row in df_pivot.iterrows():
                if row.p_val_corrected < 0.05:
                    # significant
                    ax.plot([0, 1], [row[False], row[True]], color='#5E5768', alpha=.75,
                            linewidth=1.5)
                    io.logger.info(f"{session}")

                else:
                    ax.plot([0, 1], [row[False], row[True]], color='#C8A676', alpha=.75,
                            linewidth=1.5)

                ax.scatter([-0, 1], [row[False], row[True]],
                           color=[palette[False], palette[True]],
                           alpha=0.8, edgecolor='k', linewidth=0.5, zorder=2)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Not Shuffled', 'Shuffled'])

            # ax.set_ylim([-1.25, 0.05])
            ax.set_ylim([-1.25, 0.15])
            ax.set_xlim([-0.2, 1.8])
            if i == 0:
                ax.set_ylabel('Difference distance')
                ax.set_yticks([-1.8, -1.2, -0.6, 0])

            ax.set_ylabel('Difference distance')
            ax.set_yticks([-1.2, -0.6, 0])
            ax.invert_yaxis()  # Invert y-axis for better visualization

        # PLOT SEPARATING LAYER ANALYSIS
        for idx_layer, layer_i in enumerate(['L23-SUP', 'L23-DEEP', 'L5', 'L6']):
            # After loop, concatenate
            final_summary_df = pd.concat(summaries_sl).reset_index()

            # change this to select the monkey
            final_summary_df = final_summary_df[
                final_summary_df['session'].str.startswith(monkey_name)]

            # number of sites
            num_sites = final_summary_df['session'].nunique()

            # correct the p_values
            rejected, p_vals_corrected, _, _ = multipletests(
                pvals=final_summary_df.groupby('session')[f'p_val_{cue}_sl_{layer_i}_gen']
                .mean().values,
                alpha=0.05,
                method='bonferroni')

            # keep the corrected p_values
            p_vals_h1_corrected = p_vals_corrected[:num_sites]
            p_vals_h0 = final_summary_df.groupby('session')[f'p_val_{cue}_sl_{layer_i}']\
                .mean().values

            # correct for any possible Nan (coming from no layers present at that probe)
            p_vals_h1_corrected = p_vals_h1_corrected[~np.isnan(p_vals_h1_corrected)]
            p_vals_h0 = p_vals_h0[~np.isnan(p_vals_h0)]

            # correct the number of sites as well
            num_sites = len(p_vals_h0)

            sns.set(style="ticks", context='poster')

            # Define colors
            # palette = {False: '#d0ba98', True: '#b8b6b0'}
            palette = {False: '#5E5768', True: '#C8A676'}

            for i, same_cue in enumerate([True, False]):
                ax = axes[idx_layer+1, i]
                df_event = final_summary_df[(final_summary_df['same_cue'] == same_cue) &
                                            (final_summary_df['layer_used'] == layer_i)]

                # Pivot so each session has both shuffled and not shuffled
                df_pivot = df_event.pivot(index='session', columns='shuffled',
                                          values='diff_distance')
                df_pivot = df_pivot.dropna()  # Remove incomplete pairs

                # add the corrected p_values
                if i == 0:
                    df_pivot['p_val_corrected'] = p_vals_h0
                    ax.set_title(f'SC2 {np.sum(p_vals_h0<0.05)}/{num_sites}')
                elif i == 1:
                    df_pivot['p_val_corrected'] = p_vals_h1_corrected
                    ax.set_title(f'Gen - {sum(rejected)}/{num_sites}')

                # Plot each session
                for session, row in df_pivot.iterrows():
                    if row.p_val_corrected < 0.05:
                        # significant
                        ax.plot([0, 1], [row[False], row[True]], color='#5E5768', alpha=.75,
                                linewidth=1.5)
                        io.logger.info(f"{session}")

                    else:
                        ax.plot([0, 1], [row[False], row[True]], color='#C8A676', alpha=.75,
                                linewidth=1.5)

                    ax.scatter([-0, 1], [row[False], row[True]],
                               color=[palette[False], palette[True]],
                               alpha=0.8, edgecolor='k', linewidth=0.5, zorder=2)

                ax.set_xticks([0, 1])
                ax.set_xticklabels(['Not Shuffled', 'Shuffled'])

                ax.set_ylim([-1.25, 0.15])
                ax.set_xlim([-0.2, 1.8])

                ax.set_ylabel(f'SL: {layer_i}')
                ax.set_yticks([-1.2, -0.6, 0])
                ax.invert_yaxis()  # Invert y-axis for better visualization

            sns.despine(trim=True)
            fig.tight_layout(pad=1)  # Reduce padding inside layout
            fig.subplots_adjust(wspace=0.05, hspace=0.3)  # Reduce space between subplots
            fig.savefig(
                os.path.join(PATH_FIGURES,
                             f'{monkey_name}_{cue}_diff_distance_layers_single_layer_'
                             f'{behavior}.png'),
                bbox_inches='tight', dpi=300)
