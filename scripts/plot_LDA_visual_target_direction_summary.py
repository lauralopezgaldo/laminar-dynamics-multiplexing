"""
This script opens the dataframe with the distance/accuracy for LDA at SCs and tests the hypothesis
at the SCs for the target direction.
It computes the plots per cues, so if the space exist for a given site for a given cue, it checks
whether hypothesis I is significant, correcting for multiple comparisons in each monkey.
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
    behavior = 'mvt_dir'
    cue = 'SC2'
    all_summaries = []

    current_path = os.getcwd()

    if current_path.startswith('C:'):
        server = 'W:'  # local w VPN
    else:
        server = '/envau/work'  # niolon

    PATH_FIGURES = server + \
        '/comco/lopez.l/Electrophysiology/ephy_laminar_MUA/Results/Full_trial/' \
        'lda_SCs_distance_plots/'

    # iterate on the sessions
    for SESSION in ALL_LAMINAR_NEW:
        PATH_DATA = server + \
                    '/comco/lopez.l/Electrophysiology/ephy_laminar_MUA/Results/Full_trial/' \
                    'data_paper/' + SESSION + '/'

        # iterate on the probes
        for probe in probes:
            # Open in this case the single trials aligned to SEL because the SC2 ones are not full
            file_name_mua = [i for i in os.listdir(PATH_DATA) if
                             os.path.isfile(os.path.join(PATH_DATA, i))
                             and
                             f'{SESSION}_{probe}' in i and '.csv' in i and 'cv_accuracy' in i and
                             'LDA_SCs' in i and behavior in i]
            if len(file_name_mua) == 0:
                io.logger.info(f'No MUA file found for probe {probe}')
                continue
            else:
                file_name_mua = file_name_mua[0]

            # load the dataframe
            df = pd.read_csv(os.path.join(PATH_DATA, file_name_mua))

            # check if there is a significant SEL space
            p_val_sci = \
                get_p_val_w_percentile(observations=df[(df.event_train == cue) &
                                                       (df.same_cue == True) &
                                                       (df.shuffled == False)].diff_distance,
                                       shuffled_observations=df[(df.event_train == cue) &
                                                                (df.same_cue == True) &
                                                                (df.shuffled ==
                                                                 True)].diff_distance,
                                       stat=np.mean,
                                       prediction='smaller')

            if p_val_sci > 0.05:
                io.logger.info(f'{cue}-dir space not significant for {SESSION} probe {probe}')
                continue

            # run your code here per session
            summary_df = df.groupby(['event_train', 'same_cue',
                                     'shuffled']).mean(numeric_only=True)

            # Add identifying info (e.g., session name)
            summary_df['session'] = f'{SESSION}-{probe}'

            # get the p-value for the generalization of hypothesis 1 and 2 for each session
            p_val_sci_gen = \
                get_p_val_w_percentile(observations=df[(df.event_train == cue) &
                                                       (df.same_cue == False) &
                                                       (df.shuffled == False)].diff_distance,
                                       shuffled_observations=df[(df.event_train == cue) &
                                                                (df.same_cue == False) &
                                                                (df.shuffled ==
                                                                 True)].diff_distance,
                                       stat=np.mean,
                                       prediction='smaller')

            # Add the hypothesis
            summary_df['p_val_sc2'] = p_val_sci
            summary_df['p_val_sc2_gen'] = p_val_sci_gen

            # Append to list
            all_summaries.append(summary_df)

    for monkey_name in ['Mo', 't']:
        # After loop, concatenate
        final_summary_df = pd.concat(all_summaries).reset_index()

        # change this to select the monkey
        final_summary_df = final_summary_df[final_summary_df['session'].str.startswith(monkey_name)]

        # number of sites
        num_sites = final_summary_df['session'].nunique()

        # correct the p_values
        rejected, p_vals_corrected, _, _ = multipletests(
            pvals=final_summary_df.groupby('session').p_val_sc2_gen.mean().values,
            alpha=0.05,
            method='bonferroni')

        # keep the corrected p_values
        p_vals_h1_corrected = p_vals_corrected[:num_sites]
        p_vals_h0 = final_summary_df.groupby('session').p_val_sc2.mean().values

        sns.set(style="ticks", context='poster')

        # Define colors
        # palette = {False: '#d0ba98', True: '#b8b6b0'}
        palette = {False: '#5E5768', True: '#C8A676'}
        # Get unique events
        # events = final_summary_df['event'].unique()

        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(4 * 2, 5), sharey='row')

        for i, same_cue in enumerate([True, False]):
            ax = axes[i]
            df_event = final_summary_df[(final_summary_df['same_cue'] == same_cue) &
                                        (final_summary_df['event_train'] == cue)]

            # Pivot so each session has both shuffled and not shuffled
            df_pivot = df_event.pivot(index='session', columns='shuffled', values='diff_distance')
            df_pivot = df_pivot.dropna()  # Remove incomplete pairs

            # add the corrected p_values
            if i == 0:
                df_pivot['p_val_corrected'] = p_vals_h0
                ax.set_title(cue)
            elif i == 1:
                df_pivot['p_val_corrected'] = p_vals_h1_corrected
                ax.set_title(cue + '- Gen')

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

        ax.invert_yaxis()  # Invert y-axis for better visualization
        sns.despine(trim=True)
        fig.suptitle(f'{monkey_name} {cue} - {sum(rejected)}/{num_sites}', fontsize=16)
        fig.tight_layout(pad=1)  # Reduce padding inside layout
        fig.subplots_adjust(wspace=0.05)  # Reduce space between subplots
        fig.savefig(os.path.join(PATH_FIGURES, f'{monkey_name}_{cue}_diff_distance.svg'),
                    bbox_inches='tight', dpi=300)
