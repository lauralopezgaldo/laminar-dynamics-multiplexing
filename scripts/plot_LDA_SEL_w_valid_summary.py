"""
This script opens the dataframe with the distance/accuracy for LDA at SEL and testing two hypothesis
the color hypothesis and the validity axis.
"""
import os
import warnings
import numpy as np
import h5py
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from frites import io
import seaborn as sns
from statsmodels.stats.multitest import multipletests


def perm_test(x, y, n_perms=1000, stat=np.mean):
    diff = np.abs(stat(x) - stat(y))
    pooled = np.concatenate([x, y])
    count = 0
    for n in range(n_perms):
        rng = np.random.default_rng(n)
        rng.shuffle(pooled)
        x_perm = pooled[:len(x)]
        y_perm = pooled[len(x):]
        diff_perm = np.abs(stat(x_perm) - stat(y_perm))
        if diff_perm >= diff:
            count += 1

    pv = count / n_perms
    return pv


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
    all_summaries = []

    current_path = os.getcwd()

    if current_path.startswith('C:'):
        server = 'W:'  # local w VPN
    else:
        server = '/envau/work'  # niolon

    PATH_FIGURES = server + \
        '/comco/lopez.l/Electrophysiology/ephy_laminar_MUA/Results/Full_trial/' \
        'lda_SEL_plots/'

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
                             'LDA_SEL' in i and behavior in i]
            if len(file_name_mua) == 0:
                io.logger.info(f'No MUA file found for probe {probe}')
                continue
            else:
                file_name_mua = file_name_mua[0]

            # load the dataframe
            df = pd.read_csv(os.path.join(PATH_DATA, file_name_mua))

            # check if there is a significant SEL space
            p_val_sel = \
                get_p_val_w_percentile(observations=df[(df.event == 'SEL') &
                                                       (df.shuffled == False)].diff_distance,
                                       shuffled_observations=df[(df.event == 'SEL') &
                                       (df.shuffled == True)].diff_distance, stat=np.mean,
                                       prediction='smaller')

            if p_val_sel > 0.05:
                io.logger.info(f'SEL space not significant for {SESSION} probe {probe}')
                continue

            # run your code here per session
            summary_df = df.groupby(['event', 'shuffled']).mean(numeric_only=True)

            # Add identifying info (e.g., session name)
            summary_df['session'] = f'{SESSION}-{probe}'

            # get the p-value for the generalization of hypothesis 1 and 2 for each session
            p_val_h1 = \
                get_p_val_w_percentile(observations=df[(df.event == 'SC_color') &
                                                       (df.shuffled == False)].diff_distance,
                                       shuffled_observations=df[(df.event == 'SC_color') &
                                       (df.shuffled == True)].diff_distance, stat=np.mean,
                                       prediction='smaller')
            p_val_h2 = \
                get_p_val_w_percentile(observations=df[(df.event == 'validity_axis') &
                                                       (df.shuffled == False)].diff_distance,
                                       shuffled_observations=df[(df.event == 'validity_axis') &
                                       (df.shuffled == True)].diff_distance, stat=np.mean,
                                       prediction='smaller')

            # Add the hypothesis
            summary_df['p_val_h1'] = p_val_h1
            summary_df['p_val_h2'] = p_val_h2

            # Append to list
            all_summaries.append(summary_df)

    for monkey_name in ['Mo', 't']:
        # After loop, concatenate
        final_summary_df = pd.concat(all_summaries).reset_index()
        final_summary_df = final_summary_df[final_summary_df['event'] != 'SC_valid']

        # change this to select the monkey
        final_summary_df = final_summary_df[final_summary_df['session'].str.startswith(monkey_name)]

        # Remove weird sites
        # weird_sites = ['t150303002-1', 't150319003-2']

        # number of sites
        num_sites = final_summary_df['session'].nunique()

        # correct the p_values
        rejected, p_vals_corrected, _, _ = multipletests(
            pvals=np.append(final_summary_df.groupby('session').p_val_h1.mean().values,
                            final_summary_df.groupby('session').p_val_h2.mean().values), alpha=0.05,
            method='bonferroni')

        # keep the corrected p_values
        p_vals_h1_corrected = p_vals_corrected[:num_sites]
        p_vals_h2_corrected = p_vals_corrected[num_sites:]

        sns.set(style="ticks", context='poster')

        # Define colors
        # palette = {False: '#d0ba98', True: '#b8b6b0'}
        palette = {False: '#5E5768', True: '#C8A676'}
        # Get unique events
        events = final_summary_df['event'].unique()

        # Create subplots
        fig, axes = plt.subplots(1, len(events), figsize=(4 * len(events), 5), sharey='row')

        for i, event in enumerate(['SEL', 'SC_color', 'validity_axis']):
            ax = axes[i]
            df_event = final_summary_df[final_summary_df['event'] == event]

            # Pivot so each session has both shuffled and not shuffled
            df_pivot = df_event.pivot(index='session', columns='shuffled', values='diff_distance')
            df_pivot = df_pivot.dropna()  # Remove incomplete pairs

            # add the corrected p_values
            if i == 1:
                df_pivot['p_val_corrected'] = p_vals_h1_corrected
            elif i == 2:
                df_pivot['p_val_corrected'] = p_vals_h2_corrected

            # Plot each session
            for session, row in df_pivot.iterrows():
                if event == 'SEL':
                    # significant
                    ax.plot([0, 1], [row[False], row[True]], color='#5E5768', alpha=.75,
                            linewidth=1)
                    io.logger.info(f"{session}-significant-SEL")
                elif (event != 'SEL') & (row.p_val_corrected < 0.05):
                    # significant
                    ax.plot([0, 1], [row[False], row[True]], color='#5E5768', alpha=1, linewidth=1)
                    io.logger.info(f"{session}-significant-{event}")
                else:
                    ax.plot([0, 1], [row[False], row[True]], color='#C8A676', alpha=.75,
                            linewidth=1)

                ax.scatter([-0, 1], [row[False], row[True]],
                           color=[palette[False], palette[True]],
                           alpha=0.8, edgecolor='k', linewidth=0.5, zorder=2)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Not Shuffled', 'Shuffled'])
            ax.set_title(event)
            ax.set_ylim([-0.45, 0.05])
            ax.set_xlim([-0.2, 1.5])
            if i == 0:
                ax.set_ylabel('Difference distance')
                ax.set_yticks([-0.4, -0.2, 0])

        ax.invert_yaxis()  # Invert y-axis for better visualization
        sns.despine(trim=True)
        fig.tight_layout(pad=1)  # Reduce padding inside layout
        fig.subplots_adjust(wspace=0.05)  # Reduce space between subplots
        fig.savefig(f'{PATH_FIGURES}/LDA_SEL_acoss_sites_{monkey_name}.svg', bbox_inches='tight')
