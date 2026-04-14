import os
import warnings
import numpy as np
import h5py
import pandas as pd
import xarray as xr
from frites import io
from src.preprocessing_tools import cut_mua_by_markers
from src.decoding_tools import split_train_test, run_dimensionality_reduction_analysis_crossval, \
    correct_sign_with_reference, get_lda_weight_vector, correct_consecutive_sign_flips, \
    compute_cosine_similarity, compute_shannon_entropy_xr, compute_mutual_information
from src.plotting_tools import plot_mua_per_mvt_dir_ci, plot_mua_per_trial_type_ci, \
    plot_similarity_matrix, plot_weight_evolution

# to remove the spam of pandas FutureWarning with iteritems
warnings.simplefilter(action='ignore', category=FutureWarning)


if __name__ == "__main__":
    # define the sessions to run the analysis
    ALL_LAMINAR = ['Mo180412002']

    # all possible probes
    probes = [1]

    # set the path
    PATH = 'data/'

    # define the method to run
    method = 'LDA'  # 'PCA' or 'LDA'
    behavior = 'mvt_dir'  # 'trial_type' or 'mvt_dir'
    trial_type = 2  # trial type to use to compute the mvt_dir behavior
    trial_type_labels = ['blue', 'green', 'pink']

    # specific parameters
    window_size = 0.2  # 150ms
    t_step = 0.025  # 25ms

    # loop over the sessions
    for SESSION in ALL_LAMINAR:
        # load the excel file from the PATH_Figures
        excel_file = os.path.join(PATH, f'Mua_check_channels.xlsx')

        # load the excel file
        df_mua_info = pd.read_excel(excel_file)

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
            # if there are not all trial types just skip the session
            if (len(np.unique(mua_site.trial_type.values)) < 3) & (behavior == 'trial_type'):
                continue

            # 1) Cut the MUA from TOUCH to MVT + 200 ms
            mua_site_shorter, n_times = \
                cut_mua_by_markers(mua_xr=mua_site, events_alignment=['touch', 'MVT'],
                                   t_extra=[0, 0.2])

            # 2) Select the channels of interest
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
                    mua_site_shorter.loc[dict(channels=ch)] = np.nan

                # Remove any channel that has any NaN
                has_nan = mua_site_shorter.isnull().any(dim=('trial_type', 'times'))
                mua_site_shorter = mua_site_shorter.sel(channels=~has_nan)

            # 3) Sub-select trials for training and testing the model
            seed = 0  # same seed only for reproducibility
            if behavior == 'trial_type':
                mua_train, mua_test = split_train_test(mua_xr=mua_site_shorter,
                                                       condition=behavior,
                                                       seed=seed, percent_train=0.5)
            elif behavior == 'mvt_dir':
                mua_train, mua_test = split_train_test(mua_xr=mua_site_shorter
                                                       .sel(trial_type=trial_type),
                                                       condition=behavior,
                                                       seed=seed, percent_train=0.7)
            else:
                raise ValueError('behavior should be either trial_type or mvt_dir')

            # 4) Define the parameters for the time windows
            # convert the time into samples based on the time-steps of the data
            dt = mua_site.times.values[1] - mua_site.times.values[0]
            n_times_window = int(np.ceil(window_size / dt))
            n_step = int(np.ceil(t_step / dt))

            # generate the starting indices for the time-windows
            idx_t_windows_start = np.arange(0, n_times - n_times_window, n_step)
            n_windows = len(idx_t_windows_start)

            # define dimensions and dimension labels
            n_dims = 2  # number of dimensions
            dimension_labels = [f'dim-{i}' for i in range(1, n_dims + 1)]

            # 5) Convert x-arrays to numpy to speed up the computations
            mua_train_values = mua_train.values.swapaxes(0, 1)  # n_channels, n_trials, n_times
            mua_train_labels = mua_train[behavior].values  # n_trials

            mua_test_values = mua_test.values.swapaxes(0, 1)  # n_channels, n_trials, n_times
            mua_test_labels = mua_test[behavior].values  # n_trials

            # get all the middle times of the time-windows
            all_mua_times = mua_site_shorter.times.values
            all_middle_times = np.round(all_mua_times[idx_t_windows_start +
                                                      int(n_times_window / 2)], 2)

            # 6) Iterate on the possible time-windows
            all_weights = []  # to store the weights of the PCA/LDA
            all_projections = []  # to store the projections of the data

            for i_window, idx_t_window_start in enumerate(idx_t_windows_start):
                # get the MUA to train in the time window
                mua_epoch_train = mua_train_values[:, :,
                                                   idx_t_window_start:idx_t_window_start +
                                                   n_times_window]

                # get the MUA for testing
                mua_epoch_test = mua_test_values[:, :,
                                                 idx_t_window_start:idx_t_window_start +
                                                 n_times_window]

                # Run the PCA/LDA
                weights_epoch, \
                    projections_epoch = \
                    run_dimensionality_reduction_analysis_crossval(mua_train_arr=mua_epoch_train,
                                                                   mua_test_arr=mua_epoch_test,
                                                                   method_name=method,
                                                                   num_dims=n_dims,
                                                                   behavioral_labels=\
                                                                   mua_train_labels)

                # append the results to a list
                all_weights.append(weights_epoch)
                all_projections.append(projections_epoch)

            # 7) Recreate the x-arrays
            weight_evolution = xr.DataArray(np.array(all_weights).swapaxes(0, 1),
                                            dims=('layers', 'times', 'dimensions'),
                                            coords={'layers': mua_site_shorter.layers.values,
                                                    'dimensions': dimension_labels,
                                                    'times': all_middle_times},
                                            attrs=mua_site.attrs)

            proj_single_trials_avg_xr = xr.DataArray(np.array(all_projections), dims=('times',
                                                                                      'trial_type',
                                                                                      'dimensions'),
                                                     coords={'times': all_middle_times,
                                                             'dimensions': dimension_labels,
                                                             'trial_type':
                                                                 mua_test['trial_type'].values},
                                                     attrs=mua_test.attrs)

            # add more information for the single trial labels (other behavioral conditions)
            proj_single_trials_avg_xr = \
                proj_single_trials_avg_xr.assign_coords(mvt_dir=('trial_type',
                                                                 mua_test.mvt_dir.values),
                                                        unamb_mask=('trial_type',
                                                                    mua_test.unamb_mask.values),
                                                        block=('trial_type', mua_test.block.values),
                                                        t_number=('trial_type',
                                                                  mua_test.t_number.values))

            # 8) Flip the signs when using LDA
            if method == 'LDA':
                # get the reference vector
                if behavior == 'mvt_dir':
                    event_name_ref = 'GO'
                    reference_vector_lda = get_lda_weight_vector(mua_xr=mua_train,
                                                                 event_name=event_name_ref,
                                                                 behavioral_parameter=behavior)
                    new_weight_evolution, new_projection_evolution = \
                        correct_sign_with_reference(weights_xr=weight_evolution,
                                                    projected_data_xr=proj_single_trials_avg_xr,
                                                    reference_vec=reference_vector_lda)
                elif behavior == 'trial_type':
                    new_weight_evolution, new_projection_evolution = \
                        correct_consecutive_sign_flips(weights_xr=weight_evolution,
                                                       projected_data_xr=proj_single_trials_avg_xr,
                                                       num_dims=n_dims,
                                                       behavioral_condition=behavior)
                else:
                    raise ValueError('behavior should be either trial_type or mvt_dir')

            else:
                new_weight_evolution = weight_evolution
                new_projection_evolution = proj_single_trials_avg_xr

            # compute the similarity matrix
            similarity = compute_cosine_similarity(new_weight_evolution.sel(dimensions='dim-1').T,
                                                   new_weight_evolution.sel(dimensions='dim-1').T)

            # compute shannon entropy of the weights and the single trial mutual information of the
            # projections
            new_projection_evolution = \
                compute_mutual_information(projected_data_xr=new_projection_evolution,
                                           num_dims=n_dims, behavioral_condition=behavior)
            new_weight_evolution_full = \
                compute_shannon_entropy_xr(weights_xr=new_weight_evolution, num_dims=n_dims)

            # 5) Plot the results
            # plot the evolution of the projections
            if behavior == 'mvt_dir':
                figure_conditions_projection = \
                    plot_mua_per_mvt_dir_ci(new_projection_evolution.sel(dimensions='dim-1'))

            elif behavior == 'trial_type':
                figure_conditions_projection = \
                    plot_mua_per_trial_type_ci(new_projection_evolution.sel(dimensions='dim-1'))

            # plot the similarity matrix
            plot_similarity_matrix(weight_evolution, similarity, title=f'Similarity of {SESSION}')

            # plot the weight evolution
            plot_weight_evolution(coefficients_evolution=new_weight_evolution, i_dimension=0)

            io.logger.info(f'Finished processing {SESSION} probe {probe} with {method} method!')
