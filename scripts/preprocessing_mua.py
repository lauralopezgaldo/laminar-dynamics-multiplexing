"""
This scrip generates two mne objects, one with continuous electrophysiological signal (MUA)
and the other one with annotations based on the matrices obtained from MATLAB. For MUA, it
removes the moving average in order to avoid as much as possible the drift. The data is cut in
epochs depending on the alignment marker - either outputing the full trial or a time window around
the marker.
"""

# import packages
import os
import warnings
import ast
import h5py
import pickle
import numpy as np
import pandas as pd
import mne
from frites import io
from scipy.stats import zscore
from scipy.ndimage import uniform_filter1d
from src.preprocessing_tools import open_matlab_lfp, concatenate_probes, open_matlab_behaviour,\
    open_matlab_analog, cut_in_epochs, get_layer_mask, block_from_trial, \
    get_depth_from_layer

# to remove the spam of pandas FutureWarning with iteritems
warnings.simplefilter(action='ignore', category=FutureWarning)


def preprocessing_pipeline(session, alignment='Sel', signal_type='LFP', complete_trial=True):
    """
    Build the mne epochs object and annotations with the correct trials, for one individual session.
    Use the .mat files to check the behaviour and the LFPs of each of the probes. Get the data in
    (n_trials, n_channels, n_times) with all the metadata associated, e.g. depths, electrode
    spacing, areas.
    :param session: string with the name of the session, e.g. 't150327002'
    :param alignment: string naming the marker inside the trial used for alignment. The options are
    'Sel', 'SC1', 'SC2', 'SC3', 'Go' or 'MVT'. Default is 'Sel'.
    :param signal_type: string with the type of signal to be loaded. The options are 'LFP' or 'MUA'.
    :param complete_trial: bool. If True, the epoch will contain the full trial. If False, just a
    time_window around the alignment marker will be kept.
    ----
    :return: saved 2 mne objects, one with annotations and the other with epochs: '-annot.fif',
    '-epo.fif'. The name includes the session, the alignment and the part of the trial - either full
    or t_window.
    """
    if complete_trial:
        print('Including full trial time.')

    # check where are we running the code
    current_path = os.getcwd()

    if current_path.startswith('C:'):
        server_bjorg = 'L:'  # local w VPN
        server_mua = 'W:'
    else:
        server_bjorg = '/hpc'  # niolon
        server_mua = '/envau/work/'

    path = server_mua + '/comco/lopez.l/Electrophysiology/ephy_laminar_MUA/Raw_data/'

    # doesn't work bd of HPC
    # path_analog = server_bjorg + \
    #     '/comco/kilavik.b/MatlabScripts/Behavior/Results/HandEyeMovements/Data'

    path_output = server_mua + '/comco/lopez.l/Electrophysiology/' \
                               'ephy_laminar_MUA/Results/Preprocessed_data/' + session + '/'

    # check if the preprocessing folder already exists - else create it
    if not os.path.exists(path_output):
        os.mkdir(path_output)

    # Check that the signal type is either 'LFP' or 'MUA'
    while signal_type not in ['LFP', 'MUA']:
        signal_type = input('Select the signal type: LFP or MUA.')

    # get the file names
    file_lfp_probe1 = os.path.join(path, f"{signal_type}-" + session + "-1.mat")
    file_behaviour = os.path.join(path, "behaviour-" + session + ".mat")
    file_electrodes_info = os.path.join(path, "ephydataset_info_MUA.xlsx")
    # doesn't work bd of HPC
    # file_behaviour_analog = os.path.join(path_analog, session + "_HandEyeData.mat")

    # 0) ELECTRODES' INFO FILE. Get the information of the session
    df_electrodes = pd.read_excel(file_electrodes_info, sheet_name=0)  # read the excel with pandas
    df_electrodes.set_index('SESSION', inplace=True, drop=True)  # get session as the search index

    area_1 = df_electrodes.loc[session]['PROBE_1']  # get the brain area of  the first probe

    # 0.1) Get the bad channels from the excel file as long as the signal is 'LFP'
    bad_channels = None

    # print the details of the preprocessing
    print(f'Session {session}, alignment on {alignment}. Bad channels are {bad_channels}. \n')

    # 1) BEHAVIOUR. Get the behavioural information of the session coming from the matlab structure.
    # Build a dictionary with all the available fields
    behaviour = open_matlab_behaviour(file_behaviour)

    # 2) ANALOG SIGNAL. Get the hand and eye movement given by the external devices, as well as
    # their precise timing.
    # doesn't work bd of HPC
    # behaviour_analog = open_matlab_analog(file_behaviour_analog)

    # If eliCorrect was used in OrganizeBehaviour [matlab] the number of trials between behaviour
    # and behaviour_analog will not match. For this we need a 'good_trials_mask'
    # idx of the good correct trials starting in 0
    # doesn't work bd of HPC
    good_trials_mask = behaviour['OrigCorrTrials'] - 1

    # 3) LFP. Include the LFP coming from SortChanLFP matlab script. Check if there are two probes
    # for the same session or just one. If two probes, concatenate and name the channels
    # accordingly. One recording session will correspond to one LFP object.

    if df_electrodes.loc[session]['PROBE_2'] != '-':
        file_lfp_probe2 = os.path.join(path, f"{signal_type}-" + session + "-2.mat")
        area_2 = df_electrodes.loc[session]['PROBE_2']  # get the brain area of the first probe

        channel_names, all_lfps = concatenate_probes(filename_probe1=file_lfp_probe1,
                                                     filename_probe2=file_lfp_probe2,
                                                     loc_1=area_1, loc_2=area_2)
    else:
        all_lfps = open_matlab_lfp(file_lfp_probe1)
        channel_names = np.array([area_1 + '-' + str(ch) for ch in range(1, all_lfps.shape[0] + 1)])
        area_2 = []

    # 4) FILTERING. Just remove the moving average in a big enough window to get rid of the drift
    fs_lfp = 1000  # sampling frequency of the MUA signal in Hz
    # de-mean the MUA signal with a sliding window, to minimize drift effects
    # get the running average
    window_filter = 3001  # window size in samples
    average_window = window_filter * 10
    mua_smoothed = uniform_filter1d(all_lfps, size=average_window, axis=1, mode='reflect')
    # subtract the running average
    mua_wo_drift = all_lfps - mua_smoothed
    # z-score the signal per channel
    mua_zscored = zscore(mua_wo_drift, axis=1)

    # 5) EPOCHS. Cut the continuous LFP in discrete epochs based on trial markers.
    # User should select the alignment marker. It is possible to align to any of the visual cues, or
    # to hand movement onset.
    alignments = ['Sel', 'SC1', 'SC2', 'SC3', 'Go', 'MVT']

    # average timing for the events: [touch SEL(0) SC1 SC2 SC3 GO MVT Target Rew]
    avg_event_times = behaviour['RelativeAverageTimes']

    # get the timing of the LED whenever it is possible. Not all sessions have LED argument.
    if 'SelTimLed' in behaviour:
        marker = 'TimLed'
    else:
        marker = 'Tim'
    event_markers = behaviour['Sel' + marker]  # pre-define the event markers as aligning on 'SEL'

    # ask until a valid alignment
    while alignment not in alignments:
        alignment = input('Select the alignment marker: Sel, SC1, SC2, SC3, Go or MVT.')

    if alignment == 'MVT':
        # event_markers = behaviour_analog['HandMovementOnset'][good_trials_mask].astype(int)
        # event_markers = behaviour_analog['HandMovementOnset'].astype(int)
        raise Exception('No access to the analog data from hpc')

        # io.logger.info(f'Aligning on Hand Movement Onset. Number of trials for'
        #                f'hand movement in the analog signal {len(event_markers)}.'
        #                f'Number of trials in the behaviour {len(behaviour["SelTim"])}.')
    elif alignment == 'Sel':
        event_markers = behaviour[alignment + marker]
    elif alignment == 'SC1':
        event_markers = behaviour[alignment + marker]
    elif alignment == 'SC2':
        event_markers = behaviour[alignment + marker]
    elif alignment == 'SC3':
        event_markers = behaviour[alignment + marker]
    elif alignment == 'Go':
        event_markers = behaviour[alignment + marker]

    # decide if the epoch contains the full trial. If not full trial, determine the window around
    # the event marker
    if complete_trial:  # THIS WORKS CORRECTLY ONLY FOR SEL ALIGNMENT!!
        t_start = avg_event_times[0]  # negative touch time wrt SEL(0)
        # t_end = avg_event_times[7]  # positive time when subjects reach the target wrt SEL(0)
        t_end = avg_event_times[6] + 2000
        t_min = 500  # ms - default!

        event_labels = ['start', 'touch', 'SEL', 'SC1', 'SC2', 'SC3', 'GO', 'MVT']  # event labels
        avg_event_times = np.insert(avg_event_times, 0, t_start-500)  # average event times
        event_duration = np.diff(avg_event_times[:9])/1000  # set the event duration [s]
        event_duration[2:6] = .3  # use the cue appearance as the event duration (SEL and 3 SCs) [s]

        avg_event_times = avg_event_times[0:8]/1000  # get only the ones matching the labels [s]
        trial_part = 'full'

    else:
        t_min = 0
        cue_duration = 300  # ms
        # t_window = int(input('Window size [in ms] around the event marker:'))
        t_window = 1400  # ms
        io.logger.info(f'Window size: {t_window} ms')
        t_start = -int(t_window/2)
        t_end = int(t_window/2) + cue_duration  # add the cue duration to the end of the window
        # build the event labels
        # event_labels = ['pre_cue', alignment, 'post_cue']
        event_labels = [f'pre_{alignment}', alignment, f'post_{alignment}']
        # event duration in seconds
        event_duration = [t_window/(2*1000), cue_duration, t_window/(2*1000)]
        avg_event_times = [t_start/1000, 0, t_end/1000]  # average event times in sec
        trial_part = str(t_window) + 'ms'

    mua_epochs = cut_in_epochs(mua_zscored, t_markers=event_markers, t_start=t_start, t_end=t_end,
                               t_min=t_min)

    # 6) MNE OBJECT. Build an mne object with the epochs array.
    n_trials, _, _ = mua_epochs.shape

    # Define the metadata to be included in the MNE object.
    # original trial number (before removing bad trials, both elitrials and other bad trials)
    trial_number = behaviour['OrigCorrTrials']
    trial_number_rel = behaviour['OrigRelCorrTrials']
    trial_number_unamb = behaviour['unamb']
    trial_mask_unamb = np.isin(trial_number, trial_number_unamb).astype(int)

    # get the movement direction, w/ %10 we get the last digit of the code
    mvt_direction = np.array([direction % 10 for direction in behaviour['MVTdir']]).T

    # trial type: 1, 2, 3 (Blue, Green or Pink) and block to which trials belong
    trial_type = behaviour['ttype']
    trial_block = block_from_trial(trial_type)

    # trial events contains the alignment time of each trial - on the full session time - the
    # movement direction, and the trial type.
    trial_events = np.stack((event_markers, mvt_direction, trial_type)).transpose()
    # dictionary linking the trial_type number to the label of the trial
    trial_events_id = dict(zip(np.array(['blue', 'green', 'pink']), np.unique(trial_type)))

    # layer information and depth information for both probes - if exists
    layers_probe_1 = get_layer_mask(df_layers=df_electrodes.loc[session].LAYERS_1,
                                    n_channels=df_electrodes.loc[session].CHANNELS_1)
    depths_probe_1 = get_depth_from_layer(layer_mask=layers_probe_1,
                                          electrode_spacing=df_electrodes.loc[session].S1)
    electrode_spacing_1 = df_electrodes.loc[session].S1

    if df_electrodes.loc[session]['PROBE_2'] != '-':
        layers_probe_2 = get_layer_mask(df_layers=df_electrodes.loc[session].LAYERS_2,
                                        n_channels=df_electrodes.loc[session].CHANNELS_2)
        depths_probe_2 = get_depth_from_layer(layer_mask=layers_probe_2,
                                              electrode_spacing=df_electrodes.loc[session].S2)
        electrode_spacing_2 = df_electrodes.loc[session].S2

        # build the metadata dataframe - include only the depths and layers belonging to the cortex
        metadata = {'t_number': trial_number, 't_type': trial_type, 'MVT_dir': mvt_direction,
                    't_number_rel': trial_number_rel, 't_mask_unamb': trial_mask_unamb,
                    'SC1_dir': behaviour['dirSC1'], 'SC2_dir': behaviour['dirSC2'],
                    'SC3_dir': behaviour['dirSC3'], 'block': trial_block,
                    'layers_1': [layers_probe_1[depths_probe_1 < 0]]*n_trials,
                    'layers_2': [layers_probe_2[depths_probe_2 < 0]]*n_trials,
                    'depths_1': [depths_probe_1[depths_probe_1 < 0]]*n_trials,
                    'depths_2': [depths_probe_2[depths_probe_2 < 0]]*n_trials,
                    'area_1': area_1, 'area_2': area_2, 's_1': electrode_spacing_1,
                    's_2': electrode_spacing_2, 'bb_dominance_1': df_electrodes.loc[session].BB_1,
                    'bb_dominance_2': df_electrodes.loc[session].BB_2,
                    'AP_1': df_electrodes.loc[session].AP_1,
                    'AP_2': df_electrodes.loc[session].AP_2,
                    'LAT_1': df_electrodes.loc[session].LAT_1,
                    'LAT_2': df_electrodes.loc[session].LAT_2}
    else:
        depths_probe_2 = []

        # build the metadata dataframe - include only the depths and layers belonging to the cortex
        metadata = {'t_number': trial_number, 't_type': trial_type, 'MVT_dir': mvt_direction,
                    't_number_rel': trial_number_rel, 't_mask_unamb': trial_mask_unamb,
                    'SC1_dir': behaviour['dirSC1'], 'SC2_dir': behaviour['dirSC2'],
                    'SC3_dir': behaviour['dirSC3'], 'block': trial_block,
                    'layers_1': [layers_probe_1[depths_probe_1 < 0]]*n_trials,
                    'depths_1': [depths_probe_1[depths_probe_1 < 0]]*n_trials,
                    'area_1': area_1, 's_1': electrode_spacing_1,
                    'bb_dominance_1': df_electrodes.loc[session].BB_1,
                    'AP_1': df_electrodes.loc[session].AP_1,
                    'LAT_1': df_electrodes.loc[session].LAT_1}

    all_depths = np.concatenate((depths_probe_1, depths_probe_2))  # concat depths of both probes

    df_metadata = pd.DataFrame(metadata)

    # create the three MNE objects: Annotations, info and EpochsArray
    lfp_info = mne.create_info(ch_names=list(channel_names), sfreq=fs_lfp, ch_types='seeg')
    annotations = mne.Annotations(onset=avg_event_times, duration=event_duration,
                                  description=event_labels)

    # set the zero time of the lfp as the negative time before the alignment - in seconds.
    mua_epochs_mne = mne.EpochsArray(data=mua_epochs, info=lfp_info, events=trial_events,
                                     tmin=avg_event_times[0], event_id=trial_events_id,
                                     metadata=df_metadata)

    # drop the outside channels - the ones that don't belong to the cortex from mne object
    mua_epochs_mne.drop_channels(np.array(channel_names)[all_depths >= 0])

    # 7) SAVE THE STRUCTURES
    annotations.save(os.path.join(path_output,
                                  f'{signal_type}-{session}-{alignment}-{trial_part}-annot.fif'),
                     overwrite=True)
    mua_epochs_mne.save(os.path.join(path_output,
                                     f'{signal_type}-{session}-{alignment}'
                                     f'-{trial_part}-demeaned-z_score-epo.fif'),
                        overwrite=True)

    with open(os.path.join(path_output, f'{signal_type}-{session}-metadata.pkl'), 'wb') as file:
        pickle.dump(df_metadata, file)


if __name__ == "__main__":
    # list all the sessions that we want to preprocess
    ALL_LAMINAR = ['Mo180411001', 'Mo180412002', 'Mo180626003', 'Mo180627003', 'Mo180619002',
                   'Mo180622002', 'Mo180704003', 'Mo180418002', 'Mo180419003', 'Mo180426004',
                   'Mo180601001',
                   'Mo180523002', 'Mo180705002', 'Mo180706002', 'Mo180710002',
                   'Mo180711004', 'Mo180712006', 't150303002', 't150319003', 't150423002',
                   't150430002', 't150320002',
                   't140924003', 't140930001', 't141001001', 't141008001', 't141010003',
                   't150122001', 't150123001', 't150128001', 't150204001', 't150205004',
                   't150716001',
                   'Mo180615002', 'Mo180615005', 't150327002', 't150327003',
                   't150702002', 't150702001']

    # ALIGNMENT = 'SEL' --> then COMPLETE_TRIAL = True
    for i_session in ALL_LAMINAR:
        for ALIGNMENT in ['SEL']:
            COMPLETE_TRIAL = True
            preprocessing_pipeline(session=i_session, alignment=ALIGNMENT, signal_type='MUA',
                                   complete_trial=COMPLETE_TRIAL)
            io.logger.info(f'Session {i_session} completed!')
