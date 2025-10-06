import numpy as np
from scipy.stats import spearmanr
import sys
from PyQt5 import QtWidgets
import os
import pandas as pd
import h5py
import datetime
import sys
sys.path.append("./src")

from visualpipe.analysis.speed_computation import compute_speed
from analysis.ca_imaging import CaImagingDataManager
from analysis.face_camera import FaceCamDataManager
from analysis.visual_stim import VisualStim
import utils.general_functions as general_functions
import analysis.photodiode as ptd
from gui.inputUI import InputWindow
import utils.file as file
from analysis.trial import Trial
import analysis.behavioral_states as behavioral_states
import analysis.spontaneous as spont
import params.parameters_batch as params

def visual_pipe(base_path:str=None, input_gui=False) :
    
    if input_gui :
        #---------------------------------- Launch app ----------------------
        app = QtWidgets.QApplication(sys.argv)

        # Launch the first GUI
        input_window = InputWindow()
        input_window.show()
        app.exec_()

        # Retrieve inputs from the first GUI
        inputs = input_window.get_inputs()

        # Convert inputs
        base_path = inputs["base_path"]
        compile_dir = inputs["compile_dir"]
        if os.path.exists(compile_dir):
            COMPILE = True
        else : 
            COMPILE = False
            print("No compilation or provided compile file not correct.")
        neuropil_impact_factor = float(inputs["neuropil_impact_factor"])
        F0_method = inputs["F0_method"]
        neuron_type = inputs["neuron_type"]
        starting_delay_2p = float(inputs["starting_delay_2p"])
        num_samples = int(inputs["bootstrap_nb_samples"])
        speed_threshold = inputs["speed_th"]
        facemotion_threshold = inputs["facemotion_th"]
        pupil_threshold = inputs["pupil_th"]
        pupil_threshold_type = inputs["pupil_th_type"]
        min_run_window = inputs["min_run_window"]
        min_as_window = inputs["min_as_window"]
        min_rest_window = inputs["min_rest_window"]
        print('Base path:', base_path)
        print(f'Neuron type: {neuron_type} ; F0 method: {F0_method}')

        speed_filter_kernel = 10
        motion_filter_kernel = 10
        pupil_filter_kernel = 10
        dFoF_filter_kernel = 10

    else :
        COMPILE = params.COMPILE
        compile_dir = params.compile_dir

        neuropil_impact_factor = params.neuropil_impact_factor
        F0_method = params.F0_method
        neuron_type = params.neuron_type
        starting_delay_2p = params.starting_delay_2p
        num_samples = params.num_samples

        speed_threshold = params.speed_threshold
        facemotion_threshold = params.facemotion_threshold
        pupil_threshold = params.pupil_threshold
        pupil_threshold_type = params.pupil_threshold_type

        min_run_window = params.min_run_window
        min_as_window = params.min_as_window
        min_rest_window = params.min_rest_window

        speed_filter_kernel = params.speed_filter_kernel
        motion_filter_kernel = params.motion_filter_kernel
        pupil_filter_kernel = params.pupil_filter_kernel
        dFoF_filter_kernel = params.dFoF_filter_kernel

    print('#-------------------------------------------------------------------------#')
    print(f"    Processing session: {base_path}")
    
    #---------------------------------- Get metadata ----------------------
    unique_id, global_protocol, experimenter, subject_id = file.get_metadata(base_path)
    subject_id_anibio = file.get_mouse_id(base_path, subject_id)

    #---------------------------------- Create saving folder ----------------------
    save_dir, save_fig_dir, id_version = file.create_output_folder(base_path, unique_id)
    print(f"    Saving directory: {save_dir}")

    #---------------------------------- Load Ca-Imaging data ----------------------
    ca_img_dm = CaImagingDataManager(base_path, neuropil_impact_factor, F0_method, neuron_type, starting_delay_2p)
    ca_img_dm.save_mean_image(base_path)
    ca_img_dm.save_max_proj_image(base_path)
    detected_roi = ca_img_dm._list_ROIs_idx
    print('Original number of neurons :', len(detected_roi))

    #---------------------------------- Load Camera data ----------------------------------
    print("Loading face camera data")
    timestamp_start = ptd.get_timestamp_start(base_path)
    face_cam_dm = FaceCamDataManager(base_path, timestamp_start)
    print("    ------------> Done")

    #---------------------------------- Compute speed ----------------------------------
    print("Computing speed")
    speed, speed_time_stamps = compute_speed(base_path)
    speed = np.abs(speed) #only take norm of speed
    print("    ------------> Done")

    #---------------------------------- Resample facemotion, pupil and speed traces ----------------------------------
    print("Resampling facemotion, pupil and speed traces")
    if not face_cam_dm.no_face_data :
        last_F_index = np.argmin(np.abs(ca_img_dm.time_stamps - face_cam_dm.time_stamps[-1]))
        ca_img_dm.cut_frames(last_index=last_F_index) #update metrics with new frames length
    new_time_stamps = ca_img_dm.time_stamps
    total_duration = ca_img_dm.time_stamps[-1] - ca_img_dm.time_stamps[0]
    print(f"Total duration of the recording: {total_duration} s")

    #sub sampling and filtering speed
    speed = general_functions.resample_signal(speed, 
                                            t_sample=speed_time_stamps, 
                                            new_freq=ca_img_dm.fs,
                                            interp_time=new_time_stamps,
                                            post_smoothing=2./50.)

    if not face_cam_dm.no_face_data :
        pupil = general_functions.resample_signal(face_cam_dm.pupil, 
                                                t_sample=face_cam_dm.time_stamps, 
                                                new_freq=ca_img_dm.fs, 
                                                interp_time=new_time_stamps)
        facemotion = general_functions.resample_signal(face_cam_dm.facemotion, 
                                                    t_sample=face_cam_dm.time_stamps,
                                                    new_freq=ca_img_dm.fs, 
                                                    interp_time=new_time_stamps)

        # Normalize
        pupil = general_functions.scale_trace(pupil)
        facemotion = general_functions.scale_trace(facemotion)
    else :
        facemotion, pupil = [np.nan] * len(new_time_stamps), [np.nan] * len(new_time_stamps)

    print("    ------------> Done")

    print("Calcium imaging data preprocessing")
    #---------------------------------- Detect ROIs with bad neuropils ------------------
    ca_img_dm.detect_bad_neuropils()
    kept2p_ROI = ca_img_dm._list_ROIs_idx
    print('After removing bad neuropil neurons, nb of neurons :', len(kept2p_ROI))

    #---------------------------------- Compute Fluorescence ------------------
    ca_img_dm.compute_F()
    kept_ROI_alpha = ca_img_dm._list_ROIs_idx
    print('Number of remaining neurons after alpha calculation :', len(kept_ROI_alpha))

    #---------------------------------- Calculation of F0 ----------------------
    ca_img_dm.compute_F0(percentile=10, win=60)
    kept_ROI_F0 = ca_img_dm._list_ROIs_idx
    print('Number of remaining neurons after F0 calculation  :', len(kept_ROI_F0))

    #---------------------------------- Calculation of dF over F0 ----------------------
    ca_img_dm.compute_dFoF0()

    #---------------------------------- Plot calcium imaging traces ----------------------
    ca_img_dm.plot('f0', sigma=0, mean=True, save_dir=save_fig_dir, legend=True)
    ca_img_dm.plot('fluorescence', sigma=10, save_dir=save_fig_dir, legend=True)
    ca_img_dm.plot('dFoF0', sigma=10, save_dir=save_fig_dir, legend=True)
    ca_img_dm.plot_raster('fluorescence', sigma=10, save_dir=save_fig_dir)
    ca_img_dm.plot_raster('dFoF0', sigma=10, save_dir=save_fig_dir)
    print("    ------------> Done")

    #---------------------------------- Compute correlation ----------------------
    print("Computing correlation")
    speed_corr_list = [spearmanr(speed, ROI)[0] for ROI in ca_img_dm.dFoF0]
    facemotion_corr_list = [spearmanr(facemotion, ROI)[0] for ROI in ca_img_dm.dFoF0]
    pupil_corr_list = [spearmanr(pupil, ROI)[0] for ROI in ca_img_dm.dFoF0]
    print("    ------------> Done")

    #---------------------------------- Load Photodiode data -----------------------------
    print("Loading photodiode data")
    NIdaq, acq_freq = ptd.load_and_data_extraction(base_path)
    Psignal_time, Psignal = general_functions.resample_signal(NIdaq['analog'][0],
                                                            original_freq=acq_freq,
                                                            new_freq=1000)
    print("    ------------> Done")

    #---------------------------------- Load Stimuli data ----------------------
    print("Loading visual stimuli data")
    visual_stim = VisualStim(base_path)
    protocol_df = visual_stim.protocol_df
    print("    ------------> Done")

    #---------------------------------- Set real stimuli onset with photodiode -----------------------------
    print("Realigning visual stimuli onset from photodiode")
    visual_stim.realign_from_photodiode(Psignal_time, Psignal)
    Psignal = general_functions.scale_trace(Psignal)
    print("    ------------> Done")

    #---------------------------------- Stimuli start times and durations -----------------
    print("Realigning visual stimuli onset with calcium imaging data")
    stim_time_end = list(visual_stim.real_time_onset + visual_stim.duration)
    stim_time_period = [visual_stim.real_time_onset, stim_time_end]

    F_Time_start_realigned, F_stim_init_indexes  = ptd.Find_F_stim_index(visual_stim.real_time_onset, ca_img_dm.time_stamps)
    print("    ------------> Done")

    #---------------------------------- Compute behavioral states -----------------
    print("Computing behavioral states")
    # Calculate index boundaries of dark stimulus
    idx_dark = np.argwhere(np.array(visual_stim.analyze_pupil) == 0)
    idx_lim_dark, _ = visual_stim.get_protocol_onset_index(idx_dark, F_stim_init_indexes, ca_img_dm.fs)

    min_states_window = {'run' : round(min_run_window * ca_img_dm.fs), 
                        'AS' : round(min_as_window * ca_img_dm.fs), 
                        'rest' : round(min_rest_window * ca_img_dm.fs)}

    if not face_cam_dm.no_face_data :
        # Facemotion threshold
        real_time_states_facemotion, states_window_facemotion =\
            behavioral_states.split_stages(speed, facemotion, speed_threshold, facemotion_threshold, 
                                        ca_img_dm.time_stamps, min_states_window, ca_img_dm.fs, 
                                        'std', speed_filter_kernel, motion_filter_kernel)

        behavioral_states.stage_plot(speed, facemotion, pupil, ca_img_dm.dFoF0, 
                                    ca_img_dm.time_stamps, real_time_states_facemotion, states_window_facemotion, 
                                    save_fig_dir, speed_threshold, facemotion_threshold,'std', 'facemotion', 
                                    speed_filter_kernel, motion_filter_kernel, pupil_filter_kernel, dFoF_filter_kernel,
                                    svg=False)

        run_ratio_facemotion, as_ratio_facemotion, rest_ratio_facemotion =\
            behavioral_states.time_pie(real_time_states_facemotion, total_duration, 
                                    save_fig_dir,figname="states_duration_pie_facemotion")

        # Pupil threshold
        if len(idx_lim_dark) == 0 :
            real_time_states_pupil, states_window_pupil =\
                behavioral_states.split_stages(speed, pupil, speed_threshold, pupil_threshold, 
                                            ca_img_dm.time_stamps, min_states_window, ca_img_dm.fs, 
                                            pupil_threshold_type, speed_filter_kernel, pupil_filter_kernel)

            behavioral_states.stage_plot(speed, facemotion, pupil, ca_img_dm.dFoF0, 
                                        ca_img_dm.time_stamps, real_time_states_pupil, states_window_pupil, 
                                        save_fig_dir, speed_threshold, pupil_threshold, pupil_threshold_type, 'pupil', 
                                        speed_filter_kernel, motion_filter_kernel,  pupil_filter_kernel, dFoF_filter_kernel, 
                                        svg=False)
        else :
            real_time_states_pupil, states_window_pupil =\
                behavioral_states.split_stages_mixed(speed, pupil, facemotion, idx_lim_dark,
                                                    speed_threshold, pupil_threshold, facemotion_threshold,
                                                    ca_img_dm.time_stamps, min_states_window, ca_img_dm.fs, 
                                                    pupil_threshold_type, 'std',
                                                    speed_filter_kernel, pupil_filter_kernel, motion_filter_kernel)
            
            behavioral_states.stage_plot_mixed(speed, facemotion, pupil, ca_img_dm.dFoF0, 
                                                ca_img_dm.time_stamps, real_time_states_pupil, states_window_pupil, idx_lim_dark, ca_img_dm.fs,
                                                save_fig_dir, speed_threshold, pupil_threshold, facemotion_threshold, 
                                                pupil_threshold_type, 'std', 
                                                speed_filter_kernel, motion_filter_kernel,  pupil_filter_kernel, dFoF_filter_kernel, 
                                                svg=False)

        run_ratio_pupil, as_ratio_pupil, rest_ratio_pupil =\
            behavioral_states.time_pie(real_time_states_pupil, total_duration, 
                                    save_fig_dir, figname="states_duration_pie_pupil")
    else :
        real_time_states, states_window =\
            behavioral_states.split_stages_locomotion(speed, speed_threshold, 
                                                    ca_img_dm.time_stamps, min_states_window, ca_img_dm.fs, speed_filter_kernel)

        behavioral_states.stage_plot_locomotion(speed, ca_img_dm.dFoF0, 
                                                ca_img_dm.time_stamps, real_time_states, states_window, 
                                                save_fig_dir, speed_threshold, speed_filter_kernel, 
                                                dFoF_filter_kernel,
                                                svg=False)

        run_ratio, rest_ratio =\
            behavioral_states.time_pie_locomotion(real_time_states, total_duration, 
                                    save_fig_dir,figname="states_duration_pie")
    print("    ------------> Done")

    #---------------------------------- Compute trials -----------------
    print("Computing trials")
    # Create Trial instance
    trials = Trial(ca_img_dm, visual_stim, F_stim_init_indexes, attr='dFoF0', dt_pre_stim=1, dt_post_stim=0.5)

    # Compute responsive neurons
    trials.find_responsive_rois(save_dir, folder_prefix="_".join([unique_id, id_version]))

    # Save results in file
    filename = "_".join([unique_id, id_version, 'protocol_validity_2'])
    trials.save_protocol_validity(save_dir, filename)

    # Compute contextual modulation index
    if "center-surround-cross" in visual_stim.protocol_names :
        cmi = trials.compute_cmi()
        trials.plot_cmi_hist(cmi, save_fig_dir)
        trials.plot_iso_vs_cross(save_fig_dir)
        trials.plot_surround_vs_center(save_fig_dir)
        surround_sup_cross, surround_sup_iso = trials.compute_surround_sup()
    else :
        cmi = None
        surround_sup_cross, surround_sup_iso = None, None

    # Compute the trial zscores not based on the averaged baseline but the baseline of the trace
    trial_zscores, pre_trial_zscores, post_trial_zscores = trials.compute_trial_zscores('dFoF0')

    if not face_cam_dm.no_face_data :
        real_time_states_sorted = behavioral_states.sort_dict_el(real_time_states_pupil)
    else :
        real_time_states_sorted = behavioral_states.sort_dict_el(real_time_states)

    # Sort trials by arousal states (output in trials.npy)
    trials.sort_trials_by_states(F_Time_start_realigned, real_time_states_sorted)

    # Compute averaged traces by arousal states
    trials.compute_avg_by_states()

    # Plot trials related figures
    for i in range(len(protocol_df)):    
        if visual_stim.stim_cat[i] :

            #plot trial-averaged z-score raster sorted
            trials.trial_average_rasterplot(i, savepath=save_fig_dir) 

            #plot trial-averaged z-score raster not sorted
            trials.trial_average_rasterplot(i, savepath=save_fig_dir, sort=False)

            #plot trial-averaged z-score raster sorted and normalized
            trials.trial_average_rasterplot(i, savepath=save_fig_dir, normalize=True)

            #plot trials z-score raster with paired baseline
            trials.trial_rasterplot(trial_zscores, pre_trial_zscores, post_trial_zscores, i, 'dFoF0', savepath=save_fig_dir)
            
            #plot trials z-score raster with averaged baseline
            #trials.trial_rasterplot(trials.trial_zscores, trials.pre_trial_zscores, trials.post_trial_zscores, i, trials.ca_attr, savepath=save_fig_dir)
            
            for k in range(len(ca_img_dm._list_ROIs_idx)):
                #plot trial-averaged z-score traces
                trials.plot_stim_response(i, k, save_dir, folder_prefix="_".join([unique_id, id_version]))

                #plot trial-averaged normalized with baseline traces
                trials.plot_norm_trials(i, k, save_dir, folder_prefix="_".join([unique_id, id_version]))

                #plot trial-averaged traces per arousal states
                trials.plot_trials_per_states(i, k, save_dir, folder_prefix="_".join([unique_id, id_version]))
            
            #plot trials with behavioral states
            trials.plot_stim_occurence(i, trial_zscores, pre_trial_zscores, real_time_states_sorted, F_Time_start_realigned, save_dir, folder_prefix="_".join([unique_id, id_version]))

            #plot trial-averaged traces per arousal states
            trials.plot_trials_per_states(i, k, save_dir, folder_prefix="_".join([unique_id, id_version]))

    print("    ------------> Done")

    #---------------------------------- Spontaneous behaviour ----------------------------------
    print("Computing spontaneous behaviour")
    spont_stimuli_id, _ = spont.get_spont_stim(visual_stim)
    sigma = 5

    if len(spont_stimuli_id) > 0 :
        spont_speed_corr_list, spont_facemotion_corr_list, spont_pupil_corr_list = [], [], []
        valid_neurons_speed_list, valid_neurons_facemotion_list, valid_neurons_pupil_list = [], [], []

        idx_lim_protocol, spont_stimuli_id_order, F_spontaneous = visual_stim.get_protocol_onset_index(spont_stimuli_id, F_stim_init_indexes, ca_img_dm.fs, tseries=ca_img_dm.dFoF0)

        spont_df = protocol_df.loc[spont_stimuli_id_order]

        for i, id  in enumerate(spont_stimuli_id_order):
            name_stimuli = spont_df.loc[spont_stimuli_id_order[i]]['name']
            save_spont_dir = os.path.join(save_dir, "_".join([unique_id, 'spontaneous']))
            save_spont_dir_i = os.path.join(save_spont_dir, name_stimuli)
            if not os.path.exists(save_spont_dir_i) : os.makedirs(save_spont_dir_i)

            time_stamps_spont = new_time_stamps[idx_lim_protocol[i][0]:idx_lim_protocol[i][1]]
            print(f"Spontaneous activity time {name_stimuli} {i}: from {time_stamps_spont[0]} s to {time_stamps_spont[-1]} s")

            # Speed correlation
            spont_speed_corr, valid_neurons_temp = spont.process_correlation(speed, F_spontaneous[i], time_stamps_spont, idx_lim_protocol[i], sigma, 'speed', save_spont_dir_i)
            spont_speed_corr_list.append(spont_speed_corr)
            valid_neurons_speed_list.append(valid_neurons_temp)

            if not face_cam_dm.no_face_data :
                # Facemotion correlation
                spont_facemotion_corr, valid_neurons_temp = spont.process_correlation(facemotion, F_spontaneous[i], time_stamps_spont, idx_lim_protocol[i], sigma, 'facemotion', save_spont_dir_i)
                spont_facemotion_corr_list.append(spont_facemotion_corr)
                valid_neurons_facemotion_list.append(valid_neurons_temp)

                # Pupil correlation
                if spont_df.loc[id].analyze_pupil :
                    spont_pupil_corr, valid_neurons_temp = spont.process_correlation(pupil, F_spontaneous[i], time_stamps_spont, idx_lim_protocol[i], sigma, 'pupil', save_spont_dir_i)
                    spont_pupil_corr_list.append(spont_pupil_corr)
                    valid_neurons_pupil_list.append(valid_neurons_temp)

        speed_corr, valid_neurons_speed_list2 = spont.process_multiple_protocols(spont_speed_corr_list, spont_df, valid_neurons_speed_list, 'speed', save_spont_dir, True)

        if not face_cam_dm.no_face_data :
            facemotion_corr, valid_neurons_facemotion_list2 = spont.process_multiple_protocols(spont_facemotion_corr_list, spont_df, valid_neurons_facemotion_list, 'facemotion', save_spont_dir, True)

            if len(spont_pupil_corr_list) > 0 :
                pupil_corr, valid_neurons_pupil_list2 = spont.process_multiple_protocols(spont_pupil_corr_list, spont_df, valid_neurons_pupil_list, 'pupil', save_spont_dir, True)
            else :
                nb_rois = len(ca_img_dm._list_ROIs_idx)
                nan_array = np.empty(nb_rois)
                nan_array.fill(np.nan)
                pupil_corr = nan_array
                valid_neurons_pupil_list2 = []
        
        else :
            nb_rois = len(ca_img_dm._list_ROIs_idx)
            nan_array = np.empty(nb_rois)
            nan_array.fill(np.nan)
            facemotion_corr, pupil_corr = nan_array, nan_array
            valid_neurons_facemotion_list2, valid_neurons_pupil_list2 = [], []
            
    else : 
        speed_corr, facemotion_corr, pupil_corr = np.array(speed_corr_list), np.array(facemotion_corr_list), np.array(pupil_corr_list)
        valid_neurons_speed_list2, valid_neurons_facemotion_list2, valid_neurons_pupil_list2 = None, None, None

    print("    ------------> Done")

    ################################
    print("Saving data")

    photodiode = (Psignal_time, Psignal)
    pupilAndTimeSt  = (new_time_stamps, pupil)
    fmotionAndTimeSt  = (new_time_stamps, facemotion)
    speedAndTimeSt = (new_time_stamps, speed)

    #---------------------------------- HDF5 files ----------------------------------
    filename = "_".join([unique_id, id_version, 'postprocessing']) + ".h5"
    H5_dir = os.path.join(save_dir, filename)
    hf = h5py.File(H5_dir, 'w')
    behavioral_group = hf.create_group('Behavioral')
    correlation = behavioral_group.create_group("Correlation")
    spont_correlation = behavioral_group.create_group("Spont_correlation")
    caImg_group = hf.create_group('Ca_imaging')
    caImg_full_trace = caImg_group.create_group('full_trace')
    stimuli_group = hf.create_group("Stimuli")
    rois_group = hf.create_group("ROIs")
    states_group = hf.create_group("Arousal states")
    if not face_cam_dm.no_face_data :
        states_with_pupil = states_group.create_group("Arousal states pupil")
        frame_bounds_pupil = states_with_pupil.create_group("Frame bounds")
        time_bounds_pupil = states_with_pupil.create_group("Time bounds")
        states_with_facemotion = states_group.create_group("Arousal states facemotion")
        frame_bounds_facemotion = states_with_facemotion.create_group("Frame bounds")
        time_bounds_facemotion = states_with_facemotion.create_group("Time bounds")
    else :
        frame_bounds = states_group.create_group("Frame bounds")
        time_bounds = states_group.create_group("Time bounds")

    file.create_H5_dataset(behavioral_group, [speedAndTimeSt, fmotionAndTimeSt, pupilAndTimeSt, photodiode], ['Speed', 'FaceMotion', 'Pupil', 'Photodiode'])
    file.create_H5_dataset(correlation, [speed_corr_list, facemotion_corr_list, pupil_corr_list], ['speed_corr', 'facemotion_corr', 'pupil_corr'])
    if len(spont_stimuli_id) > 0 :
        file.create_H5_dataset(spont_correlation, [spont_speed_corr_list, spont_facemotion_corr_list, spont_pupil_corr_list], ['speed_corr', 'facemotion_corr', 'pupil_corr'])
        file.create_H5_dataset(spont_correlation, [list(spont_df.name)], ['spont_protocol_name'])
        spont_valid_rois = spont_correlation.create_group("Valid_ROIs")
        file.create_H5_dataset(spont_valid_rois, [valid_neurons_speed_list2, valid_neurons_facemotion_list2, valid_neurons_pupil_list2], ['speed', 'facemotion', 'pupil'])
    caImg_group.create_dataset('Time', data=ca_img_dm.time_stamps)
    file.create_H5_dataset(caImg_full_trace, [ca_img_dm.raw_F, ca_img_dm.raw_Fneu, ca_img_dm.fluorescence, ca_img_dm.f0, ca_img_dm.dFoF0], 
                                        ['raw_F', 'raw_Fneu', 'F', 'F0', 'dFoF0'])
    file.create_H5_dataset(stimuli_group, [visual_stim.real_time_onset, F_Time_start_realigned, F_stim_init_indexes], 
                                        ['time_onset', 'time_onset_caimg_timescale', 'idx_onset_caimg_timescale'])
    if cmi is not None :
        file.create_H5_dataset(stimuli_group, [cmi, surround_sup_cross, surround_sup_iso], ['cmi', 'surround_sup_cross', 'surround_sup_iso'])
    file.create_H5_dataset(rois_group, [detected_roi, kept2p_ROI, kept_ROI_alpha, kept_ROI_F0], 
                                        ['0_original', '1_neuropil', '2_alpha', '3_F0'])
    if not face_cam_dm.no_face_data :
        file.create_H5_dataset(frame_bounds_pupil, [states_window_pupil['run'], states_window_pupil['AS'], states_window_pupil['rest']], ['Run', 'AS', 'Rest'])
        file.create_H5_dataset(time_bounds_pupil, [real_time_states_pupil['run'], real_time_states_pupil['AS'], real_time_states_pupil['rest']], ['Run', 'AS', 'Rest'])
        file.create_H5_dataset(frame_bounds_facemotion, [states_window_facemotion['run'], states_window_facemotion['AS'], states_window_facemotion['rest']], ['Run', 'AS', 'Rest'])
        file.create_H5_dataset(time_bounds_facemotion, [real_time_states_facemotion['run'], real_time_states_facemotion['AS'], real_time_states_facemotion['rest']], ['Run', 'AS', 'Rest'])
    else :
        file.create_H5_dataset(frame_bounds, [states_window['run'], states_window['rest']], ['Run', 'Rest'])
        file.create_H5_dataset(time_bounds, [real_time_states['run'], real_time_states['rest']], ['Run', 'Rest'])

    hf.close()

    #---------------------------------- Outputs ----------------------------------
    filename = "_".join([unique_id, id_version, 'visual_stim_info']) + ".xlsx"
    visual_stim.export_df_to_excel(save_dir, filename)

    filename = "_".join([unique_id, id_version, 'trials'])
    trials.save_trials(save_dir, filename)

    filename = "_".join([unique_id, id_version, 'stat.npy'])
    np.save(os.path.join(save_dir, filename), ca_img_dm.stat, allow_pickle=True)

    if COMPILE :
        data_df = pd.DataFrame({
                    "Session_id": unique_id, "Output_id": id_version, "Protocol": global_protocol, "Experimenter": experimenter, "Mouse_id": subject_id_anibio,
                    'Mean_speed' : np.nanmean(speed), 'Std_speed' : np.nanstd(speed),
                    'Mean_fmotion' : np.nanmean(facemotion), 'Std_fmotion' : np.nanstd(facemotion),
                    'Mean_pupil' : np.nanmean(pupil), 'Std_pupil' : np.nanstd(pupil),
                    'Spontaneous' : True if len(spont_stimuli_id) > 0 else False,
                    'Mean_speed_corr' : np.nanmean(speed_corr), 
                    'Mean_fmotion_corr' : np.nanmean(facemotion_corr),
                    'Mean_pupil_corr' : np.nanmean(pupil_corr), 
                    'Mean_dFoF0' : np.nanmean(ca_img_dm.dFoF0), 
                    'Run % (pupil)' : run_ratio_pupil if not face_cam_dm.no_face_data else None, 
                    'AS % (pupil)' : as_ratio_pupil if not face_cam_dm.no_face_data else None,
                    'Rest % (pupil)' : rest_ratio_pupil if not face_cam_dm.no_face_data else None,
                    'Run % (motion)' : run_ratio_facemotion if not face_cam_dm.no_face_data else None, 
                    'AS % (motion)' : as_ratio_facemotion if not face_cam_dm.no_face_data else None, 
                    'Rest % (motion)' : rest_ratio_facemotion if not face_cam_dm.no_face_data else None,
                    'Run %' : run_ratio if face_cam_dm.no_face_data else None, 
                    'Rest %' : rest_ratio if face_cam_dm.no_face_data else None, 
                    }, index=[0]).set_index("Session_id")
        file.compile_xlsx_file(data_df, compile_dir)

    settings = {"Date" : datetime.date.today(),
                "Time" : datetime.datetime.now().time(),
                "Session_id" : unique_id,
                "Neuron type" : neuron_type,
                "Neuropil impact factor" : ca_img_dm._neuropil_if,
                "F0 calculateion method" : F0_method,
                "2p starting delay" : starting_delay_2p,
                "Bootstrapping nb of samples" : num_samples,
                "Speed threshold" : f"{speed_threshold} (cm/s)",
                "Facemotion threshold" : f"{facemotion_threshold} (std)",
                "Pupil threshold" :  f"{pupil_threshold} ({pupil_threshold_type})",
                "Minimum running window" : min_run_window,
                "Minimum AS window" : min_as_window,
                "Minimum rest window" : min_rest_window,
                "Speed filter kernel" : speed_filter_kernel,
                "Motion filter kernel" : motion_filter_kernel,
                "Pupil filter kernel" : pupil_filter_kernel,
                "Fluorescence filter kernel" : dFoF_filter_kernel,
                "Analyzed folder" : base_path,
                "Saving folder" : save_dir,
                "Compile folder" : compile_dir
                }
    file.save_analysis_settings(settings, save_dir)
    
    print("    ------------> Done")

    return save_dir

if __name__ == "__main__":

    try :
        save_dir = visual_pipe(input_gui=True)
        print("Session's analysis completed successfully")
        print(f'Output folder : {save_dir}')
    except Exception as e :
        print(e)