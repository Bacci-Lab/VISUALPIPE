from Running_computation import compute_speed
from Ca_imaging import CaImagingDataManager
from face_camera import FaceCamDataManager
from visual_stim import VisualStim
import numpy as np
from scipy.stats import pearsonr
import sys
import General_functions
from Visuial_GUI import MainWindow
import Photodiode
from PyQt5 import QtWidgets
import os
from inputUI import InputWindow
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import pickle

import utils.file as file

app = QtWidgets.QApplication(sys.argv)

# Launch the first GUI
input_window = InputWindow()
input_window.show()
app.exec_()

# Retrieve inputs from the first GUI
inputs = input_window.get_inputs()

# Convert inputs
base_path = inputs["base_path"]
save_dir = inputs["save_dir"]
if not os.path.exists(save_dir) :
    os.makedirs(save_dir)
neuropil_impact_factor = float(inputs["neuropil_impact_factor"])
F0_method = inputs["F0_method"]
neuron_type = inputs["neuron_type"]
starting_delay_2p = float(inputs["starting_delay_2p"])
num_samples = int(inputs["bootstrap_nb_samples"])
print('Base path:', base_path)
print(f'Neuron type: {neuron_type} ; F0 method: {F0_method}')

#---------------------------------- Get metadata ----------------------
unique_id, global_protocol, experimenter, subject_id = file.get_metadata(base_path)
subject_id_anibio = file.get_mouse_id(base_path, subject_id)

#---------------------------------- Load Ca-Imaging data ----------------------
ca_img_dm = CaImagingDataManager(base_path, neuropil_impact_factor, F0_method, neuron_type, starting_delay_2p)
ca_img_dm.save_mean_image(save_dir)
detected_roi = ca_img_dm._list_ROIs_idx
print('Original number of neurons :', len(detected_roi))

#---------------------------------- Load Camera data ----------------------------------
timestamp_start = Photodiode.get_timestamp_start(base_path)
face_cam_dm = FaceCamDataManager(base_path, timestamp_start)

#---------------------------------- Compute speed ----------------------------------
speed, speed_time_stamps = compute_speed(base_path)

#---------------------------------- Resample facemotion, pupil and speed traces ----------------------------------
last_F_index = np.argmin(np.abs(ca_img_dm.time_stamps - face_cam_dm.time_stamps[-1]))
ca_img_dm.cut_frames(last_index=last_F_index) #update metrics with new frames length
new_time_stamps = ca_img_dm.time_stamps

#sub sampling and filtering speed
speed = General_functions.resample_signal(speed, 
                                          t_sample=speed_time_stamps, 
                                          new_freq=ca_img_dm.fs,
                                          interp_time=new_time_stamps,
                                          post_smoothing=2./50.)
pupil = General_functions.resample_signal(face_cam_dm.pupil, 
                                          t_sample=face_cam_dm.time_stamps, 
                                          new_freq=ca_img_dm.fs, 
                                          interp_time=new_time_stamps)
facemotion = General_functions.resample_signal(face_cam_dm.facemotion, 
                                               t_sample=face_cam_dm.time_stamps,
                                               new_freq=ca_img_dm.fs, 
                                               interp_time=new_time_stamps)

# Normalize
pupil = General_functions.scale_trace(pupil)
facemotion = General_functions.scale_trace(facemotion)

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
computed_F_norm = ca_img_dm.normalize_time_series("dFoF0", lower=0, upper=5)

#---------------------------------- Load Photodiode data -----------------------------
NIdaq, acq_freq = Photodiode.load_and_data_extraction(base_path)
Psignal_time, Psignal = General_functions.resample_signal(NIdaq['analog'][0],
                                                          original_freq=acq_freq,
                                                          new_freq=1000)

#---------------------------------- Load Stimuli data ----------------------
visual_stim = VisualStim(base_path)
protocol_df = visual_stim.protocol_df
print(protocol_df)

#---------------------------------- Set real stimuli onset with photodiode -----------------------------
visual_stim.realign_from_photodiode(Psignal_time, Psignal)
Psignal = General_functions.scale_trace(Psignal)

#---------------------------------- Stimuli start times and durations -----------------
stim_time_end = list(visual_stim.real_time_onset + visual_stim.duration)
stim_time_period = [visual_stim.real_time_onset, stim_time_end]

F_Time_start_realigned, F_stim_init_indexes  = Photodiode.Find_F_stim_index(visual_stim.real_time_onset, ca_img_dm.time_stamps)

#---------------------------------- Bootstrapping ----------------------------------
if not os.path.exists(os.path.join(base_path, "protocol_validity.npz")):
    protocol_validity = []
    for protocol in range(len(protocol_df)):
        chosen_protocol = protocol_df.index[protocol]
        protocol_duration = protocol_df['duration'][protocol]
        protocol_name = protocol_df['name'][protocol]
        protocol_validity_i = Photodiode.average_image(ca_img_dm.dFoF0, visual_stim.order, chosen_protocol,protocol_duration, protocol_name, F_stim_init_indexes, ca_img_dm.fs, num_samples, save_dir)
        protocol_validity.append(protocol_validity_i)
    np.savez(os.path.join(base_path, "protocol_validity.npz"), **{key: value for d in protocol_validity for key, value in d.items()})
    print(protocol_validity)

#---------------------------------- Spontaneous behaviour ----------------------------------
if 'grey-20min' in visual_stim.protocol_names :
    id_spont = protocol_df[protocol_df['name'] == 'grey-20min'].index[0]
    duration_spont = protocol_df.iloc[id_spont]["duration"]
    idx_lim_protocol, F_spontaneous = visual_stim.get_protocol_onset_index(id_spont, F_stim_init_indexes, ca_img_dm.fs, tseries=ca_img_dm.dFoF0)
    [start_spont_index, end_spont_index] = idx_lim_protocol[0]

    speed_spont = speed[start_spont_index:end_spont_index]
    facemotion_spont = facemotion[start_spont_index:end_spont_index]
    pupil_spont = pupil[start_spont_index:end_spont_index]
    time_stamps_spont = new_time_stamps[start_spont_index:end_spont_index]
    print("Spontaneous activity time: from", time_stamps_spont[0], "to", time_stamps_spont[-1])

    """ ax1 = plt.subplot(311)
    ax1.plot(time_stamps_spont, speed_spont, color='goldenrod')
    ax1.set_title('speed')
    ax1.set_xticks([])
    ax2 = plt.subplot(312)
    ax2.plot(time_stamps_spont, facemotion_spont, color='gray')
    ax2.set_title('facemotion')
    ax2.set_xticks([])
    ax3 = plt.subplot(313)
    ax3.plot(time_stamps_spont, pupil_spont, color='black')
    ax3.set_title('pupil')
    ax3.set_xlabel('Time (s)')
    plt.show() """

    # Correlation with dFoF0
    speed_corr = [pearsonr(speed_spont, ROI)[0] for ROI in F_spontaneous[:, 0, :]]
    speed_corr = [float(value) for value in speed_corr]

    facemotion_corr = [pearsonr(facemotion_spont, ROI)[0] for ROI in F_spontaneous[:, 0, :]]
    facemotion_corr = [float(value) for value in facemotion_corr]

    pupil_corr = [pearsonr(pupil_spont, ROI)[0] for ROI in F_spontaneous[:, 0, :]]
    pupil_corr = [float(value) for value in pupil_corr]

else : 
    nb_rois = len(ca_img_dm.dFoF0)
    speed_corr, facemotion_corr, pupil_corr = list(np.zeros(nb_rois)), list(np.zeros(nb_rois)), list(np.zeros(nb_rois))

################################

photodiode = (Psignal_time, Psignal)
pupilAndTimeSt  = (new_time_stamps, pupil)
fmotionAndTimeSt  = (new_time_stamps, facemotion)
speedAndTimeSt = (new_time_stamps, speed)
background_image_path = os.path.join(base_path, "Mean_image_grayscale.png")
protocol_validity_npz = np.load(os.path.join(base_path, "protocol_validity.npz"))

#---------------------------------- HDF5 files ----------------------------------
H5_dir = os.path.join(save_dir, "postprocessing.h5")
hf = h5py.File(H5_dir, 'w')
behavioral_group = hf.create_group('Behavioral')
correlation = behavioral_group.create_group("Correlation")
caImg_group = hf.create_group('Ca_imaging')
caImg_full_trace = caImg_group.create_group('full_trace')
stimuli_group = hf.create_group("Stimuli")
rois_group = hf.create_group("ROIs")

file.create_H5_dataset(behavioral_group, [speedAndTimeSt, facemotion, pupil, photodiode], ['Speed', 'FaceMotion', 'Pupil', 'Photodiode'])
if 'grey-20min' in visual_stim.protocol_names :
    file.create_H5_dataset(correlation, [speed_corr, facemotion_corr, pupil_corr], ['speed_corr', 'facemotion_corr', 'pupil_corr'])
caImg_group.create_dataset('Time', data=ca_img_dm.time_stamps)
file.create_H5_dataset(caImg_full_trace, [ca_img_dm.raw_F, ca_img_dm.raw_Fneu, ca_img_dm.fluorescence, ca_img_dm.f0, ca_img_dm.dFoF0], 
                                    ['raw_F', 'raw_Fneu', 'F', 'F0', 'dFoF0'])
file.create_H5_dataset(stimuli_group, [visual_stim.real_time_onset, F_Time_start_realigned, F_stim_init_indexes], 
                                    ['time_onset', 'time_onset_caimg_timescale', 'idx_onset_caimg_timescale'])
file.create_H5_dataset(rois_group, [detected_roi, kept2p_ROI, kept_ROI_alpha, kept_ROI_F0], 
                                    ['0_original', '1_neuropil', '2_alpha', '3_F0'])

hf.close()

#---------------------------------- Outputs ----------------------------------
file.save_pickle(ca_img_dm, save_directory=save_dir, filename='ca_img_obj')

data_df = pd.DataFrame({
            "Session_id": unique_id, "Protocol": global_protocol, "Experimenter": experimenter, "Mouse_id": subject_id_anibio,
            'Mean_speed' : np.nanmean(speed), 'Std_speed' : np.std(speed),
            'Mean_fmotion' : np.nanmean(facemotion), 'Std_fmotion' : np.std(facemotion),
            'Mean_pupil' : np.nanmean(pupil), 'Std_pupil' : np.std(pupil),
            'Spontaneous' : True if 'grey-20min' in visual_stim.protocol_names else False,
            'Mean_speed_corr' : np.nanmean(speed_corr), 'Mean_fmotion_corr' : np.nanmean(facemotion_corr), 'Mean_pupil_corr' : np.nanmean(pupil_corr), 
            'Mean_dFoF0' : np.nanmean(ca_img_dm.dFoF0)
            }, index=[0]).set_index("Session_id")
file.compile_xlsx_file(data_df, save_dir)

#---------------------------------- Second GUI ----------------------------------
main_window = MainWindow(ca_img_dm.stat, protocol_validity_npz, speed_corr, facemotion_corr, pupil_corr, computed_F_norm, ca_img_dm.time_stamps, speedAndTimeSt, fmotionAndTimeSt, pupilAndTimeSt, photodiode, stim_time_period, red_image_path, save_dir)
main_window.show()
app.exec_()