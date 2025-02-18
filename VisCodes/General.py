from Running_computation import compute_speed
from Ca_imaging import CaImagingDataManager
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

# Use inputs to process data
face_dir = os.path.join(base_path, "FaceIt")
visual_stim_path = os.path.join(base_path, "visual-stim.npy")
visual_stim = np.load(visual_stim_path, allow_pickle=True).item()

protocol_id = inputs["protocol_ids"]
protocol_name = inputs["protocol_names"]
protocol_df = pd.DataFrame({"id" : protocol_id, "name" : protocol_name})
protocol_duration = pd.DataFrame({ "id" : visual_stim['protocol_id'], "duration" : visual_stim['time_duration']}).groupby(by="id").mean()
protocol_df = protocol_df.join(protocol_duration, on="id", how="inner").set_index("id")

#---------------------------------- Load Ca-Imaging data ----------------------
ca_img_dm = CaImagingDataManager(base_path, neuropil_impact_factor, F0_method, neuron_type, starting_delay_2p)
ca_img_dm.save_mean_image(save_dir)
detected_roi = ca_img_dm._list_ROIs_idx
print('Original number of neurons :', len(detected_roi))

#---------------------------------- Load Camera data ----------------------------------
face_camera = np.load(os.path.join(base_path,"FaceCamera-summary.npy"), allow_pickle=True)
fvideo_time = face_camera.item().get('times')
faceitOutput = np.load(os.path.join(face_dir, "FaceIt.npz"), allow_pickle=True)
pupil = (faceitOutput['pupil_dilation'])
facemotion = (faceitOutput['motion_energy'])
print("len facemotion :", len(facemotion))

#---------------------------------- Compute speed ----------------------------------
speed, speed_time_stamps, last_F_index = compute_speed(base_path, ca_img_dm.fs, ca_img_dm.time_stamps)
ca_img_dm.cut_frames(last_index=last_F_index) #update metrics with new frames length

#---------------------------------- Detect ROIs with bad neuropils ------------------
ca_img_dm.detect_bad_neuropils()
kept2p_ROI = ca_img_dm._list_ROIs_idx
print('After removing bad neuropil neurons, nb of neurons :', len(kept2p_ROI))

#---------------------------------- Compute Fluorescence ------------------
ca_img_dm.compute_F()
kept_ROI_alpha = ca_img_dm._list_ROIs_idx
print('Number of remaining neurons after alpha calculation :', len(kept_ROI_alpha))

#---------------------------------- Calculation of F0 ----------------------
ca_img_dm.compute_F0(percentile= 10, win=60)
kept_ROI_F0 = ca_img_dm._list_ROIs_idx
print('Number of remaining neurons after F0 calculation  :', len(kept_ROI_F0))

#---------------------------------- Calculation of dF over F0 ----------------------
ca_img_dm.compute_dFoF0()
computed_F_norm = ca_img_dm.normalize_time_series("dFoF0", lower=0, upper=5)

#---------------------------------- Load photodiode data -----------------------------
stim_Time_start_realigned, Psignal, Psignal_time = Photodiode.realign_from_photodiode(base_path)

#---------------------------------- Downsampling Photodiode for visualization-----------------
#visual_stim, NIdaq, acq_freq = Photodiode.load_and_data_extraction(base_path)
#stim_time_durations, protocol_id, stim_start_times, interstim_times = Photodiode.extract_visual_stim_items(visual_stim)
stim_time_durations = visual_stim['time_duration']
F_Time_start_realigned, F_stim_init_indexes  = Photodiode.Find_F_stim_index(stim_Time_start_realigned, ca_img_dm.time_stamps)
stim_time_end = F_Time_start_realigned+ stim_time_durations
stim_time_end = stim_time_end.tolist()
stim_time_period = [stim_Time_start_realigned, stim_time_end]

if not os.path.exists(os.path.join(base_path, "protocol_validity.npz")):
    protocol_validity = []
    for protocol in range(len(protocol_df)):
        chosen_protocol = protocol_df.index[protocol]
        protocol_duration = protocol_df['duration'][protocol]
        protocol_name = protocol_df['name'][protocol]
        protocol_validity_i = Photodiode.average_image(ca_img_dm.dFoF0, visual_stim['protocol_id'],chosen_protocol,protocol_duration, protocol_name, F_stim_init_indexes, ca_img_dm.fs, num_samples, save_dir)
        protocol_validity.append(protocol_validity_i)
    np.savez(os.path.join(base_path, "protocol_validity.npz"), **{key: value for d in protocol_validity for key, value in d.items()})
    print(protocol_validity)

#----------------- Spontaneous behaviour -----------------
id_spont = protocol_df[protocol_df['name'] == 'grey-20min'].index[0]
duration_spont = protocol_df.iloc[id_spont]["duration"]
F_spontaneous, start_spont_index, end_spont_index = Photodiode.get_spontaneous_F(ca_img_dm.fluorescence, visual_stim['protocol_id'], id_spont, duration_spont, F_stim_init_indexes, ca_img_dm.fs)
time_start_spon_index = ca_img_dm.time_stamps[start_spont_index]
time_end_spon_index = ca_img_dm.time_stamps[end_spont_index]

fvideo_first_spont_index = np.argmin(np.abs(fvideo_time - time_start_spon_index))
fvideo_last_spont_index = np.argmin(np.abs(fvideo_time - time_end_spon_index))

speed_corr = [pearsonr(speed[start_spont_index:end_spont_index], ROI)[0] for ROI in F_spontaneous]
speed_corr = [float(value) for value in speed_corr]

Psignal = General_functions.scale_trace(Psignal)
pupil = General_functions.scale_trace(pupil)
facemotion = General_functions.scale_trace(facemotion)

facemotion_spont = facemotion[fvideo_first_spont_index:fvideo_last_spont_index]
fvideo_time_spont = fvideo_time[fvideo_first_spont_index:fvideo_last_spont_index]
print("facemotion_spont size",len(facemotion_spont))
print("fvideo_time_spont ", fvideo_time_spont[:5])
print("fvideo_time_spont ", fvideo_time_spont[-5:])

#########################
""" print("len corr_Face", len(facemotion_spont))
print("len speed[start_spont_index:end_spont_index] ", len(speed[start_spont_index:end_spont_index]))
print(facemotion_spont.shape, F_spontaneous[0].shape)
corr_Face = [pearsonr(facemotion_spont, ROI)[0] for ROI in F_spontaneous]
corr_Face = [float(value) for value in corr_Face]
plt.plot(corr_Face)
plt.show() """

""" plt.plot(fvideo_time_spont, facemotion_spont)
plt.show() """

################################

photodiode = (Psignal_time, Psignal)
pupil = (fvideo_time, pupil)
facemotion = (fvideo_time, facemotion)
speedAndTimeSt = (speed_time_stamps, speed)
print("len speed_time_stamps ", len(speed_time_stamps[start_spont_index: end_spont_index]))
print("Face_time_spo ", len(fvideo_time_spont))
background_image_path = os.path.join(base_path, "Mean_image_grayscale.png")
protocol_validity_npz = np.load(os.path.join(base_path, "protocol_validity.npz"))

# ------------------HDF5 files---------------
H5_dir = os.path.join(save_dir, "postprocessing.h5")
hf = h5py.File(H5_dir, 'w')
behavioral_group = hf.create_group('Behavioral')
correlation = behavioral_group.create_group("Correlation")
caImg_group = hf.create_group('Ca_imaging')
caImg_full_trace = caImg_group.create_group('full_trace')
rois_group = hf.create_group("ROIs")

General_functions.create_H5_dataset(behavioral_group, [speedAndTimeSt, facemotion, pupil], ['Speed', 'FaceMotion', 'Pupil'])
General_functions.create_H5_dataset(correlation, [speed_corr], ['speed_corr'])
caImg_group.create_dataset('Time', data=ca_img_dm.time_stamps)
General_functions.create_H5_dataset(caImg_full_trace, [ca_img_dm.raw_F, ca_img_dm.raw_Fneu, ca_img_dm.fluorescence, ca_img_dm.f0, ca_img_dm.dFoF0], 
                                    ['raw_F', 'raw_Fneu', 'F', 'F0', 'dFoF0'])
General_functions.create_H5_dataset(rois_group, [kept2p_ROI, kept_ROI_alpha, kept_ROI_F0], 
                                    ['1_neuropil', '2_alpha', '3_F0'])

hf.close()

#Second GUI
main_window = MainWindow(ca_img_dm.stat, protocol_validity_npz, speed_corr, computed_F_norm, ca_img_dm.time_stamps, speedAndTimeSt, facemotion, pupil, photodiode, stim_time_period, base_path, save_dir)
main_window.show()
app.exec_()