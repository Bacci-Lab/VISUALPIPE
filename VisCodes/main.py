from Running_computation import compute_speed, resample_signal
import Ca_imaging
import matplotlib.pyplot as plt
import numpy as np
import os.path
import json
import Photodiode
base_path = r"D:\Faezeh 2p2analyze\2024_09_03\16-00-59"
#inputs
neuropil_impact_factor = 0.7
F0_method = 'sliding'
neuron_type = "PYR"
save_dir = r"D:\Faezeh 2p2analyze\2024_09_03\16-00-59\fig"
num_samples = 1000
Photon_fre = 29.7597
##----------------------------------load Speed----------------------------------
speed_time_stamps, speed = compute_speed(base_path)
##----------------------------------Load Ca-Imaging data----------------------------------
F, Fneu_raw, iscell = Ca_imaging.load_Suite2p(base_path)
xml = Ca_imaging.load_xml(base_path)
F_time_stamp = xml['Green']['relativeTime']
F_time_stamp_updated = F_time_stamp + 0.100 # adding 100 ms to align F timestamps
# -----------------------------------Detect Neurons Among ROIs--------------------------
_ , detected_roi = Ca_imaging.detect_cell(iscell, F)
iscell, neuron_chosen3 = Ca_imaging.detect_bad_neuropils(detected_roi,Fneu_raw, F, iscell)
Fneu_raw, keeped_ROI = Ca_imaging.detect_cell(iscell, Fneu_raw)
F, _ = Ca_imaging.detect_cell(iscell, F)

#------------------------------------Calculation alpha------------------
if neuron_type == "PYR":
    neuropil_impact_factor, remove = Ca_imaging.calculate_alpha(F,Fneu_raw)
    #-----------------Remove Neurons with negative slope---------------
    mask = np.ones(len(F), dtype=bool)
    mask[remove]= False
    F = F[mask]
    Fneu_raw = Fneu_raw[mask]

#-------------------------Calculation of F0 ----------------------
F = F - (neuropil_impact_factor * Fneu_raw)
Fs = 30
percentile = 10
F0 = Ca_imaging.calculate_F0(F, Fs, percentile, mode= F0_method, win=60)
#-----------------Remove Neurons with F0 less than 1-----------------
zero_F0 = [i for i,val in enumerate(F0) if np.any(val < 1)]
invalid_cell_F0 = np.ones((len(F0), 2))
invalid_cell_F0[zero_F0, 0] = 0

F, _ = Ca_imaging.detect_cell(invalid_cell_F0, F)
F0, _ = Ca_imaging.detect_cell(invalid_cell_F0, F0)
Fneu_raw, Ca_imaging.detect_cell(invalid_cell_F0, Fneu_raw)
dF = Ca_imaging.deltaF_calculate(F, F0)

#--------------------------- Load photodiode data --------------------
stim_Time_start_realigned, Psignal = Photodiode.realign_from_photodiode(base_path)
visual_stim, NIdaq, Acquisition_Frequency =Photodiode.load_and_data_extraction(base_path)
time_duration, protocol_id, time_start, interstim = Photodiode.extract_visual_stim_items(visual_stim)
Flou_Time_start_realigned, F_stim_init_indexes  = Photodiode.Find_F_stim_index(stim_Time_start_realigned, F_time_stamp_updated)

protocol_id_o = [0,1,2,3,4,5,6]
protocol_duration = [3,3,1,5,2,1200,3]
protocol_name = ["moving dots","random dots", "static patch", "looming stim", "Natural Images 4 repeats", "grey 20min","drifting gratings" ]
merged_list = [
    {"id": pid, "duration": duration, "name": name}
    for pid, duration, name in zip(protocol_id_o, protocol_duration, protocol_name)]
# Photodiode.avarage_image(dF, protocol_id,3,5,'looming stim', F_stim_init_indexes,Photon_fre, num_samples, save_dir)

for protocol in range(len(merged_list)):
    chosen_protocol = merged_list[protocol]['id']
    protocol_duration = merged_list[protocol]['duration']
    protocol_name = merged_list[protocol]['name']
    Photodiode.avarage_image(dF, protocol_id,chosen_protocol,protocol_duration, protocol_name, F_stim_init_indexes,Photon_fre, num_samples, save_dir)
