from Running_computation import compute_speed, resample_signal
import Ca_imaging
import matplotlib.pyplot as plt
import numpy as np
import os.path
import json
base_path = r"D:\Faezeh 2p2analyze\2024_09_03\16-00-59"
#inputs
neuropil_impact_factor = 0.7
F0_method = 'sliding'
neuron_type = "PYR"
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


