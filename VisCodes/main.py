from Running_computation import compute_speed
import Ca_imaging
import numpy as np
from scipy.stats.stats import pearsonr
import sys
from PyQt5.QtWidgets import QApplication
from Visuial_GUI import MainWindow
import Photodiode
base_path = r"C:\Users\faezeh.rabbani\Desktop\2024_09_03\16-00-59"
#inputs
neuropil_impact_factor = 0.7
F0_method = 'sliding'
neuron_type = "PYR"
save_dir = r"C:\Users\faezeh.rabbani\Desktop\2024_09_03\16-00-59\fig2"
num_samples = 1000
Photon_fre = 29.7597

##---------------------------------- Load Ca-Imaging data ----------------------
F, Fneu_raw, iscell, stat, Mean_image = Ca_imaging.load_Suite2p(base_path)
Ca_imaging.Save_mean_Image(Mean_image, save_dir)
xml = Ca_imaging.load_xml(base_path)
F_time_stamp = xml['Green']['relativeTime']
F_time_stamp_updated = F_time_stamp + 0.100 # adding 100 ms to align F timestamps
##----------------------------------load Speed----------------------------------
speed_time_stamps, speed, last_Flourscnce_index = compute_speed(base_path,Photon_fre, F_time_stamp_updated)
##----------------------------------alligne F ----------------------------------
F = F[:, :last_Flourscnce_index]
Fneu_raw = Fneu_raw[:, :last_Flourscnce_index]
F_time_stamp_updated = F_time_stamp_updated[:last_Flourscnce_index]
# -----------------------------------Detect Neurons Among ROIs--------------------------
_ , detected_roi = Ca_imaging.detect_cell(iscell, F)
iscell, neuron_chosen3 = Ca_imaging.detect_bad_neuropils(detected_roi,Fneu_raw, F, iscell)
Fneu_raw, keeped_ROI = Ca_imaging.detect_cell(iscell, Fneu_raw)
stat, _ = Ca_imaging.detect_cell(iscell, stat)
F, _ = Ca_imaging.detect_cell(iscell, F)
#------------------------------------Calculation alpha------------------
if neuron_type == "PYR":
    neuropil_impact_factor, remove = Ca_imaging.calculate_alpha(F,Fneu_raw)
    #-----------------Remove Neurons with negative slope---------------
    mask = np.ones(len(F), dtype=bool)
    mask[remove]= False
    F = F[mask]
    Fneu_raw = Fneu_raw[mask]
    stat = stat[mask]

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
Fneu_raw, _ = Ca_imaging.detect_cell(invalid_cell_F0, Fneu_raw)
stat,_ = Ca_imaging.detect_cell(invalid_cell_F0, stat)
print("stat", stat)
dF = Ca_imaging.deltaF_calculate(F, F0)

#--------------------------- Load photodiode data --------------------
stim_Time_start_realigned, Psignal = Photodiode.realign_from_photodiode(base_path)
visual_stim, NIdaq, Acquisition_Frequency =Photodiode.load_and_data_extraction(base_path)
time_duration, protocol_id, time_start, interstim = Photodiode.extract_visual_stim_items(visual_stim)
Flou_Time_start_realigned, F_stim_init_indexes  = Photodiode.Find_F_stim_index(stim_Time_start_realigned, F_time_stamp_updated)

# protocol_id_o = [0,1,2,3,4,5,6]
# protocol_duration = [3,3,1,5,2,1200,3]
# protocol_name = ["moving dots","random dots", "static patch", "looming stim", "Natural Images 4 repeats", "grey 20min","drifting gratings" ]
# merged_list = [
#     {"id": pid, "duration": duration, "name": name}
#     for pid, duration, name in zip(protocol_id_o, protocol_duration, protocol_name)]

####Photodiode.average_image(dF, protocol_id,3,5,'looming stim', F_stim_init_indexes,Photon_fre, num_samples, save_dir)
# protocol_validity = []
# for protocol in range(len(merged_list)):
#     chosen_protocol = merged_list[protocol]['id']
#     protocol_duration = merged_list[protocol]['duration']
#     protocol_name = merged_list[protocol]['name']
#     protocol_validity_i = Photodiode.average_image(dF, protocol_id,chosen_protocol,protocol_duration, protocol_name, F_stim_init_indexes,Photon_fre, num_samples, save_dir)
#     protocol_validity.append(protocol_validity_i)

# np.savez(r"D:\Faezeh 2p2analyze\2024_09_03\16-00-59\fig2\protocol_validity.npz", **{key: value for d in protocol_validity for key, value in d.items()})
# print(protocol_validity)

F_spontaneous, start_spon_index, end_spon_index = Photodiode.get_spontaneous_F(F, protocol_id,5,1200, F_stim_init_indexes, Photon_fre)
corr = [pearsonr(speed[start_spon_index:end_spon_index], ROI)[0] for ROI in F_spontaneous]
corr_running = [float(value) for value in corr]

print("corr", corr)
# num = np.arange(len(corr))
# plt.scatter(num, corr)
# plt.show()
###############################
loaded_npz = np.load(r"C:\Users\faezeh.rabbani\Desktop\2024_09_03\16-00-59\fig2\GUI_test\protocol_validity.npz")
loaded_data = [{key: loaded_npz[key] for key in loaded_npz}]
for d in loaded_data:
    print(d['static patch'])
    Green_Cell = d['static patch']

background_image_path = r"C:\Users\faezeh.rabbani\Desktop\2024_09_03\16-00-59\fig2\GUI_test\65Mean_image_grayscale.png"
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow(stat, Green_Cell, background_image_path, loaded_data, corr_running)
    main_window.show()
    sys.exit(app.exec_())

