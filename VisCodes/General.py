from Running_computation import compute_speed
import Ca_imaging
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

if __name__ == "__main__":
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
    neuropil_impact_factor = float(inputs["neuropil_impact_factor"])
    F0_method = inputs["F0_method"]
    neuron_type = inputs["neuron_type"]
    starting_delay_2p = float(inputs["starting_delay_2p"])
    freq_2p = round(float(inputs["Photon_fre"]))

    # Use inputs to process data
    num_samples = 1000
    face_dir = os.path.join(base_path, "FaceIt")
    visual_stim_path = os.path.join(base_path, "visual-stim.npy")
    visual_stim = np.load(visual_stim_path, allow_pickle=True).item()

    protocol_id = inputs["protocol_ids"]
    protocol_name = inputs["protocol_names"]
    protocol_duration = dict(zip(visual_stim['protocol_id'], visual_stim['time_duration']))

##---------------------------------- Load Ca-Imaging data ----------------------
raw_F, raw_Fneu, iscell, stat, mean_image = Ca_imaging.load_Suite2p(base_path)
Ca_imaging.Save_mean_Image(mean_image, save_dir)
xml = Ca_imaging.load_xml(base_path)
F_time_stamp = xml['Green']['relativeTime']
F_time_stamp_updated = F_time_stamp + starting_delay_2p
##----------------------------------load Camera dta----------------------------------
face_camera = np.load(os.path.join(base_path,"FaceCamera-summary.npy"), allow_pickle=True)
fvideo_time = face_camera.item().get('times')
faceitOutput = np.load(os.path.join(face_dir, "FaceIt.npz"), allow_pickle=True)
pupil = (faceitOutput['pupil_dilation'])
facemotion = (faceitOutput['motion_energy'])
##----------------------------------load Speed----------------------------------
speed_time_stamps, speed, last_F_index = compute_speed(base_path, freq_2p, F_time_stamp_updated)
print("len speed", len(speed))
##----------------------------------alligne F ----------------------------------
raw_F = raw_F[:, :last_F_index]
raw_Fneu = raw_Fneu[:, :last_F_index]
F_time_stamp_updated = F_time_stamp_updated[:last_F_index]
# ---------------------------Detect Neurons Among ROIs------------------
_ , detected_roi = Ca_imaging.detect_cell(iscell, raw_F)
iscell, _ = Ca_imaging.detect_bad_neuropils(detected_roi,raw_Fneu, raw_F, iscell)
raw_Fneu, kept2p_ROI = Ca_imaging.detect_cell(iscell, raw_Fneu)
stat, _ = Ca_imaging.detect_cell(iscell, stat)
raw_F, _ = Ca_imaging.detect_cell(iscell, raw_F)
#------------------------------------Calculation alpha------------------
if neuron_type == "PYR":
    neuropil_impact_factor, alpha_remove = Ca_imaging.calculate_alpha(raw_F,raw_Fneu)
    #-----------------Remove Neurons with negative slope---------------
    mask = np.ones(len(raw_F), dtype=bool)
    mask[alpha_remove]= False
    raw_F = raw_F[mask]
    raw_Fneu = raw_Fneu[mask]
    stat = stat[mask]
#-------------------------Calculation of F0 ----------------------
raw_F = raw_F - (neuropil_impact_factor * raw_Fneu)
percentile = 10
f0 = Ca_imaging.calculate_F0(raw_F, freq_2p, percentile, mode= F0_method, win=60)
#-----------------Remove Neurons with F0 less than 1-----------------
invalid_F0 = [i for i,val in enumerate(f0) if np.any(val < 1)]
isvalid_F0 = np.ones((len(f0), 2))
isvalid_F0[invalid_F0, 0] = 0

raw_F, _ = Ca_imaging.detect_cell(isvalid_F0, raw_F)
f0, _ = Ca_imaging.detect_cell(isvalid_F0, f0)
raw_Fneu, _ = Ca_imaging.detect_cell(isvalid_F0, raw_Fneu)
stat,_ = Ca_imaging.detect_cell(isvalid_F0, stat)
dF = Ca_imaging.deltaF_calculate(raw_F, f0)
#---------------------------------- Load photodiode data -----------------------------
stim_Time_start_realigned, Psignal, Psignal_time = Photodiode.realign_from_photodiode(base_path)
###-------------------------Downsampling Photodiode for visualization-----------------
visual_stim, NIdaq, acq_freq = Photodiode.load_and_data_extraction(base_path)
stim_time_durations, protocol_id, stim_start_times, interstim_times = Photodiode.extract_visual_stim_items(visual_stim)
F_Time_start_realigned, F_stim_init_indexes  = Photodiode.Find_F_stim_index(stim_Time_start_realigned, F_time_stamp_updated)
stim_time_end = F_Time_start_realigned+ stim_time_durations
stim_time_end = stim_time_end.tolist()
stim_time_period = [stim_Time_start_realigned, stim_time_end]
protocol_dict = [
    {"id": pid, "duration": duration, "name": name}
    for pid, duration, name in zip(protocol_id, protocol_duration, protocol_name)]

if not os.path.exists(os.path.join(base_path, "protocol_validity.npz")):
    #Photodiode.average_image(dF, protocol_id,3,5,'looming stim', F_stim_init_indexes, freq_2p, num_samples, save_dir)
    protocol_validity = []
    for protocol in range(len(protocol_dict)):
        chosen_protocol = protocol_dict[protocol]['id']
        protocol_duration = protocol_dict[protocol]['duration']
        protocol_name = protocol_dict[protocol]['name']
        protocol_validity_i = Photodiode.average_image(dF, protocol_id,chosen_protocol,protocol_duration, protocol_name, F_stim_init_indexes, freq_2p, num_samples, save_dir)
        protocol_validity.append(protocol_validity_i)
    np.savez(os.path.join(base_path, "protocol_validity.npz"), **{key: value for d in protocol_validity for key, value in d.items()})
    print(protocol_validity)

F_spontaneous, start_spont_index, end_spont_index = Photodiode.get_spontaneous_F(raw_F, protocol_id, 5, 1200, F_stim_init_indexes, freq_2p)
time_start_spon_index = F_time_stamp_updated[start_spont_index]
time_end_spon_index = F_time_stamp_updated[end_spont_index]
########################
fvideo_first_spont_index = np.argmin(np.abs(fvideo_time - time_start_spon_index))
fvideo_last_spont_index = np.argmin(np.abs(fvideo_time - time_end_spon_index))

#########################
speed_corr = [pearsonr(speed[start_spont_index:end_spont_index], ROI)[0] for ROI in F_spontaneous]
speed_corr = [float(value) for value in speed_corr]
###############################
protocol_validity_npz = np.load(os.path.join(base_path, "protocol_validity.npz"))
loaded_data = [{key: protocol_validity_npz[key] for key in protocol_validity_npz}]
for d in loaded_data:
    Green_Cell = d['static patch']
raw_F = General_functions.normalize_time_series(raw_F, lower=0, upper=5)
speedAndTimeSt = (speed_time_stamps, speed)
Psignal = General_functions.scale_trace(Psignal)
pupil = General_functions.scale_trace(pupil)
facemotion = General_functions.scale_trace(facemotion)
facemotion_spont = facemotion[fvideo_first_spont_index:fvideo_last_spont_index]
fvideo_time_spont = fvideo_time[fvideo_first_spont_index:fvideo_last_spont_index]
print("facemotion_spont size",len(facemotion_spont))
print("fvideo_time_spont ", fvideo_time_spont[:5])
print("fvideo_time_spont ", fvideo_time_spont[-5:])

#########################
""" print("len corr_Face", len(Facemotion_spo))
print("len speed[start_spon_index:end_spon_index] ", len(speed[start_spon_index:end_spon_index]))
print(Facemotion_spo.shape, F_spontaneous[0].shape)
corr_Face = [pearsonr(Facemotion_spo, ROI)[0] for ROI in F_spontaneous]
corr_Face = [float(value) for value in corr_Face]
plt.plot(corr_Face)
plt.show() """
###############################

plt.plot(fvideo_time_spont, facemotion_spont)
plt.show()
photodiode = (Psignal_time, Psignal)
pupil = (fvideo_time, pupil)
facemotion = (fvideo_time, facemotion)
print("len speed_time_stamps ", len(speed_time_stamps[start_spont_index: end_spont_index]))
print("Face_time_spo ", len(fvideo_time_spont))
################################
background_image_path = r"Y:\raw-imaging\TESTS\Mai-An\visual_test\16-00-59\Mean_image_grayscale.png"
main_window = MainWindow(stat, Green_Cell, background_image_path, loaded_data, speed_corr, raw_F, F_time_stamp_updated, speedAndTimeSt, facemotion, pupil, photodiode, stim_time_period)
main_window.show()
app.exec_()