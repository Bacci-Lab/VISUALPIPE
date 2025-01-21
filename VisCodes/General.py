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
from init_vis import Ui_MainWindow
import matplotlib.pyplot as plt
class InputWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

    def get_inputs(self):
        """Retrieve user inputs from the first GUI."""
        return {
            "base_path": self.ui.lineEdit_data_directory.text(),
            "save_dir": self.ui.lineEdit_save_directory.text(),
            "neuropil_impact_factor": self.ui.lineEdit_Neuropil_IF.text(),
            "F0_method": self.ui.comboBox_F0_method.currentText(),
            "neuron_type": self.ui.comboBox_neural_type.currentText(),
            "starting_delay_2p": self.ui.lineEdit_starting_delay.text(),
            "Photon_fre": self.ui.lineEdit_Fre.text(),
            "protocol_ids": self.ui.protocol_numbers if hasattr(self.ui, 'protocol_numbers') else [],
            "protocol_names": self.ui.protocol_names if hasattr(self.ui, 'protocol_names') else [],
        }
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
    Photon_fre = round(float(inputs["Photon_fre"]))

    # Use inputs to process data
    num_samples = 1000
    face_dir = os.path.join(base_path, "FaceIt")
    protocol_id_o = inputs["protocol_ids"]
    protocol_name = inputs["protocol_names"]
    protocol_duration = [3, 3, 1, 5, 2, 1200, 3]

##---------------------------------- Load Ca-Imaging data ----------------------
F, Fneu_raw, iscell, stat, Mean_image = Ca_imaging.load_Suite2p(base_path)
Ca_imaging.Save_mean_Image(Mean_image, save_dir)
xml = Ca_imaging.load_xml(base_path)
F_time_stamp = xml['Green']['relativeTime']
F_time_stamp_updated = F_time_stamp + starting_delay_2p
##----------------------------------load Camera dta----------------------------------
face_camera = np.load(os.path.join(base_path,"FaceCamera-summary.npy"), allow_pickle=True)
Face_time = face_camera.item().get('times')
face = np.load(os.path.join(face_dir, "FaceIt.npz"), allow_pickle=True)
Pupil = (face['pupil_dilation'])
Facemotion = (face['motion_energy'])
##----------------------------------load Speed----------------------------------
speed_time_stamps, speed, last_Flourscnce_index = compute_speed(base_path, Photon_fre, F_time_stamp_updated)
print("len speed", len(speed))
##----------------------------------alligne F ----------------------------------
F = F[:, :last_Flourscnce_index]
Fneu_raw = Fneu_raw[:, :last_Flourscnce_index]
F_time_stamp_updated = F_time_stamp_updated[:last_Flourscnce_index]
# ---------------------------Detect Neurons Among ROIs------------------
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
percentile = 10
F0 = Ca_imaging.calculate_F0(F, Photon_fre, percentile, mode= F0_method, win=60)
#-----------------Remove Neurons with F0 less than 1-----------------
zero_F0 = [i for i,val in enumerate(F0) if np.any(val < 1)]
invalid_cell_F0 = np.ones((len(F0), 2))
invalid_cell_F0[zero_F0, 0] = 0

F, _ = Ca_imaging.detect_cell(invalid_cell_F0, F)
F0, _ = Ca_imaging.detect_cell(invalid_cell_F0, F0)
Fneu_raw, _ = Ca_imaging.detect_cell(invalid_cell_F0, Fneu_raw)
stat,_ = Ca_imaging.detect_cell(invalid_cell_F0, stat)
dF = Ca_imaging.deltaF_calculate(F, F0)
#---------------------------------- Load photodiode data -----------------------------
stim_Time_start_realigned, Psignal, Psignal_time = Photodiode.realign_from_photodiode(base_path)
###-------------------------Downsampling Photodiode for visualization-----------------
visual_stim, NIdaq, Acquisition_Frequency =Photodiode.load_and_data_extraction(base_path)
time_duration, protocol_id, time_start, interstim = Photodiode.extract_visual_stim_items(visual_stim)
Flou_Time_start_realigned, F_stim_init_indexes  = Photodiode.Find_F_stim_index(stim_Time_start_realigned, F_time_stamp_updated)
stim_time_end = Flou_Time_start_realigned+ time_duration
stim_time_end = stim_time_end.tolist()
stim_time_period = [stim_Time_start_realigned, stim_time_end]
merged_list = [
    {"id": pid, "duration": duration, "name": name}
    for pid, duration, name in zip(protocol_id_o, protocol_duration, protocol_name)]

###Photodiode.average_image(dF, protocol_id,3,5,'looming stim', F_stim_init_indexes,Photon_fre, num_samples, save_dir)
# protocol_validity = []
# for protocol in range(len(merged_list)):
#     chosen_protocol = merged_list[protocol]['id']
#     protocol_duration = merged_list[protocol]['duration']
#     protocol_name = merged_list[protocol]['name']
#     protocol_validity_i = Photodiode.average_image(dF, protocol_id,chosen_protocol,protocol_duration, protocol_name, F_stim_init_indexes,Photon_fre, num_samples, save_dir)
#     protocol_validity.append(protocol_validity_i)
# np.savez(r"D:\Faezeh 2p2analyze\2024_09_03\16-00-59\fig2\protocol_validity.npz", **{key: value for d in protocol_validity for key, value in d.items()})
# print(protocol_validity)
F_spontaneous, start_spon_index, end_spon_index = Photodiode.get_spontaneous_F(F, protocol_id, 5, 1200, F_stim_init_indexes, Photon_fre)
time_start_spon_index = F_time_stamp_updated[start_spon_index]
time_end_spon_index = F_time_stamp_updated[end_spon_index]
########################
Face_first_spon_index = np.argmin(np.abs(Face_time - time_start_spon_index))
Face_last_spon_index = np.argmin(np.abs(Face_time - time_end_spon_index))

#########################
corr_runing = [pearsonr(speed[start_spon_index:end_spon_index], ROI)[0] for ROI in F_spontaneous]
corr_running = [float(value) for value in corr_runing]
###############################
loaded_npz = np.load(r"C:\Users\faezeh.rabbani\Desktop\2024_09_03\16-00-59\fig2\GUI_test\protocol_validity.npz")
loaded_data = [{key: loaded_npz[key] for key in loaded_npz}]
for d in loaded_data:
    Green_Cell = d['static patch']
F = General_functions.normalize_time_series(F, lower=0, upper=5)
Run = (speed_time_stamps, speed)
Psignal = General_functions.scale_trace(Psignal)
Pupil = General_functions.scale_trace(Pupil)
Facemotion = General_functions.scale_trace(Facemotion)
Facemotion_spo = Facemotion[Face_first_spon_index:Face_last_spon_index]
Face_time_spo = Face_time[Face_first_spon_index:Face_last_spon_index]
print("Face_time_spo ",len(Facemotion_spo))
print("Face_time_spo ",Face_time_spo[:5])
print("Face_time_spo ",Face_time_spo[-5:])

#########################
print("len corr_Face", len(Facemotion_spo))
print("len speed[start_spon_index:end_spon_index] ", len(speed[start_spon_index:end_spon_index]))
corr_Face = [pearsonr(Facemotion_spo, ROI)[0] for ROI in F_spontaneous]
corr_Face = [float(value) for value in corr_Face]
plt.plot(corr_Face)
plt.show()
###############################

plt.plot(Face_time_spo, Facemotion_spo)
plt.show()
Photodiode = (Psignal_time, Psignal)
Pupil = (Face_time, Pupil)
Facemotion = (Face_time, Facemotion)
print("len speed_time_stamps ", len(speed_time_stamps[start_spon_index: end_spon_index]))
print("Face_time_spo ", len(Face_time_spo))
################################
background_image_path = r"C:\Users\faezeh.rabbani\Desktop\2024_09_03\16-00-59\fig2\GUI_test\65Mean_image_grayscale.png"
main_window = MainWindow(stat, Green_Cell, background_image_path, loaded_data, corr_running, F, F_time_stamp_updated, Run, Facemotion, Pupil, Photodiode, stim_time_period)
main_window.show()
app.exec_()