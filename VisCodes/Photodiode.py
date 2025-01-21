import os.path
import json
import numpy as np
import figures
import General_functions
import Running_computation
from scipy.ndimage import filters, gaussian_filter1d

def load_and_data_extraction(base_path):
    vis_stim_path = os.path.join(base_path, "visual-stim.npy")
    NIdaq_path = os.path.join(base_path, "NIdaq.npy")
    if os.path.exists(vis_stim_path):
        visual_stim = np.load(vis_stim_path, allow_pickle=True)
    else:
        raise Exception("No visual-stim.npy file exists in this directory")
    if os.path.exists(NIdaq_path):
        NIdaq = np.load(NIdaq_path, allow_pickle=True).item()
    else:
        raise Exception("No NIdaq.npy file exists in this directory")
    metadata_path = os.path.join(base_path, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as file:
            data = json.load(file)
        Acquisition_Frequency = data['NIdaq-acquisition-frequency']
    else:
        raise Exception("No JSON metadata file exists in this directory")
    return visual_stim, NIdaq, Acquisition_Frequency


def visual_stim_extraction(visual_stim):
    visual_stim = visual_stim.item()
    time_duration = visual_stim['time_duration']
    protocol_id = visual_stim['protocol_id']
    time_start = visual_stim['time_start']
    return time_duration, protocol_id, time_start

def realign_from_photodiode(base_path):
    # Calculate threshold for photodiode activity
    visual_stim, NIdaq, Acquisition_Frequency = load_and_data_extraction(base_path)
    Psignal_time, Psignal = Running_computation.resample_signal(NIdaq['analog'][0],
                                                     original_freq=Acquisition_Frequency,
                                                     pre_smoothing=2. / Acquisition_Frequency,
                                                     new_freq=1000)
    time_duration, protocol_id, time_start = visual_stim_extraction(visual_stim)

    smooth_signal = np.diff(gaussian_filter1d(np.cumsum(Psignal),7))  # integral + smooth + derivative
    smooth_signal[:1000], smooth_signal[-10:] = smooth_signal[1000], smooth_signal[-1000]  # to insure no problem at borders (of the derivative)\
    # compute signal boundaries to evaluate threshold crossing of photodiode signal
    H, bins = np.histogram(smooth_signal, bins=100)
    baseline = bins[np.argmax(H) + 1]
    #figures.Visualize_baseline(smooth_signal, baseline)
    threshold = (np.max(smooth_signal) - baseline) / 4.  # reaching 25% of peak level
    cond_thresh = (smooth_signal[1:] >= (baseline + threshold)) & (
                smooth_signal[:-1] < (baseline + threshold))
    true_indices = np.where(cond_thresh)[0]
    true_indices = [x * (1/1000) for x in true_indices]

    stim_Time_start_realigned = []
    for value in time_start:
        index = np.searchsorted(true_indices, value, side='right')
        if index < len(true_indices):
            stim_Time_start_realigned.append(true_indices[index])
    stim_Time_start_realigned = [float(val) for val in stim_Time_start_realigned]
    return stim_Time_start_realigned, Psignal, Psignal_time


def extract_visual_stim_items(visual_stim):
    visual_stim = visual_stim.item()
    time_duration = visual_stim['time_duration']
    protocol_id = visual_stim['protocol_id']
    time_start = visual_stim['time_start']
    interstim = visual_stim['interstim']
    return time_duration, protocol_id, time_start, interstim

def Find_F_stim_index(stim_Time_start_realigned, F_time_stamp_updated):
    # find the stim time in Fluorescence time-scale
    Flou_Time_start_realigned = []
    F_stim_init_indexes = []
    for value in stim_Time_start_realigned:
        index = np.searchsorted(F_time_stamp_updated, value, side='right')
        if index < len(F_time_stamp_updated):
            F_stim_init_indexes.append(index)
            Flou_Time_start_realigned.append(F_time_stamp_updated[index])
    Flou_Time_start_realigned = [float(val) for val in Flou_Time_start_realigned]
    F_stim_init_indexes = [int(val) for val in F_stim_init_indexes]
    return Flou_Time_start_realigned, F_stim_init_indexes

#################################
# period_start = []
# period_end = []
# period_interval = []
# for i in range(len(interstim)):
#     if interstim[i]>=2:
#         start = stim_Time_start_realigned[i]-1
#     else:
#         start = stim_Time_start_realigned[i] - interstim[i]/2
#     if (i < len(interstim) - 1 and interstim[i+1] >= 2) or (i == len(interstim) - 1):
#
#         end = stim_Time_start_realigned[i] + time_duration[i] + 1
#     else:
#         end = stim_Time_start_realigned[i] + time_duration[i] + interstim[i]/2
#     period_start.append(start)
#     period_end.append(end)
#     period_interval.append([period_start, period_end])
#
# color_map = {
#     0: 'blue',
#     1: 'red',
#     2: 'green',
#     3: 'yellow',
#     4: 'purple',
#     5: 'orange',
#     6: 'gray'
# }
# Psignal_time = np.arange(0, len(Psignal)/1000, 0.001)
# F_normalized = (F - np.min(F)) / (np.max(F) - np.min(F))
# Psignal_normal = (Psignal - np.min(Psignal)) / (np.max(Psignal) - np.min(Psignal))
#
# fig2, ax2 = plt.subplots()
# plt.plot(F_time_stamp_updated, F_normalized[5] )
# plt.plot(Psignal_time, Psignal_normal/6)
# for i in range(len(protocol_id)):
#     if protocol_id[i] == 3:
#         ax2.axvline(x=stim_Time_start_realigned[i], color='b', linestyle='--', alpha=0.7)
#         ax2.axvline(x=stim_Time_start_realigned[i] + time_duration[i], color='black', linestyle='--', alpha=0.7)
#         ax2.axvspan(period_start[i], period_end[i], color='gray', alpha=0.8)

#plt.show()
#
# color_names = [color_map[value] for value in protocol_id]
# fig, ax = plt.subplots()
# ax.plot(F_time_stamp_updated, F[5])
# for x in range(len(period_start)):
#     ax.axvline(x=stim_Time_start_realigned[x], color='b', linestyle='--', alpha=0.7)
#     ax.axvline(x=stim_Time_start_realigned[x] + time_duration[x], color='black', linestyle='--', alpha=0.7)
#     ax.axvspan(period_start[x], period_end[x], color=color_names[x], alpha=0.8)
# plt.show()
######################################
def get_base_line(F, F_stim_init_indexes):
    base_line = []
    for i in F_stim_init_indexes:
        F_base_i = F[i - 29:i]
        base_line.append(F_base_i)
    base_line = np.array(base_line)
    return base_line

def average_image(
    F, protocol_ids, chosen_protocol, protocol_duration_s, protocol_name,
    F_stim_init_indexes, Photon_fre, num_samples, save_dir
):
    """
    Analyze neurons for the given protocol and determine valid neurons.
    """
    print(protocol_name)
    protocol_duration = int(protocol_duration_s * Photon_fre)
    Valid_Neuron = np.zeros(len(F))
    protocol_validity = None  # Initialize protocol_validity

    for Neuron_index in range(len(F)):
        test_F = F[Neuron_index]
        base_line_F_I = get_base_line(test_F, F_stim_init_indexes)
        bootstrapped_base, fith_bootstraping_base = General_functions.bootstrap(base_line_F_I, num_samples)
        F_specific_protocol = []
        F_stim = []

        # Process the chosen protocol
        for i in range(len(protocol_ids)):
            if protocol_ids[i] == chosen_protocol:
                F_indexes = int(F_stim_init_indexes[i])
                F_specific_protocol_i = test_F[F_indexes - 29:F_indexes + protocol_duration + 29]
                F_stim_i = test_F[F_indexes:F_indexes + protocol_duration]
                F_stim.append(np.percentile(F_stim_i, 95))
                F_specific_protocol.append(F_specific_protocol_i)

        # Calculate p-value
        twenty_perc_F_stim = np.percentile(F_stim, 80)
        print("twenty_perc_F_stim",twenty_perc_F_stim)
        p_value = np.sum(fith_bootstraping_base >= twenty_perc_F_stim) / num_samples

        # Check if neuron is valid
        if p_value <= 0.05:
            Valid_Neuron[Neuron_index] = 1
            color_histo = "skyblue"
        else:
            color_histo = "thistle"

        # Generate figures
        figures.Bootstrapping_fig(
            fith_bootstraping_base, twenty_perc_F_stim, protocol_name,
            p_value, Neuron_index, color_histo, save_dir
        )
        mean_F_specific_protocol = np.mean(F_specific_protocol, 0)
        std_F_specific_protocol = np.std(F_specific_protocol, 0)
        figures.stim_period(
            protocol_duration_s, Photon_fre, mean_F_specific_protocol,
            std_F_specific_protocol, protocol_name, Neuron_index, save_dir
        )
    # Finalize protocol_validity
    protocol_validity = {protocol_name: Valid_Neuron}

    return protocol_validity


def get_spontaneous_F (F, protocol_ids,chosen_protocol, protocol_duration_s, F_stim_init_indexes,Photon_fre):
    protocol_duration = int(protocol_duration_s * Photon_fre)
    F_spontaneous = []
    for Neuron_index in range(len(F)):
        test_F = F[Neuron_index]
        for i in range(len(protocol_ids)):
            if protocol_ids[i] == chosen_protocol:
                start_spon_index = int(F_stim_init_indexes[i])
                # F_specific_protocol_i = test_F[start_spon_index-29:start_spon_index + protocol_duration + 29]
                F_spontaneous_i = test_F[start_spon_index: start_spon_index + protocol_duration]
                F_spontaneous.append(F_spontaneous_i)
    if "start_spon_index" in locals():
        end_spon_index = start_spon_index + protocol_duration
        return F_spontaneous, start_spon_index, end_spon_index
    else:
        return F_spontaneous
