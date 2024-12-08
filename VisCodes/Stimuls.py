import numpy as np
import matplotlib.pyplot as plt
import lindi
import Running_computation
import Ca_imaging
from scipy.ndimage import filters, gaussian_filter1d
base_path = r"D:\Faezeh 2p2analyze\2024_09_03\16-00-59"
F, Fneu_raw, iscell, _, _ = Ca_imaging.load_Suite2p(base_path)
xml = Ca_imaging.load_xml(base_path)
F_time_stamp = xml['Green']['relativeTime']
F_time_stamp_updated = F_time_stamp + 0.100
_ , detected_roi = Ca_imaging.detect_cell(iscell, F)
iscell, neuron_chosen3 = Ca_imaging.detect_bad_neuropils(detected_roi,Fneu_raw, F, iscell)
Fneu_raw, keeped_ROI = Ca_imaging.detect_cell(iscell, Fneu_raw)
F, _ = Ca_imaging.detect_cell(iscell, F)
print(len(F), len(F))


visual_stim = np.load(r"D:\Faezeh 2p2analyze\2024_09_03\16-00-59\visual-stim.npy", allow_pickle=True)
NIdaq_path = r"D:\Faezeh 2p2analyze\2024_09_03\16-00-59\NIdaq.npy"
NIdaq = np.load(NIdaq_path, allow_pickle=True).item()
print(NIdaq.keys())
max_episode=-1
_, Psignal =Running_computation.resample_signal(NIdaq['analog'][0],
                             original_freq=float(10000),
                             pre_smoothing=2. / float(10000),
                             new_freq=1000)

dt = NIdaq['dt']
t = np.arange(len(Psignal))*dt
# Psignal is the final photodiode signal which must be saved


data = visual_stim.item()
data_keys = data.keys()
time_duration = data['time_duration']

protocol_id = data['protocol_id']
time_start = data['time_start']

###################
smooth_signal = np.diff(gaussian_filter1d(np.cumsum(Psignal),7))  # integral + smooth + derivative
smooth_signal[:1000], smooth_signal[-10:] = smooth_signal[1000], smooth_signal[-1000]  # to insure no problem at borders (of the derivative)\
# compute signal boundaries to evaluate threshold crossing of photodiode signal
H, bins = np.histogram(smooth_signal, bins=100)
baseline = bins[np.argmax(H) + 1]
threshold = (np.max(smooth_signal) - baseline) / 4.  # reaching 25% of peak level
cond_thresh = (smooth_signal[1:] >= (baseline + threshold)) & (
            smooth_signal[:-1] < (baseline + threshold))
true_indices = np.where(cond_thresh)[0]
true_indices = [x * (1/1000) for x in true_indices]
Time_start_realigned= []
for value in time_start:
    index = np.searchsorted(true_indices, value, side='right')
    if index < len(true_indices):
        Time_start_realigned.append(true_indices[index])
Time_start_realigned = [float(val) for val in Time_start_realigned]
print("Time_start_realigned", Time_start_realigned)
from itertools import chain


#Visualize baseline
# plt.figure(figsize=(10, 6))
# plt.hist(smooth_signal, bins=100, color='skyblue', edgecolor='black', alpha=0.7, label='Smooth Signal Histogram')
# plt.axvline(baseline, color='red', linestyle='--', linewidth=2, label=f'Baseline (max bin center): {baseline:.2f}')
# plt.xlabel('Smooth Signal Value')
# plt.ylabel('Frequency')
# plt.title('Histogram of Smooth Signal with Baseline')
# plt.legend()
# plt.show()
color_map = {
    0: 'blue',
    1: 'red',
    2: 'green',
    3: 'yellow',
    4: 'purple',
    5: 'orange',
    6: 'pink'
}

# Replace each value in protocol_id with its corresponding color
color_names = [color_map[value] for value in protocol_id]

fig, ax = plt.subplots()
for x in range(len(Time_start_realigned)):
    ax.axvline(x=Time_start_realigned[x], color='yellow', linestyle='--', alpha=0.7)
    ax.axvspan(Time_start_realigned[x], Time_start_realigned[x]+(time_duration[x]), color=color_names[x], alpha=0.5)
Psignal_time = np.arange(0, len(Psignal)/1000, 0.001)
ax.plot(F_time_stamp_updated, F[25])
#ax.plot(Psignal_time, Psignal)
ax.axhline(y = (baseline + threshold), color='r', linestyle='--', linewidth=2)
plt.show()