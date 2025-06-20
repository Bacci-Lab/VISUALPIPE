import numpy as np
import h5py
from scipy import stats
import matplotlib.pyplot as plt
import os
from scipy.stats import wilcoxon

#--------------------INPUTS------------------#
data_path = r"Y:\raw-imaging\Nathan\PYR\110 male\Visual\03_03_2025\TSeries-03032025-006\2025_03_03_17-14-33_output_4"
validity_file = '2025_03_03_17-14-33_4_protocol_validity_2.npz'
trials_file = '2025_03_03_17-14-33_4_trials.npy'
save_path = data_path
file_name = 'test'
#This is the protocol you choose to select responsive neurons
protocol_validity = 'Natural-Images-4-repeats'
# Write the protocols you want to plot
protocols = ['Natural-Images-4-repeats']
#-----------------------------------------------------------------------------#





# Load the npz file
validity = np.load(os.path.join(data_path, validity_file), allow_pickle=True)
all_protocols = validity.files

# Load NPY file containing averaged z_scores before during and after stim presentation
trials = np.load(os.path.join(data_path,trials_file), allow_pickle=True).item()



print(validity.files)

valid_data = validity[protocol_validity]
valid_neurons = np.where(valid_data[:, 0] == 1)[0]

proportion_valid = len(valid_neurons)/len(valid_data[:,0])

print(f"Number of neurons responsive in {protocol_validity}: {len(valid_neurons)}")
print(f"Proportion of neurons responsive in {protocol_validity}: {proportion_valid:.2%}")


# ----------------------------------Plot average z_scores during specific protocols for all neurons of that session-----------------------------------------#
file1 = f"average_{file_name}_allneurons.jpeg"


z_score_periods = ['pre_trial_averaged_zscores', 'trial_averaged_zscores', 'post_trial_averaged_zscores']
x_labels = ['Pre-stim', 'Stim', 'Post-stim']
total_time = trials['trial_averaged_zscores'][0].shape[1] + trials['pre_trial_averaged_zscores'][0].shape[1] + trials['post_trial_averaged_zscores'][0].shape[1]
print(total_time)
# print(f" Total time:{total_time}")


y_axis = {}  # dict to store mean zscores for each protocol
sem = {}     # dict to store SEMs
frame_rate = 30
# ------------------------ Get averaged zscore for each part of the response (pre-stim-post) and assemble them-----------------------#
for protocol in protocols:
    idx = all_protocols.index(protocol)

    # Get z-scores from responsive-neurons for that protocol from pre, stim and post periods and concatenate along time

    print(f"\nProtocol: {protocol}, Index: {idx}")

    avg_trace = []
    sem_trace = []

    for period in z_score_periods:
        print(f"Period: {period}")
        zscores = trials[period][list(trials[period].keys())[idx]]  # shape: (neurons, time)
        print(f"Zscores shape:{zscores.shape}")
        avg_zscore = np.mean(zscores, axis=0)
        sem_period = stats.sem(zscores, axis=0)

        avg_trace.append(avg_zscore)
        sem_trace.append(sem_period)

    # Concatenate all 3 periods along time axis
    y_axis[protocol] = np.concatenate(avg_trace)
    sem[protocol] = np.concatenate(sem_trace)
# print(y_axis)
# print(sem)

# ---- Plot the averaged z-score for all protocols ------#
# time = np.linspace(0, total_time, total_time)
time = np.linspace(0, len(y_axis[protocols[0]]), len(y_axis[protocols[0]]))
time = time/frame_rate -1

for protocol in protocols:
    plt.plot(time, y_axis[protocol], label=protocol)
    plt.fill_between(time,
                     y_axis[protocol] - sem[protocol],
                     y_axis[protocol] + sem[protocol],
                     alpha=0.3)
plt.xticks(np.arange(-1, time[-1] + 1, 1))
plt.xlabel("Time (s)")
plt.ylabel("Average z-score for all neurons")
plt.title("Mean z-score ± SEM")
plt.legend()
plt.savefig(os.path.join(save_path, file1), dpi=300)

plt.show()






# ----------------------------------f"Plot average z_scores during specific protocols for all neurons responsive in the {protocol_validity}-----------------------------------------#
# save_path = r"P:\raw-imaging\Nathan\PYR\110 male\Visual\25_02_2025\TSeries-02252025-002\2025_02_25_14-12-40_output_4"
file2 = f"average_{file_name}_responsive.jpeg"



y_axis = {}  # dict to store mean zscores for each protocol
sem = {}     # dict to store SEMs
frame_rate = 30
magnitude = {protocol: [] for protocol in protocols}


# ------------------------ Get averaged zscore for each part of the response (pre-stim-post) and assemble them-----------------------#
for protocol in protocols:
    idx = all_protocols.index(protocol)
    print(f"\nProtocol: {protocol}, Index: {idx}")

    avg_trace = []
    sem_trace = []

    for period in z_score_periods:
        print(f"Period: {period}")
        zscores = trials[period][list(trials[period].keys())[idx]][valid_neurons, :] 
        neurons = zscores.shape[0]
        avg_zscore = np.mean(zscores, axis=0)
        sem_period = stats.sem(zscores, axis=0)

        avg_trace.append(avg_zscore)
        sem_trace.append(sem_period)
        #Compute the magnitude of the response for each neuron as the mean z-score during the stimulus period
        if period == 'trial_averaged_zscores':
            for n in range(0,neurons):
                zneuron = zscores[n, int(frame_rate*0.5):] # exclude the first 0.5 seconds because of GCaMP's slow kinetics
                magnitude[protocol].append(np.mean(zneuron))
    # Concatenate all 3 periods along time axis
    y_axis[protocol] = np.concatenate(avg_trace)
    sem[protocol] = np.concatenate(sem_trace)
# print(y_axis)
# print(sem)

# ---- Plot the averaged z-score for all protocols ------#
# time = np.linspace(0, total_time, total_time)
time = np.linspace(0, len(y_axis[protocols[0]]), len(y_axis[protocols[0]]))
time = time/frame_rate -1

for protocol in protocols:
    plt.plot(time, y_axis[protocol], label=protocol)
    plt.fill_between(time,
                     y_axis[protocol] - sem[protocol],
                     y_axis[protocol] + sem[protocol],
                     alpha=0.3)
plt.xticks(np.arange(-1, time[-1] + 1, 1))
plt.xlabel("Time (s)")
plt.ylabel(f"Average Z-score for neurons responsive to {protocol_validity}")
plt.title("Mean z-score ± SEM")
# Add text box with median and p-value
textstr = f'Nb center-responsive = {len(valid_neurons)}\n% of center-responsive= {100*proportion_valid:.1f}'

# Position text somewhere visible (top right corner)
plt.gca().text(0.95, 0.95, textstr, transform=plt.gca().transAxes,
               fontsize=9, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
plt.legend()
plt.savefig(os.path.join(save_path, file2), dpi=300)

plt.show()

# -------- Boxplot of the magnitudes of responses for each neuron in the two protocols -------#

# Prepare data
data = [magnitude[protocol] for protocol in protocols]
print(data)


plt.figure(figsize=(6,6))
plt.boxplot(data, labels=[protocol for protocol in protocols], patch_artist=True)

plt.ylabel('Response magnitude (peak z-score)')
plt.title('Distribution of neuron response magnitudes by protocol')
plt.grid(axis='y')

plt.tight_layout()
plt.show()




# -------- Calculate the CMI for this session -------#
cmi = []
protocol1 = protocols[0]
protocol2 = protocols[1]
cmi = [
    float((magnitude[protocol2][n] - magnitude[protocol1][n]) / (magnitude[protocol2][n] + magnitude[protocol1][n]))
    for n in range(neurons)
]
print(f"CMI of center_responsive neurons: {cmi}")
print(f"Mean CMI:{np.mean(cmi)}")
median_cmi = np.median(cmi)
print(f"Median CMI:{median_cmi}")

# ------ Plot the distribution of CMIs in this session ----#

file3 = f"barplot_{file_name}_responsive"
# CMI list
cmi_array = np.array(cmi)

# Define bin edges between -1.5 and +1.5, e.g. 15 bins inside
inside_bins = np.linspace(-1.5, 1.5, 16)  # 15 bins of width 0.2

# 2 extra bins for values +/-1.5
bins = np.concatenate(([-np.inf], inside_bins, [np.inf]))

# Digitize data: assign each CMI value to a bin index
bin_indices = np.digitize(cmi_array, bins)

# Count number of values per bin
counts = []
labels = []

# For bins, index 1 is <-1.5, last index is >1.5
labels.append('< -1.5')
counts.append(np.sum(bin_indices == 1))

for i in range(2, len(bins)-1):
    bin_start = bins[i-1]
    bin_end = bins[i]
    labels.append(f'{bin_start:.2f} to {bin_end:.2f}')
    counts.append(np.sum(bin_indices == i))

labels.append('> 1.5')
counts.append(np.sum(bin_indices == len(bins)))

# Plot bar plot
plt.figure(figsize=(10,5))
plt.bar(labels, counts, color='steelblue', edgecolor='black')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Number of neurons')
plt.title('Distribution of CMI values')
# Find which bin the median falls into
median_bin_idx = np.digitize(median_cmi, bins) - 1  # minus 1 to convert to zero-based index for labels/bars
# Median line x-position = center of the median bin's bar on the x-axis
median_x = median_bin_idx
# Perform two-sided Wilcoxon signed-rank test against zero median
stat, p_value = wilcoxon(cmi_array, zero_method='wilcox', alternative='two-sided', correction=False)
# Add vertical dashed line at median position
plt.axvline(median_x, color='red', linestyle='--', linewidth=2, label=f'Median = {median_cmi:.2f}, p-value ={p_value}')
# Add text box with median and p-value
textstr = f'Median = {median_cmi:.2f}\nWilcoxon p = {p_value:.3g}'

# Position text somewhere visible (top right corner)
plt.gca().text(0.95, 0.95, textstr, transform=plt.gca().transAxes,
               fontsize=12, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(save_path, file3), dpi=300)
plt.show()







# --------------------- To adapt -----------------------#
def compute_cmi():
    id_srm_cross = self.visual_stim.protocol_df[self.visual_stim.protocol_df["name"] == "center-surround-cross"].index[
        0]
    id_srm_iso = self.visual_stim.protocol_df[self.visual_stim.protocol_df["name"] == "center-surround-iso"].index[0]
    cmi = []

    for i in range(len(self.ca_img._list_ROIs_idx)):
        cross = np.mean(self.trial_fluorescence[id_srm_cross][i])
        iso = np.mean(self.trial_fluorescence[id_srm_iso][i])
        cmi_roi = (cross - iso) / (cross + iso)
        cmi.append(cmi_roi)

    return cmi

def compute_trial(self, attr='fluorescence'):

    trial_fluorescence, pre_trial_fluorescence, post_trial_fluorescence = {}, {}, {}

    for i in self.visual_stim.protocol_ids:

        if self.visual_stim.stim_cat[i]:

            trial_fluorescence_i, baselines_i, post_trial_fluorescence_i = [], [], []

            for roi_idx in range(len(self.ca_img._list_ROIs_idx)):
                roi_baselines, roi_trial_fluorescence, roi_post_trial_fluorescence = self.get_trials_trace(roi_idx,
                                                                                                           i, attr)

                baselines_i.append(roi_baselines)
                trial_fluorescence_i.append(roi_trial_fluorescence)
                post_trial_fluorescence_i.append(roi_post_trial_fluorescence)

            trial_fluorescence.update({i: np.array(trial_fluorescence_i)})
            pre_trial_fluorescence.update({i: np.array(baselines_i)})
            post_trial_fluorescence.update({i: np.array(post_trial_fluorescence_i)})

    return trial_fluorescence, pre_trial_fluorescence, post_trial_fluorescence


def compute_trial_averaged_zscores(self, trial_fluorescence, post_trial_fluorescence, average_baselines):

    trial_averaged_zscores, pre_trial_averaged_zscores, post_trial_averaged_zscores = {}, {}, {}

    for i in self.trial_fluorescence.keys() :
        average_baseline = average_baselines[i]
        average_trial = np.mean(trial_fluorescence[i], axis=1)
        average_post_trial = np.mean(post_trial_fluorescence[i], axis=1)

        trial_averaged_zscores.update({i : self.zscores(average_baseline, average_trial)})
        pre_trial_averaged_zscores.update({i : self.zscores(average_baseline, average_baseline)})
        post_trial_averaged_zscores.update({i : self.zscores(average_baseline, average_post_trial)})

    return trial_averaged_zscores, pre_trial_averaged_zscores, post_trial_averaged_zscores
