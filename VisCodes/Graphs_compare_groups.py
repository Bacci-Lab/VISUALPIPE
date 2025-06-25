import numpy as np
import h5py
from scipy import stats
import matplotlib.pyplot as plt
import os
from scipy.stats import mannwhitneyu, wilcoxon, linregress
import glob

#-----------------------INPUTS-----------------------#
## Organization of the data: folder for the protocol (e.g surround mod) > group (e.g KO and WT) > # mouse (e.g 110, 108) > session (e.g 2023-10-01) > .npz file with the validity of the neurons and .npy file with the trials
#The folder containing all subfolders
data_path = r""
save_path = data_path
# group names have to be the name of the subfolders
groups = ['WT', 'KO']
# mice ids per group (sub-subfolders)
WT_mice = ['110', '108']
KO_mice = ['109', '112']
#Will be included in all names of saved figures
fig_name = 'drifting-grating-1.0'
# Write the protocols you want to plot 
protocols = ['drifting-grating-1.0']
# List of protocol(s) used to select reponsive neurons. If contains several protocols, neurons will be selected if they are responsive to at least one of the protocols in the list.
protocol_validity = ['drifting-grating-0.2', 'drifting-grating-0.6', 'drifting-grating-1.0'] 
#Frame rate
frame_rate = 30
#----------------------------------------------------#





z_score_periods = ['pre_trial_averaged_zscores', 'trial_averaged_zscores', 'post_trial_averaged_zscores']
x_labels = ['Pre-stim', 'Stim', 'Post-stim']



def process_group(group_name, mice_list):
    magnitude = {protocol: [] for protocol in protocols} #Will contain the magnitude of the response to each protocol for each neuron
    avg_data = {protocol: [] for protocol in protocols} 
    all_neurons = 0
    stim_mean = {protocol: [] for protocol in protocols} #Will contain the mean response to the stimulus for each protocol, per session
    proportion_list = [] #Will contain the proportion of responding neurons per session
    for mouse in mice_list:
        mouse_dir = os.path.join(data_path, group_name, mouse)

        # List session subdirectories
        session_dirs = [d for d in os.listdir(mouse_dir) if os.path.isdir(os.path.join(mouse_dir, d))]
        for session in session_dirs:
            print(f"Session name: {session}")
            session_path = os.path.join(mouse_dir, session)

            # Load .npz file
            npz_files = glob.glob(os.path.join(session_path, "*.npz"))
            if len(npz_files) == 1:
                validity = np.load(npz_files[0], allow_pickle=True)
            else:
                raise FileNotFoundError(f"Expected exactly one .npz file in {session_path}, found {len(npz_files)}")      # Load .npy file
            npy_files = glob.glob(os.path.join(session_path, "*.npy"))
            if len(npy_files) == 1:
                trials = np.load(npy_files[0], allow_pickle=True).item()
            else:
                raise FileNotFoundError(f"Expected exactly one .npy file in {session_path}, found {len(npy_files)}")
            keys = list(validity.files)  # Get all keys from the validity file
            valid_data = {}  # Dictionary to store responsive neurons for each protocol
            for protocol in protocol_validity:
                if protocol in keys:
                    valid_data[protocol] = validity[protocol]
                else:
                    print(f"{protocol} does not exist in validity file.")
            valid_neuron_lists = [np.where(data[:, 0] == 1,)[0] for data in valid_data.values()] # change to -1 if you want negative responsive neurons
            valid_neurons = np.unique(np.concatenate(valid_neuron_lists))  # Get unique indices of valid neurons
            neurons = len(valid_neurons)
            print(neurons)
            proportion = 100 * len(valid_neurons) / trials['trial_averaged_zscores'][0].shape[0]
            proportion_list.append(proportion)
            print(f"Proportion responding neurons: {proportion}")
            all_neurons += neurons
            all_protocols = validity.files # List of all stimuli ("protocols") types in that session
            avg_session = []
            for i, protocol in enumerate(protocols):
                idx = all_protocols.index(protocol)

                # Get z-scores from responsive-neurons for that protocol from pre, stim and post periods and concatenate along time
                zscores_periods = [trials[period][list(trials[period].keys())[idx]][valid_neurons, :] for period in z_score_periods]
                zscores_concat = np.concatenate(zscores_periods, axis=1)
                avg_session = np.mean(zscores_concat, axis=0) # average over neurons in that session
                avg_data[protocol].append(avg_session)
                # Extract trial-averaged zscores during the stim period and give the average response of all neurons in that session
                trial_zscores = trials['trial_averaged_zscores'][list(trials['trial_averaged_zscores'].keys())[idx]][valid_neurons, :]
                mean_stim_response = np.mean(trial_zscores[:,int(frame_rate*0.5):], axis=1)  # average per neuron over time, exclude the first 0.5 seconds because of GCaMP's slow kinetics
                session_stim_mean = np.mean(mean_stim_response)  # average over neurons in that session
                stim_mean[protocol].append(session_stim_mean)
                #Store the average response of each neuron to that protocol
                for n in range(neurons):
                    zneuron = trial_zscores[n, int(frame_rate*0.5):]
                    magnitude[protocol].append(np.mean(zneuron))
    # Compute CMI for each neuron
    if len(protocols)>1:
        cmi = [
            float((magnitude[protocols[1]][n] - magnitude[protocols[0]][n]) /
                (magnitude[protocols[1]][n] + magnitude[protocols[0]][n]))
            for n in range(all_neurons)
        ]
    else:
        cmi = []
    print(f"Number of {group_name} neurons: {all_neurons}")
    # compute the surround suppression 
    suppression = []
    accepted_protocols = ['center', 'center-surround-iso', 'center-surround-cross']
    if protocols[0] in accepted_protocols and protocols[1] in accepted_protocols:
        suppression = [
                1 - float(magnitude[protocols[1]][n]) / float(magnitude[protocols[0]][n])
                if magnitude[protocols[0]][n] != 0 else np.nan
                for n in range(all_neurons)
        ]
    else:
        suppression = []

    # Concatenate all neuron arrays into one array per protocol
    for protocol in protocols:
        avg_data[protocol] = np.stack(avg_data[protocol], axis=0)

    # Compute average and SEM across neurons
    avg = {protocol: np.mean(avg_data[protocol], axis=0) for protocol in protocols} 
    sem = {protocol: stats.sem(avg_data[protocol], axis=0) for protocol in protocols}
    print(f"List of % of responsive neurons per session for {group_name}: {proportion_list}")

    return suppression, magnitude, stim_mean, all_neurons, avg, sem, cmi



# Process WT
suppression_wt, magnitude_wt, stim_wt, wt_neurons, avg_wt, sem_wt, cmi_wt = process_group('WT', WT_mice)
# Process KO
suppression_ko, magnitude_ko, stim_ko, ko_neurons, avg_ko, sem_ko, cmi_ko = process_group('KO', KO_mice)


#--------------For each neuron, plot in x the magnitude of the response to protocol 1, in y for protocol 2----------------#
if len(protocols) == 2:
    for group in groups:
        x_values = []
        y_values = []
        if group == 'WT':
            magnitude = magnitude_wt
        else:
            magnitude = magnitude_ko

        x_values = magnitude[protocols[0]]
        y_values = magnitude[protocols[1]]

        # Create the plot
        plt.plot(x_values, y_values, 'o', color='cyan' if group == 'WT' else 'orange', alpha=0.5, label=f'{group} neurons')
        slope, intercept, r_value, p_value, std_err = linregress(x_values, y_values)
        # Plot regression line
        x_fit = np.linspace(min(x_values), max(x_values), 100)
        y_fit = [slope * xi + intercept for xi in x_fit]
        plt.plot(x_fit, y_fit,
                color='blue' if group == 'WT' else 'red',
                label=f'{group} fit: y = {slope:.2f}x + {intercept:.2f}, p = {p_value:.2g}, r**2 = {(r_value)**2:.3f}')
        # Add labels
        plt.xlabel(f"Magnitude of response to {protocols[0]}")
        plt.ylabel(f"Magnitude of response to {protocols[1]}")
        plt.title("Response magnitudes (z-score) for each neuron")
    plt.legend()
    plt.savefig(os.path.join(save_path, f"{fig_name}_magnitude_response.jpeg"), dpi=300)
    plt.show()

#--------------To plot the average response during the stim period per session----------------#

colors = {'WT': 'skyblue', 'KO': 'salmon'}

fig, ax = plt.subplots(figsize=(8, 6))

x_ticks = []
x_labels = []
width = 0.6
offset = 0.3  # spacing between WT and KO

for i, protocol in enumerate(protocols):
    wt_data = stim_wt[protocol]
    ko_data = stim_ko[protocol]

    # Bar height: mean
    wt_median = np.mean(wt_data)
    ko_median = np.mean(ko_data)
    # Mann–Whitney U test
    stat, p = mannwhitneyu(wt_data, ko_data, alternative='two-sided')

    # Plot WT
    x_wt = i * 2 - offset
    ax.bar(x_wt, wt_median, width=width, color=colors['WT'], edgecolor='black', label='WT' if i == 0 else "")
    ax.scatter([x_wt] * len(wt_data), wt_data, color='black', zorder=10)

    # Plot KO
    x_ko = i * 2 + offset
    ax.bar(x_ko, ko_median, width=width, color=colors['KO'], edgecolor='black', label='KO' if i == 0 else "")
    ax.scatter([x_ko] * len(ko_data), ko_data, color='black', zorder=10)

    # Annotate p-value
    y_max = max(np.max(wt_data), np.max(ko_data)) + 0.3
    ax.plot([x_wt, x_ko], [y_max, y_max], color='black', linewidth=1.5)
    ax.text((x_wt + x_ko) / 2, y_max + 0.05, f"M-W p = {p:.3g}", ha='center', va='bottom', fontsize=11)

    # Update ticks
    x_ticks.extend([x_wt, x_ko])
    x_labels.extend([f'{protocol}\nWT', f'{protocol}\nKO'])

# Labeling
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels, rotation=0)
ax.set_ylabel('Mean stimulus response (z-score)')
ax.set_title('Session-averaged responses during stimulus')
ax.legend()
plt.tight_layout()

# Optional: save figure
plt.savefig(os.path.join(save_path, f"{fig_name}_barplot_stim_response.jpeg"), dpi=300)
plt.show()




#------------------Plot the average z-scores for each protocol across all sessions for WT and KO groups-------------------------#

# Get minimum length among all protocols
min_len = min(len(avg_wt[protocol]) for protocol in protocols)

# Generate time vector accordingly
time = np.linspace(0, min_len, min_len) / frame_rate - 1  # time in seconds

if len(protocols) == 1:
        colors = {'WT': {protocols[0]: 'blue'},
            'KO': {protocols[0]: 'red'}}
elif len(protocols) == 2:
    colors = {'WT': {protocols[0]: 'blue', protocols[1]: 'cyan'},
            'KO': {protocols[0]: 'red', protocols[1]: 'orange'}}
plt.figure(figsize=(10, 6))

for protocol in protocols:
    plt.plot(time, avg_wt[protocol][:min_len], color=colors['WT'][protocol], label=f"WT {protocol}, {wt_neurons} neurons")
    plt.fill_between(time,
                     avg_wt[protocol][:min_len] - sem_wt[protocol][:min_len],
                     avg_wt[protocol][:min_len] + sem_wt[protocol][:min_len],
                     color=colors['WT'][protocol], alpha=0.3)

    plt.plot(time, avg_ko[protocol][:min_len], color=colors['KO'][protocol], label=f"KO {protocol}, {ko_neurons} neurons")
    plt.fill_between(time,
                     avg_ko[protocol][:min_len] - sem_ko[protocol][:min_len],
                     avg_ko[protocol][:min_len] + sem_ko[protocol][:min_len],
                     color=colors['KO'][protocol], alpha=0.3)

plt.xticks(np.arange(-1, time[-1] + 1, 1))
plt.xlabel("Time (s)")
plt.ylabel(f"Average Z-score for neurons responsive to {protocol_validity}")
plt.title("Mean z-score ± SEM by Group and Protocol")
plt.legend()
plt.savefig(os.path.join(save_path, f"average_{fig_name}_wt_vs_ko.jpeg"), dpi=300)
plt.show()




###----------------Boxplot for CMI-----------###

# Assume cmi_wt and cmi_ko are lists or arrays of CMIs for WT and KO respectively
if len(protocols) == 2:
    cmi_wt_array = np.array(cmi_wt)
    cmi_ko_array = np.array(cmi_ko)

    # Median and Wilcoxon vs 0 for each group
    median_wt = np.median(cmi_wt_array)
    median_ko = np.median(cmi_ko_array)
    p_value_wt = wilcoxon(cmi_wt_array, alternative='two-sided')[1]
    p_value_ko = wilcoxon(cmi_ko_array, alternative='two-sided')[1]

    # Define bin edges between -1.5 and +1.5, e.g. 15 bins inside
    inside_bins = np.linspace(-1.5, 1.5, 16)  # 15 bins of width 0.2
    bins = np.concatenate(([-np.inf], inside_bins, [np.inf]))

    # Digitize data: assign each CMI value to a bin index
    bin_indices_wt = np.digitize(cmi_wt_array, bins)
    bin_indices_ko = np.digitize(cmi_ko_array, bins)

    # Prepare counts and labels
    counts_wt = []
    counts_ko = []
    labels = []

    labels.append('< -1.5')
    counts_wt.append(100 * np.sum(bin_indices_wt == 1) / wt_neurons)
    counts_ko.append(100 * np.sum(bin_indices_ko == 1) / ko_neurons)

    for i in range(2, len(bins)-1):
        bin_start = bins[i-1]
        bin_end = bins[i]
        labels.append(f'{bin_start:.2f} to {bin_end:.2f}')
        counts_wt.append(100 * np.sum(bin_indices_wt == i) / wt_neurons)
        counts_ko.append(100 * np.sum(bin_indices_ko == i) / ko_neurons)

    labels.append('> 1.5')
    counts_wt.append(100 * np.sum(bin_indices_wt == len(bin_indices_wt)) / wt_neurons)
    counts_ko.append(100 * np.sum(bin_indices_ko == len(bin_indices_wt)) / ko_neurons)

    # Bar plot parameters
    x = np.arange(len(labels))  # label locations
    width = 0.4  # width of the bars

    plt.figure(figsize=(12,6))
    plt.bar(x - width/2, counts_wt, width, label='WT', color='blue', edgecolor='black')
    plt.bar(x + width/2, counts_ko, width, label='KO', color='red', edgecolor='black')

    from scipy.stats import mannwhitneyu

    # Perform Mann–Whitney U test between WT and KO
    stat_mwu, p_value_mwu = mannwhitneyu(cmi_wt_array, cmi_ko_array, alternative='two-sided')

    # Add p-value annotation to the plot
    textstr = (
        f'WT median = {median_wt:.2f}, p (vs 0) = {p_value_wt:.3g}\n'
        f'KO median = {median_ko:.2f}, p (vs 0) = {p_value_ko:.3g}\n'
        f'WT vs KO (Mann–Whitney) p = {p_value_mwu:.3g}'
    )

    # Position textbox on plot
    plt.gca().text(0.95, 0.95, textstr,
                transform=plt.gca().transAxes,
                fontsize=11, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85))
    plt.legend(loc='upper right')


    # Perform Mann–Whitney U test between WT and KO
    stat_mwu, p_value_mwu = mannwhitneyu(cmi_wt_array, cmi_ko_array, alternative='two-sided')


    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel(f'% of neurons')
    plt.title('Distribution of CMI values by group')
    plt.legend()



    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"barplot_{fig_name}_cmi.jpeg"), dpi=300)
    plt.show()


###----------------Boxplot for surround suppression-----------###

# Assume cmi_wt and cmi_ko are lists or arrays of CMIs for WT and KO respectively
if len(protocols)==2:
    suppr_wt_array = np.array(suppression_wt)
    suppr_ko_array = np.array(suppression_ko)

    # Median and Wilcoxon vs 0 for each group
    median_wt = np.median(suppr_wt_array)
    median_ko = np.median(suppr_ko_array)
    p_value_wt = wilcoxon(suppr_wt_array, alternative='two-sided')[1]
    p_value_ko = wilcoxon(suppr_ko_array, alternative='two-sided')[1]

    # Define bin edges between -2 and 2, e.g. 20 bins inside
    inside_bins = np.linspace(-2, 2, 24)  # 20 bins of width 0.2
    bins = np.concatenate(([-np.inf], inside_bins, [np.inf]))

    # Digitize data: assign each surround suppression value to a bin index
    bin_indices_wt = np.digitize(suppr_wt_array, bins)
    bin_indices_ko = np.digitize(suppr_ko_array, bins)

    # Prepare counts and labels
    counts_wt = []
    counts_ko = []
    labels = []

    labels.append('< -2')
    counts_wt.append(100 * np.sum(bin_indices_wt == 1) / wt_neurons)
    counts_ko.append(100 * np.sum(bin_indices_ko == 1) / ko_neurons)

    for i in range(2, len(bins)-1):
        bin_start = bins[i-1]
        bin_end = bins[i]
        labels.append(f'{bin_start:.2f} to {bin_end:.2f}')
        counts_wt.append(100 * np.sum(bin_indices_wt == i) / wt_neurons)
        counts_ko.append(100 * np.sum(bin_indices_ko == i) / ko_neurons)

    labels.append('> 2')
    counts_wt.append(100 * np.sum(bin_indices_wt == len(bin_indices_wt)) / wt_neurons)
    counts_ko.append(100 * np.sum(bin_indices_ko == len(bin_indices_wt)) / ko_neurons)

    # Bar plot parameters
    x = np.arange(len(labels))  # label locations
    width = 0.4  # width of the bars

    plt.figure(figsize=(12,6))
    plt.bar(x - width/2, counts_wt, width, label='WT', color='blue', edgecolor='black')
    plt.bar(x + width/2, counts_ko, width, label='KO', color='red', edgecolor='black')

    from scipy.stats import mannwhitneyu

    # Perform Mann–Whitney U test between WT and KO
    stat_mwu, p_value_mwu = mannwhitneyu(suppr_wt_array, suppr_ko_array, alternative='two-sided')

    # Add p-value annotation to the plot
    textstr = (
        f'WT median = {median_wt:.2f}, p (vs 0) = {p_value_wt:.3g}\n'
        f'KO median = {median_ko:.2f}, p (vs 0) = {p_value_ko:.3g}\n'
        f'WT vs KO (Mann–Whitney) p = {p_value_mwu:.3g}'
    )

    # Position textbox on plot
    plt.gca().text(0.95, 0.95, textstr,
                transform=plt.gca().transAxes,
                fontsize=11, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85))
    plt.legend(loc='upper right')



    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel(f'% of neurons')
    plt.title('Distribution of surround suppression values by group')
    plt.legend()



plt.tight_layout()
plt.savefig(os.path.join(save_path, f"barplot_{fig_name}_surround-sup.jpeg"), dpi=300)
plt.show()


