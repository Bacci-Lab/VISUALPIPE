import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
from scipy.stats import wilcoxon

#--------------------INPUTS------------------#
data_path = r"P:\raw-imaging\Nathan\PYR\112 female\Visual\14_03_2025\TSeries-03142025-007\2025_03_14_16-02-13_output_3"
validity_file = '2025_03_14_16-02-13_3_protocol_validity_2.npz'
trials_file = '2025_03_14_16-02-13_3_trials.npy'
save_path = data_path
file_name = 'CenterVsIso'
#This is the list of stimuli you want to use to select the responsive neurons. A responsive neurons is responsive in at least one of these stimuli
protocol_validity = ['center']  
# Write the protocols you want to plot
protocols = ['center-surround-iso', 'center-surround-cross']
# Decide if you want to plot the dFoF0 baseline substraced or the z-scores
attr = 'dFoF0-baseline'  # 'dFoF0-baseline' or 'z_scores'
frame_rate = 30  # Frame rate of the imaging session in Hz
#-----------------------------------------------------------------------------#

if attr == 'dFoF0-baseline':
    z_score_periods = ['norm_averaged_baselines', 'norm_trial_averaged_ca_trace', 'norm_post_trial_averaged_ca_trace']
elif attr == 'z_scores':
    z_score_periods = ['pre_trial_averaged_zscores', 'trial_averaged_zscores', 'post_trial_averaged_zscores']


# Load the npz file
validity = np.load(os.path.join(data_path, validity_file), allow_pickle=True)
print(validity.files)
all_protocols = validity.files

# Load NPY file containing averaged z_scores before during and after stim presentation
trials = np.load(os.path.join(data_path,trials_file), allow_pickle=True).item()

# Get the duration of the pre-stimulus period
dt_prestim = trials['dt_pre_stim']

# Select the responsive neurons based on the validity file
keys = list(validity.files)  # Get all keys from the validity file
valid_data = {}  # Dictionary to store responsive neurons for each protocol
for protocol in protocol_validity:
    if protocol in keys:
        valid_data[protocol] = validity[protocol]
    else:
        print(f"{protocol} does not exist in validity file.")
valid_neuron_lists = [np.where(data[:, 0] == 1,)[0] for data in valid_data.values()] # change to -1 if you want negative responsive neurons
valid_neurons = np.unique(np.concatenate(valid_neuron_lists))  # Get unique indices of valid neurons




if attr == 'dFoF0-baseline':
    # Calculate the proportion of valid neurons
    print(np.shape(trials['norm_trial_averaged_ca_trace'][0]))
    proportion_valid = 100 * len(valid_neurons) / (trials['norm_trial_averaged_ca_trace'][0].shape[0])
elif attr == 'z_scores':
    # Calculate the proportion of valid neurons
    proportion_valid = 100 * len(valid_neurons) / (trials['trial_averaged_zscores'][0].shape[0])


print(f"Number of neurons responsive in {protocol_validity}: {len(valid_neurons)}")
print(f"Proportion of neurons responsive in {protocol_validity}: {proportion_valid:.2f}%")


#-------------------------Functions-----------------------#

def graph_averages(file_name, attr, save_path, trials, protocols, z_score_periods, frame_rate, valid_neurons, all_protocols, dt_prestim, get_valid):
    """
    Function to plot the average z-scores for all neurons if get_valid is False or only responsive neurons if get_valid is True.
    """
    # Create a figure and initialize axis
    plt.figure(figsize=(10, 6))
    if not get_valid:
        file1 = f"average_{file_name}_{attr}_allneurons.jpeg"
    elif get_valid:
        file1 = f"average_{file_name}_{attr}_responsive.jpeg"
    y_axis = {}  # dict to store mean zscores for each protocol
    sem = {}     # dict to store SEMs
    magnitude = {protocol:[] for protocol in protocols}
    neurons_count = {protocol: None for protocol in protocols}
    # Loop through each protocol and plot the average z-scores
    for protocol in protocols:
        idx = all_protocols.index(protocol)
        print(f"\nProtocol: {protocol}, Index: {idx}")
        avg_trace = []
        sem_trace = []
        
        for period in z_score_periods:
            if not get_valid:
                zscores = trials[period][list(trials[period].keys())[idx]] # shape: (neurons, time)
            elif get_valid:
                zscores = trials[period][list(trials[period].keys())[idx]][valid_neurons, :]
            neurons = zscores.shape[0]
            avg_zscore = np.mean(zscores, axis=0)
            sem_period = stats.sem(zscores, axis=0)
            avg_trace.append(avg_zscore)
            sem_trace.append(sem_period)

            #Compute the magnitude of the response for each neuron as the mean z-score during the stimulus period
            if period == 'trial_averaged_zscores' or period == 'norm_trial_averaged_ca_trace':
                for n in range(0,neurons):
                    zneuron = zscores[n, int(frame_rate*0.5):] # exclude the first 0.5 seconds because of GCaMP's slow kinetics
                    magnitude[protocol].append(np.mean(zneuron))
        neurons_count[protocol] = neurons
        # Concatenate all 3 periods along time axis
        y_axis[protocol] = np.concatenate(avg_trace)
        sem[protocol] = np.concatenate(sem_trace)

    # ---- Plot the averaged z-score for all protocols ------#
    time = np.linspace(0, len(y_axis[protocols[0]]), len(y_axis[protocols[0]]))
    time = time/frame_rate - dt_prestim # Shift the time by the duration of the baseline so the stimulus period starts at 0

    for protocol in protocols:
        plt.plot(time, y_axis[protocol], label=protocol)
        plt.fill_between(time,
                        y_axis[protocol] - sem[protocol],
                        y_axis[protocol] + sem[protocol],
                        alpha=0.3)
    plt.xticks(np.arange(-1, time[-1] + 1, 1))
    plt.xlabel("Time (s)")
    plt.ylabel(f"Average {attr}")
    if get_valid:
        plt.title(f"Mean {attr} ± SEM (Responsive Neurons)")
    elif not get_valid:
        plt.title(f"Mean {attr} ± SEM (All Neurons)")
    plt.legend()
    plt.savefig(os.path.join(save_path, file1), dpi=300)

    plt.show()

    return magnitude, neurons_count

def cmi(magnitude, neurons_count, protocols):
    """ 
    Function to calculate the contextual modulation index (CMI)
    """

    accepted_protocols = ['center', 'center-surround-iso', 'center-surround-cross', 'surround-iso_ctrl', 'surround-cross_ctrl']
    if len(protocols) > 2:
        print("You can only compute CMI for two protocols at a time. Please select two protocols from the accepted list.")
        print(f"Accepted protocols: {accepted_protocols}")
        exit()
    for protocol in protocols:
        if protocol not in accepted_protocols:
            print("You can only compute CMI for stimuli in the accepted list.")
            print(f"Accepted protocols: {accepted_protocols}")
            exit()
    cmi = []
    protocol1 = protocols[0]
    protocol2 = protocols[1]
    cmi = [
        float((magnitude[protocol2][n] - magnitude[protocol1][n]) / (magnitude[protocol2][n] + magnitude[protocol1][n]))
        for n in range(neurons_count[protocol1])
    ]
    median_cmi = np.median(cmi)

    return(cmi, median_cmi)

def plot_cmi(cmi, median_cmi, file_name, attr, save_path, get_valid):
    """
    Function to plot a boxplot of the distribution of CMIs
    """
    if get_valid:
        file3 = f"barplot_{file_name}_{attr}_responsive"
    elif not get_valid:
        file3 = f"barplot_{file_name}_{attr}_allneurons"

    plt.figure(figsize=(10,5))
    
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
    plt.bar(labels, counts, color='steelblue', edgecolor='black')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Number of neurons')
    if get_valid:
        plt.title('Distribution of CMI values (Responsive neurons)')
    elif not get_valid:
        plt.title('Distribution of CMI values (All neurons)')
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

def plot_boxplot_magnitudes(magnitude, protocols, attr, file_name, save_path, get_valid):
    """
    Plots a boxplot showing the distribution of response magnitudes for each neuron,
    grouped by protocol.

    Parameters:
    - magnitude: dict, keys are protocol names, values are lists or arrays of magnitudes
    - protocols: list of protocols to include
    - attr: string, name of the measured attribute (e.g., 'z-score')
    - file_name: string, base name for the saved figure
    - save_path: string, folder where to save the figure
    - get_valid: bool, if True, only responsive neurons were used (affects title and filename)
    """

    # Create appropriate filename
    if get_valid:
        fname = f"boxplot_{file_name}_{attr}_responsive.png"
        title = f'Distribution of neuron response magnitudes ({attr})\n(Responsive neurons)'
    else:
        fname = f"boxplot_{file_name}_{attr}_allneurons.png"
        title = f'Distribution of neuron response magnitudes ({attr})\n(All neurons)'

    # Prepare data
    data = [magnitude[protocol] for protocol in protocols]

    # Plot boxplot
    plt.figure(figsize=(6, 6))
    box = plt.boxplot(data, labels=protocols, patch_artist=True)

    # Optional: Add color
    for patch in box['boxes']:
        patch.set_facecolor('lightblue')

    plt.ylabel(f'Response magnitude (Mean of {attr} excluding the first 0.5s)')
    plt.title(title)
    plt.grid(axis='y')
    plt.tight_layout()

    # Save
    plt.savefig(os.path.join(save_path, fname), dpi=300)

    plt.show()



#---------------------------------Plots-------------------------------------------#
# Plot averages +/- SEM and get a dictionary of the magnitude of responses for all protocols

# To plot all neurons together
magnitude_all, neurons_count_all = graph_averages(file_name, attr, save_path, trials, protocols, z_score_periods, frame_rate, valid_neurons, all_protocols, dt_prestim, get_valid = False)
cmi_all, median_cmi_all = cmi(magnitude_all, neurons_count_all, protocols)
allneurons_cmi = plot_cmi(cmi_all, median_cmi_all, file_name, attr, save_path, get_valid=False)
boxplot_all = plot_boxplot_magnitudes(magnitude_all, protocols, attr, file_name, save_path, get_valid = False)

# To plot only responsive neurons
magnitude_responsive, neurons_count_responsive = graph_averages(file_name, attr, save_path, trials, protocols, z_score_periods, frame_rate, valid_neurons, all_protocols, dt_prestim, get_valid = True)
cmi_responsive, median_cmi_responsive = cmi(magnitude_responsive, neurons_count_responsive, protocols)
responsive_cmi = plot_cmi(cmi_responsive, median_cmi_responsive, file_name, attr, save_path, get_valid=True)
boxplot_responsive = plot_boxplot_magnitudes(magnitude_all, protocols, attr, file_name, save_path, get_valid = True)

