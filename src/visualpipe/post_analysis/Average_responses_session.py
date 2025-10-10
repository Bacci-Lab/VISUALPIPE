import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
from scipy.stats import wilcoxon
from scipy.stats import mannwhitneyu
import pandas as pd
import os
import sys
import seaborn as sns
sys.path.append("./src")

import visualpipe.utils.file as file
from visualpipe.analysis.ca_imaging import CaImagingDataManager

def graph_averages(file_name, attr, save_path, trials, protocols, frame_rate, valid_neurons, all_protocols, dt_prestim, get_valid):
    """
    Function to plot the average z-scores or dF/F0 - baseline for all neurons if get_valid is False or only responsive neurons if get_valid is True.
    """
    print(trials.keys())
    if attr == 'dFoF0-baseline':
        z_score_periods = ['norm_averaged_baselines', 'norm_trial_averaged_ca_trace', 'norm_post_trial_averaged_ca_trace']
    elif attr == 'z_scores':
        z_score_periods = ['pre_trial_averaged_zscores', 'trial_averaged_zscores', 'post_trial_averaged_zscores']
    
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
    
    excel_dict = {}
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
        excel_dict[f'average_{protocol}'] = y_axis[protocol]
        excel_dict[f'sem_{protocol}'] = sem[protocol]
        nb_neurons = neurons_count[protocol]

    
    # ---- Plot the averaged z-score for all protocols ------#
    time = np.linspace(0, len(y_axis[protocols[0]]), len(y_axis[protocols[0]]))
    time = time/frame_rate - dt_prestim # Shift the time by the duration of the baseline so the stimulus period starts at 0

    excel_dict['Time (s)'] = time

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

    # Create DataFrame and save to Excel
    df = pd.DataFrame(excel_dict)
    cols = ['Time (s)']+ [c for c in df.columns if c if c not in ['Time (s)']] 
    df = df[cols]
    summary_row = pd.DataFrame({'nb_neurons': [nb_neurons]})
    df = pd.concat([summary_row, df], ignore_index=True)
    excel_path = os.path.join(save_path, f"{file_name}_averages_{attr}.xlsx")
    df.to_excel(excel_path, index=False)

    return magnitude, neurons_count

def plot_per_trial(file_name, attr, save_path, trials, protocols, frame_rate, valid_neurons, all_protocols, dt_prestim, get_valid):
    """
    Function to plot the average z-scores or dF/F0 - baseline per trial for all neurons if get_valid is False or only responsive neurons if get_valid is True.
    """
    if attr == 'dFoF0-baseline':
        periods = ['pre_trial_fluorescence', 'trial_fluorescence', 'post_trial_fluorescence']
    elif attr == 'z_scores':
        periods = ['pre_trial_zscores', 'trial_zscores', 'post_trial_zscores']
    
    # One subplot per protocol
    fig, ax = plt.subplots(1, len(protocols), figsize=(6 * len(protocols), 6), squeeze=False)
    ax = ax[0]

    file1 = f"perTrial_{file_name}_{attr}_{'responsive' if get_valid else 'allneurons'}.jpeg"

    y_axis = {}  # dict to store mean zscores for each protocol
    sem = {}     # dict to store SEMs
    magnitude = {protocol:[] for protocol in protocols}
    neurons_count = {protocol: None for protocol in protocols}
    
    excel_dict = {}
    # Loop through each protocol and plot the average z-scores
    for protocol in protocols:
        idx = all_protocols.index(protocol)
        print(f"\nProtocol: {protocol}, Index: {idx}")
        y_axis[protocol] = {}
        sem[protocol] = {}
        n_trials = len(trials['trial_fluorescence'][idx][0])
        magnitude[protocol] = {trial: [] for trial in range(0,n_trials)}
        for trial in range(0,n_trials):
            avg_trial, sem_trial = [], []
            for period in periods:
                if not get_valid:
                    zscores = trials[period][list(trials[period].keys())[idx]][:,trial] # shape: (neurons, time)
                elif get_valid:
                    zscores = trials[period][list(trials[period].keys())[idx]][valid_neurons, trial] # shape: (neurons, time)
                neurons = zscores.shape[0]
                avg_zscore = np.mean(zscores, axis=0)
                sem_period = stats.sem(zscores, axis=0)
                avg_trial.append(avg_zscore)
                sem_trial.append(sem_period)

                #Compute the magnitude of the response for each neuron as the mean z-score during the stimulus period
                if period == 'trial_fluorescence' or period == 'trial_zscores':
                    for n in range(0,neurons):
                        zneuron = zscores[n, int(frame_rate*0.5):] # exclude the first 0.5 seconds because of GCaMP's slow kinetics
                        magnitude[protocol][trial].append(np.mean(zneuron))

            # Concatenate all 3 periods along time axis
            y_axis[protocol][trial] = np.concatenate(avg_trial)
            sem[protocol][trial] = np.concatenate(sem_trial)
            excel_dict[f'average_{protocol}_trial{int(trial)+1}'] = y_axis[protocol][trial]
            excel_dict[f'sem_{protocol}_trial{int(trial)+1}'] = sem[protocol][trial]
                
        neurons_count[protocol] = neurons


    
    # ---- Plot the averaged z-score for all protocols ------#
    time = np.linspace(0, len(y_axis[protocols[0]][0]), len(y_axis[protocols[0]][0]))
    time = time/frame_rate - dt_prestim # Shift the time by the duration of the baseline so the stimulus period starts at 0

    excel_dict['Time (s)'] = time

    for i, protocol in enumerate(protocols):
        excel_dict[f'nb_neurons_{protocol}'] = [neurons_count[protocol]] + ['']*(len(time)-1)
        n_trials = list(y_axis[protocol].keys())
        for trial in n_trials:
            ax[i].plot(time, y_axis[protocol][trial], label=f'Trial {int(trial)+1}')
            ax[i].fill_between(time,
                            y_axis[protocol][trial] - sem[protocol][trial],
                            y_axis[protocol][trial] + sem[protocol][trial],
                            alpha=0.4)
        ax[i].set_title(f'{protocol}, {neurons_count[protocol]} neurons')
        ax[i].axvline(0, color='k', linestyle='--', linewidth=1)
        ax[i].set_xlabel("Time (s)")
        ax[i].set_ylabel(f"Average {attr}")
        ax[i].legend()
        ax[i].set_xticks(np.arange(-1, round(time[-1]) + 1, 1))

    plt.suptitle(f"Mean {attr} ± SEM ({'Responsive Neurons' if get_valid else 'All Neurons'})")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, file1), dpi=300)
    plt.show()

    # Create DataFrame and save to Excel
    pd.DataFrame(excel_dict).to_excel(os.path.join(save_path, file1.replace('.jpeg', '.xlsx')), index=False)

    return magnitude, neurons_count


def plot_magnitude_per_trial(file_name, save_path, magnitude, protocols, get_valid):
    """
    Plot the average response magnitude (mean z-score or dF/F0 during stimulus)
    for each trial, one subplot per protocol.
    
    Parameters
    ----------
    file_name : str
        Base name for output files.
    save_path : str
        Directory to save figures and Excel output.
    magnitude : dict
        Output from plot_per_trial(), containing {protocol: {trial: [values per neuron]}}.
    protocols : list
        List of protocol names (same order as used in plot_per_trial()).
    get_valid : bool
        If True, indicates magnitudes correspond to responsive neurons only.
    """

    fig, ax = plt.subplots(1, len(protocols), figsize=(6 * len(protocols), 6), squeeze=False)
    ax = ax[0]

    file2 = f"perTrial_magnitude_{file_name}_{'responsive' if get_valid else 'allneurons'}.jpeg"

    excel_dict = {}

    for i, protocol in enumerate(protocols):
        trials = list(magnitude[protocol].keys())
        mean_mag = []
        sem_mag = []

        # Compute mean ± SEM across neurons for each trial
        for trial in trials:
            trial_values = np.array(magnitude[protocol][trial])
            mean_mag.append(np.mean(trial_values))
            sem_mag.append(stats.sem(trial_values))

        mean_mag = np.array(mean_mag)
        sem_mag = np.array(sem_mag)

        # ---- Plot ----
        ax[i].bar(range(1, len(trials) + 1), mean_mag, yerr=sem_mag, capsize=5, alpha=0.7)
        ax[i].set_title(protocol)
        ax[i].set_xlabel("Trial")
        ax[i].set_ylabel("Mean Response Magnitude")
        ax[i].set_xticks(range(1, len(trials) + 1))
        ax[i].grid(alpha=0.3)

        # Store for Excel
        excel_dict[f'mean_magnitude_{protocol}'] = mean_mag
        excel_dict[f'sem_magnitude_{protocol}'] = sem_mag

    # ---- Global formatting ----
    plt.suptitle(f"Response Magnitude per Trial ({'Responsive' if get_valid else 'All'} Neurons)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, file2), dpi=300)
    plt.show()

    # ---- Save to Excel ----
    pd.DataFrame(excel_dict).to_excel(os.path.join(save_path, file2.replace('.jpeg', '.xlsx')), index=False)

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

def plot_cmi(cmi, median_cmi=None, file_name='', attr='dFoF0-baseline', save_path='', get_valid=False):
    """
    Function to plot a boxplot of the distribution of CMIs
    """

    if median_cmi is None:
        median_cmi = np.median(cmi)

    if get_valid:
        file3 = f"barplot_{file_name}_{attr}_responsive"
    elif not get_valid:
        file3 = f"barplot_{file_name}_{attr}_allneurons"
    
    edgecolor='black'
    cmi_plot = [el if np.abs(el) < 1.5 else 2 if el > 1.5 else -2 for el in cmi]

    _, p_value = wilcoxon(cmi, zero_method='wilcox', alternative='two-sided', correction=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.histplot(cmi_plot, bins=12, binrange=(-1.5, 1.5), color='white', edgecolor=edgecolor, element='step', ax=ax)
    sns.histplot(cmi_plot, bins=1, binrange=(1.875, 2.125), color='white', edgecolor=edgecolor, ax=ax)
    sns.histplot(cmi_plot, bins=1, binrange=(-2.125, -1.875), color='white', edgecolor=edgecolor, ax=ax)
    ax.set_xticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], labels=['<-1.5', '-1.5', '-1', '-0.5', '0', '0.5', '1', '1.5', '>1.5'])
    ax.set_ylabel('Count of neurons')
    ax.set_title('Contextual Modulation Index')
    ax.axvline(median_cmi, color='black', linestyle='--', linewidth=1.5, label='median') # Add vertical dashed line at median position
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    plt.tight_layout()

    textstr = f'{len(cmi)} neurons'
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='left')
    
    textstr = f'Median = {median_cmi:.2f}\nWilcoxon p = {p_value:.3g}'
    plt.gca().text(0.95, 0.95, textstr, transform=plt.gca().transAxes,
                   fontsize=10, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    if get_valid:
        plt.title('Distribution of CMI values (Responsive neurons)')
    elif not get_valid:
        plt.title('Distribution of CMI values (All neurons)')

    fig.savefig(os.path.join(save_path, file3), dpi=300)
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
    box = plt.boxplot(data, tick_labels=protocols, patch_artist=True)

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


def plot_cdf_magnitudes(magnitude, protocols, attr, file_name, save_path, get_valid):
    """
    Plots CDFs of neuron response magnitudes for each protocol
    and runs pairwise statistical comparisons.

    Parameters:
    - magnitude: dict, keys are protocol names, values are lists of magnitudes
    - protocols: list of protocols to include
    - attr: string, how fluorescence is measured (e.g., 'z_score' or 'dFoF0-baseline')
    - file_name: string, base name for the saved figure
    - save_path: string, folder where to save the figure
    - get_valid: bool, if True, only responsive neurons were used (affects title and filename)
    """

    # Filename & title
    if get_valid:
        fname = f"cdf_{file_name}_{attr}_responsive.png"
        title = f'Cumulative distribution of neuron response magnitudes ({attr})\n(Responsive neurons)'
    else:
        fname = f"cdf_{file_name}_{attr}_allneurons.png"
        title = f'Cumulative distribution of neuron response magnitudes ({attr})\n(All neurons)'

    plt.figure(figsize=(6, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(protocols)))
    stats_text = []

    # Plot CDF for each protocol
    for i, protocol in enumerate(protocols):
        data = np.array(magnitude[protocol])
        data_sorted = np.sort(data)
        cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
        plt.plot(data_sorted, cdf, label=protocol, color=colors[i], lw=2)

    # Perform statistical test (pairwise Mann–Whitney U)
    for i in range(len(protocols)):
        for j in range(i + 1, len(protocols)):
            p = mannwhitneyu(magnitude[protocols[i]], magnitude[protocols[j]], alternative='two-sided').pvalue
            # Put text in bottom-right corner of the axes
            plt.text(
                0.95, 0.05,  # relative position in axes coords
                f"p = {p:.3e}",
                transform=plt.gca().transAxes,
                fontsize=10,
                ha='right', va='bottom',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none")
            )

    # Add stats text on the plot
    y_pos = 0.95
    for line in stats_text:
        plt.text(1.05, y_pos, line, transform=plt.gca().transAxes, fontsize=8, va='top')
        y_pos -= 0.05

    # Styling
    plt.xlabel(f'Response magnitude (Mean of {attr} excluding first 0.5s)')
    plt.ylabel('Cumulative probability')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save
    plt.savefig(os.path.join(save_path, fname), dpi=300)
    plt.show()


if __name__ == "__main__":

    #--------------------INPUTS------------------#
    base_path = r"Y:\raw-imaging\Nathan\VIPcre-CB1tdTom\1st batch\fod+fog female\Visual\2025_03_28\TSeries-03282025-005"
    id_version = '4'
    file_name = 'test'

    # Get metadata to define data path
    unique_id, global_protocol, experimenter, subject_id = file.get_metadata(base_path)
    data_path = os.path.join(base_path, "_".join([unique_id, 'output', id_version]))

    # Define save path
    save_path = r'Y:\raw-imaging\Nathan\VIPcre-CB1tdTom\1st batch\fod+fog female\Visual\2025_03_28\TSeries-03282025-005\2025_03_28_15-45-51_output_2\Post-analysis'

    # --------------------------------- Define stimulus of interests --------------------------------- #
    #This is the list of stimuli you want to use to select the responsive neurons. A responsive neurons is responsive in at least one of these stimuli
    protocol_validity = ['center-grating-0.05-90.0', 'center-grating-0.19-90.0']  
    # Write the protocols you want to plot
    protocols = ['center-grating-0.05-90.0', 'center-grating-0.19-90.0']
    # Decide if you want to plot the dFoF0 baseline substraced or the z-scores
    attr = 'dFoF0-baseline'  # 'dFoF0-baseline' or 'z_scores'

    # --------------------------------- Load data --------------------------------- #

    # Load the npz file
    validity = np.load(os.path.join(data_path, "_".join([unique_id, id_version, 'protocol_validity_2.npz'])), allow_pickle=True)
    all_protocols = validity.files

    # Load NPY file containing averaged z_scores before during and after stim presentation
    trials = np.load(os.path.join(data_path, "_".join([unique_id, id_version, 'trials.npy'])), allow_pickle=True).item()

    # Get frame rate
    ca_img = CaImagingDataManager(base_path)
    frame_rate = ca_img.fs

    # Load xlsx file with visual stimuli info
    stimuli_df = pd.read_excel(os.path.join(data_path, "_".join([unique_id, id_version, 'visual_stim_info.xlsx'])), engine='openpyxl').set_index('id')

    # Get the duration of the pre-stimulus period
    dt_prestim = trials['dt_pre_stim']

    # ------------------------- Select the responsive neurons based on the validity file ------------------------- #
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
        proportion_valid = 100 * len(valid_neurons) / (trials['norm_trial_averaged_ca_trace'][0].shape[0])
    elif attr == 'z_scores':
        # Calculate the proportion of valid neurons
        proportion_valid = 100 * len(valid_neurons) / (trials['trial_averaged_zscores'][0].shape[0])


    print(f"Number of neurons responsive in {protocol_validity}: {len(valid_neurons)}")
    print(f"Proportion of neurons responsive in {protocol_validity}: {proportion_valid:.2f}%")

    #---------------------------------Plots-------------------------------------------#
    # Plot averages +/- SEM and get a dictionary of the magnitude of responses for all protocols

    """     # To plot all neurons together
    magnitude_all, neurons_count_all = graph_averages(file_name, attr, save_path, trials, protocols, frame_rate, valid_neurons, all_protocols, dt_prestim, get_valid = False)
    cmi_all, median_cmi_all = cmi(magnitude_all, neurons_count_all, protocols)
    allneurons_cmi = plot_cmi(cmi_all, median_cmi_all, file_name, attr, save_path, get_valid=False)
    boxplot_all = plot_boxplot_magnitudes(magnitude_all, protocols, attr, file_name, save_path, get_valid = False) """

    # To plot only responsive neurons
    magnitude_responsive, neurons_count_responsive = graph_averages(file_name, attr, save_path, trials, protocols, frame_rate, valid_neurons, all_protocols, dt_prestim, get_valid = True)
    cmi_responsive, median_cmi_responsive = cmi(magnitude_responsive, neurons_count_responsive, protocols)
    responsive_cmi = plot_cmi(cmi_responsive, median_cmi_responsive, file_name, attr, save_path, get_valid=True)
    boxplot_responsive = plot_boxplot_magnitudes(magnitude_responsive, protocols, attr, file_name, save_path, get_valid = True)
    cdf_= plot_cdf_magnitudes(magnitude_responsive, protocols, attr, file_name, save_path, get_valid=True)
    magnitude_trial, neurons_count = plot_per_trial(file_name, attr, save_path, trials, protocols, frame_rate, valid_neurons, all_protocols, dt_prestim, get_valid = True)
    plot_magnitude_per_trial(file_name, save_path, magnitude_trial, protocols, get_valid = True)