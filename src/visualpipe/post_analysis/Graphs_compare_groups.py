import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
from scipy.stats import mannwhitneyu, wilcoxon, linregress
import glob

def process_group(group_name, mice_list, attr):
    if attr == 'dFoF0-baseline':
        z_score_periods = ['norm_averaged_baselines', 'norm_trial_averaged_ca_trace', 'norm_post_trial_averaged_ca_trace']
    elif attr == 'z_scores':
        z_score_periods = ['pre_trial_averaged_zscores', 'trial_averaged_zscores', 'post_trial_averaged_zscores']
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
            if attr == 'dFoF0-baseline':
                proportion = 100 * len(valid_neurons) / trials['norm_trial_averaged_ca_trace'][0].shape[0]
                proportion_list.append(proportion)
            elif attr == 'z_scores':
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
                if attr == 'dFoF0-baseline':
                    trial_zscores = trials['norm_trial_averaged_ca_trace'][list(trials['norm_trial_averaged_ca_trace'].keys())[idx]][valid_neurons, :]
                elif attr == 'z_scores':
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

    return suppression, magnitude, stim_mean, all_neurons, avg, sem, cmi, proportion_list

def XY_magnitudes(groups, magnitude_wt, magnitude_ko, protocols, protocol_validity, save_path, attr):
    """
    Function to extract and plot the x and y values for the magnitude of the response to each protocol for both groups.
    """
    if len(protocols) != 2:
        raise ValueError("This function is designed to work with exactly two protocols.")
    x_values = []
    y_values = []
    for group, magnitude in zip(groups, [magnitude_wt, magnitude_ko]):
        x_values = magnitude[protocols[0]]
        y_values = magnitude[protocols[1]]
        # Create the plot
        plt.plot(x_values, y_values, 'o', color='cyan' if group == groups[0] else 'orange', alpha=0.5, label=f'{group} neurons')
        slope, intercept, r_value, p_value, std_err = linregress(x_values, y_values)
        # Plot regression line
        x_fit = np.linspace(min(x_values), max(x_values), 100)
        y_fit = [slope * xi + intercept for xi in x_fit]
        plt.plot(x_fit, y_fit,
                color='blue' if group == groups[0] else 'red',
                label=f'{group} fit: y = {slope:.2f}x + {intercept:.2f}, p = {p_value:.2g}, r**2 = {(r_value)**2:.3f}')
    # Add labels
    plt.xlabel(f"Magnitude of response to {protocols[0]}")
    plt.ylabel(f"Magnitude of response to {protocols[1]}")
    plt.title(f"Response magnitudes ({attr}) for {protocol_validity}-responsive neurons")
    plt.legend()
    plt.savefig(os.path.join(save_path, f"{fig_name}_magnitude_response_{attr}.jpeg"), dpi=300)
    plt.show()

def plot_perc_responsive(groups, proportions_wt, proportions_ko, save_path, fig_name):
    """
    Function to plot the % of responsive neurons per session for WT and KO groups.
    """
    colors = {groups[0]: 'skyblue', groups[1]: 'salmon'}
    fig, ax = plt.subplots(figsize=(8, 6))

    x_ticks = []
    x_labels = []
    width = 0.6
    offset = 0.3  # spacing between WT and KO

    # Bar height: mean
    wt_mean = np.mean(proportions_wt)
    ko_mean = np.mean(proportions_ko)
    # Mann–Whitney U test
    stat, p = mannwhitneyu(proportions_wt, proportions_ko, alternative='two-sided')

    # Plot WT
    ax.bar(1, wt_mean, width=width, color=colors[groups[0]], edgecolor='black', label=groups[0])
    ax.scatter([1] * len(proportions_wt), proportions_wt, color='black', zorder=10)

    # Plot KO
    ax.bar(2, ko_mean, width=width, color=colors[groups[1]], edgecolor='black', label=groups[1])
    ax.scatter([2] * len(proportions_ko), proportions_ko, color='black', zorder=10)

    # Annotate p-value
    y_max = max(np.max(proportions_wt), np.max(proportions_ko)) + 0.3
    ax.plot([1,2], [y_max, y_max], color='black', linewidth=1.5)
    ax.text((3) / 2, y_max + 0.05, f"M-W p = {p:.3g}", ha='center', va='bottom', fontsize=11)

    # Update ticks
    x_ticks.extend([1,2])
    x_labels.extend(groups)

    # Labeling
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_ylabel(f'% of {protocol_validity}-responsive neurons')
    ax.set_title(f'% of responsive neurons per session')
    ax.legend()
    
    plt.tight_layout()
    
    # save figure
    plt.savefig(os.path.join(save_path, f"{fig_name}_barplot_%responsive.jpeg"), dpi=300)
    plt.show()

def plot_avg_session(groups, attr, save_path, fig_name, stim_wt, stim_ko, protocols):
    
    colors = {groups[0]: 'skyblue', groups[1]: 'salmon'}

    fig, ax = plt.subplots(figsize=(8, 6))

    x_ticks = []
    x_labels = []
    width = 0.6
    offset = 0.3  # spacing between WT and KO

    for i, protocol in enumerate(protocols):
        wt_data = stim_wt[protocol]
        ko_data = stim_ko[protocol]

        # Bar height: mean
        wt_mean = np.mean(wt_data)
        ko_mean = np.mean(ko_data)
        # Mann–Whitney U test
        stat, p = mannwhitneyu(wt_data, ko_data, alternative='two-sided')

        # Plot WT
        x_wt = i * 2 - offset
        ax.bar(x_wt, wt_mean, width=width, color=colors[groups[0]], edgecolor='black', label=groups[0] if i == 0 else "")
        ax.scatter([x_wt] * len(wt_data), wt_data, color='black', zorder=10)

        # Plot KO
        x_ko = i * 2 + offset
        ax.bar(x_ko, ko_mean, width=width, color=colors[groups[1]], edgecolor='black', label=groups[1] if i == 0 else "")
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
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_ylabel(f'Mean stimulus response ({attr})')
    ax.set_title('Session-averaged responses during stimulus')
    ax.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(save_path, f"{fig_name}_barplot_stim_response_{attr}.jpeg"), dpi=300)
    plt.show()

def graph_averages(frame_rate, groups, fig_name, attr, save_path, protocols, protocol_validity, avg_wt, sem_wt, avg_ko, sem_ko, wt_neurons, ko_neurons):
    """
    Function to plot the average z-scores or dFoF0 for responsive neurons.
    """
    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(18, 18))
    # Select colors for each group and protocol
    if len(protocols) == 1:
        colors = {groups[0]: {protocols[0]: 'blue'},
            groups[1]: {protocols[0]: 'red'}}
    elif len(protocols) == 2:
        colors = {groups[0]: {protocols[0]: 'blue', protocols[1]: 'cyan'},
                groups[1]: {protocols[0]: 'red', protocols[1]: 'orange'}}
    # color map for graphs separating groups
    color_map = plt.get_cmap('tab10')  # You can use 'tab10', 'Set2', etc.
    protocol_colors = {protocol: color_map(i) for i, protocol in enumerate(protocols)}

    for i,group in enumerate(groups):
        if group == groups[0]:
            avg = avg_wt
            sem = sem_wt
            neurons = wt_neurons
        elif group == groups[1]:
            avg = avg_ko
            sem = sem_ko
            neurons = ko_neurons
        # Get minimum length among all protocols
        min_len = min(len(avg[protocol]) for protocol in protocols)

        # Generate time vector accordingly
        time = np.linspace(0, min_len, min_len) / frame_rate - 1  # time in seconds

        for protocol in protocols:
            axs[1,0].plot(time, avg[protocol][:min_len], color=colors[group][protocol], label=f"{group}, {protocol}, {neurons} neurons")
            axs[1,0].fill_between(time,
                            avg[protocol][:min_len] - sem[protocol][:min_len],
                            avg[protocol][:min_len] + sem[protocol][:min_len],
                            color=colors[group][protocol], alpha=0.3)
            
        # plot each group separately
        for protocol in protocols:
            axs[0,i].plot(time, avg[protocol][:min_len], color=protocol_colors[protocol], label=f"{group} {protocol}, {neurons} neurons")
            axs[0,i].fill_between(time,
                            avg[protocol][:min_len] - sem[protocol][:min_len],
                            avg[protocol][:min_len] + sem[protocol][:min_len], color=protocol_colors[protocol],
                            alpha=0.3)
        axs[0,i].set_xticks(np.arange(-1, time[-1] + 1, 1))
        axs[0,i].set_xlabel("Time (s)")
        axs[0,i].set_ylabel(f"Average {attr} for neurons responsive to {protocol_validity}")
        axs[0,i].set_title(f"Mean {attr} ± SEM by Protocol for {group}")
        axs[0, i].legend(loc='upper left', bbox_to_anchor=(1.05, 1))


    axs[1,0].set_xticks(np.arange(-1, time[-1] + 1, 1))
    axs[1,0].set_xlabel("Time (s)")
    axs[1,0].set_ylabel(f"Average {attr} for neurons responsive to {protocol_validity}")
    axs[1,0].set_title(f"Mean {attr} ± SEM by Group and Protocol")
    axs[1,0].legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    # Hide the unused subplot (bottom-right)
    axs[1, 1].axis('off')
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.2)
    plt.savefig(os.path.join(save_path, f"{fig_name}_averages_{attr}.jpeg"), dpi=300, bbox_inches='tight')
    plt.show()

def boxplot(list1, list2, neurons1, neurons2, groups, protocols, save_path, fig_name, attr, variable = "CMI"):
    """
    Function to plot a boxplot comparing the distribution of two groups.
    variable: 'CMI' or 'suppression_index'
    """
    if len (protocols)!= 2:
        raise ValueError("This function is designed to work with exactly two protocols.")
    
    array1 = np.array(list1)
    array2 = np.array(list2)
    median1 = np.median(array1)
    median2 = np.median(array2)
    p_value_1 = wilcoxon(array1, alternative='two-sided')[1]
    p_value_2 = wilcoxon(array2, alternative='two-sided')[1]

    # Define bin edges 
    outside_labels = []
    if variable == "CMI":
        inside_bins = np.linspace(-1.5, 1.5, 16)  
        outside_labels = ['< -1.5', '> 1.5']
    elif variable == "suppression_index":
        inside_bins = np.linspace(-2, 2, 24)
        outside_labels = ['< -2', '> 2']
    bins = np.concatenate(([-np.inf], inside_bins, [np.inf]))

    bin_indices_1 = np.digitize(array1, bins)
    bin_indices_2 = np.digitize(array2, bins)

    counts_1 = []
    counts_2 = []
    labels = []
    # Prepare counts and labels
    # negative outside bin 
    labels.append(outside_labels[0])
    counts_1.append(100 * np.sum(bin_indices_1 == 1) / neurons1)
    counts_2.append(100 * np.sum(bin_indices_2 == 1) / neurons2)

    # inside bins
    for i in range(2, len(bins)-1):
        bin_start = bins[i-1]
        bin_end = bins[i]
        labels.append(f'{bin_start:.2f} to {bin_end:.2f}')
        counts_1.append(100 * np.sum(bin_indices_1 == i) / neurons1)
        counts_2.append(100 * np.sum(bin_indices_2 == i) / neurons2)

    # positive outside bin 
    labels.append(outside_labels[1])
    counts_1.append(100 * np.sum(bin_indices_1 == len(bin_indices_1)) / neurons1)
    counts_2.append(100 * np.sum(bin_indices_2 == len(bin_indices_2)) / neurons2)

    # Bar plot parameters
    x = np.arange(len(labels))  # label locations
    width = 0.4  # width of the bars

    plt.figure(figsize=(12,6))
    plt.bar(x - width/2, counts_1, width, label=groups[0], color='blue', edgecolor='black')
    plt.bar(x + width/2, counts_2, width, label=groups[1], color='red', edgecolor='black')

    # Perform Mann–Whitney U test between WT and KO
    stat_mwu, p_value_mwu = mannwhitneyu(array1, array2, alternative='two-sided')

    # Add p-value annotation to the plot
    textstr = (
        f'{groups[0]} median = {median1:.2f}, p (vs 0) = {p_value_1:.3g}\n'
        f'{groups[1]} median = {median2:.2f}, p (vs 0) = {p_value_2:.3g}\n'
        f'{groups[0]} vs {groups[1]} (Mann–Whitney) p = {p_value_mwu:.3g}'
    )

    # Position textbox on plot
    plt.gca().text(0.95, 0.95, textstr,
                transform=plt.gca().transAxes,
                fontsize=11, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85))
    plt.legend(loc='upper right')


    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel(f'% of neurons')
    plt.title(f"{variable} for {groups[0]} vs {groups[1]}")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"barplot_{fig_name}_{variable}_{attr}.jpeg"), dpi=300)
    plt.show()

if __name__ == "__main__":
    
    #-----------------------INPUTS-----------------------#
    ## Organization of the data: folder for the protocol (e.g surround mod) > group (e.g KO and WT) > # mouse (e.g 110, 108) > session (e.g 2023-10-01) > .npz file with the validity of the neurons and .npy file with the trials
    #The folder containing all subfolders
    data_path = r"Y:\raw-imaging\Nathan\PYR\surround_mod"
    save_path = data_path
    # group names have to be the name of the subfolders
    groups = ['WT', 'KO']
    # mice ids per group (sub-subfolders)
    WT_mice = ['110', '108']
    KO_mice = ['109', '112']
    #Will be included in all names of saved figures
    fig_name = 'test'
    # Write the protocols you want to plot 
    protocols = ['center', 'center-surround-iso']  
    # List of protocol(s) used to select reponsive neurons. If contains several protocols, neurons will be selected if they are responsive to at least one of the protocols in the list.
    protocol_validity = ['center'] 
    #Frame rate
    frame_rate = 30
    # Decide if you want to plot the dFoF0 baseline substraced or the z-scores
    attr = 'dFoF0-baseline'  # 'dFoF0-baseline' or 'z_scores'
    #----------------------------------------------------#

    if attr == 'dFoF0-baseline':
        z_score_periods = ['norm_averaged_baselines', 'norm_trial_averaged_ca_trace', 'norm_post_trial_averaged_ca_trace']
    elif attr == 'z_scores':
        z_score_periods = ['pre_trial_averaged_zscores', 'trial_averaged_zscores', 'post_trial_averaged_zscores']

    x_labels = ['Pre-stim', 'Stim', 'Post-stim']

    #-------------------Call the functions to process and plot the data-------------------#
    # Process WT
    suppression_wt, magnitude_wt, stim_wt, wt_neurons, avg_wt, sem_wt, cmi_wt, proportions_wt = process_group('WT', WT_mice, attr)
    # Process KO
    suppression_ko, magnitude_ko, stim_ko, ko_neurons, avg_ko, sem_ko, cmi_ko, proportions_ko = process_group('KO', KO_mice, attr)
    #XY plot of the magnitudes of the responses to the two protocols
    XY_magnitudes(groups, magnitude_wt, magnitude_ko, protocols, protocol_validity, save_path, attr)
    # Plot the % of responsive neurons per session
    plot_perc_responsive(groups, proportions_wt, proportions_ko, save_path, fig_name)
    # Plot the average response during the stim period per session
    plot_avg_session(groups, attr, save_path, fig_name, stim_wt, stim_ko, protocols)
    # Plot the average z-scores or dFoF0-baseline trace for responsive neurons
    graph_averages(frame_rate, groups, fig_name, attr, save_path, protocols, protocol_validity, avg_wt, sem_wt, avg_ko, sem_ko, wt_neurons, ko_neurons)
    #plot the distribution of CMI 
    boxplot(cmi_wt, cmi_ko, wt_neurons, ko_neurons, groups, protocols, save_path, fig_name, attr, variable = "CMI")
    #plot the distribution of suppression index
    boxplot(suppression_wt, suppression_ko, wt_neurons, ko_neurons, groups, protocols, save_path, fig_name, attr, variable = "suppression_index")