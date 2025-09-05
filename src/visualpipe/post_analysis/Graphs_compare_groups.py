import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
from scipy.stats import mannwhitneyu, wilcoxon, linregress
import glob
import pandas as pd
import seaborn as sns
import sys
from scipy.ndimage import gaussian_filter1d
sys.path.append("./src")

import visualpipe.post_analysis.utils as utils

def get_valid_neurons_session(validity, protocol_list):
   # Dictionary to store responsive neurons for each protocol
    valid_data = {protocol : validity[protocol] 
                 if protocol in validity.files 
                 else print(f"{protocol} does not exist in validity file.") 
                 for protocol in protocol_list}
    
    valid_neuron_lists = [np.where(data[:, 0] == 1,)[0] for data in valid_data.values()] # change to -1 if you want negative responsive neurons
    valid_neurons = np.unique(np.concatenate(valid_neuron_lists))  # Get unique indices of valid neurons
    
    return valid_neurons

def get_centered_neurons(stimuli_df, neurons_list, trials, attr, frame_rate = 30, plot = False):
    """
    Function to get the indices of neurons that have their maximal response in the center stimulus.
    """
    stimuli = [
        'quick-spatial-mapping-up-left', 'quick-spatial-mapping-up', 
        'quick-spatial-mapping-up-right', 'quick-spatial-mapping-left',
        'quick-spatial-mapping-center', 'quick-spatial-mapping-right',
        'quick-spatial-mapping-down-left', 'quick-spatial-mapping-down',
        'quick-spatial-mapping-down-right'
    ]
    if attr == 'dFoF0-baseline':
        period_names = ['norm_averaged_baselines', 'norm_trial_averaged_ca_trace', 'norm_post_trial_averaged_ca_trace']
    elif attr == 'z_scores':
        period_names = ['pre_trial_averaged_zscores', 'trial_averaged_zscores', 'post_trial_averaged_zscores']
    else:
        raise ValueError(f"Unknown attribute: {attr}")
           
    if not all(stim in stimuli_df['name'].values for stim in stimuli):
        missing = [stim for stim in stimuli if stim not in stimuli_df['name'].values]
        raise ValueError(f"Missing stimuli for selecting centered neurons in this session: {missing}")
    
    centered_neurons = []
    not_centered = []
    for neuron in neurons_list:  # Iterate over all neurons
        magnitudes_neuron = {stimulus: [] for stimulus in stimuli}
        # build magnitude dictionary for the mapping stimuli
        for stimulus in stimuli:
            stimulus_id = stimuli_df[stimuli_df.name == stimulus].index[0]
            trial_neuron = trials[period_names[1]][stimulus_id][neuron, int(frame_rate*0.5):]  # Exclude first 0.5s
            magnitudes_neuron[stimulus] = np.mean(trial_neuron)
        # Find the stimulus with max response
        max_stimulus = max(magnitudes_neuron, key=magnitudes_neuron.get)
        if max_stimulus == 'quick-spatial-mapping-center':
            centered_neurons.append(neuron)
        else:
            not_centered.append(neuron)
    proportion_centered = 100*len(centered_neurons)/trials[period_names[1]][0].shape[0]

    if plot:
        def plot_for_neurons(neurons, title):
            fig, axes = plt.subplots(3, 3, figsize=(15, 12))
            axes = axes.flatten()
            # First, collect all traces to find global min/max
            all_avg_traces = []
            for stim in stimuli:
                stim_id = stimuli_df[stimuli_df.name == stim].index[0]
                traces = np.concatenate([trials[period][stim_id][neurons, :] for period in period_names], axis = 1)
                avg_trace = np.mean(traces, axis=0)
                all_avg_traces.append(avg_trace)

            time = np.linspace(0, len(all_avg_traces[0]), len(all_avg_traces[0])) / frame_rate - 1
            # Determine global min and max
            global_min = min([trace.min() for trace in all_avg_traces])
            global_max = max([trace.max() for trace in all_avg_traces])

            # Plot each stimulus
            for i, stim in enumerate(stimuli):   
                axes[i].plot(time, all_avg_traces[i])
                axes[i].set_title(stim.replace('quick-spatial-mapping-', ''))
                axes[i].set_xlabel('Time (s)')
                axes[i].set_xticks(np.arange(-1, time[-1] + 1, 1))
                axes[i].set_ylabel('Average dF/F0 - baseline')
                axes[i].set_ylim(global_min, global_max)  # same scale for all subplots

                # Add % text to center plot only
                if stim == 'quick-spatial-mapping-center':
                    axes[i].text(
                        0.95, 0.95,  # x, y in axes fraction coordinates
                        f'{proportion_centered:.1f}%\nCentered', 
                        transform=axes[i].transAxes,
                        fontsize=9,
                        fontweight='bold',
                        ha='right', 
                        va='top',
                        color='red'
                    )

            plt.suptitle(title, fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

        # Plot centered neurons
        plot_for_neurons(centered_neurons, 'Centered Neurons')

        # Plot not centered neurons
        plot_for_neurons(not_centered, 'Not Centered Neurons') 
    return centered_neurons, not_centered

def compute_cmi(magnitude, protocol_cross='center-surround-cross', protocol_iso='center-surround-iso'):
    """Compute the CMI (Center Magnitude Index) for the specified protocols.
    
    Args:
        magnitude (dict): Dictionary containing the magnitude of responses for each protocol.
        stimuli_df (pd.DataFrame): DataFrame containing the visual stimuli information.
        protocol_cross (str): Name of the cross protocol.
        protocol_iso (str): Name of the iso protocol.
    
    Returns:
        float: The computed CMI value.
    """
    accepted_protocols = ['center', 'center-surround-iso', 'center-surround-cross', 'surround-iso_ctrl', 'surround-cross_ctrl']
    
    if protocol_cross not in accepted_protocols :
        raise ValueError(f"Protocol '{protocol_cross}' is not in the accepted protocols: {accepted_protocols}")
    if protocol_cross not in magnitude.keys():
        raise ValueError(f"Protocol '{protocol_cross}' is not in the magnitude keys: {magnitude.keys()}")
    if protocol_iso not in accepted_protocols :
        raise ValueError(f"Protocol '{protocol_iso}' is not in the accepted protocols: {accepted_protocols}")
    if protocol_iso not in magnitude.keys():
        raise ValueError(f"Protocol '{protocol_iso}' is not in the magnitude keys: {magnitude.keys()}")
        
    cmi = (magnitude[protocol_cross] - magnitude[protocol_iso]) / (magnitude[protocol_cross] + magnitude[protocol_iso])
        
    return cmi

def compute_suppression(magnitude, protocol_surround='center-surround-iso'):
    """Compute the suppression index for the specified protocols.
    
    Args:
        magnitude (dict): Dictionary containing the magnitude of responses for each protocol.
        protocol_cross (str): Name of the cross protocol.
        protocol_iso (str): Name of the iso protocol.
    
    Returns:
        float: The computed suppression index value.
    """

    accepted_protocols = ['center-surround-iso', 'center-surround-cross']
    suppression = []
    if protocol_surround not in accepted_protocols :
        print(f"Protocol '{protocol_surround}' is not in the accepted protocols: {accepted_protocols}")
    elif protocol_surround not in magnitude.keys():
        print(f"Protocol '{protocol_surround}' is not in the magnitude keys: {magnitude.keys()}")
    elif 'center' not in magnitude.keys():
        print(f"Protocol 'center' is not in the magnitude keys: {magnitude.keys()}")
        
    else:
        suppression = 1 - magnitude[protocol_surround] / magnitude['center']

    return suppression

def process_group(df, groups_id, attr, valid_sub_protocols, sub_protocols, protocol_name, get_centered):
    if attr == 'dFoF0-baseline':
        period_names = ['norm_averaged_baselines', 'norm_trial_averaged_ca_trace', 'norm_post_trial_averaged_ca_trace']
    elif attr == 'z_scores':
        period_names = ['pre_trial_averaged_zscores', 'trial_averaged_zscores', 'post_trial_averaged_zscores']
    
    suppression_groups, magnitude_groups, stim_groups, nb_neurons, avg_groups, sem_groups, cmi_groups, proportions_groups, individual_groups = [], [], [], [], [], [], [], [], []

    for key in groups_id.keys():

        print(f"\n-------------------------- Processing {key} group --------------------------")

        all_neurons = 0
        magnitude = {protocol: [] for protocol in sub_protocols} #Will contain the magnitude of the response to each protocol for each neuron
        avg_data = {protocol: [] for protocol in sub_protocols} 
        stim_mean = {protocol: [] for protocol in sub_protocols} #Will contain the mean response to the stimulus for each protocol, per session
        single_neurons_group = {protocol: [] for protocol in sub_protocols} #Will contain the individual traces of each neuron for each protocol
        proportion_list = [] #Will contain the proportion of responding neurons per session

        df_filtered = df[df["Genotype"] == key]
        for k in range(len(df_filtered)):
            mouse_id = df_filtered["Mouse_id"].iloc[k]
            session_id = df_filtered["Session_id"].iloc[k]
            output_id = df_filtered["Output_id"].iloc[k]
            session_path = os.path.join(df_filtered["Session_path"].iloc[k], f"{session_id}_output_{output_id}")

            print(f"\nSession id: {session_id}\n  Mouse id : {mouse_id}\n     Session path: {session_path}")

            validity, trials, stimuli_df = utils.load_data_session(session_path)

            valid_neurons = get_valid_neurons_session(validity, valid_sub_protocols)
            
            if not get_centered:
                neurons = len(valid_neurons)
                all_neurons += neurons
                proportion = 100 * len(valid_neurons) / trials[period_names[1]][0].shape[0]
                proportion_list.append(proportion)
                print(f"Proportion responding neurons: {proportion}, Number of responding neurons: {neurons}")
            else:
                centered_neurons, non_centered = get_centered_neurons(stimuli_df, valid_neurons, trials, attr, frame_rate = 30, plot = False)
                all_neurons += len(centered_neurons)
                proportion = 100 * len(centered_neurons) / trials[period_names[1]][0].shape[0]
                proportion_list.append(proportion)
                print(f"Proportion of centered neurons: {proportion}, Number of centered neurons: {len(centered_neurons)}")
                valid_neurons = centered_neurons

            
            for protocol in sub_protocols:

                stim_id = stimuli_df[stimuli_df.name == protocol].index[0]

                # Get z-scores from responsive-neurons for that protocol from pre, stim and post periods and concatenate along time
                traces_sep = [trials[period][stim_id][valid_neurons, :] for period in period_names]
                traces_concat = np.concatenate(traces_sep, axis=1)
                # Merge individual traces into group-level container
                if len(single_neurons_group[protocol]) == 0:
                    single_neurons_group[protocol] = traces_concat
                else:
                    single_neurons_group[protocol] = np.vstack([single_neurons_group[protocol], traces_concat])

                avg_session_trace = np.mean(traces_concat, axis=0) # average over neurons in that session
                avg_data[protocol].append(avg_session_trace)

                # Extract trial-averaged zscores during the stim period and give the average response of all neurons in that session (exclude the first 0.5 seconds because of GCaMP's slow kinetics)
                trial_traces = trials[period_names[1]][stim_id][valid_neurons, int(frame_rate*0.5):]
                mean_trial_value_per_neurons = np.mean(trial_traces, axis=1)  # average per neuron over time, 
                mean_trial_value_session = np.mean(mean_trial_value_per_neurons)  # average over neurons in that session
                stim_mean[protocol].append(mean_trial_value_session)
                
                #Store the average response of each neuron to that protocol
                magnitude[protocol].append(mean_trial_value_per_neurons)
        for protocol in magnitude.keys():
            magnitude[protocol] = np.concatenate(magnitude[protocol])

        if protocol_name == "surround-mod" and len(sub_protocols) == 2:
            cmi = compute_cmi(magnitude, sub_protocols[1], sub_protocols[0])
            suppression = compute_suppression(magnitude, sub_protocols[1])
        else :
            cmi, suppression = [], []

        print(f"\nNumber of {key} neurons: {all_neurons}")

        # Concatenate all neuron arrays into one array per protocol
        for protocol in sub_protocols:
            avg_data[protocol] = np.stack(avg_data[protocol], axis=0)

        # Compute average and SEM across neurons
        avg = {protocol: np.mean(avg_data[protocol], axis=0) for protocol in sub_protocols} 
        sem = {protocol: stats.sem(avg_data[protocol], axis=0) for protocol in sub_protocols}
        print(f"List of % of responsive neurons per session for {key}: {proportion_list}")

        suppression_groups.append(suppression)
        magnitude_groups.append(magnitude)
        stim_groups.append(stim_mean)
        nb_neurons.append(all_neurons)
        avg_groups.append(avg)
        sem_groups.append(sem)
        cmi_groups.append(cmi)
        proportions_groups.append(proportion_list)
        individual_groups.append(single_neurons_group)

    return suppression_groups, magnitude_groups, stim_groups, nb_neurons, avg_groups, sem_groups, cmi_groups, proportions_groups, individual_groups

def XY_magnitudes(groups_id, magnitude_groups, sub_protocols, protocol_validity, save_path, attr):
    """
    Function to extract and plot the x and y values for the magnitude of the response to each protocol for both groups.
    """

    color = {'WT': 'skyblue', 'KO': 'orange'}
    if len(sub_protocols) == 2:
        protocol_x = sub_protocols[0]
        protocol_y = sub_protocols[1]
        for group in groups_id.keys():

            magnitude = magnitude_groups[groups_id[group]]
            x_values = magnitude[protocol_x]
            y_values = magnitude[protocol_y]
            # Clamp values
            x_values = np.clip(x_values, -0.1, 0.5)
            y_values = np.clip(y_values, -0.1, 0.5)
            # Define tick positions
            ticks = np.arange(-0.1, 0.6, 0.1)
            # Define tick labels (same length as ticks)
            tick_labels = ['<-0.1'] + [f"{t:.1f}" for t in ticks[1:-1]] + ['>0.5']
            """ slope, intercept, r_value, p_value, _ = linregress(x_values, y_values)
            x_fit = np.linspace(min(x_values), max(x_values), 100)
            y_fit = slope * x_fit + intercept """

            # Create the plot
            plt.scatter(x_values, y_values, marker='o', c=color[group], alpha=0.5, label=f'{group} neurons')
            # Plot regression line
            
            #plt.plot(x_fit, y_fit, color=color[group], label=f'{group} fit: y = {slope:.2f}x + {intercept:.2f}, p = {p_value:.2g}, r**2 = {(r_value)**2:.3f}')
        
        # Add labels
        # Add dashed x=y line
        lims = [-0.1, 0.5]
        plt.plot(lims, lims, 'k--', alpha=0.7, label="x = y")

        plt.xlabel(f"Magnitude of response to {protocol_x}")
        plt.ylabel(f"Magnitude of response to {protocol_y}")
        plt.xticks(ticks, tick_labels)
        plt.yticks(ticks, tick_labels)
        plt.title(f"Response magnitudes ({attr}) for {str(protocol_validity)+'-responsive' if not get_centered else 'centered'} neurons")
        plt.legend()
        plt.savefig(os.path.join(save_path, f"{fig_name}_magnitude_response_{attr}.jpeg"), dpi=300)
        plt.show()
    else:
        print(f"XY plot is not available for {len(sub_protocols)} protocols. Please select 2 protocols to compare.")
        print(f"Current protocols: {sub_protocols}")
        return None

def plot_perc_responsive(groups_id, proportions_groups, save_path, fig_name):
    """
    Function to plot the % of responsive neurons per session for WT and KO groups.
    """
    color = {'WT': 'skyblue', 'KO': 'orange'}
    x_ticks = []
    x_labels = []
    width = 0.5
    offset = 0.75  # spacing between WT and KO

    fig, ax = plt.subplots(figsize=(6, 6))

    for i, key in enumerate(groups_id.keys()):
        
        x = offset * i
        x_ticks.append(x)
        x_labels.append(key)
        proportions = proportions_groups[groups_id[key]]
        mean_proportion = np.mean(proportions) # Bar height: mean

        ax.bar(x, mean_proportion, width=width, color=color[key], edgecolor='black', label=key)
        ax.scatter([x] * len(proportions), proportions, color='black', zorder=10)

    if len(proportions_groups) == 2 :
        # Mann–Whitney U test
        stat, p = mannwhitneyu(proportions_groups[0], proportions_groups[1], alternative='two-sided')
    
        # Annotate p-value
        y_max = max(np.max(proportions_groups[0]), np.max(proportions_groups[1])) + 1
        ax.plot(x_ticks, [y_max, y_max], color='black', linewidth=1.5)
        ax.text(offset/2, y_max + 0.05, f"M-W p = {p:.3g}", ha='center', va='bottom', fontsize=11)

    # Labeling
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, ha='right')
    ax.set_ylabel('Proportion of neurons')
    ax.set_title(f'% of {str(valid_sub_protocols)+'-responsive' if not get_centered else 'centered'} neurons')
    #ax.legend()
    plt.tight_layout()
    
    # save figure
    fig.savefig(os.path.join(save_path, f"{fig_name}_barplot_%responsive.jpeg"), dpi=300)
    plt.show()

def plot_avg_session(groups_id, stim_groups, attr, save_path, fig_name, protocols):
    
    colors = {0: 'skyblue', 1: 'orange', 2: 'salmon', 3: 'grey'}

    fig, ax = plt.subplots(figsize=(8, 7))

    x_ticks = []
    x_labels = []
    width = 0.5
    offset = (len(groups_id.keys()) + 1) * width # spacing between groups
    excel_data = {}
    for i, protocol in enumerate(protocols):

        for k, group in enumerate(groups_id.keys()):
            x = offset * i + width * k
            x_ticks.append(x)
            x_labels.append(f'{protocol}\n{group}')

            stim_mean = stim_groups[groups_id[group]][protocol]
            stim_global_mean = np.mean(stim_mean)
            col_name = f"{group}_{protocol}_session_avg"
            excel_data[col_name] = stim_mean

            ax.bar(x, stim_global_mean, color=colors[k], width=width, edgecolor='black', label=group if i == 0 else "")
            ax.scatter([x] * len(stim_mean), stim_mean, color='black', zorder=10)

        if len(groups_id) == 2 :
            # Mann–Whitney U test
            stat, p = mannwhitneyu(stim_groups[0][protocol], stim_groups[1][protocol], alternative='two-sided')

            # Annotate p-value
            y_max = max(np.max(stim_groups[0][protocol]), np.max(stim_groups[1][protocol])) + 0.2
            ax.plot([x_ticks[2*i], x_ticks[2*i+1]], [y_max, y_max], color='black', linewidth=1.5)
            ax.text(np.mean([x_ticks[2*i], x_ticks[2*i+1]]), y_max + 0.005, f"M-W p = {p:.3g}", ha='center', va='bottom', fontsize=11)

    # Labeling
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_ylabel(f'Mean stimulus response ({attr})')
    ax.set_title('Session-averaged responses during stimulus')
    ax.legend()
    plt.tight_layout()

    fig.savefig(os.path.join(save_path, f"{fig_name}_barplot_stim_response_{attr}.jpeg"), dpi=300)
    plt.show()

    #Save values in Excel
    # Find the longest list length
    max_len = max(len(v) for v in excel_data.values())

    # Pad shorter lists with None
    padded = {k: v + [None]*(max_len - len(v)) for k, v in excel_data.items()}
    excel_path = os.path.join(save_path, f"{fig_name}_avgs_PerSession_{attr}.xlsx")
    pd.DataFrame(padded).to_excel(excel_path, index=False)

def graph_averages(frame_rate, groups_id, fig_name, attr, save_path, protocols, protocol_validity, avg_groups, sem_groups, nb_neurons):
    """
    Function to plot the average z-scores or dFoF0 for responsive neurons.
    """
    groups = list(groups_id.keys())
    # Select colors for each group and protocol
    if len(protocols) == 1:
        colors = {groups[0]: {protocols[0]: 'skyblue'},
                  groups[1]: {protocols[0]: 'orange'}}
    elif len(protocols) == 2:
        colors = {groups[0]: {protocols[0]: 'skyblue', protocols[1]: 'steelblue'},
                  groups[1]: {protocols[0]: 'orange', protocols[1]: 'peru'}}

    # Create a figure with subplots
    fig, axs = plt.subplots(1, 3, figsize=(3*8.5, 7))
    # Build a dict for DataFrame export
    excel_dict = {}
    for i, group in enumerate(groups):
        avg = avg_groups[groups_id[group]]
        sem = sem_groups[groups_id[group]]
        neurons = nb_neurons[groups_id[group]]
        
        # Get minimum length among all protocols
        min_len = min(len(avg[protocol]) for protocol in protocols)

        # Generate time vector accordingly
        time = np.linspace(0, min_len, min_len) / frame_rate - 1  # time in seconds
        # Save time only once
        if "Time (s)" not in excel_dict:
            excel_dict["Time (s)"] = time

        for protocol in protocols:
            # Store data for Excel
            # Add avg & sem columns to dict
            avg_col_name = f"{group}_{protocol}_avg"
            sem_col_name = f"{group}_{protocol}_sem"
            excel_dict[avg_col_name] = avg[protocol][:min_len]
            excel_dict[sem_col_name] = sem[protocol][:min_len]

            axs[2].plot(time, avg[protocol][:min_len], color=colors[group][protocol], label=f"{group}, {protocol}, {neurons} neurons")
            axs[2].fill_between(time,
                            avg[protocol][:min_len] - sem[protocol][:min_len],
                            avg[protocol][:min_len] + sem[protocol][:min_len],
                            color=colors[group][protocol], alpha=0.3)
            
        # plot each group separately
        for protocol in protocols:
            axs[i].plot(time, avg[protocol][:min_len], color=colors[group][protocol], label=f"{group} {protocol}, {neurons} neurons")
            axs[i].fill_between(time,
                            avg[protocol][:min_len] - sem[protocol][:min_len],
                            avg[protocol][:min_len] + sem[protocol][:min_len], color=colors[group][protocol],
                            alpha=0.3)
        axs[i].set_xticks(np.arange(-1, time[-1] + 1, 1))
        axs[i].set_xlabel("Time (s)")
        axs[i].set_ylabel('Average dF/F0 - baseline' if attr == 'dFoF0-baseline' else 'Average z-scored dF/F0')
        axs[i].set_title(f"Average {attr} for {str(protocol_validity)+'-responsive neurons' if not get_centered else 'centered neurons'} for {group}s")
        axs[i].legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    axs[2].set_xticks(np.arange(-1, time[-1] + 1, 1))
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel('Average dF/F0 - baseline' if attr == 'dFoF0-baseline' else 'Average z-scored dF/F0')
    axs[2].set_title(f"Average {attr} for {str(protocol_validity)+'-responsive neurons' if not get_centered else 'centered neurons'} comparing groups")
    axs[2].legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    # Hide the unused subplot (bottom-right)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.2)
    fig.savefig(os.path.join(save_path, f"{fig_name}_averages_{attr}.jpeg"), dpi=300, bbox_inches='tight')
    plt.show()

    # Create DataFrame and save to Excel
    df = pd.DataFrame(excel_dict)
    excel_path = os.path.join(save_path, f"{fig_name}_averages_{attr}.xlsx")
    df.to_excel(excel_path, index=False)

def histplot(sub_protocols, list1, list2, groups, save_path, fig_name, attr, variable="CMI"):
    """
    Function to plot a histogram comparing the distribution of two groups.
    variable: 'CMI' or 'suppression_index'
    """
    edgecolor = 'black'
    medians = []
    if len(sub_protocols) == 2:
        labels_list = []
        if variable == "CMI":
            for l in [list1, list2]:
                bins = [-float('inf'), -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, float('inf')]
                labels = ['<-1.5', '-1.5 to -1.25', '-1.25 to -1', '-1 to -0.75', '-0.75 to -0.5', '-0.5 to -0.25', '-0.25 to 0', '0 to 0.25', '0.25 to 0.5', '0.5 to 0.75', '0.75 to 1', '1 to 1.25', '1.25 to 1.5', '>1.5']
                labeled = pd.cut(l, bins=bins, labels=labels)
                labels_list += labeled.astype(str).tolist()
        elif variable == "suppression_index" and 'center' in sub_protocols:
            for l in [list1, list2]:
                bins = [-float('inf'), -2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, float('inf')]
                labels = ['<-2', '-2 to -1.75', '-1.75 to -1.5', '-1.5 to -1.25', '-1.25 to -1', '-1 to -0.75', '-0.75 to -0.5', '-0.5 to -0.25', '-0.25 to 0', '0 to 0.25', '0.25 to 0.5', '0.5 to 0.75', '0.75 to 1', '1 to 1.25', '1.25 to 1.5', '1.5 to 1.75', '1.75 to 2', '>2']
                labeled = pd.cut(l, bins=bins, labels=labels)
                labels_list += labeled.astype(str).tolist()
        else :
            raise Exception("Variable must be 'CMI' or 'suppression_index', and sub_protocols must contain 'center' for suppression index.")

        # Perform Mann–Whitney U test between WT and KO
        stat_mwu, p_value_mwu = mannwhitneyu(np.array(list1), np.array(list2), alternative='two-sided')
        p_value_1 = wilcoxon(np.array(list1), alternative='two-sided')[1]
        p_value_2 = wilcoxon(np.array(list2), alternative='two-sided')[1]
        
        genotype = [groups[0]] * len(list1) + [groups[1]] * len(list2)
        df = pd.DataFrame({"Genotype" : genotype, variable : pd.Categorical(labels_list, categories=labels, ordered=True)})

        fig, ax = plt.subplots(figsize=(8, 7))
       
        sns.histplot(df, x=variable, hue="Genotype", common_norm=False, shrink=.8, stat="percent", element='step', ax=ax, alpha=0)
        plt.ylabel(f'% of neurons')
        plt.xticks(rotation=45)
        plt.title(f"{variable} for {groups[0]} vs {groups[1]}")
        plt.tight_layout()
        textstr = (
            f'{groups[0]} median = {np.median(list1):.2f}, p (vs 0) = {p_value_1:.3g}\n'
            f'{groups[1]} median = {np.median(list2):.2f}, p (vs 0) = {p_value_2:.3g}\n'
            f'{groups[0]} vs {groups[1]} (Mann–Whitney) p = {p_value_mwu:.3g}'
        )
        # Position textbox on plot
        plt.gca().text(0.99, 0.97, textstr,
                    transform=plt.gca().transAxes,
                    fontsize=11, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85))

        fig.savefig(os.path.join(save_path, f"barplot_{fig_name}_{variable}_{attr}.jpeg"), dpi=300)
        plt.show()

        # Count neurons per bin per genotype
        bin_counts = df.groupby(["Genotype", variable], observed=False).size().reset_index(name="Count")

        # Pivot so that each column is a genotype
        pivot_df = bin_counts.pivot(index=variable, columns="Genotype", values="Count").fillna(0)

        # Optionally convert counts to percentages
        pivot_df_percent = pivot_df.div(pivot_df.sum(axis=0), axis=1) * 100

        # Save percentages to Excel
        with pd.ExcelWriter(os.path.join(save_path, f"{fig_name}_{variable}_{attr}.xlsx")) as writer:
            pivot_df_percent.to_excel(writer, sheet_name="Percentages")
    else:
        print(f"Histogram is not available for {len(sub_protocols)} protocols. Please select 2 protocols to compare.")
        print(f"Current protocols: {sub_protocols}")
        return None
    

def representative_traces(frame_rate, suppression_groups, cmi_groups, magnitude_groups, groups_id,
                          individual_groups, sub_protocols, attr, save_path, fig_name,
                          variable='suppression_index'):
    if not any(cmi_groups) or not any(suppression_groups):
        print("No CMI or suppression index data available to plot representative traces.")
        return
    excel_dict = {}
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    if len(groups_id) == 1:  # ensure ax is iterable
        ax = [ax]

    for i, group in enumerate(groups_id.keys()):
        id_group = groups_id[group]
        if variable == 'suppression_index':
            cmi = suppression_groups[id_group]
        elif variable == 'CMI':
            cmi = cmi_groups[id_group]
        else:
            raise ValueError(f"Unknown variable: {variable}")

        # Find representative trace (closest to median)
        median = np.round(np.median(cmi), 2)
        print(f"Group {group}: median {variable} = {median}")
        cmi_id = np.where(np.round(cmi, 2) == median)[0][0]
        rep_cmi = cmi[cmi_id]

        indiv_traces = individual_groups[id_group]
        rep_trace = {protocol: indiv_traces[protocol][cmi_id] for protocol in sub_protocols}
        max_group = np.max([np.percentile(rep_trace[protocol], 95) for protocol in sub_protocols])  # Get max value for normalization

        # Get minimum length among all protocols
        min_len = min(len(trace) for trace in rep_trace.values())

        # Generate time vector
        time = np.linspace(0, min_len, min_len) / frame_rate - 1
        if "Time (s)" not in excel_dict:
            excel_dict["Time (s)"] = time

        for protocol in sub_protocols:
            col_name = f"_{group}_{protocol}_representative"
            excel_dict[col_name] = rep_trace[protocol][:min_len]
            ax[0,i].plot(time, gaussian_filter1d(rep_trace[protocol][:min_len] / max_group, sigma=1), color = 'skyblue' if protocol == sub_protocols[0] else 'orange',
                       label=f"{group} {protocol}", lw=2)

        ax[0,i].legend()
        leg = ax[0,i].legend()
        # get legend bounding box in axes coordinates
        bb = leg.get_window_extent().transformed(ax[0,i].transAxes.inverted())
        ax[0,i].set_title(f"{group}")
        ax[0,i].set_ylabel(f"{attr} normalized to the highest response")
        ax[0,i].set_xlabel("Time (s)")
        x = bb.x1   # right edge of legend
        y = bb.y0 - 0.05  # just below legend (tweak offset)
        ax[0,i].text(
            0.7, 0.02,
            f"Representative {variable} = {rep_cmi:.2f}",
            transform=ax[0,i].transAxes,
            fontsize=9,
            ha='right', va='bottom',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                    alpha=0.7, edgecolor="none")
        )
        magnitudes = [magnitude_groups[id_group][protocol][cmi_id] for protocol in sub_protocols]
        ax[1,i].bar([protocol for protocol in sub_protocols], magnitudes/np.max(magnitudes), color = ['skyblue', 'orange'], width = 0.3)
        ax[1,i].set_ylabel(f"Response magnitude ({attr}) normalized to the highest response")
        ax[1,i].set_title(f"{group} - Response magnitudes")
    plt.tight_layout()
    fig.savefig(os.path.join(save_path, f"{fig_name}_representative_traces_{variable}_{attr}.jpeg"),
                dpi=300, bbox_inches='tight')
    plt.show()

    # Save to Excel
    df = pd.DataFrame(excel_dict)
    df.to_excel(os.path.join(save_path, f"{fig_name}_representative_traces_{variable}_{attr}.xlsx"),
                index=False)


def plot_cdf_magnitudes(groups_id, magnitude_groups, sub_protocols, attr, file_name, save_path):
    """
    Plots CDFs of neuron response magnitudes for each protocol
    and runs pairwise statistical comparisons.

    Parameters:
    - magnitude: dict, keys are protocol names, values are lists of magnitudes
    - protocols: list of protocols to include
    - attr: string, how fluorescence is measured (e.g., 'z_score' or 'dFoF0-baseline')
    - file_name: string, base name for the saved figure
    - save_path: string, folder where to save the figure
    - get_valid: bool, if True, only responsive neurons were used (affects title and filename)"""
    groups = list(groups_id.keys())
    fname = f"cdf_{file_name}_{attr}_{groups[0]}_vs_{groups[1]}_responsiveNeurons.png"
    title = f'Cumulative distribution of neuron response magnitudes ({attr})\n(Responsive neurons)'
    plt.figure(figsize=(6, 6))
    colors = {groups[0]: 'skyblue',
                groups[1]: 'orange'}
    stats_text = []
    if len(sub_protocols) > 1:
        print(f"Cannot plot CDFs for more than 1 protocol. Please select 1 protocol to compare WT and KOs.")
        return None
    else:
        protocol = sub_protocols[0] 
    
        for group in groups_id.keys():
            group_idx = groups_id[group]

        # Check data availability
            if protocol not in list(magnitude_groups[group_idx].keys()):
                print(f"Protocol '{protocol}' not found in magnitude data for group '{group}'. Skipping.")
                continue
            magnitude = magnitude_groups[groups_id[group]][protocol]

            # Plot CDF
            data = np.array(magnitude)
            data_sorted = np.sort(data)
            cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
            plt.plot(data_sorted, cdf, label=f"{group} {protocol}", color=colors[group], lw=2)

        # Perform statistical test (pairwise Mann–Whitney U)
        p = mannwhitneyu(magnitude_groups[0][protocol], magnitude_groups[1][protocol], alternative='two-sided').pvalue
        # Put text in bottom-right corner of the axes
        plt.text(0.95, 0.05,  # relative position in axes coords
            f"p = {p:.3e}",
            transform=plt.gca().transAxes,
            fontsize=10,
            ha='right', va='bottom',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"))
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

    #-----------------------INPUTS-----------------------#

    excel_sheet_path = r"Y:\raw-imaging\Nathan\PYR\Nathan_sessions_visualpipe.xlsx"
    save_path = r"Y:\raw-imaging\Nathan\PYR\vision_survey\Analysis"
    
    #Will be included in all names of saved figures
    fig_name = 'looming'

    #Name of the protocol to analyze (e.g. 'surround-mod', 'visual-survey'...)
    protocol_name = "vision-survey"

    # Write the protocols you want to plot 
    sub_protocols = ['looming-stim']  
    # List of protocol(s) used to select responsive neurons. If contains several protocols, neurons will be selected if they are responsive to at least one of the protocols in the list.
    valid_sub_protocols = ['looming-stim'] 
    ''''quick-spatial-mapping-center', 'quick-spatial-mapping-left', 'quick-spatial-mapping-right',
        'quick-spatial-mapping-up', 'quick-spatial-mapping-down',
        'quick-spatial-mapping-up-left', 'quick-spatial-mapping-up-right',
        'quick-spatial-mapping-down-left', 'quick-spatial-mapping-down-right'''
    
    #Frame rate
    frame_rate = 30

    # Decide if you want to plot the dFoF0 baseline substraced or the z-scores
    attr = 'dFoF0-baseline'  # 'dFoF0-baseline' or 'z_scores'

    # Decide if you want to only keep neurons that are centered
    get_centered = True

    #----------------------------------------------------#
    df = utils.load_excel_sheet(excel_sheet_path, protocol_name)

    groups_id = {'WT': 0, 'KO': 1}

    suppression_groups, magnitude_groups, stim_groups, nb_neurons, avg_groups, sem_groups, cmi_groups, proportions_groups, individual_groups = process_group(df, groups_id, attr, valid_sub_protocols, sub_protocols, protocol_name, get_centered = False) 
    #representative_traces(frame_rate, suppression_groups, cmi_groups, magnitude_groups, groups_id,
    #                      individual_groups, sub_protocols, attr, save_path, fig_name, variable='CMI')
    

    
     #-------------------Call the functions to process and plot the data-------------------#

    #XY plot of the magnitudes of the responses to the two protocols
    XY_magnitudes(groups_id, magnitude_groups, sub_protocols, valid_sub_protocols, save_path, attr)
    # Plot the % of responsive neurons per session
    plot_perc_responsive(groups_id, proportions_groups, save_path, fig_name)
    # Plot the average response during the stim period per session
    plot_avg_session(groups_id, stim_groups, attr, save_path, fig_name, sub_protocols)
    # Plot the average z-scores or dFoF0-baseline trace for responsive neurons
    graph_averages(frame_rate, groups_id, fig_name, attr, save_path, sub_protocols, valid_sub_protocols, avg_groups, sem_groups, nb_neurons)
    #plot the distribution of CMI 
    histplot(sub_protocols, cmi_groups[0], cmi_groups[1], list(groups_id.keys()), save_path, fig_name, attr, variable = "CMI")
    #plot the distribution of suppression index
    histplot(sub_protocols, suppression_groups[0], suppression_groups[1], list(groups_id.keys()), save_path, fig_name, attr, variable="suppression_index")
    # Plot CDFs of neuron response magnitudes comparing groups
    plot_cdf_magnitudes(groups_id, magnitude_groups, sub_protocols, attr, fig_name, save_path) 
