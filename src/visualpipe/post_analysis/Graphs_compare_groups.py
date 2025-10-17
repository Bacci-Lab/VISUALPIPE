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
                 if protocol in validity
                 else print(f"{protocol} does not exist in validity file.") 
                 for protocol in protocol_list}
    
    valid_neuron_lists = [np.where(data[:, 0] == 1,)[0] for data in valid_data.values()] # change to -1 if you want negative responsive neurons
    valid_neurons = np.unique(np.concatenate(valid_neuron_lists))  # Get unique indices of valid neurons
    
    return valid_neurons

def get_centered_neurons(stimuli_df, neurons_list, trials, attr, plot, frame_rate = 30):
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

def process_group(df, groups_id, attr, valid_sub_protocols, sub_protocols, protocol_name, get_centered, plot):
    #Define trial period names based on attribute
    if attr == 'dFoF0-baseline':
        period_names = ['norm_averaged_baselines', 'norm_trial_averaged_ca_trace', 'norm_post_trial_averaged_ca_trace']
        trial_periods = ['pre_trial_fluorescence', 'trial_fluorescence', 'post_trial_fluorescence']
    elif attr == 'z_scores':
        period_names = ['pre_trial_averaged_zscores', 'trial_averaged_zscores', 'post_trial_averaged_zscores']
        trial_periods = ['pre_trial_zscores', 'trial_zscores', 'post_trial_zscores']

    # Initialize group-level containers
    suppression_groups, magnitude_groups, stim_groups, nb_neurons, avg_groups, sem_groups, cmi_groups, proportions_groups, individual_groups = [], [], [], [], [], [], [], [], []
    perTrials_groups = {group: {} for group in list(groups_id.keys())} #Will contain the average response per trial for each protocol, per group
    mag_trials = {group: {} for group in list(groups_id.keys())} #Will contain the magnitude of the response to each protocol for each trial of each protocol, per group
    sem_trials = {group: {} for group in list(groups_id.keys())} #Will contain the SEM of the response to each protocol for each trial of each protocol, per group

    # Temporary containers to accumulate per-session trial-vectors
    mag_trials_sessions = {group: {protocol: [] for protocol in sub_protocols} for group in list(groups_id.keys())}

    #loop over groups (e.g. WT and KO)
    for key in groups_id.keys():
        perTrials = {protocol: {} for protocol in sub_protocols} #Will contain the average response per trial for each protocol, for one group

        print(f"\n-------------------------- Processing {key} group --------------------------")

        all_neurons = 0
        #Initialize protocol-level containers
        magnitude = {protocol: [] for protocol in sub_protocols} #magnitude of the response to each protocol for each neuron
        avg_data = {protocol: [] for protocol in sub_protocols} #trial-averaged traces per protocol, per session
        stim_mean = {protocol: [] for protocol in sub_protocols} #mean magnitude of response to for each protocol, per session
        single_neurons_group = {protocol: [] for protocol in sub_protocols} #individual traces of each neuron for each protocol
        proportion_list = [] #proportion of responsive neurons per session

        df_filtered = df[df["Genotype"] == key]
        for k in range(len(df_filtered)):
            #get the session path
            mouse_id = df_filtered["Mouse_id"].iloc[k]
            session_id = df_filtered["Session_id"].iloc[k]
            output_id = df_filtered["Output_id"].iloc[k]
            session_path = os.path.join(df_filtered["Session_path"].iloc[k], f"{session_id}_output_{output_id}")

            print(f"\nSession id: {session_id}\n  Mouse id : {mouse_id}\n     Session path: {session_path}")

            validity, trials, stimuli_df = utils.load_data_session(session_path) #load data of that session

            # Special case of looming protocol 100% contrast which has different naming between vision-survey and looming-sweeping-log protocols
            stimuli_df['name'] = stimuli_df['name'].replace('looming-stim', 'looming-stim-log-1.0')
            for stim_name in list(validity.keys()):  #same in the validity file
                if stim_name == 'looming-stim':
                    validity['looming-stim-log-1.0'] = validity.pop(stim_name)

            valid_neurons = get_valid_neurons_session(validity, valid_sub_protocols) #get indices of responsive neurons (based on the validy-protocol(s)) for that session
            
            # filter centered neurons if required
            if not get_centered:
                neurons = len(valid_neurons)
                all_neurons += neurons
                proportion = 100 * len(valid_neurons) / trials[period_names[1]][0].shape[0]
                proportion_list.append(proportion)
                print(f"Proportion responding neurons: {proportion}, Number of responsive neurons: {neurons}")
            else:
                centered_neurons, non_centered = get_centered_neurons(stimuli_df, valid_neurons, trials, attr, plot, frame_rate = 30)
                all_neurons += len(centered_neurons)
                proportion = 100 * len(centered_neurons) / trials[period_names[1]][0].shape[0]
                proportion_list.append(proportion)
                print(f"Proportion of centered neurons: {proportion}, Number of centered neurons: {len(centered_neurons)}")
                valid_neurons = centered_neurons

            
            for protocol in sub_protocols:
                
                stim_id = stimuli_df[stimuli_df.name == protocol].index[0]
                n_trials = trials[trial_periods[1]][stim_id].shape[1]

                # Get traces from responsive-neurons for that protocol from pre, stim and post periods and concatenate along time
                traces_sep = [trials[period][stim_id][valid_neurons, :] for period in period_names]
                traces_concat = np.concatenate(traces_sep, axis=1)
                # Merge individual traces into group-level container
                if len(single_neurons_group[protocol]) == 0:
                    single_neurons_group[protocol] = traces_concat
                else:
                    min_len = min(single_neurons_group[protocol].shape[1], traces_concat.shape[1])
                    single_neurons_group[protocol] = np.vstack([
                        single_neurons_group[protocol][:, :min_len],
                        traces_concat[:, :min_len]])
                    #single_neurons_group[protocol] = np.vstack([single_neurons_group[protocol], traces_concat])

                avg_session_trace = np.mean(traces_concat, axis=0) # average trace of all neurons in that session and for that protocol
                avg_data[protocol].append(avg_session_trace)

                # Calculate the average response magnitude across all neurons in that session (mean of response from 0.5s after stim onset to end of stim)
                stim_traces = trials[period_names[1]][stim_id][valid_neurons, int(frame_rate*0.5):]
                mean_trial_value_per_neurons = np.mean(stim_traces, axis=1)  # average per neuron over time, 
                mean_trial_value_session = np.mean(mean_trial_value_per_neurons)  # average over neurons in that session
                stim_mean[protocol].append(mean_trial_value_session)
                
                #Store the average response of each neuron to that protocol
                magnitude[protocol].append(mean_trial_value_per_neurons)
                # Now compute the average response per trial for each neuron, subtracting the baseline of that trial
                session_mag_list = []
                for trial in range(0,n_trials):
                    avg_trial = []
                    # Only compute baseline if attr == 'dFoF0-baseline'
                    if attr == 'dFoF0-baseline':
                        baseline = np.mean(trials['pre_trial_fluorescence'][stim_id][valid_neurons, trial, :], axis=1)

                    for trial_period in trial_periods:
                        trial_trace = trials[trial_period][stim_id][valid_neurons, trial, :]
                        if attr == 'dFoF0-baseline':
                            trial_trace_baselined = trial_trace - baseline[:, np.newaxis]
                        elif attr == 'z_scores':
                            trial_trace_baselined = trial_trace  # already normalized
                        if trial_period == trial_periods[1]:  # only compute magnitude during stimulus period
                            response_magnitudes = np.mean(trial_trace_baselined[:, int(frame_rate * 0.5):], axis=1)
                            mag_trial = np.mean(response_magnitudes)
                            session_mag_list.append(mag_trial)
                        avg_period = np.mean(trial_trace_baselined, axis=0) # average over neurons of the trial period (pre, stim or post)
                        avg_trial.append(avg_period)
                    avg_trial = np.concatenate(avg_trial, axis=0) #concatenate all periods of that trial
                    perTrials[protocol][trial] = avg_trial # store the average response of that trial for that protocol
                mag_trials_sessions[key][protocol].append(session_mag_list)
        perTrials_groups[key] = perTrials # store the average response per trial for each protocol, for that group
        
        
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
            #avg_data[protocol] = np.stack(avg_data[protocol], axis=0)
            min_len = min(arr.shape[-1] for arr in avg_data[protocol])

            # Truncate all arrays to that length
            trimmed_arrays = [arr[..., :min_len] for arr in avg_data[protocol]]

            # Now stack safely
            avg_data[protocol] = np.stack(trimmed_arrays, axis=0)
            sessions_mag = mag_trials_sessions[key][protocol]  # list of lists: n_sessions x n_trials

            # Convert to array (n_sessions, n_trials)
            mag_arr = np.array(sessions_mag)  # shape (n_sessions, n_trials)
            # Compute mean across sessions for each trial
            mag_mean_per_trial = np.mean(mag_arr, axis=0)
            # Compute SEM across sessions for each trial (use ddof=0; stats.sem does nansafe handling)
            mag_sem_per_trial = stats.sem(mag_arr, axis=0)

            mag_trials[key][protocol] = mag_mean_per_trial.tolist()
            sem_trials[key][protocol] = mag_sem_per_trial.tolist()


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
        print(mag_trials)
        

    return suppression_groups, magnitude_groups, stim_groups, nb_neurons, avg_groups, sem_groups, cmi_groups, proportions_groups, individual_groups, perTrials_groups, mag_trials, sem_trials

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
    Function to plot the % of responsive neurons per session for WT and KO groups,
    and save the values in an Excel file.
    """
    palette = ['skyblue', 'orange', 'green', 'red']  # add more colors if needed
    group_keys = list(groups_id.keys())
    color = {k: palette[i] for i, k in enumerate(group_keys)}
    x_ticks = []
    x_labels = []
    width = 0.5
    offset = 0.75  # spacing between WT and KO

    fig, ax = plt.subplots(figsize=(6, 6))
    results = []

    for i, key in enumerate(groups_id.keys()):
        x = offset * i
        x_ticks.append(x)
        x_labels.append(key)
        proportions = proportions_groups[groups_id[key]]
        mean_proportion = np.mean(proportions)  # Bar height: mean

        # Plot
        ax.bar(x, mean_proportion, width=width, color=color[key], edgecolor='black', label=key)
        ax.scatter([x] * len(proportions), proportions, color='black', zorder=10)

        # Store values
        for p in proportions:
            results.append({
                "Group": key,
                "Proportion": p
            })
        results.append({
            "Group": key,
            "Proportion": "Mean",
            "Value": mean_proportion
        })

    # Statistics
    stats_result = {}
    if len(proportions_groups) == 2:
        stat, p = mannwhitneyu(proportions_groups[0], proportions_groups[1], alternative='two-sided')
        stats_result = {"Test": "Mann-Whitney U", "U-statistic": stat, "p-value": p}

        # Annotate p-value
        y_max = max(np.max(proportions_groups[0]), np.max(proportions_groups[1])) + 1
        ax.plot(x_ticks, [y_max, y_max], color='black', linewidth=1.5)
        ax.text(offset/2, y_max + 0.05, f"M-W p = {p:.3g}", ha='center', va='bottom', fontsize=11)

    # Labeling
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, ha='right')
    ax.set_ylabel('Proportion of neurons')
    ax.set_title(f'% of {str(valid_sub_protocols)+"-responsive" if not get_centered else "centered"} neurons')
    plt.tight_layout()

    # Save figure
    fig.savefig(os.path.join(save_path, f"{fig_name}_barplot_%responsive.jpeg"), dpi=300)
    plt.show()

    # Save Excel
    df = pd.DataFrame(results)
    excel_path = os.path.join(save_path, f"{fig_name}_%responsive.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Values", index=False)
        if stats_result:
            pd.DataFrame([stats_result]).to_excel(writer, sheet_name="Statistics", index=False)

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
    group_palette = ['skyblue', 'orange', 'green', 'red', 'purple']  
    protocol_palette = ['steelblue', 'peru', 'yellow', 'pink']
    #create a figure with subplots 
    n_groups = len(groups)
    n_subplots = n_groups + 1  if n_groups >= 2 else n_groups
    fig, axs = plt.subplots(1, n_subplots, figsize=(n_subplots*8.5, 7))
    # Ensure axs is always a 1D array, even if n_subplots == 1
    if n_subplots == 1:
        axs = np.array([axs])

    # Build a dict for DataFrame export
    excel_dict = {}
    colors = {}
    for i, group in enumerate(groups):
        colors[group] = {}
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
        if n_groups>=2:
            for protocol in protocols:
                group_color = group_palette[i % len(group_palette)]
                # Store data for Excel
                # Add avg & sem columns to dict
                avg_col_name = f"{group}_{protocol}_avg"
                sem_col_name = f"{group}_{protocol}_sem"
                excel_dict[avg_col_name] = avg[protocol][:min_len].tolist()
                excel_dict[sem_col_name] = sem[protocol][:min_len].tolist()

                axs[-1].plot(time, avg[protocol][:min_len], color=group_color, label=f"{group}, {protocol}, {neurons} neurons")
                axs[-1].fill_between(time,
                                avg[protocol][:min_len] - sem[protocol][:min_len],
                                avg[protocol][:min_len] + sem[protocol][:min_len],
                                color=group_color, alpha=0.3)
            
        # plot each group separately
        for j,protocol in enumerate(protocols):
            if n_groups ==1:
                avg_col_name = f"{group}_{protocol}_avg"
                sem_col_name = f"{group}_{protocol}_sem"
                excel_dict[avg_col_name] = avg[protocol][:min_len].tolist()
                excel_dict[sem_col_name] = sem[protocol][:min_len].tolist()
            protocol_color = protocol_palette[j % len(protocol_palette)]
            axs[i].plot(time, avg[protocol][:min_len], color=protocol_color, label=f"{group} {protocol}, {neurons} neurons")
            axs[i].fill_between(time,
                            avg[protocol][:min_len] - sem[protocol][:min_len],
                            avg[protocol][:min_len] + sem[protocol][:min_len], color=protocol_color,
                            alpha=0.3)
        axs[i].set_xticks(np.arange(-1, time[-1] + 1, 1))
        axs[i].set_xlabel("Time (s)")
        axs[i].set_ylabel('Average dF/F0 - baseline' if attr == 'dFoF0-baseline' else 'Average z-scored dF/F0')
        axs[i].set_title(f"Average {attr} for {str(protocol_validity)+'-responsive neurons' if not get_centered else 'centered neurons'} for {group}s")
        axs[i].legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    if n_groups>=2:
        axs[-1].set_xticks(np.arange(-1, time[-1] + 1, 1))
        axs[-1].set_xlabel("Time (s)")
        axs[-1].set_ylabel('Average dF/F0 - baseline' if attr == 'dFoF0-baseline' else 'Average z-scored dF/F0')
        axs[-1].set_title(f"Average {attr} for {str(protocol_validity)+'-responsive neurons' if not get_centered else 'centered neurons'} comparing groups")
        axs[-1].legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    # Hide the unused subplot (bottom-right)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.2)
    fig.savefig(os.path.join(save_path, f"{fig_name}_averages_{attr}.jpeg"), dpi=300, bbox_inches='tight')
    plt.show()

    # Create DataFrame and save to Excel
    df = pd.DataFrame(excel_dict)
    excel_path = os.path.join(save_path, f"{fig_name}_averages_{attr}.xlsx")
    df.to_excel(excel_path, index=False)

"""

def plot_per_trial(groups_id, nb_neurons, perTrials_groups, sub_protocols, frame_rate, dt_prestim, fig_name, attr, save_path):
    
    Function to plot the average z-scores or dF/F0 - baseline per trial for all neurons if get_valid is False or only responsive neurons if get_valid is True.
    
    
    excel_dict = {}
    for group in groups_id.keys():
        # One subplot per protocol
        fig, ax = plt.subplots(1, len(sub_protocols), figsize=(6 * len(sub_protocols), 6), squeeze=False)
        ax = ax[0]
        file1 = f"{group}_perTrial_{fig_name}_{attr}_responsive.jpeg"
        neurons = nb_neurons[groups_id[group]]
        for i, protocol in enumerate(sub_protocols):
            time = np.linspace(0, len(perTrials_groups[group][protocol][0]), len(perTrials_groups[group][protocol][0]))
            time = time/frame_rate - dt_prestim # Shift the time by the duration of the baseline so the stimulus period starts at 0
            excel_dict['Time (s)'] = time
            excel_dict[f'nb_neurons_{protocol}'] = [neurons] + ['']*(len(time)-1)
            n_trials = perTrials_groups[group][protocol].keys().__len__()
            print(n_trials)
            for trial in range(n_trials):
                trial_trace = perTrials_groups[group][protocol][trial]
                excel_dict[f'{group}_{protocol}_trial{int(trial)+1}'] = trial_trace
                ax[i].plot(time, trial_trace, label=f'Trial {int(trial)+1}')
            ax[i].set_title(f'{protocol}')
            ax[i].axvline(0, color='k', linestyle='--', linewidth=1)
            ax[i].set_xlabel("Time (s)")
            ax[i].set_ylabel(f"Average {attr}")
            ax[i].legend()
            ax[i].set_xticks(np.arange(-1, round(time[-1]) + 1, 1))
        plt.suptitle(f"Mean {attr} ({neurons} responsive neurons) per trial for {group}s")
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, file1), dpi=300)
        plt.show()

        # Create DataFrame and save to Excel
        pd.DataFrame(excel_dict).to_excel(os.path.join(save_path, file1.replace('.jpeg', '.xlsx')), index=False)"""

def plot_per_trial(groups_id, nb_neurons, perTrials_groups, sub_protocols, frame_rate, dt_prestim, fig_name, attr, save_path):
    """
    Function to plot the average z-scores or dF/F0 - baseline per trial for all neurons if get_valid is False or only responsive neurons if get_valid is True.
    """
    excel_dict = {}
    for group in groups_id.keys():
        # One subplot per protocol
        fig, ax = plt.subplots(1, len(sub_protocols), figsize=(6 * len(sub_protocols), 6), squeeze=False)
        ax = ax[0]
        file1 = f"{group}_perTrial_{fig_name}_{attr}_responsive.jpeg"
        neurons = nb_neurons[groups_id[group]]

        for i, protocol in enumerate(sub_protocols):
            # Get trial traces
            trials = list(perTrials_groups[group][protocol].values())

            # Find minimal trial length
            min_len = min(len(tr) for tr in trials)
            # Truncate all to same length
            trials = [tr[:min_len] for tr in trials]

            # Build time vector
            time = np.linspace(0, min_len, min_len) / frame_rate - dt_prestim

            # Save to excel_dict (note: same length for all)
            excel_dict['Time (s)'] = time
            excel_dict[f'nb_neurons_{protocol}'] = [neurons] + [''] * (len(time) - 1)

            n_trials = len(trials)
            print(f"{group} - {protocol}: {n_trials} trials (length={min_len})")

            for trial_idx, trial_trace in enumerate(trials):
                excel_dict[f'{group}_{protocol}_trial{trial_idx + 1}'] = trial_trace
                ax[i].plot(time, trial_trace, label=f'Trial {trial_idx + 1}')

            ax[i].set_title(f'{protocol}')
            ax[i].axvline(0, color='k', linestyle='--', linewidth=1)
            ax[i].set_xlabel("Time (s)")
            ax[i].set_ylabel(f"Average {attr}")
            ax[i].legend()
            ax[i].set_xticks(np.arange(-1, round(time[-1]) + 1, 1))

        plt.suptitle(f"Mean {attr} ({neurons} responsive neurons) per trial for {group}s")
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, file1), dpi=300)
        plt.show()

        # ✅ Ensure consistent column lengths
        min_len_all = min(len(v) for v in excel_dict.values())
        excel_dict = {k: v[:min_len_all] for k, v in excel_dict.items()}

        # Create DataFrame and save to Excel
        pd.DataFrame(excel_dict).to_excel(
            os.path.join(save_path, file1.replace('.jpeg', '.xlsx')),
            index=False
        )

def magnitude_per_trial(fig_name, save_path, nb_neurons, mag_trials, sem_trials, sub_protocols, groups_id):
    """
    Plot the average response magnitude (mean z-score or dF/F0 during stimulus)
    for each trial, one subplot per protocol.
    Each group is plotted side-by-side for comparison.
    Also saves ONE Excel file with one sheet per protocol.

    Parameters
    ----------
    fig_name : str
        Identifier for the saved files.
    save_path : str
        Folder where plots and Excel files are saved.
    nb_neurons : list
        List containing the number of neurons per group.
    mag_trials : dict
        mag_trials[group][protocol] = list of mean magnitudes per trial.
    sem_trials : dict
        sem_trials[group][protocol] = list of SEMs per trial.
    sub_protocols : list
        List of protocol names.
    groups_id : dict
        Dictionary of groups, e.g., {'WT': 0, 'KO': 1}.
    get_valid : bool
        Whether only valid (responsive) neurons were used.
    """

    fig, ax = plt.subplots(1, len(sub_protocols), figsize=(6 * len(sub_protocols), 6), squeeze=False)
    ax = ax[0]

    file_base = f"perTrial_magnitude_{fig_name}"
    excel_path = os.path.join(save_path, f"{file_base}.xlsx")

    # Colors and bar settings
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    bar_width = 0.35

    # Dictionary to hold DataFrames per protocol for Excel export
    excel_dfs = {}

    # Loop over protocols

    for i, protocol in enumerate(sub_protocols):
        trial_numbers = None
        excel_rows = []  # Each row = one group (e.g., WT, KO)
        columns = []

        for g_idx, group in enumerate(groups_id.keys()):
            magnitude = mag_trials[group][protocol]
            sem = sem_trials[group][protocol]
            neuron_count = nb_neurons[g_idx]
            n_trials = len(magnitude)
            trial_numbers = np.arange(1, n_trials + 1)

            # Plot bars side by side
            ax[i].bar(
                trial_numbers + g_idx * bar_width,
                magnitude,
                yerr=sem,
                capsize=4,
                width=bar_width,
                alpha=0.7,
                label=group,
                color=colors[g_idx % len(colors)]
            )

            # Build one Excel row (mean, sem pairs + neuron count)
            group_row = []
            for t in range(n_trials):
                group_row += [magnitude[t], sem[t]]
                # Build column names only once (first group)
                if g_idx == 0:
                    columns += [f"mean_trial{t+1}", f"sem_trial{t+1}"]

            group_row.append(neuron_count)
            excel_rows.append(group_row)

        # Add neuron count column name
        columns.append("nb_neurons")


        excel_dfs[protocol] = pd.DataFrame(excel_rows, index=list(groups_id.keys()), columns=columns)

        # Format axes
        ax[i].set_title(protocol)
        ax[i].set_xlabel("Trial")
        ax[i].set_ylabel("Mean Response Magnitude")
        ax[i].set_xticks(trial_numbers + bar_width / 2)
        ax[i].set_xticklabels(trial_numbers)
        ax[i].grid(alpha=0.3)
        ax[i].legend(title="Group")

    # --- Save figure ---
    plt.suptitle(f"Response Magnitude per Trial")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{file_base}.jpeg"), dpi=300)
    plt.show()

    # --- Save one Excel file with multiple sheets ---
    with pd.ExcelWriter(excel_path) as writer:
        for protocol, df in excel_dfs.items():
            df.to_excel(writer, sheet_name=protocol, index_label='Group')

    print(f"✅ Plots and Excel file saved:\n{excel_path}")


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
    fname = f"cdf_{file_name}_{attr}_responsiveNeurons.png"
    title = f'Cumulative distribution of neuron response magnitudes ({attr})\n(Responsive neurons)'
    plt.figure(figsize=(6, 6))
    palette = ['skyblue', 'orange', 'green', 'red', 'purple']
    colors = {group: palette[i % len(palette)] for i, group in enumerate(groups)}
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
        if len(groups) == 2:
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

    excel_sheet_path = r"Y:\raw-imaging\Nathan\Nathan_sessions_visualpipe.xlsx"
    save_path = r"Y:\raw-imaging\Nathan\PYR\Visualpipe_postanalysis\looming-sweeping-log\Analysis"
    
    #Will be included in all names of saved figures
    fig_name = 'looming-stim_test'

    #Name of the protocol to analyze (e.g. 'surround-mod', 'visual-survey'...)
    protocol_name = "looming-sweeping-log"

    # Write the protocols you want to plot 
    sub_protocols = ['looming-stim-log-1.0']  
    # List of protocol(s) used to select responsive neurons. If contains several protocols, neurons will be selected if they are responsive to at least one of the protocols in the list.
    valid_sub_protocols = ['looming-stim-log-1.0'] 
    '''quick-spatial-mapping-center', 'quick-spatial-mapping-left', 'quick-spatial-mapping-right',
        'quick-spatial-mapping-up', 'quick-spatial-mapping-down',
        'quick-spatial-mapping-up-left', 'quick-spatial-mapping-up-right',
        'quick-spatial-mapping-down-left', 'quick-spatial-mapping-down-right'''
    
    
    #Frame rate
    frame_rate = 30

    #dt_prestim: duration of the baseline period before stimulus onset (in seconds)
    dt_prestim = 1

    # Decide if you want to plot the dFoF0 baseline substraced or the z-scores
    attr = 'dFoF0-baseline'  # 'dFoF0-baseline' or 'z_scores'

    # Decide if you want to only keep neurons that are centered
    get_centered = False  # True or False

    #----------------------------------------------------#
    df = utils.load_excel_sheet(excel_sheet_path, protocol_name)

    groups_id = {'WT': 0, 'KO': 1}  # keys are group names, e.g 'WT': 0, 'KO': 1

    suppression_groups, magnitude_groups, stim_groups, nb_neurons, avg_groups, sem_groups, cmi_groups, proportions_groups, individual_groups, perTrials_groups, mag_trials, sem_trials = process_group(df, groups_id, attr, valid_sub_protocols, sub_protocols, protocol_name, get_centered = get_centered, plot=False) 
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
    if len(list(groups_id.keys())) == 2 and len(sub_protocols) == 2:
        histplot(sub_protocols, cmi_groups[0], cmi_groups[1], list(groups_id.keys()), save_path, fig_name, attr, variable = "CMI")
    #plot the distribution of suppression index
    if len(list(groups_id.keys())) == 2 and 'center' in sub_protocols:
        histplot(sub_protocols, suppression_groups[0], suppression_groups[1], list(groups_id.keys()), save_path, fig_name, attr, variable="suppression_index")
    # Plot CDFs of neuron response magnitudes comparing groups
    plot_cdf_magnitudes(groups_id, magnitude_groups, sub_protocols, attr, fig_name, save_path) 
    plot_per_trial(groups_id, nb_neurons, perTrials_groups, sub_protocols, frame_rate, dt_prestim, fig_name, attr, save_path)
    magnitude_per_trial(fig_name, save_path, nb_neurons, mag_trials, sem_trials, sub_protocols, groups_id)
