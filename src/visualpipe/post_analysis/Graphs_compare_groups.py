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
import re
sys.path.append("./src")

import visualpipe.post_analysis.utils as utils

# --------------------- Data loading and selection ----------------------------- #

def load_session_data(session_path):
    validity, trials, stimuli_df = utils.load_data_session(session_path)

    # Rename looming protocol if needed
    stimuli_df['name'] = stimuli_df['name'].replace('looming-stim', 'looming-stim-log-1.0')
    if 'looming-stim' in validity:
        validity['looming-stim-log-1.0'] = validity.pop('looming-stim')

    return validity, trials, stimuli_df

def get_period_names(attr):
    if attr == 'dFoF0-baseline':
        period_names = ['norm_averaged_baselines', 'norm_trial_averaged_ca_trace', 'norm_post_trial_averaged_ca_trace']
        trial_periods = ['pre_trial_fluorescence', 'trial_fluorescence', 'post_trial_fluorescence']
    elif attr == 'z_scores':
        period_names = ['pre_trial_averaged_zscores', 'trial_averaged_zscores', 'post_trial_averaged_zscores']
        trial_periods = ['pre_trial_zscores', 'trial_zscores', 'post_trial_zscores']
    else:
        raise ValueError(f"Unknown attribute: {attr}")
    return period_names, trial_periods



def get_valid_neurons(validity, protocol_groups, selection_method='any', group_name='looming'):
    """
    Select responsive neurons using a single input format: 
    protocol_groups = {group_name: [list of protocols], ...}
    
    selection_method:
        'any'  -> neurons responsive to at least one protocol in any group
        'only' -> neurons exclusive to a specific group (provide group_name)
        'and'  -> neurons shared between groups
    """
    if selection_method not in ('any', 'only', 'and'):
        raise ValueError("selection_method must be 'any', 'only', or 'and'.")
    
    if (selection_method == 'only' or selection_method == 'any') and group_name is None:
        raise ValueError("For 'only' and 'any', you must provide group_name.")

    exclusives, intersections, total_union = populations_overlap(validity, protocol_groups)

    if selection_method == 'any':
        return np.array(list(total_union))
    elif selection_method == 'only':
        if group_name not in protocol_groups:
            raise ValueError(f"'{group_name}' not in protocol_groups.")
        return np.array(list(exclusives[group_name]))
    elif selection_method == 'and':
        all_intersections = set().union(*intersections.values())
        return np.array(list(all_intersections))

def populations_overlap(validity, protocol_groups):
    group_neurons = {}
    for group_name, protocols in protocol_groups.items():
        combined = set()
        for prot in protocols:
            print(prot)
            if prot in validity:
                data = validity[prot]
                valid_neurons = set(np.where(data[:, 0] == 1)[0])
                combined |= valid_neurons
            else:
                print(f"{prot} does not exist in validity file.")
        group_neurons[group_name] = combined

    group_names = list(group_neurons.keys())
    exclusives = {}
    intersections = {}

    # Compute exclusives
    for g_name in group_names:
        other_sets = [group_neurons[other] for other in group_names if other != g_name]
        only_g = group_neurons[g_name] - set().union(*other_sets)
        exclusives[g_name] = only_g

    # Compute pairwise intersections
    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            g_i = group_names[i]
            g_j = group_names[j]
            overlap = group_neurons[g_i] & group_neurons[g_j]
            key = f"{g_i} & {g_j}"
            intersections[key] = overlap

    total_union = set().union(*group_neurons.values())
    return exclusives, intersections, total_union


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
    period_names, _ = get_period_names(attr)
           
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

def select_neurons(validity, valid_sub_protocols, selection_method, group_name,
                   get_centered, stimuli_df, trials, period_names, attr, plot):
    valid_neurons = get_valid_neurons(validity, valid_sub_protocols,
                                      selection_method=selection_method,
                                      group_name=group_name)
    
    # Convert to 1D int array
    if isinstance(valid_neurons, set):
        valid_neurons = np.array(list(valid_neurons), dtype=int)
    else:
        valid_neurons = np.atleast_1d(valid_neurons).astype(int)
    
    if get_centered:
        centered_neurons, non_centered = get_centered_neurons(stimuli_df, valid_neurons, trials,
                                                             attr, plot, frame_rate=30)
        valid_neurons = centered_neurons
    
    proportion = 100 * len(valid_neurons) / trials[period_names[1]][0].shape[0]
    return valid_neurons, proportion


# --------------------- Compute variables ----------------------------- #

def compute_magnitude(frame_rate, trace, magnitude_method = 'mean'):
    if magnitude_method == 'mean':
        trace = trace[:,int(0.5*frame_rate):]
        magnitudes = np.mean(trace, axis=1)

    elif magnitude_method == 'peak':
        magnitudes = np.max(trace, axis=1)

    elif magnitude_method == 'filtered_peak':
        trace = gaussian_filter1d(trace, sigma=1, axis=1)
        magnitudes = np.max(trace, axis=1)

    elif magnitude_method == 'auc':
        magnitudes = np.trapezoid(trace, axis=1, dx=1/frame_rate)
    else:
        raise('You need to use an available amplitude calculation method: mean, peak, filtered_peak, auc')
    
    return magnitudes



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


def preferred_contrast(groups_id, mag_per_session, sub_protocols):
    """
    Calculate, for each group and protocol, the percentage of neurons that prefer that protocol per session.
    
    Returns:
        preferred_contrasts_grouped[group][protocol] = list of percentages across sessions
    """
    preferred_contrasts_grouped = {group: {contrast: [] for contrast in sub_protocols} for group in groups_id.keys()}
    for group in groups_id.keys():
        group_idx = groups_id[group]
        group_sessions = mag_per_session[group_idx]
        n_sessions = len(group_sessions[sub_protocols[0]])

        for session_idx in range(n_sessions):
            # Initialize counter for each protocol
            preferred_counts = {contrast: 0 for contrast in sub_protocols}

            # neuron_mags: n_neurons x n_protocols
            neuron_mags = np.array([group_sessions[protocol][session_idx] for protocol in sub_protocols]).T
            neurons = neuron_mags.shape[0]
            print(neurons)

            # Count preferred protocol per neuron
            for neuron in range(neurons):
                magnitudes = neuron_mags[neuron]
                preferred_contrast = sub_protocols[np.argmax(magnitudes)]
                preferred_counts[preferred_contrast] += 1

            # Convert counts to percentages and store
            for contrast in sub_protocols:
                pct = 100 * preferred_counts[contrast] / neurons if neurons > 0 else 0
                preferred_contrasts_grouped[group][contrast].append(pct)

    return preferred_contrasts_grouped


def adaptation_index(sub_protocols, groups_id, mag_trial_indiv, first=3, last=3):
    groups = list(groups_id.keys())
    AI = {group: {} for group in list(groups_id.keys())}
    for group in groups:
        for protocol in sub_protocols:
            magnitudes = mag_trial_indiv[group][protocol]
            if magnitudes.shape[1] < first + last:
                raise ValueError(f"Number of trials in early and late exceeds the total number of trials. n_trials: {magnitudes.shape[1]}")
            early = magnitudes[:, :first]
            early = np.mean(early, axis=1)
            late = magnitudes[:, -last:]
            late = np.mean(late, axis=1)
            index = np.divide(early - late, early + late, out=np.zeros_like(early), where=(early + late)!=0)
            AI[group][protocol]=index # adaptation index values range from -1 to 1
    
    return AI


def process_group(df, groups_id, attr, valid_sub_protocols, sub_protocols, protocol_name, selection_method, group_name, magnitude_method, get_centered, plot):
    #Define trial period names based on attribute
    period_names, trial_periods = get_period_names(attr)
    # Initialize group-level containers
    suppression_groups, magnitude_groups, stim_groups, nb_neurons, avg_groups, sem_groups, cmi_groups, proportions_groups, individual_groups, mag_per_session = [], [], [], [], [], [], [], [], [], []
    perTrials_groups = {group: {} for group in list(groups_id.keys())} #Will contain the average response per trial for each protocol, per group
    mag_trials = {group: {} for group in list(groups_id.keys())} #Will contain the magnitude of the response to each protocol for each trial of each protocol, per group
    sem_trials = {group: {} for group in list(groups_id.keys())} #Will contain the SEM of the response to each protocol for each trial of each protocol, per group
    mag_trial_indiv = {group: {} for group in list(groups_id.keys())} # individual magnitudes per trials per neuron for all protocols

    # Temporary containers to accumulate per-session trial-vectors
    mag_trials_sessions = {group: {protocol: [] for protocol in sub_protocols} for group in list(groups_id.keys())}

    #loop over groups (e.g. WT and KO)
    for key in groups_id.keys():
        df_filtered = df[df["Genotype"] == key] 

        print(f"\n-------------------------- Processing {key} group --------------------------")

        all_neurons = 0
        #Initialize protocol-level containers
        magnitude = {protocol: [] for protocol in sub_protocols} #magnitude of the response to each protocol for each neuron
        avg_data = {protocol: [] for protocol in sub_protocols} #trial-averaged traces per protocol, per session
        stim_mean = {protocol: [] for protocol in sub_protocols} #mean magnitude of response to for each protocol, per session
        single_neurons_group = {protocol: [] for protocol in sub_protocols} #individual traces of each neuron for each protocol
        single_neurons_session = {protocol: [] for protocol in sub_protocols}  #individual traces of each neuron for each protocol, per session
        proportion_list = [] #proportion of responsive neurons per session
        perTrials = {protocol: {} for protocol in sub_protocols} #Will contain the average response per trial for each protocol, for one group

        
        for k in range(len(df_filtered)):
            #get the session path
            mouse_id = df_filtered["Mouse_id"].iloc[k]
            session_id = df_filtered["Session_id"].iloc[k]
            output_id = df_filtered["Output_id"].iloc[k]
            session_path = os.path.join(df_filtered["Session_path"].iloc[k], f"{session_id}_output_{output_id}")

            print(f"\nSession id: {session_id}\n  Mouse id : {mouse_id}\n     Session path: {session_path}")

            validity, trials, stimuli_df = load_session_data(session_path)

            valid_neurons, proportion = select_neurons(validity, valid_sub_protocols, selection_method, group_name,
                   get_centered, stimuli_df, trials, period_names, attr, plot) #extract responsive neurons (and centered if get_centered = True)
            
            all_neurons+=len(valid_neurons)
            proportion_list.append(proportion)
            print(f"Proportion of centered responsive neurons: {proportion}, Number of centered responsive neurons: {len(valid_neurons)}"
                    if get_centered else
                    f"Proportion of responsive neurons: {proportion}, Number of responsive neurons: {len(valid_neurons)}")

            for protocol in sub_protocols:
                
                stim_id = stimuli_df[stimuli_df.name == protocol].index[0]
                n_trials = trials[trial_periods[1]][stim_id].shape[1]
                all_magnitudes = np.zeros((len(valid_neurons), n_trials))  # each row = a neuron, each column = trial

                # Get traces from responsive-neurons for that protocol from pre, stim and post periods and concatenate along time
                traces_sep = [trials[period][stim_id][valid_neurons, :] for period in period_names]
                traces_concat = np.concatenate(traces_sep, axis=1)
                magnitude_per_neuron = compute_magnitude(frame_rate, traces_concat, magnitude_method)
                single_neurons_session[protocol].append(magnitude_per_neuron)  
                # Merge individual traces into group-level container
                if len(single_neurons_group[protocol]) == 0:
                    single_neurons_group[protocol] = traces_concat
                else:
                    min_len = min(single_neurons_group[protocol].shape[1], traces_concat.shape[1])
                    single_neurons_group[protocol] = np.vstack([
                        single_neurons_group[protocol][:, :min_len],
                        traces_concat[:, :min_len]])

                avg_session_trace = np.mean(traces_concat, axis=0) # average trace of all neurons in that session and for that protocol
                avg_data[protocol].append(avg_session_trace)

                stim_traces = trials[period_names[1]][stim_id][valid_neurons, :] #extract traces in the stim period for all neurons in that session
                mag_per_neuron = compute_magnitude(frame_rate, stim_traces, magnitude_method)
                # Calculate the average response magnitude across all neurons in that session (method determined by magnitude_method)
                mean_magnitude_session = np.mean(mag_per_neuron)  # average over neurons in that session
                stim_mean[protocol].append(mean_magnitude_session)
                #Store the average response of each neuron to that protocol
                magnitude[protocol].append(mag_per_neuron) #magnitude values for this protocol per neuron

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
                            response_magnitudes = compute_magnitude(frame_rate, trial_trace_baselined, magnitude_method)
                            all_magnitudes[:, trial] = response_magnitudes
                            mag_trial = np.mean(response_magnitudes)
                            session_mag_list.append(mag_trial)
                        avg_period = np.mean(trial_trace_baselined, axis=0) # average over neurons of the trial period (pre, stim or post)
                        avg_trial.append(avg_period)
                    avg_trial = np.concatenate(avg_trial, axis=0) #concatenate all periods of that trial
                    perTrials[protocol][trial] = avg_trial # store the average response of that trial for that protocol
                mag_trials_sessions[key][protocol].append(session_mag_list)
                if protocol not in mag_trial_indiv[key]:
                    mag_trial_indiv[key][protocol] = [all_magnitudes]
                else:
                    mag_trial_indiv[key][protocol].append(all_magnitudes)
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

            mag_trial_indiv[key][protocol] = np.concatenate(mag_trial_indiv[key][protocol], axis=0) #concatenate individual neuron values from different sessions


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
        mag_per_session.append(single_neurons_session)
        

    return suppression_groups, magnitude_groups, stim_groups, nb_neurons, avg_groups, sem_groups, cmi_groups, proportions_groups, individual_groups, perTrials_groups, mag_trials, sem_trials, mag_per_session, mag_trial_indiv



# ------------------- Data plotting --------------------------- #


def plot_protocol_overlap(groups_id, df, protocol_groups, save_path, fig_name):
    overlaps = {}

    # === Compute overlaps per session ===
    for group in groups_id.keys():
        overlaps[group] = {}
        df_filtered = df[df["Genotype"] == group]

        for k in range(len(df_filtered)):
            mouse_id = df_filtered["Mouse_id"].iloc[k]
            session_id = df_filtered["Session_id"].iloc[k]
            output_id = df_filtered["Output_id"].iloc[k]
            session_path = os.path.join(df_filtered["Session_path"].iloc[k],
                                        f"{session_id}_output_{output_id}")

            validity, _, _ = utils.load_data_session(session_path)
            exclusives, intersections, total_union = populations_overlap(validity, protocol_groups)

            # Exclusives
            for protocol in exclusives.keys():
                percent = 100 * len(exclusives[protocol]) / len(total_union)
                overlaps[group].setdefault(protocol, []).append(percent)

            # Intersections
            for intersect in intersections.keys():
                percent = 100 * len(intersections[intersect]) / len(total_union)
                overlaps[group].setdefault(intersect, []).append(percent)

    # === Prepare plot ===
    all_keys = sorted(
        set([key for group in overlaps.values() for key in group.keys()])
    )

    x = np.arange(len(all_keys))  # positions for each condition
    width = 0.2                    # horizontal offset for each group
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {0: 'skyblue', 1: 'orange', 2: 'salmon', 3: 'grey'}
    n_groups = len(groups_id)
    offsets = np.linspace(-width*(n_groups-1)/2, width*(n_groups-1)/2, n_groups)

    # === Plot individual points ===
    for g_idx, group in enumerate(groups_id.keys()):
        color = colors[g_idx]
        for k, key in enumerate(all_keys):
            points = overlaps[group].get(key, [])
            ax.scatter(np.full(len(points), x[k] + offsets[g_idx]),
                       points,
                       color=color, label=group if k == 0 else "", alpha=0.8, s=60)

    # === Styling ===
    ax.set_xticks(x)
    ax.set_xticklabels(all_keys, rotation=45, ha='right')
    ax.set_ylabel('% of neurons')
    ax.set_title('Exclusive and overlapping responsive neuron populations (individual sessions)')
    ax.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # === Save figure ===
    fig.savefig(os.path.join(save_path, f"{fig_name}_ProtocolOverlaps.png"), dpi=300)
    plt.show()

    # === Save Excel ===
    # Convert overlaps dict to DataFrame with keys as rows and groups as columns
    df_out = pd.DataFrame(index=all_keys, columns=groups_id.keys())
    for group in groups_id.keys():
        for key in all_keys:
            values = overlaps[group].get(key, [])
            df_out.loc[key, group] = ', '.join(f'{v:.2f}' for v in values)

    excel_path = os.path.join(save_path, f"{fig_name}_ProtocolOverlaps.xlsx")
    df_out.to_excel(excel_path)
    print(f"Excel file saved to: {excel_path}")

    return overlaps


def plot_MI_control(groups_id, magnitude_groups, sub_protocols, save_path, fig_name, attr, contrasts):
    pattern = r'[-]([\d.]+)$'
    
    # Extract contrast values and protocol types
    contrast_values = []
    protocol_type = []
    for protocol in sub_protocols:
        stim = protocol.split('-')[0:-1]
        stim = '-'.join(stim)
        match = re.search(pattern, protocol)
        if match:
            contrast = float(match.group(1))
            contrast_values.append(contrast)
        if stim not in protocol_type:
            protocol_type.append(stim)
    contrast_values = sorted(np.unique(contrast_values))
    
    # Create a single figure
    fig, ax = plt.subplots(1, len(contrasts), figsize=(6 * len(sub_protocols), 6), squeeze=False)
    with pd.ExcelWriter(os.path.join(save_path, f"{fig_name}_MItocontrol_{attr}.xlsx")) as writer:
        for i,contrast in enumerate(contrast_values):
            protocol1 = f'{protocol_type[0]}-{contrast}'
            protocol2 = f'{protocol_type[1]}-{contrast}'
            print(f"Computing MI at contrast {contrast} using protocols {protocol1} and {protocol2}")
            # Loop over groups and plot on the same axes
            MI_groups = {group: [] for group in groups_id.keys()}
            for group in groups_id.keys():
                magnitude = magnitude_groups[groups_id[group]]
                MI_list = []
                MI = (magnitude[protocol2] - magnitude[protocol1])/ (magnitude[protocol2] + magnitude[protocol1])
                MI_groups[group] = MI
            MI_WT = MI_groups[list(groups_id.keys())[0]]
            MI_KO = MI_groups[list(groups_id.keys())[1]]
            genotype = [list(groups_id.keys())[0]] * len(MI_WT) + [list(groups_id.keys())[1]] * len(MI_KO)
            labels = ['<-1.5', '-1.5 to -1.25', '-1.25 to -1', '-1 to -0.75', '-0.75 to -0.5', '-0.5 to -0.25', '-0.25 to 0', '0 to 0.25', '0.25 to 0.5', '0.5 to 0.75', '0.75 to 1', '1 to 1.25', '1.25 to 1.5', '>1.5']
            bins = [-float('inf'), -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, float('inf')]
            labels_list = []
            for l in [MI_WT, MI_KO]:
                labeled = pd.cut(l, bins=bins, labels=labels)
                labels_list += labeled.astype(str).tolist()
            df = pd.DataFrame({"Genotype" : genotype, "MI" : pd.Categorical(labels_list, categories=labels, ordered=True)})
            ax[0, i].set_title(f'Contrast: {contrast}')
            sns.histplot(df, x="MI", hue="Genotype", common_norm = False, shrink=.8,
                stat="percent", element='step', ax=ax[0, i], alpha=0)
            ax[0, i].set_ylabel('Proportion of neurons (%)')
            ax[0, i].tick_params(rotation=45)
            stat_mwu, p_value_mwu = mannwhitneyu(np.array(MI_WT), np.array(MI_KO), alternative='two-sided')
            p_value_1 = wilcoxon(np.array(MI_WT), alternative='two-sided')[1]
            p_value_2 = wilcoxon(np.array(MI_KO), alternative='two-sided')[1]
            textstr = (
                f'{list(groups_id.keys())[0]} median = {np.median(MI_WT):.2f}, p (vs 0) = {p_value_1:.3g}\n'
                f'{list(groups_id.keys())[1]} median = {np.median(MI_KO):.2f}, p (vs 0) = {p_value_2:.3g}\n'
                f'{list(groups_id.keys())[0]} vs {list(groups_id.keys())[1]} (Mann–Whitney) p = {p_value_mwu:.3g}'
            )
            # Position textbox on plot
            ax[0, i].text(0.99, 0.97, textstr,
                        transform=ax[0, i].transAxes,
                        fontsize=9, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85))
            
            #save in an excel sheet
            # Count neurons per bin per genotype
            bin_counts = df.groupby(["Genotype", "MI"], observed=False).size().reset_index(name="Count")

            # Pivot so that each column is a genotype
            pivot_df = bin_counts.pivot(index="MI", columns="Genotype", values="Count").fillna(0)

            # Optionally convert counts to percentages
            pivot_df_percent = pivot_df.div(pivot_df.sum(axis=0), axis=1) * 100
            # Save to Excel sheet for this contrast
            pivot_df_percent.to_excel(writer, sheet_name=f"contrast_{contrast}")
    fig.suptitle(f"MI for {protocol_type[1]} vs {protocol_type[0]} for {list(groups_id.keys())[0]} and {list(groups_id.keys())[1]} neurons",
             fontsize=14) 
    fig.tight_layout()
    
    
    fig.savefig(os.path.join(save_path, f"{fig_name}_MItoControl_{attr}.jpeg"), dpi=300)
    plt.show()


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

def mean_mag_per_protocol(groups_id, magnitude_groups, sub_protocols, save_path, fig_name, attr):
    """
    Plot the mean ± SEM of response magnitude to each protocol for both groups,
    and save individual neuron values in a separate Excel sheet.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import os

    palette = ['skyblue', 'orange', 'green', 'red']  # extend if needed
    group_keys = list(groups_id.keys())
    color = {k: palette[i % len(palette)] for i, k in enumerate(group_keys)}
    width = 0.5
    offset = (len(group_keys) + 1) * width  # spacing between protocols

    fig, ax = plt.subplots(figsize=(8, 6))

    x_ticks = []
    x_labels = []
    summary_data = []

    # Prepare dict for individual neuron values
    indiv_values = {}

    for protocol in sub_protocols:
        for key in group_keys:
            magnitudes = magnitude_groups[groups_id[key]][protocol]
            mean_magnitude = np.mean(magnitudes)
            sem_magnitude = np.std(magnitudes) / np.sqrt(len(magnitudes))
            n_neurons = len(magnitudes)

            # Store stats for summary
            summary_data.append({
                "Protocol": protocol,
                "Group": key,
                "Mean": mean_magnitude,
                "SEM": sem_magnitude,
                "N_neurons": n_neurons
            })

            # Store individual values for Excel
            col_name = f"{key}_{protocol}"
            indiv_values[col_name] = magnitudes

            # Plot
            i = sub_protocols.index(protocol)
            j = group_keys.index(key)
            x = offset * i + width * j
            x_ticks.append(x)
            x_labels.append(f'{protocol}\n{key}')
            ax.bar(x, mean_magnitude, width=width, color=color[key], edgecolor='black', label=key if i == 0 else "")
            ax.errorbar(x, mean_magnitude, yerr=sem_magnitude, fmt='none', ecolor='black', capsize=5, linewidth=1.2)

    # Labeling
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_ylabel('Magnitude of response')
    ax.set_title(f'Mean ± SEM of response magnitude ({attr})')
    ax.legend()

    plt.tight_layout()

    # Save figure
    fig.savefig(os.path.join(save_path, f"{fig_name}_mean_magnitude_{attr}.jpeg"), dpi=300)
    plt.show()

    # Convert summary to DataFrame
    df_summary = pd.DataFrame(summary_data)

    # Convert individual values to DataFrame (align lengths)
    max_len = max(len(v) for v in indiv_values.values())
    for k in indiv_values:
        # Pad with NaN so all columns have the same length
        if len(indiv_values[k]) < max_len:
            indiv_values[k] = np.pad(indiv_values[k], (0, max_len - len(indiv_values[k])), constant_values=np.nan)
    df_indiv = pd.DataFrame(indiv_values)

    # Save to Excel with two sheets
    excel_path = os.path.join(save_path, f"{fig_name}_mean_magnitude_{attr}.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
        df_indiv.to_excel(writer, sheet_name='IndividualValues', index=False)


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
    min_groups = None
    for i, group in enumerate(groups):
        colors[group] = {}
        avg = avg_groups[groups_id[group]]
        sem = sem_groups[groups_id[group]]
        neurons = nb_neurons[groups_id[group]]
        
        # Get minimum length among all protocols
        min_len = min(len(avg[protocol]) for protocol in protocols)
        if min_groups is None:
            min_groups = min_len
        elif min_len < min_groups:
            min_groups = min_len

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

    for col in excel_dict.keys():
        excel_dict[col] = excel_dict[col][:min_groups]
    # Create DataFrame and save to Excel
    df = pd.DataFrame(excel_dict)
    excel_path = os.path.join(save_path, f"{fig_name}_averages_{attr}.xlsx")
    df.to_excel(excel_path, index=False)


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
    if len(sub_protocols) == 2 or variable == 'AI':
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
        elif variable == 'AI':
            for l in [list1, list2]:
                bins = np.linspace(-1, 1, 17)
                labels = [f"{round(bins[i],2)} to {round(bins[i+1],2)}" for i in range(len(bins)-1)]
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

        fig.savefig(os.path.join(save_path, f"histplot_{fig_name}_{variable}_{attr}.jpeg"), dpi=300)
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


def plot_cdf_magnitudes(groups_id, magnitude_groups, sub_protocols, attr, magnitude_method, file_name, save_path):
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
        plt.xlabel(f'Response magnitude ({magnitude_method})')
        plt.ylabel('Cumulative probability')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save
        plt.savefig(os.path.join(save_path, fname), dpi=300)
        plt.show()


def perc_pref_contrast(groups_id, mag_per_session, sub_protocols, attr, fig_name, save_path):
    """
    Plot mean ± SEM of % neurons preferring each contrast across sessions for each group.
    """
    # Compute per-session percentages
    preferred_contrasts = preferred_contrast(groups_id, mag_per_session, sub_protocols)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    width = 0.35
    x = np.arange(len(sub_protocols))

    for i, group in enumerate(groups_id.keys()):
        means, sems = [], []
        for protocol in sub_protocols:
            # preferred_contrasts[group][protocol] = list of percentages per session
            session_values = np.array(preferred_contrasts[group][protocol])
            means.append(np.mean(session_values))
            sems.append(stats.sem(session_values))
        
        ax.bar(x + i * width, means, width, yerr=sems, capsize=5, label=group)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(sub_protocols)
    ax.set_ylabel('Percentage of Neurons (%)')
    ax.set_title(f'Preferred Contrast Distribution ({attr})')
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(save_path, f"{fig_name}_preferred_contrast_{attr}.jpeg"), dpi=300)
    plt.show()

    # ---------------- Save per-session percentages to Excel ----------------
    excel_path = os.path.join(save_path, f"{fig_name}_preferred_contrast_{attr}.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        for group in groups_id.keys():
            # Build DataFrame: rows = sessions, columns = protocols
            n_sessions = len(preferred_contrasts[group][sub_protocols[0]])
            data = {protocol: preferred_contrasts[group][protocol] for protocol in sub_protocols}
            df = pd.DataFrame(data)
            df.index.name = 'Session'
            df.to_excel(writer, sheet_name=group)


def plot_adaptation_index(sub_protocols, groups_id, mag_trial_indiv, attr, fig_name, save_path, first=3, last=3):
    groups = list(groups_id.keys())
    AI = adaptation_index(sub_protocols, groups_id, mag_trial_indiv, first, last)
    
    for protocol in sub_protocols: 
        list1 = AI[groups[0]][protocol]
        list2 = AI[groups[1]][protocol]
        
        histplot(
            sub_protocols=sub_protocols,
            list1=list1,
            list2=list2,
            groups=groups,
            save_path=save_path,
            fig_name=f"{fig_name}_{protocol}",
            attr=attr,
            variable="AI"
        )
    


    
if __name__ == "__main__":

    #-----------------------INPUTS-----------------------#

    excel_sheet_path = r"Y:\raw-imaging\Nathan\Nathan_sessions_visualpipe.xlsx"
    save_path = r"Y:\raw-imaging\Nathan\PYR\Visualpipe_postanalysis\looming-sweeping-log\Analysis"
    
    #Will be included in all names of saved figures
    fig_name = 'test'

    #Name of the physion protocol to analyze (e.g. 'surround-mod', 'visual-survey'...)
    protocol_name = "looming-sweeping-log"

    # Write the protocols you want to plot 
    sub_protocols = ['looming-stim-log-1.0']  
    # Method od selection of responsive neurons: 'any', 'only' or 'and'
       # selection_method:
       # 'any'  -> neurons responsive to at least one protocol in any group
       # 'only' -> neurons exclusive to a specific group (provide group_name)
       # 'and'  -> neurons shared between groups
    selection_method = 'any'
    # For the methods 'only' and 'any': you should put the key of the group of protocols you are interested in from valid_sub_protocols. If you want to use method 'and', put None
    group_name = 'looming'
    # Dict of protocol(s) used to select responsive neurons. 
    valid_sub_protocols = {'looming': ['looming-stim-log-0.0', 'looming-stim-log-0.1', 'looming-stim-log-0.4','looming-stim-log-1.0']} 
    # Example of correct valid_sub_protocols {'looming': ['looming-stim-log-0.0', 'looming-stim-log-0.1', 'looming-stim-log-0.4','looming-stim-log-1.0']} 
    '''quick-spatial-mapping-center', 'quick-spatial-mapping-left', 'quick-spatial-mapping-right',
        'quick-spatial-mapping-up', 'quick-spatial-mapping-down',
        'quick-spatial-mapping-up-left', 'quick-spatial-mapping-up-right',
        'quick-spatial-mapping-down-left', 'quick-spatial-mapping-down-right'''
    'black-sweeping-log-0.0', 'black-sweeping-log-0.1', 'black-sweeping-log-0.4','black-sweeping-log-1.0'
    'white-sweeping-log-0.0', 'white-sweeping-log-0.1', 'white-sweeping-log-0.4','white-sweeping-log-1.0', 'black-sweeping-log-0.0', 'black-sweeping-log-0.1', 'black-sweeping-log-0.4','black-sweeping-log-1.0'
    'looming-stim-log-0.0', 'looming-stim-log-0.1', 'looming-stim-log-0.4','looming-stim-log-1.0'
    'black-sweeping-log-0.0', 'black-sweeping-log-0.1', 'black-sweeping-log-0.4','black-sweeping-log-1.0'
    'dimming-circle-log-0.0', 'dimming-circle-log-0.1', 'dimming-circle-log-0.4','dimming-circle-log-1.0'


    #Frame rate
    frame_rate = 30

    #dt_prestim: duration of the baseline period before stimulus onset (in seconds)
    dt_prestim = 1

    # Decide if you want to plot the dFoF0 baseline substraced or the z-scores
    attr = 'dFoF0-baseline'  # 'dFoF0-baseline' or 'z_scores'

    # Decide if you want to only keep neurons that are centered
    get_centered = False  # True or False

    # Decide on the way to calculate the amplitude of response
    magnitude_method = 'auc' #'auc', 'peak' or 'filtered_peak', 'mean'

    #----------------------------------------------------#
    df = utils.load_excel_sheet(excel_sheet_path, protocol_name)

    groups_id = {'WT': 0, 'KO': 1}  # keys are group names, e.g 'WT': 0, 'KO': 1

    suppression_groups, magnitude_groups, stim_groups, nb_neurons, avg_groups, sem_groups, cmi_groups, proportions_groups, individual_groups, perTrials_groups, mag_trials, sem_trials, mag_per_session, mag_trial_indiv = process_group(df, groups_id, attr, valid_sub_protocols, sub_protocols, protocol_name, selection_method, group_name, magnitude_method, get_centered, plot=False) 
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
    #plot_cdf_magnitudes(groups_id, magnitude_groups, sub_protocols, attr, magnitude_method, fig_name, save_path) 
    #plot_per_trial(groups_id, nb_neurons, perTrials_groups, sub_protocols, frame_rate, dt_prestim, fig_name, attr, save_path)
    #magnitude_per_trial(fig_name, save_path, nb_neurons, mag_trials, sem_trials, sub_protocols, groups_id)
    #mean_mag_per_protocol(groups_id, magnitude_groups, sub_protocols, save_path, fig_name, attr)
    #plot_MI_control(groups_id, magnitude_groups, sub_protocols, save_path, fig_name, attr, contrasts=[0.05, 0.14, 0.37, 1.0])
    #perc_pref_contrast(groups_id, mag_per_session, sub_protocols, attr, fig_name, save_path)
    #plot_adaptation_index(sub_protocols, groups_id, mag_trial_indiv, attr, fig_name, save_path, first=3, last=3)

    #plot_protocol_overlap(groups_id, df, valid_sub_protocols, save_path, fig_name)


