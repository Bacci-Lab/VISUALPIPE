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
from openpyxl import Workbook
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

    stimuli_df['name'] = stimuli_df['name'].replace('center', 'center-20-1.0')
    if 'center' in validity:
        validity['center-20-1.0'] = validity.pop('center')

    stimuli_df['name'] = stimuli_df['name'].replace('center-surround-cross', 'center-surround_high_contrast-cross-20.0-1.0')
    if 'center-surround-cross' in validity:
        validity['center-surround_high_contrast-cross-20.0-1.0'] = validity.pop('center-surround-cross')

    stimuli_df['name'] = stimuli_df['name'].replace('center-surround-iso', 'center-surround_high_contrast-iso-1.0')
    if 'center-surround-iso' in validity:
        validity['center-surround_high_contrast-iso-1.0'] = validity.pop('center-surround-iso')

    stimuli_df['name'] = stimuli_df['name'].replace('surround-iso_ctrl', 'surround-iso_ctrl-20-1.0')
    if 'surround-iso_ctrl' in validity:
        validity['surround-iso_ctrl-20-1.0'] = validity.pop('surround-iso_ctrl')

    stimuli_df['name'] = stimuli_df['name'].replace('surround-cross_ctrl', 'surround-cross_ctrl-20-1.0')
    if 'surround-cross_ctrl' in validity:
        validity['surround-cross_ctrl-20-1.0'] = validity.pop('surround-cross_ctrl')

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

def find_state_trial(trials, stimuli_df, sub_protocols, period_names):
    rest_dict, AS_dict, run_dict = {protocol: [] for protocol in sub_protocols}, {protocol: [] for protocol in sub_protocols}, {protocol: [] for protocol in sub_protocols}
    for stimulus in sub_protocols:
        stim_id = stimuli_df[stimuli_df.name == stimulus].index[0]
        states = trials['arousal_states'][stim_id]
        states = np.array(states)  # convert to numpy array for indexing
        rest = np.where(states == 'rest')[0]
        AS = np.where(states == 'AS')[0]
        run = np.where(states == 'run')[0]
        rest_dict[stimulus] = rest
        AS_dict[stimulus] = AS
        run_dict[stimulus] = run
        print(f"{stimulus}: number of trials for rest:{len(rest)}, AS:{len(AS)}, run:{len(run)}")
    return rest_dict, AS_dict, run_dict

def parse_size_contrast(protocol_name):
    """
    Extract size (deg) and contrast (0-1) from protocol string.
    Example: 'size-tuning-contrast-log-20-0.22' -> (20, 0.22)
    """
    m = re.search(r'size-tuning-contrast-log-(\d+)-([\d\.]+)', protocol_name)
    if m:
        size = float(m.group(1))
        contrast = float(m.group(2))
        return size, contrast
    else:
        raise ValueError(f"Cannot parse size/contrast from {protocol_name}")


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
            if prot in validity:
                data = validity[prot]
                valid_neurons = set(np.where(data[:, 0] == 1)[0]) # change to -1 if you want negatively responsive neurons
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


def get_centered_neurons(stimuli_df, neurons_list, trials, attr, plot, direction = 'max', frame_rate = 30):
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
        if direction == 'max':
            max_stimulus = max(magnitudes_neuron, key=magnitudes_neuron.get)
        elif direction == 'min':
            max_stimulus = min(magnitudes_neuron, key=magnitudes_neuron.get)
        if max_stimulus == 'quick-spatial-mapping-center':
            centered_neurons.append(neuron)
        else:
            not_centered.append(neuron)
    proportion_centered = 100*len(centered_neurons)/trials[period_names[1]][0].shape[0]

    if plot and len(centered_neurons)!=0:
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

def select_neurons(red_path, validity, valid_sub_protocols, selection_method, group_name,
                   get_centered, stimuli_df, trials, period_names, attr, plot, direction):
    valid_neurons = get_valid_neurons(validity, valid_sub_protocols,
                                      selection_method=selection_method,
                                      group_name=group_name)
    
    # Convert to 1D int array
    if isinstance(valid_neurons, set):
        valid_neurons = np.array(list(valid_neurons), dtype=int)
    else:
        valid_neurons = np.atleast_1d(valid_neurons).astype(int)
    if red_path is not None:
        red_neurons = np.load(red_path)
        valid_neurons = np.intersect1d(valid_neurons, red_neurons)

    if get_centered:
        centered_neurons, non_centered = get_centered_neurons(stimuli_df, valid_neurons, trials,
                                                             attr, plot, direction, frame_rate=30)
        valid_neurons = centered_neurons

    if red_path is not None:
        proportion = 100 * len(valid_neurons) / len(red_neurons)
    else:
        proportion = 100 * len(valid_neurons) / trials[period_names[1]][0].shape[0]
    return valid_neurons, proportion

def normalize_magnitudes(groups_id, sub_protocols, rest_groups, AS_groups, run_groups):
    """
    Normalize AS and RUN magnitudes relative to the maximum mean response
    across protocols in REST, per neuron, handling missing neurons/states.

    Parameters
    ----------
    groups_id : dict
        Group names and their indices.
    sub_protocols : list
        List of protocol names.
    rest_groups, AS_groups, run_groups : list of dicts
        Each dict: protocol -> array of mean magnitudes per neuron.

    Returns
    -------
    rest_norm, AS_norm, run_norm : lists of dicts
        Same structure, normalized per neuron relative to their max REST magnitude.
    """
    rest_norm, AS_norm, run_norm = [], [], []

    for g in range(len(groups_id)):
        rest = rest_groups[g]
        AS   = AS_groups[g]
        run  = run_groups[g]

        # Determine the max number of neurons across REST protocols
        max_neurons = max((arr.size for arr in rest.values()), default=0)

        # Stack REST protocols safely with NaNs for missing neurons
        rest_stack = np.full((len(sub_protocols), max_neurons), np.nan)
        for i, p in enumerate(sub_protocols):
            arr = rest.get(p, np.array([]))
            rest_stack[i, :arr.size] = arr

        # Max per neuron across REST protocols (ignores NaNs)
        rest_max = np.nanmax(rest_stack, axis=0)
        rest_max[rest_max == 0] = np.nan  # avoid division by zero

        # Function to safely normalize arrays per neuron
        def normalize_dict(state_dict):
            normed = {}
            for p in sub_protocols:
                arr = state_dict.get(p, np.array([]))
                if arr.size > 0:
                    normed[p] = arr / rest_max[:arr.size]
                else:
                    normed[p] = np.array([])  # keep empty if no neurons
            return normed

        rest_norm.append(normalize_dict(rest))
        AS_norm.append(normalize_dict(AS))
        run_norm.append(normalize_dict(run))

    return rest_norm, AS_norm, run_norm

def reshape_mag_per_session(mag_per_session, sub_protocols, groups_id):
    """
    Convert flat mag_per_session into neuron x contrast x size arrays per session,
    normalizing each neuron to its maximum response across all contrasts and sizes.

    Returns:
        reshaped[group_name][session_idx] = array of shape n_neurons x n_contrasts x n_sizes (normalized)
        contrasts = sorted list of unique contrasts
        sizes = sorted list of unique sizes
    """
    # Extract unique sizes and contrasts
    sizes, contrasts = [], []
    for proto in sub_protocols:
        s, c = parse_size_contrast(proto)
        sizes.append(s)
        contrasts.append(c)
    sizes = np.unique(sizes)
    contrasts = np.unique(contrasts)

    reshaped = {group: [] for group in groups_id}
    for group in groups_id:
        group_idx = groups_id[group]
        n_sessions = len(mag_per_session[group_idx][sub_protocols[0]])

        for session_idx in range(n_sessions):
            n_neurons = len(mag_per_session[group_idx][sub_protocols[0]][session_idx])
            array_session = np.zeros((n_neurons, len(contrasts), len(sizes)))

            # Fill the array
            for proto in sub_protocols:
                s, c = parse_size_contrast(proto)
                size_idx = np.where(sizes == s)[0][0]
                contrast_idx = np.where(contrasts == c)[0][0]
                array_session[:, contrast_idx, size_idx] = mag_per_session[group_idx][proto][session_idx]

            # --- Normalize each neuron to its maximum response ---
            max_per_neuron = array_session.max(axis=(1, 2), keepdims=True)
            max_per_neuron[max_per_neuron == 0] = 1  # avoid division by zero
            array_session /= max_per_neuron

            reshaped[group].append(array_session)

    return reshaped, contrasts, sizes


def export_pref_pct_to_excel(pref_contrast_pct, pref_size_pct, contrasts, sizes, group_id, save_path):
    """
    Export preferred contrast and size percentages to a single Excel file per group,
    with two sheets: 'Contrast' and 'Size'.
    
    Arguments:
        pref_contrast_pct: dict[group][contrast][session_idx] = percentage
        pref_size_pct: dict[group][size][session_idx] = percentage
        contrasts: list of contrast values
        sizes: list of size values
        group_names: list of groups to export (e.g., ['WT', 'KO'])
        save_path: folder to save Excel files
    """
    os.makedirs(save_path, exist_ok=True)

    for group in groups_id.keys():
        # Preferred contrast sheet
        contrast_dict = {f'Contrast_{c}': pref_contrast_pct[group][c] for c in contrasts}
        df_contrast = pd.DataFrame(contrast_dict)

        # Preferred size sheet
        size_dict = {f'Size_{s}': pref_size_pct[group][s] for s in sizes}
        df_size = pd.DataFrame(size_dict)

        # Excel writer with two sheets
        excel_file = os.path.join(save_path, f'preferred_pct_{group}.xlsx')
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            df_contrast.to_excel(writer, sheet_name='Contrast', index=False)
            df_size.to_excel(writer, sheet_name='Size', index=False)

        print(f"Saved preferred percentages Excel (two sheets) for {group}: {excel_file}")



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



def process_group(df, groups_id, attr, valid_sub_protocols, sub_protocols, protocol_name, selection_method, group_name, frame_rate, magnitude_method, get_centered, plot, red_ch = 'green', direction = 'max'):
    #Define trial period names based on attribute
    period_names, trial_periods = get_period_names(attr)
    # Initialize group-level containers
    magnitude_groups, nb_neurons, avg_groups, sem_groups, proportions_groups, individual_groups, rest_groups, AS_groups, run_groups = [], [], [], [], [], [], [], [], []
    mag_trial_indiv = {group: {} for group in list(groups_id.keys())} # individual magnitudes per trials per neuron for all protocols


    #loop over groups (e.g. WT and KO)
    for key in groups_id.keys():
        
        df_filtered = df[df["Genotype"] == key] 

        print(f"\n-------------------------- Processing {key} group --------------------------")

        all_neurons = 0
        #Initialize protocol-level containers
        magnitude = {protocol: [] for protocol in sub_protocols} #magnitude of the response to each protocol for each neuron
        avg_data = {protocol: [] for protocol in sub_protocols} #trial-averaged traces per protocol, per session
        single_neurons_group = {protocol: [] for protocol in sub_protocols} #individual traces of each neuron for each protocol
        single_neurons_session = {protocol: [] for protocol in sub_protocols}  #individual traces of each neuron for each protocol, per session
        rest_mags, AS_mags, run_mags = {protocol: [] for protocol in sub_protocols}, {protocol: [] for protocol in sub_protocols}, {protocol: [] for protocol in sub_protocols} #magnitude of the response to each protocol for each neuron, separated by arousal state
        proportion_list = [] #proportion of responsive neurons per session

        
        for k in range(len(df_filtered)):
            #get the session path
            mouse_id = df_filtered["Mouse_id"].iloc[k]
            session_id = df_filtered["Session_id"].iloc[k]
            output_id = df_filtered["Output_id"].iloc[k]
            session_path = os.path.join(df_filtered["Session_path"].iloc[k], f"{session_id}_output_{output_id}")

            print(f"\nSession id: {session_id}\n  Mouse id : {mouse_id}\n     Session path: {session_path}")

            validity, trials, stimuli_df = load_session_data(session_path)
            rest, AS, run = find_state_trial(trials, stimuli_df, sub_protocols, period_names)
            if red_ch == 'red-green':
                red_path = os.path.join(session_path, 'red_channel/red_green_cells.npy')
                if not os.path.exists(red_path):
                    print('No red-green cells file found, skipping this session')
                    continue
            else:
                red_path = None
            valid_neurons, proportion = select_neurons(red_path, validity, valid_sub_protocols, selection_method, group_name,
                   get_centered, stimuli_df, trials, period_names, attr, plot, direction = 'max') #extract responsive neurons (and centered if get_centered = True)
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
                #Store the average response of each neuron to that protocol
                magnitude[protocol].append(mag_per_neuron) #magnitude values for this protocol per neuron

                
                for trial in range(0,n_trials):
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
                # Only compute mean if there are trials for this state
                if len(rest) > 0:
                    rest_mean = np.mean(all_magnitudes[:, rest[protocol]], axis=1)
                    
                else:
                    rest_mean = np.full(len(valid_neurons), np.nan)
                rest_mags[protocol].append(rest_mean)
                if len(AS) > 0:
                    AS_mean = np.mean(all_magnitudes[:, AS[protocol]], axis=1)
                else:
                    AS_mean = np.full(len(valid_neurons), np.nan)
                AS_mags[protocol].append(AS_mean)
                if len(run) > 0:
                    run_mean = np.mean(all_magnitudes[:, run[protocol]], axis=1)
                else:
                    run_mean = np.full(len(valid_neurons), np.nan)
                run_mags[protocol].append(run_mean)
                if protocol not in mag_trial_indiv[key]:
                    mag_trial_indiv[key][protocol] = [all_magnitudes]
                else:
                    mag_trial_indiv[key][protocol].append(all_magnitudes)
        #perTrials_groups[key] = perTrials # store the average response per trial for each protocol, for that group
        
        for protocol in magnitude.keys():
            magnitude[protocol] = np.concatenate(magnitude[protocol])


        print(f"\nNumber of {key} neurons: {all_neurons}")

        # Concatenate all neuron arrays into one array per protocol
        for protocol in sub_protocols:
            #avg_data[protocol] = np.stack(avg_data[protocol], axis=0)
            min_len = min(arr.shape[-1] for arr in avg_data[protocol])

            # Truncate all arrays to that length
            trimmed_arrays = [arr[..., :min_len] for arr in avg_data[protocol]]

            # Now stack safely
            avg_data[protocol] = np.stack(trimmed_arrays, axis=0)


            mag_trial_indiv[key][protocol] = np.concatenate(mag_trial_indiv[key][protocol], axis=0) #concatenate individual neuron values from different sessions"""
        
            rest_mags[protocol] = np.concatenate(rest_mags[protocol], axis=0)
            AS_mags[protocol]   = np.concatenate(AS_mags[protocol], axis=0)
            run_mags[protocol]  = np.concatenate(run_mags[protocol], axis=0)

        # Compute average and SEM across neurons
        avg = {protocol: np.mean(avg_data[protocol], axis=0) for protocol in sub_protocols} 
        sem = {protocol: stats.sem(avg_data[protocol], axis=0) for protocol in sub_protocols}
        print(f"List of % of responsive neurons per session for {key}: {proportion_list}")

        magnitude_groups.append(magnitude)
        nb_neurons.append(all_neurons)
        avg_groups.append(avg)
        sem_groups.append(sem)
        proportions_groups.append(proportion_list)
        individual_groups.append(single_neurons_group)
        rest_groups.append(rest_mags)
        AS_groups.append(AS_mags) 
        run_groups.append(run_mags)  
        

    return magnitude_groups, nb_neurons, avg_groups, sem_groups, proportions_groups, individual_groups, mag_trial_indiv, rest_groups, AS_groups, run_groups

def evoked_modulation_index(rest_mags, AS_mags, run_mags, sub_protocols, groups_id):
    Run_mod_g, AS_mod_g = [], []
    for group in range(len(groups_id)):
        Run_mod_protocols, AS_mod_protocols = {protocol: [] for protocol in sub_protocols}, {protocol: [] for protocol in sub_protocols}
        for protocol in sub_protocols:
            rest = rest_mags[group][protocol]
            AS = AS_mags[group][protocol]
            run = run_mags[group][protocol]
            LMI = (run - rest)/(rest + run)
            ASMI = (AS - rest)/(rest + AS)
            Run_mod_protocols[protocol] = LMI
            AS_mod_protocols[protocol] = ASMI
        Run_mod_g.append(Run_mod_protocols)
        AS_mod_g.append(AS_mod_protocols)
    return Run_mod_g, AS_mod_g

# ------------------- Data plotting --------------------------- #

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


def mean_mag_per_protocol(groups_id, state_groups, sub_protocols, save_path, fig_name, attr, state):
    """
    Plot the mean ± SEM of response magnitude to each protocol for both groups,
    and save individual neuron values in a separate Excel sheet.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import os

    palette = ['orange', 'skyblue', 'green', 'red']  # extend if needed
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
            magnitudes = state_groups[groups_id[key]][protocol]
            mean_magnitude = np.nanmean(magnitudes)
            n_neurons = np.sum(~np.isnan(magnitudes))
            sem_magnitude = np.nanstd(magnitudes) / np.sqrt(n_neurons)

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
    ax.set_title(f'Mean ± SEM of response magnitude ({attr}) - {state}')
    ax.legend()

    plt.tight_layout()

    # Save figure
    fig.savefig(os.path.join(save_path, f"{fig_name}_{state}_mean_magnitude_{attr}.jpeg"), dpi=300)
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
    excel_path = os.path.join(save_path, f"{fig_name}_{state}_mean_magnitude_{attr}.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
        df_indiv.to_excel(writer, sheet_name='IndividualValues', index=False)
    

def representative_traces(frame_rate, suppression_groups, cmi_groups, magnitude_groups, groups_id,
                          individual_groups, sub_protocols, attr, save_path, fig_name,
                          variable='suppression_index'):
    if not cmi_groups or not suppression_groups:
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

        # Compute median
        median = np.median(cmi)
        print(f"Group {group}: median {variable} = {median:.2f}")

        # Select traces within ±0.2 around the median
        mask = np.where(np.abs(cmi - median) <= 0.02)[0]

        if len(mask) == 0:
            print(f"⚠️ No traces within ±0.2 of the median for group {group}. Using closest value instead.")
            mask = [np.argmin(np.abs(cmi - median))]

        candidate_ids = mask

        # Compute baseline noise for each candidate trace
        noise_values = []
        baseline_len = int(1 * frame_rate)   # = 30 samples for 30 Hz

        for idx in candidate_ids:
            noise_per_protocol = []
            for protocol in sub_protocols:
                trace = individual_groups[id_group][protocol][idx]

                # First second STD
                noise_per_protocol.append(np.std(trace[:baseline_len]))

            # Mean noise across protocols
            noise_values.append(np.mean(noise_per_protocol))

        # Representative = least noisy trace
        cmi_id = candidate_ids[np.argmin(noise_values)]
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
            excel_dict[col_name] = rep_trace[protocol][:min_len]/max_group
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
    

def representative_traces_joint(frame_rate,
                                                            suppression_groups,
                                                            cmi_groups,
                                                            magnitude_groups,
                                                            groups_id,
                                                            individual_groups,
                                                            sub_protocols,
                                                            attr,
                                                            save_path,
                                                            fig_name):
    
    if not cmi_groups or not suppression_groups:
        print("No CMI or suppression index data available.")
        return

    excel_dict = {}
    fig, ax = plt.subplots(2, len(groups_id), figsize=(14, 8))
    baseline_len = int(1 * frame_rate)

    center_protocol = sub_protocols[0]  # normalize everything to this

    for i, group in enumerate(groups_id.keys()):
        id_group = groups_id[group]

        SI = suppression_groups[id_group]
        CMI = cmi_groups[id_group]

        median_SI = np.median(SI)
        median_CMI = np.median(CMI)

        # --- Compute Euclidean distance to joint median ---
        distances = np.sqrt((SI - median_SI)**2 + (CMI - median_CMI)**2)

        # Consider top 5% closest ROIs
        threshold = np.percentile(distances, 5)
        candidate_ids = np.where(distances <= threshold)[0]
        if len(candidate_ids) == 0:
            candidate_ids = [np.argmin(distances)]

        # --- Select least noisy among candidates ---
        noise_values = []
        for idx in candidate_ids:
            noise_per_protocol = []
            for protocol in sub_protocols:
                trace = individual_groups[id_group][protocol][idx]
                #noise_per_protocol.append(np.std(trace[:baseline_len]))
                baseline_std = np.std(trace[:baseline_len])
                response_amp = np.percentile(trace, 95) - np.mean(trace[:baseline_len])
                snr = response_amp / baseline_std if baseline_std > 0 else 0
                noise_per_protocol.append(snr)
            noise_values.append(np.mean(noise_per_protocol))

        rep_idx = candidate_ids[np.argmin(noise_values)]
        rep_SI = SI[rep_idx]
        rep_CMI = CMI[rep_idx]

        rep_trace = {protocol: individual_groups[id_group][protocol][rep_idx]
                     for protocol in sub_protocols}

        # --- Normalize by center protocol ---
        center_max = np.percentile(rep_trace[center_protocol], 95)
        min_len = min(len(t) for t in rep_trace.values())
        time = np.linspace(0, min_len, min_len) / frame_rate - 1

        if "Time (s)" not in excel_dict:
            excel_dict["Time (s)"] = time

        for protocol in sub_protocols:
            normalized = rep_trace[protocol][:min_len] / center_max
            excel_dict[f"{group}_{protocol}_representative"] = normalized

            ax[0, i].plot(
                time,
                gaussian_filter1d(normalized, sigma=1),
                lw=2,
                label=protocol
            )

        ax[0, i].set_title(f"{group} — Representative ROI")
        ax[0, i].set_ylabel(f"{attr} normalized to {center_protocol}")
        ax[0, i].set_xlabel("Time (s)")
        ax[0, i].legend()

        ax[0, i].text(
            0.98, 0.02,
            f"SI = {rep_SI:.2f}, CMI = {rep_CMI:.2f}",
            transform=ax[0, i].transAxes,
            ha='right', va='bottom',
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
        )

        # --- Bar plot of magnitudes, normalized by center protocol ---
        center_mag = magnitude_groups[id_group][center_protocol][rep_idx]
        magnitudes = [magnitude_groups[id_group][protocol][rep_idx] / center_mag
                      for protocol in sub_protocols]

        ax[1, i].bar([protocol for protocol in sub_protocols],
                     magnitudes,
                     color=['skyblue', 'orange'], width=0.3)
        ax[1, i].set_ylabel(f"Response magnitude ({attr}) normalized to {center_protocol}")
        ax[1, i].set_title(f"{group} — Response magnitudes")

    plt.tight_layout()
    fig.savefig(
        os.path.join(save_path,
                     f"{fig_name}_representative_traces_joint_least_noisy_center_protocol_{attr}.jpeg"),
        dpi=300,
        bbox_inches='tight'
    )
    plt.show()

    df = pd.DataFrame(excel_dict)
    df.to_excel(
        os.path.join(save_path,
                     f"{fig_name}_representative_traces_joint_least_noisy_center_protocol_{attr}.xlsx"),
        index=False
    )





    
if __name__ == "__main__":

    #-----------------------INPUTS-----------------------#

    excel_sheet_path = r"Y:\raw-imaging\Nathan\Nathan_sessions_visualpipe_PYR.xlsx"
    save_path = r"Y:\raw-imaging\Nathan\PYR\Visualpipe_postanalysis\size-tuning-log\Analysis"
    
    #Will be included in all names of saved figures
    fig_name = 'size_tuning_70°22%'

    #Name of the physion protocol to analyze (e.g. 'surround-mod', 'visual-survey'...)
    protocol_name = "size-tuning-protocol-log"

    # Write the protocols you want to plot 
    sub_protocols = ['size-tuning-contrast-log-35-0.22']
       # 'any'  -> neurons responsive to at least one protocol in any group
       # 'only' -> neurons exclusive to a specific group (provide group_name)
       # 'and'  -> neurons shared between groups
    selection_method = 'any'
    # For the methods 'only' and 'any': you should put the key of the group of protocols you are interested in from valid_sub_protocols. If you want to use method 'and', put None
    group_name = 'size'
    # Dict of protocol(s) used to select responsive neurons. 
    valid_sub_protocols = {'size': ['size-tuning-contrast-log-35-0.22']}

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
    'size-tuning-contrast-log-5-0.05', 'size-tuning-contrast-log-5-0.11', 'size-tuning-contrast-log-5-0.22','size-tuning-contrast-log-5-0.47','size-tuning-contrast-log-5-1.0',
    'size-tuning-contrast-log-10-0.05', 'size-tuning-contrast-log-10-0.11', 'size-tuning-contrast-log-10-0.22','size-tuning-contrast-log-10-0.47','size-tuning-contrast-log-10-1.0',
    'size-tuning-contrast-log-15-0.05', 'size-tuning-contrast-log-15-0.11', 'size-tuning-contrast-log-15-0.22','size-tuning-contrast-log-15-0.47','size-tuning-contrast-log-15-1.0',
    'size-tuning-contrast-log-20-0.05', 'size-tuning-contrast-log-20-0.11', 'size-tuning-contrast-log-20-0.22','size-tuning-contrast-log-20-0.47','size-tuning-contrast-log-20-1.0',
    'size-tuning-contrast-log-25-0.05', 'size-tuning-contrast-log-25-0.11', 'size-tuning-contrast-log-25-0.22','size-tuning-contrast-log-25-0.47','size-tuning-contrast-log-25-1.0',
    'size-tuning-contrast-log-30-0.05', 'size-tuning-contrast-log-30-0.11', 'size-tuning-contrast-log-30-0.22','size-tuning-contrast-log-30-0.47','size-tuning-contrast-log-30-1.0',
    'size-tuning-contrast-log-35-0.05', 'size-tuning-contrast-log-35-0.11', 'size-tuning-contrast-log-35-0.22','size-tuning-contrast-log-35-0.47','size-tuning-contrast-log-35-1.0',
    'size-tuning-contrast-log-40-0.05', 'size-tuning-contrast-log-40-0.11', 'size-tuning-contrast-log-40-0.22','size-tuning-contrast-log-40-0.47','size-tuning-contrast-log-40-1.0'


    #Frame rate
    frame_rate = 30

    #dt_prestim: duration of the baseline period before stimulus onset (in seconds)
    dt_prestim = 1

    # Decide if you want to plot the dFoF0 baseline substraced or the z-scores
    attr = 'z_scores'  # 'dFoF0-baseline' or 'z_scores'

    # Decide if you want to only keep neurons that are centered
    get_centered = True  # True or False

    #Decide if you want to plot green only neurons, or red-green ones
    color_ch = 'green'  # 'green' or 'red-green'

    # Decide on the way to calculate the amplitude of response
    magnitude_method = 'mean' #'auc', 'peak' or 'filtered_peak', 'mean'


    #----------------------------------------------------#
    df = utils.load_excel_sheet(excel_sheet_path, protocol_name)

    groups_id = {'WT': 0, 'KO': 1}  # keys are group names, e.g 'WT': 0, 'KO': 1

    magnitude_groups, nb_neurons, avg_groups, sem_groups, proportions_groups, individual_groups, mag_trial_indiv, rest_groups, AS_groups, run_groups = process_group(df, groups_id, attr, valid_sub_protocols, sub_protocols, protocol_name, selection_method, group_name, frame_rate, magnitude_method, get_centered, plot=False, red_ch = color_ch, direction = 'max') 
    
    
    rest_norm, AS_norm, run_norms = normalize_magnitudes(groups_id, sub_protocols, rest_groups, AS_groups, run_groups) 
    #representative_traces(frame_rate, suppression_groups, cmi_groups, magnitude_groups, groups_id,
    #                      individual_groups, sub_protocols, attr, save_path, fig_name, variable="CMI")
    #representative_traces_joint(frame_rate, suppression_groups, cmi_groups, magnitude_groups, groups_id,
    #                            individual_groups, sub_protocols, attr, save_path, fig_name)

    
     #-------------------Call the functions to process and plot the data-------------------#

    # Plot the average z-scores or dFoF0-baseline trace for responsive neurons
    #graph_averages(frame_rate, groups_id, fig_name, attr, save_path, sub_protocols, valid_sub_protocols, avg_groups, sem_groups, nb_neurons)
    #mean_mag_per_protocol(groups_id, rest_norm, sub_protocols, save_path, fig_name, attr, state='rest')
    #mean_mag_per_protocol(groups_id, AS_norm, sub_protocols, save_path, fig_name, attr, state='AS')
    #mean_mag_per_protocol(groups_id, run_norms, sub_protocols, save_path, fig_name, attr, state='run')

    Run_mod_g, AS_mod_g = evoked_modulation_index(rest_groups, AS_groups, run_groups, sub_protocols, groups_id)
    mean_mag_per_protocol(groups_id, Run_mod_g, sub_protocols, save_path, fig_name, attr, state='LMI')
    mean_mag_per_protocol(groups_id, AS_mod_g, sub_protocols, save_path, fig_name, attr, state='ASMI')




