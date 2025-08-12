import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
from scipy.stats import mannwhitneyu, wilcoxon, linregress
import glob
import pandas as pd
import seaborn as sns
import sys
sys.path.append("./src")

def load_excel_sheet(excel_sheet_path, protocol_name):
    df = pd.read_excel(excel_sheet_path)
    df = df[df["Protocol"] == protocol_name]
    df = df[df["Analyze"] == 1]
    duplicates = df.duplicated(subset=['Session_id'], keep='first')

    if np.sum(duplicates) > 0 : 
        print(f"There is/are {np.sum(duplicates)} duplicated session(s) in the excel file. Please remove them or set 'Analyze' to 0 in the excel file. The first occurence will be kept automatically if nothing is specified.")
        print(f"    Duplicated session(s) : {df[duplicates]['Session_id'].unique()}")
        df = df[~duplicates] # Remove duplicates

    print(f"Genotype : {df.Genotype.unique()}")
    print(f"Mice included : {df.Mouse_id.unique()}")
    
    return df

def load_data_session(path) :
    """    Load the validity and trials data from the specified path.
    Args:
        path (str): Path to the directory containing the .npz and .npy files.
    Returns:
        validity (dict): Dictionary containing the validity data loaded from the .npz file.
        trials (dict): Dictionary containing the trials data loaded from the .npy file.
    Raises:
        FileNotFoundError: If the expected .npz or .npy files are not found in the specified path.
    """
    
    # Load the npz file
    npz_files = glob.glob(os.path.join(path, "*protocol_validity_2.npz"))
    if len(npz_files) == 1:
        validity = np.load(npz_files[0], allow_pickle=True)
    else:
        raise FileNotFoundError(f"Expected exactly one .npz file in {path}, found {len(npz_files)} files")      
    
    # Load .npy file
    npy_files = glob.glob(os.path.join(path, "*trials.npy"))
    if len(npy_files) == 1:
        trials = np.load(npy_files[0], allow_pickle=True).item()
    else:
        raise FileNotFoundError(f"Expected exactly one .npy file in {path}, found {len(npy_files)}")
    
    # Load xlsx file with visual stimuli info
    xlsx_files = glob.glob(os.path.join(path, "*visual_stim_info.xlsx"))
    if len(xlsx_files) == 1:
        stimuli_df = pd.read_excel(os.path.join(path, xlsx_files[0]), engine='openpyxl').set_index('id')
    else:
        raise FileNotFoundError(f"Expected exactly one .xlsx file in {path}, found {len(xlsx_files)} files")   

    return validity, trials, stimuli_df

def get_valid_neurons_session(validity, protocol_list):
   # Dictionary to store responsive neurons for each protocol
    valid_data = {protocol : validity[protocol] 
                 if protocol in validity.files 
                 else print(f"{protocol} does not exist in validity file.") 
                 for protocol in protocol_list}
    
    valid_neuron_lists = [np.where(data[:, 0] == 1,)[0] for data in valid_data.values()] # change to -1 if you want negative responsive neurons
    valid_neurons = np.unique(np.concatenate(valid_neuron_lists))  # Get unique indices of valid neurons
    
    return valid_neurons

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

def compute_suppression(magnitude, protocol_surround=''):
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

def process_group(df, groups_id, attr, valid_sub_protocols, sub_protocols, protocol_name):

    if attr == 'dFoF0-baseline':
        period_names = ['norm_averaged_baselines', 'norm_trial_averaged_ca_trace', 'norm_post_trial_averaged_ca_trace']
    elif attr == 'z_scores':
        period_names = ['pre_trial_averaged_zscores', 'trial_averaged_zscores', 'post_trial_averaged_zscores']
    
    suppression_groups, magnitude_groups, stim_groups, nb_neurons, avg_groups, sem_groups, cmi_groups, proportions_groups = [], [], [], [], [], [], [], []

    for key in groups_id.keys():

        print(f"\n-------------------------- Processing {key} group --------------------------")

        all_neurons = 0
        magnitude = {protocol: [] for protocol in sub_protocols} #Will contain the magnitude of the response to each protocol for each neuron
        avg_data = {protocol: [] for protocol in sub_protocols} 
        stim_mean = {protocol: [] for protocol in sub_protocols} #Will contain the mean response to the stimulus for each protocol, per session
        proportion_list = [] #Will contain the proportion of responding neurons per session

        df_filtered = df[df["Genotype"] == key]

        for k in range(len(df_filtered)):
            mouse_id = df_filtered["Mouse_id"].iloc[k]
            session_id = df_filtered["Session_id"].iloc[k]
            output_id = df_filtered["Output_id"].iloc[k]
            session_path = os.path.join(df_filtered["Session_path"].iloc[k], f"{session_id}_output_{output_id}")

            print(f"\nSession id: {session_id}\n  Mouse id : {mouse_id}\n     Session path: {session_path}")

            validity, trials, stimuli_df = load_data_session(session_path)

            valid_neurons = get_valid_neurons_session(validity, valid_sub_protocols)
            neurons = len(valid_neurons)
            all_neurons += neurons
            
            proportion = 100 * len(valid_neurons) / trials[period_names[1]][0].shape[0]
            proportion_list.append(proportion)
            print(f"Proportion responding neurons: {proportion}, Number of responding neurons: {neurons}")
            
            for protocol in sub_protocols:

                stim_id = stimuli_df[stimuli_df.name == protocol].index[0]

                # Get z-scores from responsive-neurons for that protocol from pre, stim and post periods and concatenate along time
                traces_sep = [trials[period][stim_id][valid_neurons, :] for period in period_names]
                traces_concat = np.concatenate(traces_sep, axis=1)
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

    return suppression_groups, magnitude_groups, stim_groups, nb_neurons, avg_groups, sem_groups, cmi_groups, proportions_groups

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

            slope, intercept, r_value, p_value, _ = linregress(x_values, y_values)
            x_fit = np.linspace(min(x_values), max(x_values), 100)
            y_fit = slope * x_fit + intercept

            # Create the plot
            plt.scatter(x_values, y_values, marker='o', c=color[group], alpha=0.5, label=f'{group} neurons')
            # Plot regression line
            
            plt.plot(x_fit, y_fit, color=color[group], label=f'{group} fit: y = {slope:.2f}x + {intercept:.2f}, p = {p_value:.2g}, r**2 = {(r_value)**2:.3f}')
        
        # Add labels
        plt.xlabel(f"Magnitude of response to {protocol_x}")
        plt.ylabel(f"Magnitude of response to {protocol_y}")
        plt.title(f"Response magnitudes ({attr}) for {protocol_validity}-responsive neurons")
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
    ax.set_ylabel(f'% of {valid_sub_protocols}-responsive neurons')
    ax.set_title(f'% of responsive neurons per session')
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

    for i, protocol in enumerate(protocols):

        for k, group in enumerate(groups_id.keys()):
            x = offset * i + width * k
            x_ticks.append(x)
            x_labels.append(f'{protocol}\n{group}')

            stim_mean = stim_groups[groups_id[group]][protocol]
            stim_global_mean = np.mean(stim_mean)

            ax.bar(x, stim_global_mean, color=colors[k], width=width, edgecolor='black', label=group if i == 0 else "")
            ax.scatter([x] * len(stim_mean), stim_mean, color='black', zorder=10)

        if len(groups_id) == 2 :
            # Mann–Whitney U test
            stat, p = mannwhitneyu(stim_groups[0][protocol], stim_groups[0][protocol], alternative='two-sided')

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
    
    for i, group in enumerate(groups):
        avg = avg_groups[groups_id[group]]
        sem = sem_groups[groups_id[group]]
        neurons = nb_neurons[groups_id[group]]
        
        # Get minimum length among all protocols
        min_len = min(len(avg[protocol]) for protocol in protocols)

        # Generate time vector accordingly
        time = np.linspace(0, min_len, min_len) / frame_rate - 1  # time in seconds

        for protocol in protocols:
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
        axs[i].set_ylabel(f"Average {attr} for neurons responsive to {protocol_validity}")
        axs[i].set_title(f"Mean {attr} ± SEM by Protocol for {group}")
        axs[i].legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    axs[2].set_xticks(np.arange(-1, time[-1] + 1, 1))
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel(f"Average {attr} for neurons responsive to {protocol_validity}")
    axs[2].set_title(f"Mean {attr} ± SEM by Group and Protocol")
    axs[2].legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    # Hide the unused subplot (bottom-right)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.2)
    fig.savefig(os.path.join(save_path, f"{fig_name}_averages_{attr}.jpeg"), dpi=300, bbox_inches='tight')
    plt.show()

def histplot(sub_protocols, list1, list2, groups, save_path, fig_name, attr, variable="CMI"):
    """
    Function to plot a histogram comparing the distribution of two groups.
    variable: 'CMI' or 'suppression_index'
    """
    edgecolor = 'black'
    labels_list = []
    if len(sub_protocols) == 2:
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
        
        genotype = ["WT"] * len(list1) + ["KO"] * len(list2)
        df = pd.DataFrame({"Genotype" : genotype, variable : pd.Categorical(labels_list, categories=labels, ordered=True)})

        fig, ax = plt.subplots(figsize=(12, 7))
        sns.histplot(df, x=variable, hue="Genotype", multiple="dodge", edgecolor=edgecolor, ax=ax, shrink=.8)
        plt.ylabel(f'% of neurons')
        plt.xticks(rotation=45, ha='right')
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
    else:
        print(f"Histogram is not available for {len(sub_protocols)} protocols. Please select 2 protocols to compare.")
        print(f"Current protocols: {sub_protocols}")
        return None
    
if __name__ == "__main__":

    #-----------------------INPUTS-----------------------#

    excel_sheet_path = "./src/visualpipe/post_analysis/Nathan_sessions.xlsx"
    save_path = r"Y:\raw-imaging\Nathan\PYR\surround_mod\Analysis"
    
    #Will be included in all names of saved figures
    fig_name = 'CenterVsfbRF_Iso'

    #Name of the protocol to analyze (e.g. 'surround-mod', 'visual-survey'...)
    protocol_name = "surround-mod"

    # Write the protocols you want to plot 
    sub_protocols = ['center', 'surround-iso_ctrl']  
    # List of protocol(s) used to select reponsive neurons. If contains several protocols, neurons will be selected if they are responsive to at least one of the protocols in the list.
    valid_sub_protocols = ['center'] 

    #Frame rate
    frame_rate = 30

    # Decide if you want to plot the dFoF0 baseline substraced or the z-scores
    attr = 'dFoF0-baseline'  # 'dFoF0-baseline' or 'z_scores'

    #----------------------------------------------------#
    df = load_excel_sheet(excel_sheet_path, protocol_name)

    groups_id = {'WT': 0, 'KO': 1}

    suppression_groups, magnitude_groups, stim_groups, nb_neurons, avg_groups, sem_groups, cmi_groups, proportions_groups = process_group(df, groups_id, attr, valid_sub_protocols, sub_protocols, protocol_name)

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