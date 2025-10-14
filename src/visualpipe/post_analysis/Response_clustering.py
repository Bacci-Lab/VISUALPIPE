import os
os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances
from scipy.stats import mannwhitneyu, wilcoxon, linregress
import glob
from kneed import KneeLocator 
import math
import pandas as pd
import sys
sys.path.append("./src")

import visualpipe.post_analysis.utils as utils

def get_valid_neurons_session(validity, protocol):
    """
    Return the indices of responsive neurons for a given protocol in a session.
    
    Parameters
    ----------
    validity : dict
        Dictionary containing the validity data loaded from the .npz file.
    protocol : str
        Name of the protocol (stimulus) to select the responsive neurons for.
    
    Returns
    -------
    valid_neurons : array-like
        Indices of responsive neurons for the given protocol.
    """
    # Dictionary to store responsive neurons for each protocol
    valid_data = {protocol : validity[protocol] 
                 if protocol in validity 
                 else print(f"{protocol} does not exist in validity file.") }
    
    valid_neuron_lists = [np.where(data[:, 0] == 1,)[0] for data in valid_data.values()] # change to -1 if you want negative responsive neurons
    valid_neurons = np.unique(np.concatenate(valid_neuron_lists))  # Get unique indices of valid neurons
    
    return valid_neurons

def process_group(df:pd.DataFrame, groups_id:dict, sub_protocol:str, frame_rate:float, attr:str='z-scores'):
    """
    Process all neurons from a given group of mice and compute their 
    (1) normalized traces to be used for clustering, 
    (2) mean and SEM of the average traces across sessions, 
    and (3) list of all neurons.

    Parameters
    ----------
    data_path : str
        Path to the directory containing the data.
    protocol : str
        Name of the protocol for which the neurons should be processed.
    group_name : str
        Name of the group of mice.
    frame_rate : float
        Frame rate in Hz.
    mice_list : list, optional
        List of mice to process. If None, all mice in the group will be processed.

    Returns
    -------
    normalized_traces : array, shape (n_neurons, n_timepoints)
        Normalized traces for all neurons in the group.
    single_traces : array, shape (n_neurons, n_timepoints)
        Traces for all neurons in the group without normalization.
    magnitude : array, shape (n_neurons,)
        Magnitude of the response for each neuron.
    response_mean : array, shape (n_neurons,)
        Mean response for each neuron.
    all_neurons : int
        Total number of neurons in the group.
    mouse_avg_zscore : array, shape (n_timepoints,)
        Average trace across all neurons in the group.
    mouse_sem_zscore : array, shape (n_timepoints,)
        SEM of the traces across all neurons in the group.
    """

    if attr == 'dFoF0-baseline':
        period_names = ['norm_averaged_baselines', 'norm_trial_averaged_ca_trace', 'norm_post_trial_averaged_ca_trace']
    elif attr == 'z-scores':
        period_names = ['pre_trial_averaged_zscores', 'trial_averaged_zscores', 'post_trial_averaged_zscores']

    normalized_traces_groups, single_traces_groups, magnitude_groups, response_mean_groups, all_neurons_groups, mouse_avg_zscore_groups, mouse_sem_zscore_groups = [], [], [], [], [], [], []

    for key in groups_id.keys():
        print(f"\n-------------------------- Processing {key} group --------------------------")

        magnitude = []
        avg_data = []
        all_neurons = 0
        stim_mean = []
        single_traces = []

        df_filtered = df[df["Genotype"] == key]

        for k in range(len(df_filtered)):
            mouse_id = df_filtered["Mouse_id"].iloc[k]
            session_id = df_filtered["Session_id"].iloc[k]
            output_id = df_filtered["Output_id"].iloc[k]
            session_path = os.path.join(df_filtered["Session_path"].iloc[k], f"{session_id}_output_{output_id}")

            print(f"\nSession id: {session_id}\n  Mouse id : {mouse_id}\n     Session path: {session_path}")

            validity, trials, stimuli_df = utils.load_data_session(session_path)

            # Determine which neurons are valid
            valid_neurons = get_valid_neurons_session(validity, sub_protocol)
            nb_valid_neurons = len(valid_neurons)
            all_neurons += nb_valid_neurons

            proportion = 100 * nb_valid_neurons / trials['trial_averaged_zscores'][0].shape[0]
            print(f"    Proportion of {sub_protocol} responding neurons: {proportion}")
            
            stim_id = stimuli_df[stimuli_df.name == sub_protocol].index[0]

            traces_sep = [trials[period][stim_id][valid_neurons, :] for period in period_names]
            traces_concat = np.concatenate(traces_sep, axis=1)
            single_traces.append(traces_concat)

            avg_session_trace = np.mean(traces_concat, axis=0)
            avg_data.append(avg_session_trace)
            
            # Extract trial-averaged traces and compute peak magnitud
            trial_traces = np.array(trials[period_names[1]][stim_id][valid_neurons, int(frame_rate*0.5):])
            mean_stim_response = np.mean(trial_traces, axis=1)  # average per neuron over time
            session_stim_mean = np.mean(mean_stim_response)  # average over neurons in that session
            stim_mean.append(session_stim_mean)

            idx_max_neurons = np.argmax(np.abs(trial_traces), axis=1) # peak response
            for i, idx in enumerate(idx_max_neurons) :
                magnitude.append(trial_traces[i][idx])
    
        #This one will be used to plot average responses per cluster (not normalized)
        single_traces = np.concatenate(single_traces, axis=0)
        magnitudes_array = np.array(magnitude)
        
        #This one will be used to cluster only based on response shape (normalized)
        normalized_traces = single_traces / magnitudes_array.reshape(-1, 1)
        print(f"\nNumber of {key} neurons: {all_neurons}")

        # Concatenate all neuron arrays into one array per protocol
        avg_data = np.stack(avg_data, axis=0)

        # Compute average and SEM across neurons
        mouse_avg_zscore = np.mean(avg_data, axis=0)
        mouse_sem_zscore = stats.sem(avg_data, axis=0)

        normalized_traces_groups.append(normalized_traces)
        single_traces_groups.append(single_traces)
        magnitude_groups.append(magnitudes_array)
        response_mean_groups.append(stim_mean)
        all_neurons_groups.append(all_neurons)
        mouse_avg_zscore_groups.append(mouse_avg_zscore)
        mouse_sem_zscore_groups.append(mouse_sem_zscore)

    return normalized_traces_groups, single_traces_groups, magnitude_groups, response_mean_groups, all_neurons_groups, mouse_avg_zscore_groups, mouse_sem_zscore_groups

def dunn_index_pointwise(X, labels):
    """
    Compute the Dunn Index based on pointwise distances (original formulation).

    Parameters:
        X: np.ndarray of shape (n_samples, n_features)
        labels: np.ndarray of shape (n_samples,)

    Returns:
        Dunn index (float)
    """
    unique_clusters = np.unique(labels)
    n_clusters = len(unique_clusters)

    # Store intra- and inter-cluster distances
    intra_dists = []
    inter_dists = []

    for i, ci in enumerate(unique_clusters):
        Xi = X[labels == ci]
        # Intra-cluster diameter
        if len(Xi) > 1:
            intra = np.max(cdist(Xi, Xi))
        else:
            intra = 0  # singleton cluster
        intra_dists.append(intra)

        # Inter-cluster distances to other clusters
        for j, cj in enumerate(unique_clusters):
            if j <= i:
                continue
            Xj = X[labels == cj]
            inter = np.min(cdist(Xi, Xj))
            inter_dists.append(inter)

    min_inter = np.min(inter_dists)
    max_intra = np.max(intra_dists)

    if max_intra == 0:
        return np.inf

    return min_inter / max_intra

def plot_dunn_index(k_range, dunn_scores, group, save_path=None, fig_name='', show=True): 
    """
    Plot the Dunn Index for a range of cluster numbers.

    Parameters:
        k_range (list or array): range of cluster numbers
        dunn_scores (list or array): corresponding Dunn Index scores
        group (str): name of the group, for plotting
        save_path (str, optional): path to save the plot, defaults to None
        fig_name (str, optional): prefix for the figure name, defaults to ''
        show (bool, optional): whether to show the plot, defaults to True
    """
    plt.plot(k_range, dunn_scores, marker='o')
    plt.xlabel("Number of clusters")
    plt.ylabel(f"Dunn Index for {group}")
    plt.title(f"Dunn Index vs Number of Clusters for {group}")
    if save_path is not None:
        plt.savefig(os.path.join(save_path, f"{fig_name}_{group}_dunn_index.jpg"))
    if show:
        plt.show()
    plt.close()

def elbow_method(var=None, data=None, var_type='wcss', group='', max_nb_clusters=10, save_path=None, fig_name='', show=True):
    """
    Use the elbow method to determine the optimal number of clusters for a given dataset.

    The elbow method is a heuristic method for determining the number of clusters in a dataset by plotting the distortion (wcss) or inter-cluster distance as a function of the number of clusters.

    Parameters:
        var (list or array, optional): the distortion or inter-cluster distances to plot, defaults to None
        data (array-like, optional): the data to cluster, defaults to None
        var_type (str, optional): the type of the distortion or inter-cluster distances, either 'wcss' or 'interdistance', defaults to 'wcss'
        group (str, optional): the name of the group, for plotting, defaults to ''
        max_nb_clusters (int, optional): the maximum number of clusters to consider, defaults to 10
        save_path (str, optional): path to save the plot, defaults to None
        fig_name (str, optional): prefix for the figure name, defaults to ''
        show (bool, optional): whether to show the plot, defaults to True

    Returns:
        int: the optimal number of clusters given by the elbow method
    """
    k_range = range(2, max_nb_clusters+1)

    if var is None :
        if data is not None:
            var = []
            for i in k_range:
                kmeans = KMeans(n_clusters=i, n_init=10, random_state=0)
                kmeans.fit(data)
                if var_type == 'wcss':
                    var.append(kmeans.inertia_)
                elif var_type == 'interdistance' :
                    centroids = kmeans.cluster_centers_  # Get centroids from the fitted k-means model
                    intercluster_distances = pairwise_distances(centroids) # Compute pairwise distances between centroids (Euclidean by default)
                    unique_distances = intercluster_distances[np.triu_indices_from(intercluster_distances, k=1)]  # Extract upper triangle (excluding diagonal) to avoid redundancy
                    var.append(np.sum(unique_distances))
        else :
            raise Exception("You must provide either var or X")

    if var_type == 'wcss':
        direction = 'decreasing'
    elif var_type == 'interdistance' :
        direction = 'increasing'
    else :
        raise Exception("var_type must be either wcss or interdistance")

    knee = KneeLocator(k_range, var, curve='convex', direction=direction)
    k_elbow = knee.knee or max_nb_clusters

    plt.plot(k_range, var, marker='o')
    plt.axvline(k_elbow, color='g', linestyle='--', label=f'Elbow k={k_elbow}')
    plt.xlabel("Number of clusters")
    plt.ylabel(f"{var_type} ({group})")
    plt.title(f"Elbow Method for {group}")
    if save_path is not None:
        plt.savefig(os.path.join(save_path, f"{fig_name}_{var_type}_{group}_elbow_method.jpg"))
    if show:
        plt.show()
    plt.close()

    return k_elbow

def find_nb_clusters(traces, max_clusters:int=10, save_path=None, fig_name='', group='all', show=True) :
    """
    Determine the optimal number of clusters for given data using Dunn Index, inertia, and inter-cluster distance metrics.

    Parameters:
        traces (np.ndarray): The data to cluster, with shape (n_samples, n_features).
        max_clusters (int, optional): Maximum number of clusters to consider. Defaults to 10.
        save_path (str, optional): Path to save the plots, if specified. Defaults to None.
        fig_name (str, optional): Prefix for the figure name when saving plots. Defaults to ''.
        group (str, optional): Name of the group for labeling plots. Defaults to 'all'.
        show (bool, optional): Whether to display the plots. Defaults to True.

    Returns:
        tuple: Contains the optimal number of clusters based on inertia (elbow method) and inter-cluster distance.
    """

    dunn_scores = []
    inertias = []
    interdistance = []

    k_range = list(range(2, max_clusters+1))

    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=50).fit(traces)
        labels = kmeans.labels_
        dunn_scores.append(dunn_index_pointwise(traces, labels))
        inertias.append(kmeans.inertia_) #this is the sum of squared distances of all points to their closest cluster centroid
        
        # Get centroids from the fitted k-means model
        centroids = kmeans.cluster_centers_
        # Compute pairwise distances between centroids (Euclidean by default)
        intercluster_distances = pairwise_distances(centroids)

        # Extract upper triangle (excluding diagonal) to avoid redundancy
        unique_distances = intercluster_distances[np.triu_indices_from(intercluster_distances, k=1)]
        interdistance.append(np.sum(unique_distances))

    k_elbow_inertia = elbow_method(var=inertias, group=group, var_type='wcss', max_nb_clusters=max_clusters, save_path=save_path, fig_name=fig_name, show=show)

    k_elbow_interdist = elbow_method(var=interdistance, group=group, var_type='interdistance', max_nb_clusters=max_clusters,save_path=save_path, fig_name=fig_name,show=show)

    plot_dunn_index(k_range, dunn_scores, group, save_path=save_path, fig_name=fig_name, show=show)

    return k_elbow_inertia, k_elbow_interdist

def get_avg_responses_clusters(traces, cluster_data, time, nclusters, labels, xticks=None, attr='z-scores',
                                group='all', save_path=None, fig_name='', show=True):

    if attr == 'z-scores':
        ylabel = 'Z-scored ΔF/F0'
    elif attr == 'dFoF0-baseline':
        ylabel = 'ΔF/F0 baseline substracted'
    else:
        raise Exception("attr must be either 'z-scores' or 'dFoF0-baseline'")

    colors = plt.get_cmap('tab10', nclusters)  # get distinct colors

    fig = plt.figure()
    for k in range(nclusters):
        cluster_k = traces[labels == k]
        mean_k = np.mean(cluster_k, axis=0)
        sem_k = stats.sem(cluster_k, axis=0)
        label_k = f'Cluster {k} (n={cluster_k.shape[0]})'
        cluster_data[group][k] = {'mean': mean_k, 'sem': sem_k, 'n': cluster_k.shape[0]}

        plt.fill_between(time, mean_k - sem_k, mean_k + sem_k,
                        alpha=0.3, color=colors(k), label=label_k)
        plt.plot(time, mean_k, color=colors(k))

    plt.xlabel('Time (s)')
    plt.ylabel(ylabel)
    plt.title(f'Average Response per Cluster for {group}')
    if xticks is not None: plt.xticks(xticks)
    plt.legend()
    
    if save_path is not None:
        fig.savefig(os.path.join(save_path, f"{fig_name}_{group}_{nclusters}clusters_averaged.jpg"))
    if show :
        plt.show()
    plt.close()

    return cluster_data

def compare_clusters_traces(wt_cluster, ko_cluster, cluster_data, time, xticks=None, attr='z-scores', save_path=None, fig_name='', show=True):
    """
    Compare average traces of two clusters, one from WT and one from KO.

    Parameters:
        wt_cluster (int): Index of the WT cluster.
        ko_cluster (int): Index of the KO cluster.
        cluster_data (dict): Dictionary containing the data for each cluster, with keys 'WT' and 'KO'.
        time (array): Array of time points.
        xticks (array, optional): Array of x-ticks. Defaults to None.
        attr (str, optional): Type of data to plot. Can be either 'z-scores' or 'dFoF0-baseline'. Defaults to 'z-scores'.
        save_path (str, optional): Path to save the figure. Defaults to None.
        fig_name (str, optional): Prefix for the figure name. Defaults to ''.
        show (bool, optional): Whether to display the plot. Defaults to True.

    Returns:
        None
    """
    if attr == 'z-scores':
        ylabel = 'Z-scored ΔF/F0'
    elif attr == 'dFoF0-baseline':
        ylabel = 'ΔF/F0 baseline substracted'
    else:
        raise Exception("attr must be either 'z-scores' or 'dFoF0-baseline'")
    
    fig = plt.figure()

    # WT
    mean_wt = cluster_data['WT'][wt_cluster]['mean']
    sem_wt = cluster_data['WT'][wt_cluster]['sem']
    n_wt = cluster_data['WT'][wt_cluster]['n']
    plt.plot(time, mean_wt, label=f'WT Cluster {wt_cluster} (n={n_wt})', color='blue')
    plt.fill_between(time, mean_wt - sem_wt, mean_wt + sem_wt, color='blue', alpha=0.3)

    # KO
    mean_ko = cluster_data['KO'][ko_cluster]['mean']
    sem_ko = cluster_data['KO'][ko_cluster]['sem']
    n_ko = cluster_data['KO'][ko_cluster]['n']
    plt.plot(time, mean_ko, label=f'KO Cluster {ko_cluster} (n={n_ko})', color='red')
    plt.fill_between(time, mean_ko - sem_ko, mean_ko + sem_ko, color='red', alpha=0.3)

    plt.xlabel('Time (s)')
    plt.ylabel(ylabel)
    plt.title(f'Comparison: WT Cluster {wt_cluster} vs KO Cluster {ko_cluster}')
    plt.xticks(xticks)
    plt.legend()

    if save_path is not None:
        fig.savefig(os.path.join(save_path, f"{fig_name}_WT{wt_cluster}_vs_KO{ko_cluster}.jpg"))
    if show :
        plt.show()
    plt.close()

def pie_chart_clusters(percentages, n_clusters_joint, group, attr, save_path=None, fig_name='', show=True):
    """
    Plot a pie chart of the percentage of neurons in each cluster for a given group.

    Parameters
    ----------
    percentages : array-like
        An array of percentages, where each element is the percentage of neurons
        in a given cluster.
    n_clusters_joint : int
        The number of clusters.
    group : str
        The group name, either 'WT' or 'KO'.
    save_path : str, optional
        The path to save the figure. If None, the figure will not be saved.
    fig_name : str, optional
        The name of the figure. If None, a default name will be used.
    show : bool, optional
        Whether to display the plot. Defaults to True.
    """
    fig = plt.figure()
    
    plt.pie(percentages, labels=[f'Cluster {i}' for i in range(n_clusters_joint)],
            autopct='%1.1f%%', colors=plt.cm.tab10.colors[:n_clusters_joint], startangle=90)
    plt.title(f'{group}: % of neurons in each cluster')
    plt.axis('equal')

    if save_path is not None:
        file_name = f"{fig_name}_{group}_cluster_pie_{attr}.jpg"
        fig.savefig(os.path.join(save_path, file_name))
    if show:
        plt.show()
    plt.close()

def plot_avg_cluster_traces_group(cluster_id, joint_cluster_data, time, attr, xticks=None, save_path=None, fig_name='', show=True):
    """
    Plot the average traces for a given cluster in both WT and KO groups to compare their responses.

    Parameters
    ----------
    cluster_id : int
        The ID of the cluster to be plotted.
    joint_cluster_data : dict
        A dictionary containing the means and SEMs of the traces for each group.
    time : array-like
        A time array.
    xticks : array-like, optional
        Optional xticks for the plot.
    save_path : str, optional
        The path to save the figure. If None, the figure will not be saved.
    fig_name : str, optional
        The name of the figure. If None, a default name will be used.
    show : bool, optional
        Whether to show the plot. Default is True.

    Returns
    -------
    None
    """
    mean_wt = joint_cluster_data['WT']['mean']
    sem_wt = joint_cluster_data['WT']['sem']
    n_wt = joint_cluster_data['WT']['n']

    mean_ko = joint_cluster_data['KO']['mean']
    sem_ko = joint_cluster_data['KO']['sem']
    n_ko = joint_cluster_data['KO']['n']

    fig = plt.figure(figsize=(8, 5))
    plt.fill_between(time, mean_wt - sem_wt, mean_wt + sem_wt, alpha=0.3, color='blue', label=f'WT (n={n_wt})')
    plt.plot(time, mean_wt, color='blue')

    plt.fill_between(time, mean_ko - sem_ko, mean_ko + sem_ko, alpha=0.3, color='red', label=f'KO (n={n_ko})')
    plt.plot(time, mean_ko, color='red')

    plt.title(f'Cluster {cluster_id}: Average {attr} (WT vs KO)')
    plt.xlabel('Time (s)')
    plt.ylabel('Z-scored ΔF/F')
    plt.xticks(xticks)
    plt.legend()

    if save_path is not None :
        fig.savefig(os.path.join(save_path, f"{fig_name}_cluster{cluster_id}_WTvsKO_{attr}.jpg"))
    if show:
        plt.show()
    plt.close()
    
def plot_raster_cluster_group(cluster_id, norm_traces, idx_wt, idx_ko, time, attr, xticks=None, save_path=None, fig_name='', show=True):
    """
    Plot raster plots of normalized neural traces for two groups of neurons in the same cluster.

    Parameters
    ----------
    cluster_id : int
        The id of the cluster to plot.
    norm_traces : array-like
        2D array of normalized traces for the two groups of neurons. The first dimension corresponds to the neuron, the second to the time.
    idx_wt : array-like
        Indices of the neurons in the WT group.
    idx_ko : array-like
        Indices of the neurons in the KO group.
    time : array-like
        Time points of the traces.
    xticks : array-like, optional
        Time points to display on the x-axis.
    save_path : str, optional
        Path to save the figure to.
    fig_name : str, optional
        Base name of the figure.
    show : bool, optional
        Whether to display the figure.

    Returns
    -------
    None
    """
    vmin, vmax = np.nanmin(norm_traces), np.nanmax(norm_traces)
    if np.abs(vmin) > np.abs(vmax) :
        lim = math.floor(np.abs(vmin)*100) * 0.01
    else : 
        lim = math.floor(np.abs(vmax)*100) * 0.01

    fig, axs = plt.subplots(1,2,figsize=(10, 6))

    for i, group, idx in zip(range(2), ['WT', 'KO'], [idx_wt, idx_ko]) :
        norm = norm_traces[idx]
        axs[i].imshow(norm, aspect='auto', extent=[time[0], time[-1], 0, norm.shape[0]],
                      cmap='RdBu_r', vmin=-lim, vmax=lim)
        axs[i].set_title(f'Cluster {cluster_id} - {group} ({norm.shape[0]} neurons) - {attr}')
        axs[i].set_ylabel('Neuron')
        axs[i].set_xlabel('Time (s)')
        if xticks is not None :
            axs[i].set_xticks(xticks)
        axs[i].set_yticks(np.arange(0, norm.shape[0]-1, 10))  # y-ticks for raster plots
    
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(os.path.join(save_path, f"{fig_name}_cluster{cluster_id}_rasters_{attr}.jpg"), dpi=300)
    if show:
        plt.show()
    plt.close()
    
if __name__ == "__main__":

    excel_sheet_path = r"Y:\raw-imaging\Nathan\Nathan_sessions_visualpipe.xlsx"
    save_path = r"Y:\raw-imaging\Nathan\PYR\Visualpipe_postanalysis\vision_survey\Analysis"

    #Will be included in all names of saved figures
    fig_name = 'Looming-stim-1stSessions'

    #Name of the protocol to analyze (e.g. 'surround-mod', 'visual-survey'...)
    protocol_name = 'vision-survey'

    # Write the stimulus type you want to use for clustering
    sub_protocol = 'looming-stim'

    attr='dFoF0-baseline'  # 'z-scores' or 'dFoF0-baseline'

    #Frame rate
    frame_rate = 30

    max_clusters = 10

    #-------------------------------------------------------------------------#

    df = utils.load_excel_sheet(excel_sheet_path, protocol_name)

    groups_id = {'WT': 0, 'KO': 1}

    normalized_traces_groups, single_traces_groups, magnitude_groups, response_mean_groups, all_neurons_groups, mouse_avg_zscore_groups, mouse_sem_zscore_groups = process_group(df, groups_id, sub_protocol, frame_rate, attr=attr)

    """
    #------------------- Cluster the two groups separately -------------------#

    #To determine the ideal number of clusters
    for group in groups_id.keys():
        traces = normalized_traces_groups[groups_id[group]]
        find_nb_clusters(traces, max_clusters=max_clusters, save_path=save_path, fig_name=fig_name, group=group, show=False)
     
    # Choose a number of clusters
    k_clusters = {'WT': 5, 'KO': 4}  # Example: you can set different numbers for each group
    
    #------------------- Run KMeans clustering
    
    # Dictionaries to store cluster mean and SEM for each group
    cluster_data = {'WT': {}, 'KO': {}}
    time = (np.arange(single_traces_groups[0].shape[1]) / frame_rate) -1 
    xticks = np.arange(-1, time[-1] + 1, 1)  # ticks every 1 second

    for group in groups_id.keys():
        norm = normalized_traces_groups[groups_id[group]]
        traces = single_traces_groups[groups_id[group]]
        nclusters = k_clusters[group]

        kmeans = KMeans(n_clusters=nclusters, n_init=100).fit(norm)
        labels = kmeans.labels_

        cluster_data = get_avg_responses_clusters(traces, cluster_data, time, nclusters, labels, xticks=xticks, attr=attr, group=group, save_path=save_path, fig_name=fig_name, show=False)

    #------------------- 🆕 Plot comparison between Cluster n from WT and Cluster k from KO
    wt_cluster = 0  # <-- change this to the WT cluster index you want to compare
    ko_cluster = 3  # <-- change this to the KO cluster index you want to compare

    compare_clusters_traces(wt_cluster, ko_cluster, cluster_data, time, xticks=xticks, attr=attr, save_path=save_path, fig_name=fig_name, show=False) """



    #-------------To cluster WT and KO together and then compare % of neurons in each cluster------------#

    #To determine the ideal number of clusters
    norm_traces = np.concatenate(normalized_traces_groups, axis=0)
    find_nb_clusters(norm_traces, max_clusters=max_clusters, save_path=save_path, fig_name=fig_name, show=False)

    # Set number of clusters for joint clustering
    n_clusters_joint = 4  # choose based on elbow/Dunn index as before

    #------------------- Run KMeans clustering on the combined data
    kmeans = KMeans(n_clusters=n_clusters_joint, n_init=50).fit(norm_traces)
    cluster_labels = kmeans.labels_
    
    all_traces = np.concatenate(single_traces_groups, axis=0)
    group_labels = np.array(['WT'] * all_neurons_groups[groups_id['WT']] + ['KO'] * all_neurons_groups[groups_id['KO']])

    # Time axis
    time = (np.arange(all_traces.shape[1]) / frame_rate) - 1
    xticks = np.arange(-1, time[-1] + 1, 1)

    # For storing cluster data
    joint_cluster_data = {}
    wt_cluster_counts = np.zeros(len(np.unique(cluster_labels)), dtype=int)
    ko_cluster_counts = np.zeros(len(np.unique(cluster_labels)), dtype=int)

    # Plot per-cluster average traces for WT and KO
    for k in range(n_clusters_joint):

        idx_cluster = np.where(cluster_labels == k)[0]
        wt_cluster_counts[k] = np.sum(group_labels[idx_cluster] == 'WT')
        ko_cluster_counts[k] = np.sum(group_labels[idx_cluster] == 'KO')

        idx_wt = idx_cluster[group_labels[idx_cluster] == 'WT']
        idx_ko = idx_cluster[group_labels[idx_cluster] == 'KO']
        traces_wt_k = all_traces[idx_wt]
        traces_ko_k = all_traces[idx_ko]
        mean_wt = np.mean(traces_wt_k, axis=0) if len(idx_wt) > 0 else np.zeros(all_traces.shape[1])
        sem_wt = stats.sem(traces_wt_k, axis=0) if len(idx_wt) > 1 else np.zeros(all_traces.shape[1])
        mean_ko = np.mean(traces_ko_k, axis=0) if len(idx_ko) > 0 else np.zeros(all_traces.shape[1])
        sem_ko = stats.sem(traces_ko_k, axis=0) if len(idx_ko) > 1 else np.zeros(all_traces.shape[1])
        joint_cluster_data[k] = {
            'WT': {'mean': mean_wt, 'sem': sem_wt, 'n': len(idx_wt)},
            'KO': {'mean': mean_ko, 'sem': sem_ko, 'n': len(idx_ko)}
        }

        # Plot raster plots
        plot_raster_cluster_group(k, norm_traces, idx_wt, idx_ko, time, attr, xticks, save_path, fig_name, show=True)
        # Plot traces
        plot_avg_cluster_traces_group(k, joint_cluster_data[k], time, attr, xticks, save_path, fig_name, show=True)

    # Normalize to percentages
    wt_percentages = 100 * wt_cluster_counts / np.sum(group_labels == 'WT')
    ko_percentages = 100 * ko_cluster_counts / np.sum(group_labels == 'KO')
    cluster_proportions = {
        'WT': wt_percentages,
        'KO': ko_percentages
    }
    print(cluster_proportions)
    
    # Plot pie chart for WT
    pie_chart_clusters(wt_percentages, n_clusters_joint, 'WT', attr = attr, save_path=save_path, fig_name=fig_name, show=True)

    # Plot pie chart for KO
    pie_chart_clusters(ko_percentages, n_clusters_joint, 'KO', attr = attr, save_path=save_path, fig_name=fig_name, show=True) 
