import numpy as np
import h5py
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.stats import mannwhitneyu, wilcoxon, linregress
import glob
from kneed import KneeLocator 


## Organization of the data: put validity2.npz and trials.npy files in a folder, and seperate the groups in subfolders (e.g WT and KO)
#The folder containing all subfolders
data_path = r"Y:\raw-imaging\Nathan\PYR\vision_survey"
save_path = data_path
# groups as to be the name of the subfolders
groups = ['WT', 'KO']
# mice per group (sub-subfolders
WT_mice = ['110','108']
KO_mice = ['109','112']
#Will be included in all figure names
fig_name = 'looming-stim'
# Write the protocols you want to plot
protocols = ['looming-stim']
# Protocol used to select reponsive neurons
protocol_validity = 'looming-stim'
#Frame rate
frame_rate = 30



z_score_periods = ['pre_trial_averaged_zscores', 'trial_averaged_zscores', 'post_trial_averaged_zscores']
x_labels = ['Pre-stim', 'Stim', 'Post-stim']

import numpy as np
import os
import glob

# ... (keep your imports and initial variables as is)

def process_group(group_name, mice_list):
    magnitude = {protocol: [] for protocol in protocols}
    avg_data = {protocol: [] for protocol in protocols}
    all_neurons = 0
    stim_mean = {protocol: [] for protocol in protocols}
    single_traces = {protocol: [] for protocol in protocols}
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
            keys = list(trials['trial_averaged_zscores'].keys())

            # Determine which neurons are valid
            valid_data = validity[protocol_validity]
            valid_neurons = np.where(valid_data[:, 0] == 1)[0]
            neurons = len(valid_neurons)
            proportion = 100 * len(valid_neurons) / trials['trial_averaged_zscores'][0].shape[0]
            #print(f"Proportion responding neurons: {proportion}")
            all_neurons += neurons
            all_protocols = validity.files
            avg_session = []
            for protocol in protocols:
                idx = all_protocols.index(protocol)

                # Get z-scores from defined periods and concatenate along time
                zscores_periods = [trials[period][list(trials[period].keys())[idx]][valid_neurons, :] for period in z_score_periods]
                zscores_concat = np.concatenate(zscores_periods, axis=1)
                single_traces[protocol].append(zscores_concat)
                avg_session = np.mean(zscores_concat, axis=0)
                avg_data[protocol].append(avg_session)
                # Extract trial-averaged zscores and compute peak magnitude
                trial_zscores = trials['trial_averaged_zscores'][list(trials['trial_averaged_zscores'].keys())[idx]][valid_neurons, :]
                mean_stim_response = np.mean(trial_zscores[:,int(frame_rate*0.5):], axis=1)  # average per neuron over time
                session_stim_mean = np.mean(mean_stim_response)  # average over neurons in that session
                stim_mean[protocol].append(session_stim_mean)
                for n in range(neurons):
                    zneuron = trial_zscores[n, int(frame_rate*0.5):]
                    magnitude[protocol].append(np.mean(zneuron))
    single_traces = np.concatenate(single_traces['looming-stim'], axis=0)
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
    #if protocols[0] in accepted_protocols and protocols[1] in accepted_protocols:
    #    suppression = [
    #            1 - float(magnitude[protocols[1]][n]) / float(magnitude[protocols[0]][n])
    #            if magnitude[protocols[0]][n] != 0 else np.nan
    #            for n in range(all_neurons)
    #    ]
    #else:
    #    suppression = []

    suppression = []
    # Concatenate all neuron arrays into one array per protocol
    for protocol in protocols:
        avg_data[protocol] = np.stack(avg_data[protocol], axis=0)

    # Compute average and SEM across neurons
    avg = {protocol: np.mean(avg_data[protocol], axis=0) for protocol in protocols}
    sem = {protocol: stats.sem(avg_data[protocol], axis=0) for protocol in protocols}

    return single_traces, suppression, magnitude, stim_mean, all_neurons, avg, sem, cmi



# Process WT
traces_wt, suppression_wt, magnitude_wt, stim_wt, wt_neurons, avg_wt, sem_wt, cmi_wt = process_group('WT', WT_mice)
# Process KO
traces_ko, suppression_ko, magnitude_ko, stim_ko, ko_neurons, avg_ko, sem_ko, cmi_ko = process_group('KO', KO_mice)
print(traces_wt)
print(len(traces_wt))

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

#To determine the ideal number of clusters

""" for group in groups:
    dunn_scores = []
    inertias = []
    if group == 'WT':
        traces = traces_wt
    elif group == 'KO':
        traces = traces_ko
    k_range = list(range(2, 10))
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(traces)
        labels = kmeans.labels_
        dunn_scores.append(dunn_index_pointwise(traces, labels))
        inertias.append(kmeans.inertia_)
    knee = KneeLocator(k_range, inertias, curve='convex', direction='decreasing')
    k_elbow = knee.knee or k_range[-1]
    plt.plot(k_range, dunn_scores, marker='o')
    plt.xlabel("Number of clusters")
    plt.ylabel(f"Dunn Index for {group}")
    plt.title(f"Dunn Index vs Number of Clusters for {group}")
    plt.savefig(os.path.join(save_path, f"{fig_name}_{group}_dunn_index.jpg"))
    plt.show()

    plt.plot(k_range, inertias, marker='o')
    plt.axvline(k_elbow, color='g', linestyle='--', label=f'Elbow k={k_elbow}')
    plt.xlabel("Number of clusters")
    plt.ylabel(f"Inertias for {group}")
    plt.title(f"Inertias vs Number of Clusters for {group}")
    plt.savefig(os.path.join(save_path, f"{fig_name}_{group}_inertias.jpg"))
    plt.show() """


# After choosing a number of clusters, we can run KMeans clustering
k_clusters = {'WT': 4, 'KO': 4}  # Example: you can set different numbers for each group
time = (np.arange(traces_wt.shape[1]) / frame_rate) -1 
print(time)
xticks = np.arange(-1, time[-1] + 1, 1)  # ticks every 1 second

# Dictionaries to store cluster mean and SEM for each group
cluster_data = {'WT': {}, 'KO': {}}

for group in groups:
    if group == 'WT':
        traces = traces_wt
        nclusters = k_clusters['WT']
    elif group == 'KO':
        traces = traces_ko
        nclusters = k_clusters['KO']
    kmeans = KMeans(n_clusters=nclusters, random_state=0).fit(traces)
    labels = kmeans.labels_

    colors = plt.cm.get_cmap('tab10', nclusters)  # get distinct colors

    plt.figure()
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
    plt.ylabel('Z-scored ΔF/F')
    plt.title(f'Average Response per Cluster for {group}')
    plt.xticks(xticks)
    plt.legend()
    plt.savefig(os.path.join(save_path, f"{fig_name}_{group}_{k_clusters[group]}clusters_averaged.jpg"))
    plt.show()



# 🆕 Plot comparison between Cluster n from WT and Cluster k from KO
wt_cluster = 0  # <-- change this to the WT cluster index you want to compare
ko_cluster = 3  # <-- change this to the KO cluster index you want to compare

plt.figure()

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
plt.ylabel('Z-scored ΔF/F')
plt.title(f'Comparison: WT Cluster {wt_cluster} vs KO Cluster {ko_cluster}')
plt.xticks(xticks)
plt.legend()
plt.savefig(os.path.join(save_path, f"{fig_name}_WT{wt_cluster}_vs_KO{ko_cluster}.jpg"))
plt.show()






