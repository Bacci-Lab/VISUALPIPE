import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances
from scipy.stats import mannwhitneyu, wilcoxon, linregress
import glob
from kneed import KneeLocator 
import os
import math

def process_group(group_name, mice_list):
    magnitude = {protocol: []}
    avg_data = {protocol: []}
    all_neurons = 0
    stim_mean = {protocol: []}
    single_traces = {protocol: []}
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
            all_protocols = validity.files
            # Determine which neurons are valid
            valid_data = validity[protocol_validity]
            #valid_neurons = np.where(np.isin(valid_data[:, 0], [1, -1]))[0] #if you want to look at both positive and negative responsive neurons
            valid_neurons = np.where(valid_data[:, 0] == 1)[0]    #if you only want to look at positive responsive neurons
            neurons = len(valid_neurons)
            proportion = 100 * len(valid_neurons) / trials['trial_averaged_zscores'][0].shape[0]
            print(f"Proportion of {protocol_validity}-responding neurons: {proportion}")
            all_neurons += neurons

            avg_session = []
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
            #Get the magnitude of the response (minimum or maximum in pre, stim or post periods), for normalization later
            for zneuron in zscores_concat:
                magnitude[protocol].append(np.max(np.abs(zneuron)))
    
    #This one will be used to plot average responses per cluster (not normalized)
    single_traces = np.concatenate(single_traces[f'{protocol}'], axis=0)
    magnitudes_array = np.array(magnitude[protocol])
    #This one will be used to cluster only based on response shape (normalized)
    normalized_traces = single_traces / magnitudes_array.reshape(-1, 1)
    print(f"Number of {group_name} neurons: {all_neurons}")

    # Concatenate all neuron arrays into one array per protocol
    avg_data[protocol] = np.stack(avg_data[protocol], axis=0)

    # Compute average and SEM across neurons
    avg = {protocol: np.mean(avg_data[protocol], axis=0)}
    sem = {protocol: stats.sem(avg_data[protocol], axis=0)}

    return normalized_traces, single_traces, magnitude, stim_mean, all_neurons, avg, sem

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

if __name__ == "__main__":

    ## Organization of the data: put validity2.npz and trials.npy files in a folder, and seperate the groups in subfolders (e.g WT and KO)
    #The folder containing all subfolders
    data_path = r"Y:\raw-imaging\Nathan\PYR\vision_survey"
    save_path = data_path
    # groups as to be the name of the subfolders
    groups = ['WT', 'KO']
    # mice per group (sub-subfolders)
    WT_mice = ['110','108']
    KO_mice = ['109','112']
    #Will be included in all figure names
    fig_name = 'static-patch-0deg-together-PositiveOnly'
    # Write the stimulus type you want to use for clustering
    protocol = 'static-patch-0'
    # Protocol used to select responsive neurons
    protocol_validity = 'static-patch-0'
    #Frame rate
    frame_rate = 30

    z_score_periods = ['pre_trial_averaged_zscores', 'trial_averaged_zscores', 'post_trial_averaged_zscores']
    x_labels = ['Pre-stim', 'Stim', 'Post-stim']

    # Process WT
    norm_wt, traces_wt, magnitude_wt, stim_wt, wt_neurons, avg_wt, sem_wt = process_group('WT', WT_mice)
    # Process KO
    norm_ko, traces_ko, magnitude_ko, stim_ko, ko_neurons, avg_ko, sem_ko = process_group('KO', KO_mice)

    #------------------- Cluster the two groups separately -------------------#

    #To determine the ideal number of clusters

    """for group in groups:
        dunn_scores = []
        inertias = []
        interdistance = []
        if group == 'WT':
            traces = norm_wt
        elif group == 'KO':
            traces = norm_ko
        k_range = list(range(2, 10))
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
        knee1 = KneeLocator(k_range, inertias, curve='convex', direction='decreasing')
        k_elbow1 = knee1.knee or k_range[-1]
        knee2 = KneeLocator(k_range, interdistance, curve='convex', direction='increasing')
        k_elbow2 = knee2.knee or k_range[-1]
        plt.plot(k_range, dunn_scores, marker='o')
        plt.xlabel("Number of clusters")
        plt.ylabel(f"Dunn Index for {group}")
        plt.title(f"Dunn Index vs Number of Clusters for {group}")
        plt.savefig(os.path.join(save_path, f"{fig_name}_{group}_dunn_index.jpg"))
        plt.show()

        plt.plot(k_range, inertias, marker='o')
        plt.axvline(k_elbow1, color='g', linestyle='--', label=f'Elbow k={k_elbow1}')
        plt.xlabel("Number of clusters")
        plt.ylabel(f"Inertias for {group}")
        plt.title(f"Inertias vs Number of Clusters for {group}")
        plt.savefig(os.path.join(save_path, f"{fig_name}_{group}_inertias.jpg"))
        plt.show() 

        plt.plot(k_range, interdistance, marker='o')
        plt.axvline(k_elbow2, color='g', linestyle='--', label=f'Elbow k={k_elbow2}')
        plt.xlabel("Number of clusters")
        plt.ylabel(f"Intercluster distances for {group}")
        plt.title(f"Intercluster distances vs Number of Clusters for {group}")
        #plt.savefig(os.path.join(save_path, f"{fig_name}_{group}_interclusterDist.jpg"))
        plt.show() """

    """# After choosing a number of clusters, we can run KMeans clustering
    k_clusters = {'WT': 5, 'KO': 4}  # Example: you can set different numbers for each group
    time = (np.arange(traces_wt.shape[1]) / frame_rate) -1 
    print(time)
    xticks = np.arange(-1, time[-1] + 1, 1)  # ticks every 1 second

    # Dictionaries to store cluster mean and SEM for each group
    cluster_data = {'WT': {}, 'KO': {}}

    for group in groups:
        if group == 'WT':
            norm = norm_wt
            plot_traces = traces_wt
            nclusters = k_clusters['WT']
        elif group == 'KO':
            norm = norm_ko
            plot_traces = traces_ko
            nclusters = k_clusters['KO']
        kmeans = KMeans(n_clusters=nclusters, n_init=100).fit(norm)
        labels = kmeans.labels_

        colors = plt.cm.get_cmap('tab10', nclusters)  # get distinct colors

        plt.figure()
        for k in range(nclusters):
            cluster_k = plot_traces[labels == k]
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
        plt.show()"""



    """# 🆕 Plot comparison between Cluster n from WT and Cluster k from KO
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






    """





    #-------------To cluster WT and KO together and then compare % of neurons in each cluster------------#

    # Concatenate WT and KO traces
    norm_traces = np.concatenate([norm_wt, norm_ko], axis=0)
    all_traces = np.concatenate([traces_wt, traces_ko], axis=0)
    group_labels = np.array(['WT'] * norm_wt.shape[0] + ['KO'] * norm_ko.shape[0])

    #determine the best number of clusters for joint clustering
    dunn_scores = []
    inertias = []
    interdistance = []
    k_range = list(range(2, 10))
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=100).fit(norm_traces) #Does 100 iterations with different centroid initializations to find the best one (lowest inertia)
        labels = kmeans.labels_
        dunn_scores.append(dunn_index_pointwise(norm_traces, labels))
        inertias.append(kmeans.inertia_)
        # Get centroids from the fitted k-means model
        centroids = kmeans.cluster_centers_
        # Compute pairwise distances between centroids (Euclidean by default)
        intercluster_distances = pairwise_distances(centroids)
        # Extract upper triangle (excluding diagonal) to avoid redundancy
        unique_distances = intercluster_distances[np.triu_indices_from(intercluster_distances, k=1)]
        interdistance.append(np.sum(unique_distances))
    knee1 = KneeLocator(k_range, inertias, curve='convex', direction='decreasing')
    k_elbow1 = knee1.knee or k_range[-1]
    knee2 = KneeLocator(k_range, interdistance, curve='convex', direction='increasing')
    k_elbow2 = knee2.knee or k_range[-1]
    plt.plot(k_range, dunn_scores, marker='o')
    plt.xlabel("Number of clusters")
    plt.ylabel(f"Dunn Index")
    plt.title(f"Dunn Index vs Number of Clusters both groups combined")
    plt.savefig(os.path.join(save_path, f"{fig_name}_combined_dunn_index.jpg"))
    plt.show()

    plt.plot(k_range, inertias, marker='o')
    plt.axvline(k_elbow1, color='g', linestyle='--', label=f'Elbow k={k_elbow1}')
    plt.xlabel("Number of clusters")
    plt.ylabel(f"Inertias")
    plt.title(f"Inertias vs Number of Clusters for both groups combined")
    plt.savefig(os.path.join(save_path, f"{fig_name}_combined_inertias.jpg"))
    plt.show() 

    plt.plot(k_range, interdistance, marker='o')
    plt.axvline(k_elbow2, color='g', linestyle='--', label=f'Elbow k={k_elbow2}')
    plt.xlabel("Number of clusters")
    plt.ylabel(f"Intercluster distances")
    plt.title(f"Intercluster distances vs Number of Clusters for both groups combined")
    plt.savefig(os.path.join(save_path, f"{fig_name}_combined_DistInterCluster.jpg"))
    plt.show() 





    # Set number of clusters for joint clustering
    n_clusters_joint = 5  # choose based on elbow/Dunn index as before

    # Run KMeans clustering on the combined data
    kmeans = KMeans(n_clusters=n_clusters_joint, n_init=50).fit(norm_traces)
    cluster_labels = kmeans.labels_

    # Time axis
    time = (np.arange(all_traces.shape[1]) / frame_rate) - 1
    xticks = np.arange(-1, time[-1] + 1, 1)
    wt_y_ticks = np.arange(0, norm_wt.shape[0], 1)  # y-ticks for raster plots
    ko_y_ticks = np.arange(0, norm_ko.shape[0], 1)  # y-ticks for raster plots

    # Get color map
    colors = plt.cm.get_cmap('tab10', n_clusters_joint)

    # For storing cluster data
    joint_cluster_data = {}
    wt_cluster_counts = np.zeros(len(np.unique(cluster_labels)), dtype=int)
    print(wt_cluster_counts)
    ko_cluster_counts = np.zeros(len(np.unique(cluster_labels)), dtype=int)
    print(ko_cluster_counts)
    # Plot per-cluster average traces for WT and KO
    for k in range(n_clusters_joint):
        fig, axs = plt.subplots(1,2,figsize=(10, 6))
        idx_cluster = np.where(cluster_labels == k)[0]
        idx_wt = idx_cluster[group_labels[idx_cluster] == 'WT']
        idx_ko = idx_cluster[group_labels[idx_cluster] == 'KO']
        wt_cluster_counts[k] = np.sum(group_labels[idx_cluster] == 'WT')
        ko_cluster_counts[k] = np.sum(group_labels[idx_cluster] == 'KO')

        traces_wt_k = all_traces[idx_wt]
        traces_ko_k = all_traces[idx_ko]
        norm_wt_k = norm_traces[idx_wt]
        norm_ko_k = norm_traces[idx_ko]

        vmin, vmax = np.nanmin(norm_traces), np.nanmax(norm_traces)
        if np.abs(vmin) > np.abs(vmax) :
            lim = math.floor(np.abs(vmin)*100) * 0.01
        else : 
            lim = math.floor(np.abs(vmax)*100) * 0.01
        im0 = axs[0].imshow(norm_wt_k, aspect='auto', extent=[time[0], time[-1], 0, norm_wt_k.shape[0]],
                        cmap='RdBu_r', vmin=-lim, vmax=lim)
        axs[0].set_title(f'Cluster {k+1} – WT ({norm_wt_k.shape[0]} neurons)')
        axs[0].set_ylabel('Neuron')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_xticks(xticks)
        axs[0].set_yticks(np.arange(0, norm_wt_k.shape[0]-1, 10))  # y-ticks for raster plots

        im1 = axs[1].imshow(norm_ko_k, aspect='auto', extent=[time[0], time[-1], 0, norm_ko_k.shape[0]],
                            cmap='RdBu_r', vmin=-lim, vmax=lim)
        axs[1].set_title(f'Cluster {k+1} – KO ({norm_ko_k.shape[0]} neurons)')
        axs[1].set_ylabel('Neuron')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_xticks(xticks)
        axs[1].set_yticks(np.arange(0, norm_ko_k.shape[0]-1, 10))  # y-ticks for raster plots
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{fig_name}_cluster{k}_rasters.jpg"), dpi=300) 
        plt.show()  
            
        mean_wt = np.mean(traces_wt_k, axis=0) if len(idx_wt) > 0 else np.zeros(all_traces.shape[1])
        sem_wt = stats.sem(traces_wt_k, axis=0) if len(idx_wt) > 1 else np.zeros(all_traces.shape[1])
        mean_ko = np.mean(traces_ko_k, axis=0) if len(idx_ko) > 0 else np.zeros(all_traces.shape[1])
        sem_ko = stats.sem(traces_ko_k, axis=0) if len(idx_ko) > 1 else np.zeros(all_traces.shape[1])

        joint_cluster_data[k] = {
            'WT': {'mean': mean_wt, 'sem': sem_wt, 'n': len(idx_wt)},
            'KO': {'mean': mean_ko, 'sem': sem_ko, 'n': len(idx_ko)}
        }

        # Plot traces
        plt.figure(figsize=(8, 5))
        plt.fill_between(time, mean_wt - sem_wt, mean_wt + sem_wt, alpha=0.3, color='blue', label=f'WT (n={len(idx_wt)})')
        plt.plot(time, mean_wt, color='blue')

        plt.fill_between(time, mean_ko - sem_ko, mean_ko + sem_ko, alpha=0.3, color='red', label=f'KO (n={len(idx_ko)})')
        plt.plot(time, mean_ko, color='red')

        plt.title(f'Cluster {k}: Average Z-score Traces (WT vs KO)')
        plt.xlabel('Time (s)')
        plt.ylabel('Z-scored ΔF/F')
        plt.xticks(xticks)
        plt.legend()
        plt.savefig(os.path.join(save_path, f"{fig_name}_cluster{k}_WTvsKO.jpg"))
        plt.show()

    # Normalize to percentages
    wt_percentages = 100 * wt_cluster_counts / np.sum(group_labels == 'WT')
    ko_percentages = 100 * ko_cluster_counts / np.sum(group_labels == 'KO')
    cluster_proportions = {
        'WT': wt_percentages,
        'KO': ko_percentages
    }
    print(cluster_proportions)
    # Plot pie chart for WT
    plt.figure()
    plt.pie(wt_percentages, labels=[f'Cluster {i}' for i in range(n_clusters_joint)],
            autopct='%1.1f%%', colors=plt.cm.tab10.colors[:n_clusters_joint], startangle=90)
    plt.title('WT: % of neurons in each cluster')
    plt.axis('equal')
    plt.savefig(os.path.join(save_path, f"{fig_name}_WT_cluster_pie.jpg"))
    plt.show()

    # Plot pie chart for KO
    plt.figure()
    plt.pie(ko_percentages, labels=[f'Cluster {i}' for i in range(n_clusters_joint)],
            autopct='%1.1f%%', colors=plt.cm.tab10.colors[:n_clusters_joint], startangle=90)
    plt.title('KO: % of neurons in each cluster')
    plt.axis('equal')
    plt.savefig(os.path.join(save_path, f"{fig_name}_KO_cluster_pie.jpg"))
    plt.show()


