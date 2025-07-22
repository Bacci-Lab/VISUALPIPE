import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import copy
from scipy.stats import spearmanr, zscore
from scipy.ndimage import gaussian_filter1d
from visual_stim import VisualStim

def get_spont_stim(visual_stim:VisualStim):
    spont_stimuli_name = []
    spont_stimuli_id = []
    analyze_pupil = []

    for stimuli_name in visual_stim.protocol_names :
        if 'grey' in stimuli_name :
            if stimuli_name not in spont_stimuli_name :
                spont_stimuli_name.append(stimuli_name)
                analyze_pupil.append(1)
        elif 'black' in stimuli_name :
            if stimuli_name not in spont_stimuli_name :
                spont_stimuli_name.append(stimuli_name)
                analyze_pupil.append(0)
    
    for el in spont_stimuli_name :
        id_spont = visual_stim.protocol_df[visual_stim.protocol_df['name'] == el].index[0]
        spont_stimuli_id.append(id_spont)
    
    return spont_stimuli_id, analyze_pupil

def compute_spont_corr(behavior_spont, F_spontaneous, time_stamps_spont, sigma=0, label='', save_spont_dir='', permutation=True):

    if sigma > 0 :
        F_spontaneous = gaussian_filter1d(F_spontaneous, sigma, axis=1)
        behavior_spont = gaussian_filter1d(behavior_spont, sigma)

    # Correlation with dFoF0
    spont_behavior_corr = [spearmanr(behavior_spont, ROI)[0] for ROI in F_spontaneous]
    spont_behavior_corr = [float(value) for value in spont_behavior_corr]

    if permutation :
        valid_neurons = permutation_test(F_spontaneous, behavior_spont, spont_behavior_corr, time_stamps_spont, label=label, savefolder=os.path.join(save_spont_dir, f"permutation_{label}"))
        return spont_behavior_corr, valid_neurons
    
    else :
        return spont_behavior_corr, None

def permutation_test(fluorescence, beh_trace, corr, time, samples=1000, label='', savefolder=''):
    
    beh_trace_norm = beh_trace/np.max(beh_trace)
    fluorescence_norm = [i/np.max(i) for i in fluorescence]

    random_list = np.random.choice(np.arange(1, len(beh_trace_norm)), samples, replace=False)
    beh_trace_shuffled = [np.roll(beh_trace_norm, i) for i in random_list]
    
    valid_neurons = []

    for s in tqdm(range(len(fluorescence_norm)), desc=label + " permutation processing"):
        null_corr = np.array([spearmanr(beh_trace_shuffled[i], fluorescence_norm[s])[0] for i in range(samples)])
        p_value = np.sum(np.abs(null_corr) >= np.abs(corr[s])) / samples

        plot_permutation(null_corr, corr[s], p_value, s, beh_trace_norm, fluorescence_norm, time, label, samples, savefolder)

        if p_value <= 0.05:
            valid_neurons.append(s)

    return valid_neurons

def plot_permutation(null_corr, real_corr, p_value, roi_id, beh_trace, fluorescence, time, label, samples, savefolder):

    weights = np.ones(samples) / samples
    if 'speed' in label :
        color2 = 'goldenrod'
    elif 'pupil' in label :
        color2 = 'black'
    else :
        color2 = 'gray'

    sns.set_theme()

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 7), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f'ROI {roi_id}')

    if p_value > 0.05 :
        color = "thistle"
    else :
        color = "skyblue"
    ax.hist(null_corr, weights=weights * 100, bins=30, alpha=0.7, edgecolor='white', color=color, label='Shuffled correlation')
    ax.axvline(real_corr, color='black', linestyle='dashed', linewidth=2, label='Observed correlation')
    ax.set_xlabel('Correlation Coefficient')
    ax.set_ylabel('Count (in %)')
    ax.set_title(f'Permutation Test ({label})')
    ax.legend(loc='upper right', fontsize='small')
    ax.annotate(f'p-value = {p_value:.3f}', xy=(0.02, 0.98), xycoords='axes fraction', fontsize=9, va='top', ha='left')
    
    #-------------------------------
    ax2.plot(time, fluorescence[roi_id], alpha=0.6, label=r'$\Delta$F/F')
    ax2.plot(time, beh_trace, color=color2, alpha=0.6, label=label)
    ax2.set_yticks([])
    ax2.margins(x=0)
    ax2.set_xlabel("Time(s)")
    ax2.legend(loc='upper right', fontsize='small')
    plt.tight_layout(h_pad=1.5)

    #-------------------------------
    figname = f'ROI_{roi_id}_{label}_permutation'
    if not os.path.exists(savefolder) :
        os.makedirs(savefolder)
    savepath = os.path.join(savefolder, figname)
    fig.savefig(savepath)
    plt.close(fig)

def get_valid_neurons(valid_neurons_list):

    if len(valid_neurons_list) > 1 :
        set_overlap = set(valid_neurons_list[0]).intersection(valid_neurons_list[1])
        for i in range(1, len(valid_neurons_list) - 1):
            set_overlap = set(set_overlap).intersection(valid_neurons_list[i+1])
        valid_neurons = list(set_overlap)
    else :
        valid_neurons = valid_neurons_list[0]

    return valid_neurons

def pie_plot(nb_valid_ROIs, nb_invalid_ROIs, save_folder='', label='', color_valid='skyblue'):

    fig = plt.figure(figsize=(7, 5))
    labels = ["Correlated ROIs", "Uncorrelated ROIs"]
    sizes = [nb_valid_ROIs, nb_invalid_ROIs]
    colors = [color_valid, 'silver']
    plt.pie(sizes, labels=[f'{label} ({size})' for label, size in zip(labels, sizes)], colors=colors,
            autopct='%1.1f%%')
    plt.title(f"Permutation test: {label}")
    save_direction = os.path.join(save_folder,  f"pie_{label}_permutation.png")
    fig.savefig(save_direction, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)

def colormap_perm_test(time, dF, x, valid_neurons, corr, sigma=0, label:str=None, save_path=None):
    
    idx_sorted = np.flip(np.argsort(np.array(corr)[valid_neurons]))
    id_roi_sorted = np.arange(len(corr))[valid_neurons][idx_sorted]
    dF_valid_corr_sorted = dF[valid_neurons][idx_sorted]
    mean_dF = np.mean(dF[valid_neurons], axis=0)

    if sigma > 0:
        x = gaussian_filter1d(x, sigma)
        mean_dF = gaussian_filter1d(mean_dF, sigma)
        dF_valid_corr_sorted = gaussian_filter1d(dF_valid_corr_sorted, sigma, axis=1)

    nb_plot_ROIs = np.min([len(valid_neurons), 3])
    label_order = ['Most ', '2nd most ', '3rd most ']
    height_ratios = [2, 2, 9]
    for i in range(nb_plot_ROIs) :
        height_ratios.append(2)

    fig = plt.figure(figsize=(11, 10))
    gs = fig.add_gridspec(3 + nb_plot_ROIs, 2, height_ratios=height_ratios, width_ratios=[25,1], hspace=0.7, wspace=0.05)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[2, 0])
    ax2b = fig.add_subplot(gs[2, 1])
    
    # Var
    if label=='speed':
        ax0.set_ylabel('cm/s')
        c = 'goldenrod'
        label2 = label
    elif label=='pupil':
        c = 'black'
        x = zscore(x)
        label2 = label + ' z-score'
    elif label=='facemotion':
        c = 'gray'
        x = zscore(x)
        label2 = label + ' z-score'
    else :
        label2=''
    ax0.plot(time, x, color=c)
    ax0.set_xticks([])
    ax0.margins(x=0)
    ax0.set_facecolor("white")
    ax0.set_title(label2)

    ax1.plot(time, mean_dF)
    ax1.set_xticks([])
    ax1.margins(x=0)
    ax1.set_facecolor("white")
    ax1.set_title(r'$\Delta$F/F mean on ROIs which passed the ' + label + ' permutation test')

    c = ax2.pcolormesh(np.flip(dF_valid_corr_sorted, axis=0), cmap='Greys')
    ax2.set_ylabel('Sorted neurons')
    ax2.margins(x=0)
    ax2.set_facecolor("white")
    ax2.set_title('Neuronal activity (sorted by ' + label + ' correlation)')
    fig.colorbar(c, cax=ax2b)

    for i in range(nb_plot_ROIs):
        ax3 = fig.add_subplot(gs[3+i, 0])

        ax3.plot(time, dF_valid_corr_sorted[i])
        ax3.margins(x=0)
        ax3.set_facecolor("white")
        ax3.set_title(label_order[i] + label + r" correlated neuron's normalized $\Delta$F/F : ROI " + f"{id_roi_sorted[i]}", horizontalalignment='center')
        
        if i == nb_plot_ROIs - 1 : 
            ax3.set_xlabel('Time (s)')
        else :
            ax3.set_xticks([])
    
    if nb_plot_ROIs == 0 :
        ax2.set_xlabel('Frame')
    else :
        ax2.set_xticks([])

    if save_path != None :
        save_direction = os.path.join(save_path, label +"_permT_neural_activity.png")
        fig.savefig(save_direction, bbox_inches='tight', pad_inches=0.3)
    else :
        plt.show()
    
    plt.close(fig)

def process_correlation(x, fluorescence, time, idx_lim_protocol, sigma=0, label:str='', save_path:str='', plot:bool=True):
    """
    Compute the correlation between x (e.g. behavior) and fluorescence in a given time window
    and performs a permutation test to identify which neurons are significantly correlated to x.
    Also plot correlation and the pie chart of correlated neurons if plotis set to True.

    Parameters
    ----------
    x : array
        Behavioral signal.
    fluorescence : array
        Fluorescence activity.
    time : array
        Time of the fluorescence activity.
    start_spont_index : int
        Start index of the spontaneous activity.
    end_spont_index : int
        End index of the spontaneous activity.
    sigma : int
        Sigma of the Gaussian filter applied to the behavioral signal.
    label : str
        Label of the behavioral signal.
    save_path : str
        Path to save the figure.
    plot : bool
        If True, plot the correlation and the pie chart of correlated neurons.

    Returns
    -------
    corr : array
        Correlation between fluorescence activity and behavioral signal.
    valid_neurons : array
        Indices of the neurons that passed the permutation test.
    """
    
    [start_spont_index, end_spont_index] = idx_lim_protocol
    x_spont = x[start_spont_index:end_spont_index]
    if fluorescence.shape[1] > x_spont.shape[0] :
        fluorescence = fluorescence[:, start_spont_index:end_spont_index]
    if time.shape[0] > x_spont.shape[0] :
        time = time[start_spont_index:end_spont_index]
    
    corr, valid_neurons = compute_spont_corr(x_spont, fluorescence, time, sigma, label, save_path)
    if plot :
        colormap_perm_test(time, fluorescence, x_spont, valid_neurons, corr, sigma=sigma, label=label, save_path=save_path)
        pie_plot(len(valid_neurons), len(corr) - len(valid_neurons), save_path, label)

    return corr, valid_neurons 

def process_multiple_protocols(spont_corr_list, spont_df, valid_neurons_list, label:str='', save_path:str='', plot:bool=True):

    # Compute the mean correlation weighted by the duration of each protocol
    if label == 'pupil' :
        spont_dt_pupil = spont_df[spont_df.analyze_pupil == 1].duration
        corr = np.dot(np.array(spont_corr_list).T, spont_dt_pupil/np.sum(spont_dt_pupil))
    else :
        corr = np.dot(np.array(spont_corr_list).T, spont_df.duration/np.sum(spont_df.duration))
    
    # Valid neurons should be valid for all protocols
    valid_neurons_speed = get_valid_neurons(valid_neurons_list)
    
    # Build array of valid neurons for each protocol for the HDF5 file
    valid_neurons_list_save = np.zeros((len(corr), len(valid_neurons_list)))
    for i in range(len(valid_neurons_list)) : 
        valid_neurons_list_save[valid_neurons_list[i], i] = 1

    if plot :
        pie_plot(len(valid_neurons_speed), len(corr) - len(valid_neurons_speed), save_path, label)

    return corr, valid_neurons_list_save