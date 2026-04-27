import matplotlib.pyplot as plt
import numpy as np
import os.path
from scipy import stats
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

def Visualize_baseline(smooth_signal, baseline):
    plt.figure(figsize=(10, 6))
    plt.hist(smooth_signal, bins=100, color='skyblue', edgecolor='black', alpha=0.7, label='Smooth Signal Histogram')
    plt.axvline(baseline, color='red', linestyle='--', linewidth=2, label=f'Baseline (max bin center): {baseline:.2f}')
    plt.xlabel('Smooth Signal Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Smooth Signal with Baseline')
    plt.legend()
    plt.show()

def Bootstrapping_fig(bootstrapped_data, real_stim_mean, protocol_name, p_value, Neuron_index,color_histo, save_dir, file_prefix=''):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(bootstrapped_data, bins=30, alpha=0.5, edgecolor='white', color=color_histo,
            label='bootstraped baseline')
    ax.axvline(real_stim_mean, color='darkmagenta', linestyle='dashed', linewidth=2, label='maximum 20% stimulus')
    ax.set_xlabel('5th percentile dF/F')
    ax.set_ylabel('Count')
    ax.set_title(f'Neuron {Neuron_index}  ({protocol_name})')
    ax.annotate(f'p-value: {p_value:.3f}', xy=(0.02, 0.98), xycoords='axes fraction', fontsize=9,
                va='top', ha='left')
    ax.legend()
    fig_name = " Bootstrapping Neuron " + str(Neuron_index)
    foldername = "_".join(list(filter(None, [file_prefix, protocol_name])))
    save_direction = os.path.join(save_dir, foldername)
    isExist1 = os.path.exists(save_direction)
    if isExist1:
        pass
    else:
        os.mkdir(save_direction)
    save_direction = os.path.join(save_direction, fig_name)
    fig.savefig(save_direction)
    plt.close(fig)

def box_plot(F_base, F_stim, save_dir, protocol_name, Neuron_index):
    data = [F_base, F_stim]

    stds = [np.std(F_base), np.std(F_stim)]
    fig, ax = plt.subplots(figsize=(8, 6))
    box = ax.boxplot(data, labels=['F_base', 'F_stim'], patch_artist=True)
    for patch in box['boxes']:
        patch.set_alpha(0.5)

    # Overlay data points
    for i, dataset in enumerate(data, start=1):
        ax.scatter([i] * len(dataset), dataset, color='blue', alpha=0.5, label='Data points' if i == 1 else "")

    # Customize plot
    ax.set_title('Comparison of F_base and F_stim')
    ax.set_ylabel('Mean Values')
    ax.grid(True)
    ax.legend()
    plt.show()

    fig_name = " Box plot Neuron " + str(Neuron_index)
    save_direction = os.path.join(save_dir, protocol_name)
    isExist1 = os.path.exists(save_direction)
    if isExist1:
        pass
    else:
        os.mkdir(save_direction)
    save_direction = os.path.join(save_direction, fig_name)
    fig.savefig(save_direction)
    plt.close(fig)

def stim_period(protocol_duration_s,Photon_fre, mean_F_specific_protocol,std_F_specific_protocol, protocol_name, Neuron_index, save_dir, file_prefix=''):
    protocol_duration = int(protocol_duration_s  * Photon_fre) + 58
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axvline(x=0, color='orchid', linestyle='--', alpha=0.7, linewidth=2,
               label='stim start')
    ax.axvline(x= protocol_duration_s, color='red', linestyle='--', linewidth=2,
               alpha=0.7, label='stim end')

    time = np.linspace(-29 / Photon_fre, (protocol_duration - 30) /Photon_fre, protocol_duration)
    ax.plot(time, mean_F_specific_protocol, color='black', label='Mean', linewidth=2)
    ax.fill_between(time,
                    mean_F_specific_protocol - std_F_specific_protocol,
                    mean_F_specific_protocol + std_F_specific_protocol,
                    color='gray', alpha=0.5, label='Standard Deviation')
    fig_name = " test Neuron " + str(Neuron_index)
    ax.margins(x=0)
    ax.set_xlabel("Time(s)")
    ax.set_ylabel(r"$\Delta$F/$F_0$")
    ax.set_title(protocol_name + '\n' + fig_name)
    ax.legend(bbox_to_anchor=(0, 1), loc='upper left', frameon=False)
    foldername = "_".join(list(filter(None, [file_prefix, protocol_name])))
    save_direction1 = os.path.join(save_dir, foldername)
    isExist1 = os.path.exists(save_direction1)
    if isExist1:
        pass
    else:
        os.mkdir(save_direction1)
    save_direction = os.path.join(save_direction1, fig_name)
    fig.savefig(save_direction)
    plt.close(fig)

def general_figure(time, neural_traces, behavioral_data, filter_kernel, speed_corr_list, save_dir):

    custom_params = {"axes.spines.right": False, "axes.spines.top": False, "axes.spines.bottom": False, "axes.spines.left": False}
    sns.set_theme(style="white", rc=custom_params)
    color = {'Speed' : 'goldenrod', 'Pupil' : 'black', 'Facemotion' : 'gray', r'Normalized $\Delta$F/F mean' : 'dodgerblue'}

    idx_sorted = np.argsort(speed_corr_list)
    neural_traces_sorted = neural_traces[idx_sorted]
    mean_dF = gaussian_filter1d(np.mean(neural_traces, axis=0), filter_kernel['Neuronal activity'])

    for key in behavioral_data.keys():
        behavioral_data[key] = gaussian_filter1d(behavioral_data[key], filter_kernel[key])

    behavioral_data.update({r'Normalized $\Delta$F/F mean' : mean_dF})

    fig = plt.figure(figsize=(24, 20))
    gs = fig.add_gridspec(len(behavioral_data.keys())*3+12, 45)

    # Behavioral data and mean dF/F0
    for i, key in enumerate(behavioral_data.keys()):

        ax = fig.add_subplot(gs[i*3:i*3+2, :43])
        ax.set_title(key, fontsize=25, y=1.0, horizontalalignment='center')
        ax.plot(time, behavioral_data[key], linewidth=4, color=color[key])
        ax.set_xticks([])
        ax.margins(x=0)
        if key == 'Speed' :
            ax.set_ylabel('cm/s', fontsize=20)
        ax.set_facecolor("white")
        plt.yticks(fontsize=20)

    # Neuronal activity map
    ax1 = fig.add_subplot(gs[len(behavioral_data.keys())*3:len(behavioral_data.keys())*3+11, :43])
    ax1.set_title('Normalized neuronal activity (sorted by speed correlation)', fontsize=25, y=1.0, horizontalalignment='center')
    ax1.pcolormesh(neural_traces_sorted, cmap='Greys')
    ax1.set_xlabel('Time (s)', fontsize=20)
    ax1.margins(x=0)
    ax1.set_ylabel('Neuron', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    m = ax1.pcolormesh(neural_traces_sorted, cmap='Greys')
    ax1b = fig.add_subplot(gs[len(behavioral_data.keys())*3:len(behavioral_data.keys())*3+11, 44:45])
    fig.colorbar(m, cax=ax1b)
    plt.yticks(fontsize=20)

    save_path = os.path.join(save_dir, "general_figure.png")
    fig.savefig(save_path)
    plt.close(fig)