import matplotlib.pyplot as plt
import numpy as np
import os.path
from scipy import stats
def Visualize_baseline(smooth_signal, baseline):
    plt.figure(figsize=(10, 6))
    plt.hist(smooth_signal, bins=100, color='skyblue', edgecolor='black', alpha=0.7, label='Smooth Signal Histogram')
    plt.axvline(baseline, color='red', linestyle='--', linewidth=2, label=f'Baseline (max bin center): {baseline:.2f}')
    plt.xlabel('Smooth Signal Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Smooth Signal with Baseline')
    plt.legend()
    plt.show()

def Bootstrapping_fig(bootstrapped_data, real_stim_mean, protocol_name, p_value, Neuron_index,color_histo, save_dir):
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
    save_direction = os.path.join(save_dir, protocol_name)
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

def stim_period(protocol_duration_s,Photon_fre, mean_F_specific_protocol,std_F_specific_protocol, protocol_name, Neuron_index, save_dir):
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
    ax.set_ylabel("raw F")
    ax.set_title(protocol_name + '\n' + fig_name)
    ax.legend(bbox_to_anchor=(0, 1), loc='upper left', frameon=False)
    save_direction1 = os.path.join(save_dir, protocol_name)
    isExist1 = os.path.exists(save_direction1)
    if isExist1:
        pass
    else:
        os.mkdir(save_direction1)
    save_direction = os.path.join(save_direction1, fig_name)
    fig.savefig(save_direction)
    plt.close(fig)
