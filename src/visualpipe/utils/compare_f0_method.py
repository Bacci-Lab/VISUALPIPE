#%% Imports
import sys
sys.path.append("./src/visualpipe")
sys.path.append("../")

import numpy as np
import os
import matplotlib.pyplot as plt
import math
import pandas as pd

from analysis.ca_imaging import CaImagingDataManager
from analysis.visual_stim import VisualStim
from analysis.trial import Trial
import analysis.photodiode as ptd
import utils.general_functions as general_functions

#%% Set data folderpath
# Filepath should be the result data folder in which the HDF5 file is
folderpath = r"\\iss\bacci\raw-imaging\Adrianna\experiments\NDNF\2025_06_25\15-19-24"
foldername = "compare_f0_method"
savepath = os.path.join(folderpath, foldername)
os.makedirs(savepath, exist_ok=True)

# Parameters to set
neuron_type = 'Other' # 'PYR' or 'Other'
stimuli = ["drifting-grating-0.2", "drifting-grating-0.6", "drifting-grating-1.0", "looming-stim"] # ["all"] or name of specific stimuli to analyze

#%% Load visual stimuli and photodiode data
print("Loading photodiode data")
NIdaq, acq_freq = ptd.load_and_data_extraction(folderpath)
Psignal_time, Psignal = general_functions.resample_signal(NIdaq['analog'][0],
                                                          original_freq=acq_freq,
                                                          new_freq=1000)

print("Loading visual stimuli data")
visual_stim = VisualStim(folderpath)
protocol_df = visual_stim.protocol_df
visual_stim.realign_from_photodiode(Psignal_time, Psignal)

#%% Calcium Imaging with sliding hamming
F0_method = 'hamming'
ca_img_hamming = CaImagingDataManager(folderpath, f0_method=F0_method, neuron_type=neuron_type)
detected_roi = ca_img_hamming._list_ROIs_idx
print('Original number of neurons :', len(detected_roi))

#---------------------------------- Detect ROIs with bad neuropils ------------------
ca_img_hamming.detect_bad_neuropils()
print('After removing bad neuropil neurons, nb of neurons :', len(ca_img_hamming._list_ROIs_idx))

#---------------------------------- Compute Fluorescence ------------------
ca_img_hamming.compute_F()
print('Number of remaining neurons after alpha calculation :', len(ca_img_hamming._list_ROIs_idx))

#---------------------------------- Calculation of F0 ----------------------
ca_img_hamming.compute_F0(percentile=10, win=60)
print('Number of remaining neurons after F0 calculation  :', len(ca_img_hamming._list_ROIs_idx))

#---------------------------------- Calculation of dF over F0 ----------------------
ca_img_hamming.compute_dFoF0()

print("Percentage of neurons kept : ", len(ca_img_hamming._list_ROIs_idx)/len(detected_roi)*100)
print("    ------------> Hamming Done")

#%% Calcium Imaging with sliding
F0_method = 'sliding'
ca_img_sliding = CaImagingDataManager(folderpath, f0_method=F0_method, neuron_type=neuron_type)
detected_roi = ca_img_sliding._list_ROIs_idx
print('Original number of neurons :', len(detected_roi))

#---------------------------------- Detect ROIs with bad neuropils ------------------
ca_img_sliding.detect_bad_neuropils()
print('After removing bad neuropil neurons, nb of neurons :', len(ca_img_sliding._list_ROIs_idx))

#---------------------------------- Compute Fluorescence ------------------
ca_img_sliding.compute_F()
print('Number of remaining neurons after alpha calculation :', len(ca_img_sliding._list_ROIs_idx))

#---------------------------------- Calculation of F0 ----------------------
ca_img_sliding.compute_F0(percentile=10, win=60)
print('Number of remaining neurons after F0 calculation  :', len(ca_img_sliding._list_ROIs_idx))

#---------------------------------- Calculation of dF over F0 ----------------------
ca_img_sliding.compute_dFoF0()

print("Percentage of neurons kept : ", len(ca_img_sliding._list_ROIs_idx)/len(detected_roi)*100)
print("    ------------> Slicing Done")

#%% Create Trials
_, ca_onset_indexes = ptd.Find_F_stim_index(visual_stim.real_time_onset, ca_img_hamming.time_stamps)

trials_hamming = Trial(ca_img_hamming, visual_stim, ca_onset_indexes, attr='dFoF0', dt_pre_stim=1, dt_post_stim=0.5)
trials_sliding = Trial(ca_img_sliding, visual_stim, ca_onset_indexes, attr='dFoF0', dt_pre_stim=1, dt_post_stim=0.5)

list_ROIs_idx_common = list(set(ca_img_sliding._list_ROIs_idx) & set(ca_img_hamming._list_ROIs_idx))
ROIs_idxs_h = [list(ca_img_hamming._list_ROIs_idx).index(i) for i in list_ROIs_idx_common]
ROIs_idxs_s = [list(ca_img_sliding._list_ROIs_idx).index(i) for i in list_ROIs_idx_common]

df = pd.DataFrame({'suite2p_idx': list_ROIs_idx_common,
                   'ROIs_idx_hamming': ROIs_idxs_h,
                   'ROIs_idx_sliding': ROIs_idxs_s})
df.to_excel(os.path.join(savepath, 'equivalence_ROIs_ids.xlsx'), index=False)

#%% stimuli selection

if "all" not in stimuli :
    stimuli_id_list = [protocol_df['name'].tolist().index(stimuli[i]) for i in range(len(stimuli))]
    if len(stimuli_id_list) == 0 :
        raise ValueError(f"Stimuli {stimuli} not found in protocol_df")
else :
    stimuli_id_list = [i for i in range(len(visual_stim.stim_cat)) if visual_stim.stim_cat[i] == 1]

#%% plot rastermaps
for stimuli_id in stimuli_id_list:
    stim_dt = trials_hamming.visual_stim.protocol_df['duration'][stimuli_id]
    stimuli_name = trials_hamming.visual_stim.protocol_df['name'][stimuli_id]
    stimuli_onset = trials_hamming.pre_trial_fluorescence[stimuli_id].shape[2]

    os.makedirs(os.path.join(savepath, stimuli_name), exist_ok=True)

    print(f"Plotting comparison for stimulus {stimuli_name}...")

    #loop for multiple ROIs
    for i in range(len(list_ROIs_idx_common)) :

        print(f"    ROI {list_ROIs_idx_common[i]}")
        roi_id_h = ROIs_idxs_h[i]
        roi_id_s = ROIs_idxs_s[i]

        data_hamming =\
            np.concatenate((trials_hamming.pre_trial_fluorescence[stimuli_id][roi_id_h], 
                            trials_hamming.trial_fluorescence[stimuli_id][roi_id_h], 
                            trials_hamming.post_trial_fluorescence[stimuli_id][roi_id_h]), axis=1)

        data_sliding =\
            np.concatenate((trials_sliding.pre_trial_fluorescence[stimuli_id][roi_id_s], 
                            trials_sliding.trial_fluorescence[stimuli_id][roi_id_s], 
                            trials_sliding.post_trial_fluorescence[stimuli_id][roi_id_s]), axis=1)

        time = (np.arange(data_hamming.shape[1]) - stimuli_onset) / ca_img_hamming.fs

        vmin, vmax = np.nanmin([data_hamming, data_sliding]), np.nanmax([data_hamming, data_sliding])

        # plot
        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(20, 8), sharex=True)
        im = ax[0].pcolormesh(time, np.arange(data_hamming.shape[0]), data_hamming, cmap='Greys', vmin=vmin, vmax=vmax)
        ax[1].pcolormesh(time, np.arange(data_sliding.shape[0]), data_sliding, cmap='Greys', vmin=vmin, vmax=vmax)
        vmin2 = np.min(data_hamming - data_sliding) if np.min(data_hamming - data_sliding) !=0 else -0.1
        vmax2 = np.max(data_hamming - data_sliding) if np.max(data_hamming - data_sliding) !=0 else -0.1
        lim = np.max([np.abs(vmin2), np.abs(vmax2)])
        im2 = ax[2].pcolormesh(time, np.arange(data_sliding.shape[0]), data_hamming - data_sliding, cmap='Reds')
        for j in range(3):
            ax[j].axvline(x=0, color='black', linestyle='--')
            ax[j].axvline(x=stim_dt, color='black', linestyle='--')
        ax[0].set_ylabel('Trial number')
        ax[1].set_ylabel('Trial number')
        ax[2].set_ylabel('Trial number')
        ax[2].set_xlabel('Time (s)')
        ax[0].set_title('Hamming Method')
        ax[1].set_title('Sliding Method')
        ax[2].set_title('dF/F0 difference between hamming and sliding method')
        fig.colorbar(im, ax=[ax[0], ax[1]])
        fig.colorbar(im2, ax=ax[2])
        fig.suptitle(f'ROI {list_ROIs_idx_common[i]} comparison for stimulus : {stimuli_name}')
        fig.savefig(os.path.join(savepath, stimuli_name, f'{stimuli_name}_roi_{list_ROIs_idx_common[i]}_raster.png'))
        plt.close(fig)

        trials_sort_diff = np.argsort(np.abs(np.sum(data_hamming - data_sliding, axis=1)))[::-1]
        if len(trials_sort_diff) > 10 :
            trials_sort_diff = trials_sort_diff[:10]
        
        fig, ax = plt.subplots(len(trials_sort_diff)+1, 1, figsize=(7, (len(trials_sort_diff)+1)*1.5), sharex=True)
        for j in range(len(trials_sort_diff)) :
            ax[j].plot(time, data_sliding[trials_sort_diff[j]], label='sliding', color='steelblue')
            ax[j].plot(time, data_hamming[trials_sort_diff[j]], label='hamming', color='darkorange')
            ax[j].set_frame_on(False)
            ax[j].set_ylabel('dF/F0')
            ax[j].axvline(x=0, color='black', linestyle='--')
            ax[j].axvline(x=stim_dt, color='black', linestyle='--')
            ax[j].set_title(f'Trial {trials_sort_diff[j]+1}')
        ax[j+1].plot(time, np.mean(data_sliding, axis=0), label='sliding', color='steelblue')
        ax[j+1].plot(time, np.mean(data_hamming, axis=0), label='hamming', color='darkorange')
        ax[j+1].axvline(x=0, color='black', linestyle='--')
        ax[j+1].axvline(x=stim_dt, color='black', linestyle='--')
        ax[j+1].set_frame_on(False)
        ax[j+1].set_title(f'Average over trials')
        ax[j+1].set_xlabel('Time (s)')
        ax[0].legend()
        fig.suptitle(f'Trials with highest dF/F0 difference for ROI {list_ROIs_idx_common[i]} for {stimuli_name}')
        fig.tight_layout(pad=2.0)
        fig.savefig(os.path.join(savepath, stimuli_name, f'{stimuli_name}_roi_{list_ROIs_idx_common[i]}.png'))
        plt.close(fig)

    print("--> Done!")