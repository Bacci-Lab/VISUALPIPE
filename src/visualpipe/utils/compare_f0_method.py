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
folderpath = r"path/to/your/data/folder"
foldername = "compare_f0_method"
savepath = os.path.join(folderpath, foldername)
os.makedirs(savepath, exist_ok=True)

# Parameters to set
neuron_type = 'Other' # 'PYR' or 'Other' (calculation of alpha)
#stimuli = ["drifting-grating-0.2", "drifting-grating-0.6", "drifting-grating-1.0", "looming-stim"] # ["all"], names of specific stimuli to analyze or None
stimuli = None

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
""" F0_method = 'hamming'
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
print("    ------------> Hamming Done") """

#%% Calcium Imaging 1
F0_method = 'sliding'
window = 60  # seconds
sig = 60  # seconds
description1 = 'Sliding window 60s'

ca_img_1 = CaImagingDataManager(folderpath, f0_method=F0_method, neuron_type=neuron_type)
detected_roi = ca_img_1._list_ROIs_idx
print('Original number of neurons :', len(detected_roi))

#---------------------------------- Detect ROIs with bad neuropils ------------------
ca_img_1.detect_bad_neuropils()
print('After removing bad neuropil neurons, nb of neurons :', len(ca_img_1._list_ROIs_idx))

#---------------------------------- Compute Fluorescence ------------------
ca_img_1.compute_F()
print('Number of remaining neurons after alpha calculation :', len(ca_img_1._list_ROIs_idx))

#---------------------------------- Calculation of F0 ----------------------
ca_img_1.compute_F0(percentile=10, win=window, sig=sig)
print('Number of remaining neurons after F0 calculation  :', len(ca_img_1._list_ROIs_idx))

#---------------------------------- Calculation of dF over F0 ----------------------
ca_img_1.compute_dFoF0()

print("Percentage of neurons kept : ", len(ca_img_1._list_ROIs_idx)/len(detected_roi)*100)
print(f"    ------------> {description1} Done")

#%% Calcium Imaging 2
F0_method = 'sliding'
window = 300  # seconds
sig = 60  # seconds
description2 = 'Sliding window 300s'

ca_img_2 = CaImagingDataManager(folderpath, f0_method=F0_method, neuron_type=neuron_type)
detected_roi = ca_img_2._list_ROIs_idx
print('Original number of neurons :', len(detected_roi))

#---------------------------------- Detect ROIs with bad neuropils ------------------
ca_img_2.detect_bad_neuropils()
print('After removing bad neuropil neurons, nb of neurons :', len(ca_img_2._list_ROIs_idx))

#---------------------------------- Compute Fluorescence ------------------
ca_img_2.compute_F()
print('Number of remaining neurons after alpha calculation :', len(ca_img_2._list_ROIs_idx))

#---------------------------------- Calculation of F0 ----------------------
ca_img_2.compute_F0(percentile=10, win=window, sig=sig)
print('Number of remaining neurons after F0 calculation  :', len(ca_img_2._list_ROIs_idx))

#---------------------------------- Calculation of dF over F0 ----------------------
ca_img_2.compute_dFoF0()

print("Percentage of neurons kept : ", len(ca_img_2._list_ROIs_idx)/len(detected_roi)*100)
print(f"    ------------> {description2} Done")

#%% Find common ROIs between both methods
list_ROIs_idx_common = list(set(ca_img_1._list_ROIs_idx) & set(ca_img_2._list_ROIs_idx))
ROIs_idxs_1 = [list(ca_img_1._list_ROIs_idx).index(i) for i in list_ROIs_idx_common]
ROIs_idxs_2 = [list(ca_img_2._list_ROIs_idx).index(i) for i in list_ROIs_idx_common]

df = pd.DataFrame({'suite2p_idx': list_ROIs_idx_common,
                   f'ROIs_idx_1 ({description1})': ROIs_idxs_1,
                   f'ROIs_idx_2 ({description2})': ROIs_idxs_2})
df.to_excel(os.path.join(savepath, 'equivalence_ROIs_ids.xlsx'), index=False)

#%% stimuli selection
if stimuli is None :
    print("No stimuli specified, analyzing entire trace.")
    stimuli_id_list = []
elif "all" not in stimuli :
    stimuli_id_list = [protocol_df['name'].tolist().index(stimuli[i]) for i in range(len(stimuli))]
    if len(stimuli_id_list) == 0 :
        raise ValueError(f"Stimuli {stimuli} not found in protocol_df")
else :
    stimuli_id_list = [i for i in range(len(visual_stim.stim_cat)) if visual_stim.stim_cat[i] == 1]

#%% Plot rastermaps
if stimuli_id_list != [] :

    # create trials
    _, ca_onset_indexes = ptd.Find_F_stim_index(visual_stim.real_time_onset, ca_img_1.time_stamps)
    trials_1 = Trial(ca_img_1, visual_stim, ca_onset_indexes, attr='dFoF0', dt_pre_stim=1, dt_post_stim=0.5)
    trials_2 = Trial(ca_img_2, visual_stim, ca_onset_indexes, attr='dFoF0', dt_pre_stim=1, dt_post_stim=0.5)

    # loop over stimuli
    for stimuli_id in stimuli_id_list:
        stim_dt = trials_1.visual_stim.protocol_df['duration'][stimuli_id]
        stimuli_name = trials_1.visual_stim.protocol_df['name'][stimuli_id]
        stimuli_onset = trials_1.pre_trial_fluorescence[stimuli_id].shape[2]

        os.makedirs(os.path.join(savepath, stimuli_name), exist_ok=True)

        print(f"Plotting comparison for stimulus {stimuli_name}...")

        #loop for multiple ROIs
        for i in range(len(list_ROIs_idx_common)) :

            print(f"    ROI {list_ROIs_idx_common[i]}")
            roi_id_1 = ROIs_idxs_1[i]
            roi_id_2 = ROIs_idxs_2[i]

            data_1 =\
                np.concatenate((trials_1.pre_trial_fluorescence[stimuli_id][roi_id_1], 
                                trials_1.trial_fluorescence[stimuli_id][roi_id_1], 
                                trials_1.post_trial_fluorescence[stimuli_id][roi_id_1]), axis=1)

            data_2 =\
                np.concatenate((trials_2.pre_trial_fluorescence[stimuli_id][roi_id_2], 
                                trials_2.trial_fluorescence[stimuli_id][roi_id_2], 
                                trials_2.post_trial_fluorescence[stimuli_id][roi_id_2]), axis=1)

            time = (np.arange(data_1.shape[1]) - stimuli_onset) / ca_img_1.fs

            vmin, vmax = np.nanmin([data_1, data_2]), np.nanmax([data_1, data_2])

            # plot rastermaps
            fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(20, 8), sharex=True)
            im = ax[0].pcolormesh(time, np.arange(data_1.shape[0]), data_1, cmap='Greys', vmin=vmin, vmax=vmax)
            ax[1].pcolormesh(time, np.arange(data_2.shape[0]), data_2, cmap='Greys', vmin=vmin, vmax=vmax)
            vmin2 = np.min(data_1 - data_2) if np.min(data_1 - data_2) !=0 else -0.1
            vmax2 = np.max(data_1 - data_2) if np.max(data_1 - data_2) !=0 else -0.1
            lim = np.max([np.abs(vmin2), np.abs(vmax2)])
            im2 = ax[2].pcolormesh(time, np.arange(data_2.shape[0]), data_1 - data_2, cmap='Reds')
            for j in range(3):
                ax[j].axvline(x=0, color='black', linestyle='--')
                ax[j].axvline(x=stim_dt, color='black', linestyle='--')
            ax[0].set_ylabel('Trial number')
            ax[1].set_ylabel('Trial number')
            ax[2].set_ylabel('Trial number')
            ax[2].set_xlabel('Time (s)')
            ax[0].set_title(f'{description1} Method')
            ax[1].set_title(f'{description2} Method')
            ax[2].set_title('dF/F0 difference between traces 1 and 2')
            fig.colorbar(im, ax=[ax[0], ax[1]])
            fig.colorbar(im2, ax=ax[2])
            fig.suptitle(f'ROI {list_ROIs_idx_common[i]} comparison for stimulus : {stimuli_name}')
            fig.savefig(os.path.join(savepath, stimuli_name, f'{stimuli_name}_roi_{list_ROIs_idx_common[i]}_raster.png'))
            plt.close(fig)

            # plot trials with highest difference
            trials_sort_diff = np.argsort(np.abs(np.sum(data_1 - data_2, axis=1)))[::-1]
            if len(trials_sort_diff) > 10 :
                trials_sort_diff = trials_sort_diff[:10]
            
            fig, ax = plt.subplots(len(trials_sort_diff)+1, 1, figsize=(7, (len(trials_sort_diff)+1)*1.5), sharex=True)
            for j in range(len(trials_sort_diff)) :
                ax[j].plot(time, data_2[trials_sort_diff[j]], label=description2, color='steelblue')
                ax[j].plot(time, data_1[trials_sort_diff[j]], label=description1, color='darkorange')
                ax[j].set_frame_on(False)
                ax[j].set_ylabel('dF/F0')
                ax[j].axvline(x=0, color='black', linestyle='--')
                ax[j].axvline(x=stim_dt, color='black', linestyle='--')
                ax[j].set_title(f'Trial {trials_sort_diff[j]+1}')
            ax[j+1].plot(time, np.mean(data_2, axis=0), label=description2, color='steelblue')
            ax[j+1].plot(time, np.mean(data_1, axis=0), label=description1, color='darkorange')
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

#%% Plotting comparison of F0 methods on entire trace
print("Plotting comparison of F0 methods on entire trace...")
t = ca_img_1.time_stamps

for i in range(len(list_ROIs_idx_common)) :

    print(f"    ROI {list_ROIs_idx_common[i]}")
    roi_id_1 = ROIs_idxs_1[i]
    roi_id_2 = ROIs_idxs_2[i]

    raw_f = ca_img_1.raw_F[roi_id_1]
    
    fig, ax = plt.subplots(2, 1, figsize=(len(t)*0.0008, 8), sharex=True)
    ax[0].plot(t, raw_f, alpha=0.5, label='raw fluorescence', color='gray')
    ax[0].plot(t, ca_img_1.f0[roi_id_1], label=f'f0 {description1}', color='steelblue')
    ax[0].plot(t, ca_img_2.f0[roi_id_1], label=f'f0 {description2}', color='darkorange')
    ax[1].plot(t, ca_img_1.dFoF0[roi_id_1], alpha=0.5, label=f'dF/F0 {description1}', color='steelblue')
    ax[1].plot(t, ca_img_2.dFoF0[roi_id_2], alpha=0.5, label=f'dF/F0 {description2}', color='darkorange')
    ax[1].set_xlabel('Time (s)')
    ax[0].legend()
    ax[1].legend()
    ax[0].set_frame_on(False)
    ax[1].set_frame_on(False)
    plt.suptitle(f'ROI {list_ROIs_idx_common[i]}')
    plt.tight_layout()
    fig.savefig(os.path.join(savepath, f'roi_{list_ROIs_idx_common[i]}.png'))
    plt.close(fig)
    #plt.show()