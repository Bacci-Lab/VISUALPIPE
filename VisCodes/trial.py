import numpy as np
from Ca_imaging import CaImagingDataManager
from visual_stim import VisualStim
import matplotlib.pyplot as plt
import math
import os
from scipy.ndimage import gaussian_filter1d
from matplotlib.colors import LinearSegmentedColormap

class Trial(object):

    def __init__(self, ca_img: CaImagingDataManager, visual_stim: VisualStim, ca_onset_idxes, attr='fluorescence', dt_pre_stim=0.5, dt_post_stim=0):
        self.ca_img = ca_img
        self.visual_stim = visual_stim
        self.ca_onset_idxes = np.array(ca_onset_idxes)
        self.ca_attr = attr
        self.dt_pre_stim = dt_pre_stim
        self.dt_post_stim = dt_post_stim

        self.trial_fluorescence, self.pre_trial_fluorescence = self.compute_trial(self.ca_attr, self.dt_pre_stim, self.dt_post_stim)
        self.average_baselines = self.compute_average_baselines(self.pre_trial_fluorescence)
        self.trial_averaged_zscores, self.pre_trial_averaged_zscores = self.compute_trial_averaged_zscores(self.trial_fluorescence, self.average_baselines)
        self.trial_zscores, self.pre_trial_zscores = self.compute_trial_zscores_avb()
        
    def get_baseline(self, roi_id, stimulus_id, attr='fluorescence', dt_pre=0.5):
        """
        For a fixed stimulus and neuron, get the baseline of each stimuli occurence.

        roi_id: int - ROI id
        stimulus_id: int - stimulus id
        attr: string - attribute of calcium imaging ('raw_F', 'fluorescence' or 'dFoF0')
        df_pre: float - time period in second to calculate baseline
        """
        if attr not in ['raw_F', 'fluorescence', 'dFoF0'] :
            raise Exception("Choose a valid calcium trace.")
        
        if hasattr(self.ca_img, attr) : 
            trace = getattr(self.ca_img, attr)
        else : 
            raise Exception("Ca_imaging object has no attribute " + attr)

        baseline_list = []
        onset_idx_list = self.ca_onset_idxes[self.visual_stim.stimuli_idx[stimulus_id]]
        nb_frames = round(dt_pre * self.ca_img.fs)

        for i in onset_idx_list:
            baseline_i = trace[roi_id][i-nb_frames : i]
            baseline_list.append(baseline_i)

        return np.array(baseline_list)

    def get_trial_trace(self, roi_id, stimulus_id, attr='fluorescence', dt_post_stim=0):
        """
        For a fixed stimulus and neuron, get the calcium imaging trace of each stimuli occurence.

        roi_id: int - ROI id
        stimulus_id: int - stimulus id
        attr: string - attribute of calcium imaging ('raw_F', 'fluorescence' or 'dFoF0')
        dt_post_stim: float - time period in second after the end of the stimulus to include in trial 
        """
        if attr not in ['raw_F', 'fluorescence', 'dFoF0'] :
            raise Exception("Choose a valid calcium trace.")
        
        if hasattr(self.ca_img, attr) : 
            trace = getattr(self.ca_img, attr)
        else : 
            raise Exception("Ca_imaging object has no attribute " + attr)

        trial_trace_list = []
        onset_idx_list = self.ca_onset_idxes[self.visual_stim.stimuli_idx[stimulus_id]]
        stim_dt = self.visual_stim.protocol_df['duration'][stimulus_id]
        trial_nb_frames = round((stim_dt + dt_post_stim) * self.ca_img.fs)

        for i in onset_idx_list:
            trial_trace_i = trace[roi_id][i : i + trial_nb_frames]
            trial_trace_list.append(trial_trace_i)
        
        return np.array(trial_trace_list)

    def zscores(self, baselines, traces):
        return np.array([(traces[i, :] - np.mean(baselines, axis=1)[i]) / np.std(baselines, axis=1)[i] for i in range(traces.shape[0])])
    
    def zscores2(self, baselines, traces):
        return np.array([(traces[i, :] - np.mean(baselines)) / np.std(baselines) for i in range(traces.shape[0])])

    def compute_trial(self, attr='fluorescence', dt_pre_stim=0.5, dt_post_stim=0):

        trial_fluorescence, pre_trial_fluorescence = {}, {}

        for i in self.visual_stim.stimuli_idx.keys() :
            trial_fluorescence_i, baselines_i = [], []

            for roi_idx in range(len(self.ca_img._list_ROIs_idx)):
                roi_baselines = self.get_baseline(roi_idx, i, attr, dt_pre_stim)
                roi_trial_fluorescence = self.get_trial_trace(roi_idx, i, attr, dt_post_stim)

                baselines_i.append(roi_baselines)
                trial_fluorescence_i.append(roi_trial_fluorescence)

            trial_fluorescence.update({i : np.array(trial_fluorescence_i)})
            pre_trial_fluorescence.update({i : np.array(baselines_i)})

        return trial_fluorescence, pre_trial_fluorescence
    
    def compute_average_baselines(self, pre_trial_fluorescence) :
        average_baselines = {}

        for i in self.visual_stim.stimuli_idx.keys() :
            average_baseline_i = np.mean(pre_trial_fluorescence[i], axis=1)
            average_baselines.update({i: average_baseline_i})

        return average_baselines

    def compute_trial_zscores(self, attr='fluorescence', dt_pre_stim=0.5, dt_post_stim=0):

        trial_zscores, pre_trial_zscores = {}, {}

        for i in self.visual_stim.stimuli_idx.keys() :
            roi_zscores, roi_f0_zscores = [], []

            for roi_idx in range(len(self.ca_img._list_ROIs_idx)):
                roi_baselines = self.get_baseline(roi_idx, i, attr, dt_pre_stim)
                roi_trial_fluorescence = self.get_trial_trace(roi_idx, i, attr, dt_post_stim)

                roi_zscore = self.zscores(roi_baselines, roi_trial_fluorescence)
                roi_f0_zscore = self.zscores(roi_baselines, roi_baselines)
            
                roi_zscores.append(roi_zscore)
                roi_f0_zscores.append(roi_f0_zscore)

            trial_zscores.update({i : np.array(roi_zscores)})
            pre_trial_zscores.update({i : np.array(roi_f0_zscores)})

        return trial_zscores, pre_trial_zscores

    def compute_trial_zscores_avb(self):
        trial_zscores, pre_trial_zscores = {}, {}

        for i in self.visual_stim.stimuli_idx.keys() :
            roi_zscores, roi_f0_zscores = [], []

            for roi_idx in range(len(self.ca_img._list_ROIs_idx)):
                roi_baselines = self.get_baseline(roi_idx, i, self.ca_attr, self.dt_pre_stim)
                roi_trial_fluorescence = self.get_trial_trace(roi_idx, i, self.ca_attr, self.dt_post_stim)

                roi_zscore = self.zscores2(self.average_baselines[i][roi_idx], roi_trial_fluorescence)
                roi_f0_zscore = self.zscores2(self.average_baselines[i][roi_idx], roi_baselines)
            
                roi_zscores.append(roi_zscore)
                roi_f0_zscores.append(roi_f0_zscore)

            trial_zscores.update({i : np.array(roi_zscores)})
            pre_trial_zscores.update({i : np.array(roi_f0_zscores)})

        return trial_zscores, pre_trial_zscores

    def compute_trial_averaged_zscores(self, trial_fluorescence, average_baselines):

        trial_averaged_zscores, pre_trial_averaged_zscores = {}, {}

        for i in self.visual_stim.stimuli_idx.keys() :
            average_baseline = average_baselines[i]
            average_trial = np.mean(trial_fluorescence[i], axis=1)

            trial_averaged_zscores.update({i : self.zscores(average_baseline, average_trial)})
            pre_trial_averaged_zscores.update({i : self.zscores(average_baseline, average_baseline)})

        return trial_averaged_zscores, pre_trial_averaged_zscores

    def trial_average_rasterplot(self, stimuli_id, savepath='') :
        
        stimuli_name = self.visual_stim.protocol_df['name'][stimuli_id]
        stimuli_onset = self.pre_trial_averaged_zscores[stimuli_id].shape[1]
        data = np.concatenate((self.pre_trial_averaged_zscores[stimuli_id], self.trial_averaged_zscores[stimuli_id]), axis=1)
        time = (np.arange(data.shape[1]) - stimuli_onset) / self.ca_img.fs

        vmin, vmax = np.nanmin(data), np.nanmax(data)
        if np.abs(vmin) > np.abs(vmax) :
            lim = math.floor(np.abs(vmin)*100) * 0.01
        else : 
            lim = math.floor(np.abs(vmax)*100) * 0.01

        fig = plt.figure(figsize=(10, 6))
        im = plt.pcolormesh(time, np.arange(data.shape[0]), data, cmap='RdBu_r', vmin=-lim, vmax=lim)
        plt.colorbar(im, label=self.ca_attr + " z-score")
        plt.axvline(0, color='black', linestyle='--')
        plt.xlabel('Time (s)')
        plt.ylabel('Neuron')
        plt.title(stimuli_name)

        fig.savefig(os.path.join(savepath, stimuli_name + "_trial_average_rasterplot.png"))
        plt.close(fig)

    def trial_rasterplot(self, trial_zscores, pre_trial_zscores, stimuli_id, attr='fluorescence', savepath='') :
        
        stimuli_name = self.visual_stim.protocol_df['name'][stimuli_id]
        stimuli_onset = pre_trial_zscores[stimuli_id].shape[2]
        data = np.concatenate((pre_trial_zscores[stimuli_id], trial_zscores[stimuli_id]), axis=2).reshape((-1, trial_zscores[stimuli_id].shape[2]+pre_trial_zscores[stimuli_id].shape[2]))
        time = (np.arange(data.shape[1]) - stimuli_onset) / self.ca_img.fs

        vmin, vmax = np.nanmin(data), np.nanmax(data)
        if np.abs(vmin) > np.abs(vmax) :
            lim = math.floor(np.abs(vmin)*100) * 0.01
        else : 
            lim = math.floor(np.abs(vmax)*100) * 0.01

        fig = plt.figure(figsize=(15, 15))
        im = plt.pcolormesh(time, np.arange(data.shape[0]), data, cmap='RdBu_r', vmin=-lim, vmax=lim)
        plt.colorbar(im, label=attr + " z-score")
        plt.axvline(0, color='black', linestyle='--')
        if trial_zscores[stimuli_id].shape[1] > 1 :
            for i in range(1, trial_zscores[stimuli_id].shape[0]): 
                plt.axhline(i*trial_zscores[stimuli_id].shape[1]+0.5, color='gray', linestyle=':', linewidth=0.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Stimulus occurence per neuron')
        plt.title(stimuli_name)

        fig.savefig(os.path.join(savepath, stimuli_name + "_trial_rasterplot.png"))
        plt.close(fig)