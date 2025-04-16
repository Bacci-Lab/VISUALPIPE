import numpy as np
from Ca_imaging import CaImagingDataManager
from visual_stim import VisualStim
import matplotlib.pyplot as plt
import math
import os
from scipy.ndimage import gaussian_filter1d
from matplotlib.colors import LinearSegmentedColormap
from sklearn import metrics

class Trial(object):

    def __init__(self, ca_img: CaImagingDataManager, visual_stim: VisualStim, ca_onset_idxes, attr='fluorescence', dt_pre_stim=0.5, dt_post_stim=0, dt_post_stim_plot=None):

        self.ca_img = ca_img
        self.visual_stim = visual_stim
        self.ca_onset_idxes = np.array(ca_onset_idxes)
        self.ca_attr = attr
        self.dt_pre_stim = dt_pre_stim
        max_post_stim = np.min(visual_stim.interstim) - dt_pre_stim
        if dt_post_stim <= max_post_stim :
            self.dt_post_stim = dt_post_stim
        else : 
            raise Exception(f'dt_post_stim must be between 0 and {max_post_stim} s.')
        if dt_post_stim_plot is None :
            self.dt_post_stim_plot = max_post_stim - dt_post_stim
        else :
            self.dt_post_stim_plot = np.min([max_post_stim - dt_post_stim, dt_post_stim_plot])
        
        self.trial_fluorescence, self.pre_trial_fluorescence, self.post_trial_fluorescence = self.compute_trial(self.ca_attr)
        self.average_baselines = self.compute_average_baselines(self.pre_trial_fluorescence)
        self.trial_averaged_zscores, self.pre_trial_averaged_zscores, self.post_trial_averaged_zscores =\
              self.compute_trial_averaged_zscores(self.trial_fluorescence, self.post_trial_fluorescence, self.average_baselines)
        self.trial_zscores, self.pre_trial_zscores, self.post_trial_zscores = self.compute_trial_zscores_avb()
        self.trial_response_bounds, self.responsive = self.find_responsive_rois()
        
    def get_trials_trace(self, roi_id, stimulus_id, attr='fluorescence'):
        """
        For a fixed stimulus and neuron, get the calcium imaging baseline, trial trace and post-stimulus trace of each stimuli occurence.

        roi_id: int - ROI id
        stimulus_id: int - stimulus id
        attr: string - attribute of calcium imaging ('raw_F', 'fluorescence' or 'dFoF0')
        """

        if attr not in ['raw_F', 'fluorescence', 'dFoF0'] :
            raise Exception("Choose a valid calcium trace.")
        
        if hasattr(self.ca_img, attr) : 
            trace = getattr(self.ca_img, attr)
        else : 
            raise Exception("Ca_imaging object has no attribute " + attr)

        baseline_list = []
        trial_trace_list = []
        ptrial_trace_list = []

        onset_idx_list = self.ca_onset_idxes[self.visual_stim.stimuli_idx[stimulus_id]]
        stim_dt = self.visual_stim.protocol_df['duration'][stimulus_id]
        pre_trial_nb_frames = round(self.dt_pre_stim * self.ca_img.fs)
        trial_nb_frames = round((stim_dt + self.dt_post_stim) * self.ca_img.fs)
        ptrial_nb_frames = round(self.dt_post_stim_plot * self.ca_img.fs)

        for i in onset_idx_list:
            baseline_i = trace[roi_id][i-pre_trial_nb_frames : i]
            baseline_list.append(baseline_i)

            trial_trace_i = trace[roi_id][i : i + trial_nb_frames]
            trial_trace_list.append(trial_trace_i)

            trial_trace_i = trace[roi_id][i + trial_nb_frames: i + trial_nb_frames + ptrial_nb_frames]
            ptrial_trace_list.append(trial_trace_i)
        
        return np.array(baseline_list), np.array(trial_trace_list), np.array(ptrial_trace_list)

    def zscores(self, baselines, traces):
        return np.array([(traces[i, :] - np.mean(baselines, axis=1)[i]) / np.std(baselines, axis=1)[i] for i in range(traces.shape[0])])
    
    def zscores2(self, baselines, traces):
        return np.array([(traces[i, :] - np.mean(baselines)) / np.std(baselines) for i in range(traces.shape[0])])

    def compute_trial(self, attr='fluorescence'):

        trial_fluorescence, pre_trial_fluorescence, post_trial_fluorescence = {}, {}, {}

        for i in self.visual_stim.protocol_ids :

            if self.visual_stim.stim_cat[i] :

                trial_fluorescence_i, baselines_i, post_trial_fluorescence_i = [], [], []

                for roi_idx in range(len(self.ca_img._list_ROIs_idx)):
                    roi_baselines, roi_trial_fluorescence, roi_post_trial_fluorescence = self.get_trials_trace(roi_idx, i, attr)

                    baselines_i.append(roi_baselines)
                    trial_fluorescence_i.append(roi_trial_fluorescence)
                    post_trial_fluorescence_i.append(roi_post_trial_fluorescence)

                trial_fluorescence.update({i : np.array(trial_fluorescence_i)})
                pre_trial_fluorescence.update({i : np.array(baselines_i)})
                post_trial_fluorescence.update({i : np.array(post_trial_fluorescence_i)})

        return trial_fluorescence, pre_trial_fluorescence, post_trial_fluorescence
    
    def compute_average_baselines(self, pre_trial_fluorescence) :
        average_baselines = {}

        for i in self.trial_fluorescence.keys() :
            average_baseline_i = np.mean(pre_trial_fluorescence[i], axis=1)
            average_baselines.update({i: average_baseline_i})

        return average_baselines

    def compute_trial_zscores(self, attr='fluorescence'):

        trial_zscores, pre_trial_zscores, post_trial_zscores = {}, {}, {}

        for i in self.trial_fluorescence.keys() :
            roi_zscores, roi_f0_zscores, roi_post_zscores = [], [], []

            for roi_idx in range(len(self.ca_img._list_ROIs_idx)):
                roi_baselines, roi_trial_fluorescence, roi_post_trial_fluorescence = self.get_trials_trace(roi_idx, i, attr)

                roi_zscore = self.zscores(roi_baselines, roi_trial_fluorescence)
                roi_f0_zscore = self.zscores(roi_baselines, roi_baselines)
                roi_post_zscore = self.zscores(roi_baselines, roi_post_trial_fluorescence)
            
                roi_zscores.append(roi_zscore)
                roi_f0_zscores.append(roi_f0_zscore)
                roi_post_zscores.append(roi_post_zscore)

            trial_zscores.update({i : np.array(roi_zscores)})
            pre_trial_zscores.update({i : np.array(roi_f0_zscores)})
            post_trial_zscores.update({i : np.array(roi_post_zscores)})

        return trial_zscores, pre_trial_zscores, post_trial_zscores

    def compute_trial_zscores_avb(self):
        trial_zscores, pre_trial_zscores, post_trial_zscores = {}, {}, {}

        for i in self.trial_fluorescence.keys() :
            roi_zscores, roi_f0_zscores, roi_post_zscores = [], [], []

            for roi_idx in range(len(self.ca_img._list_ROIs_idx)):
                roi_baselines, roi_trial_fluorescence, roi_post_trial_fluorescence = self.get_trials_trace(roi_idx, i, self.ca_attr)

                roi_zscore = self.zscores2(self.average_baselines[i][roi_idx], roi_trial_fluorescence)
                roi_f0_zscore = self.zscores2(self.average_baselines[i][roi_idx], roi_baselines)
                roi_post_zscore = self.zscores2(self.average_baselines[i][roi_idx], roi_post_trial_fluorescence)
            
                roi_zscores.append(roi_zscore)
                roi_f0_zscores.append(roi_f0_zscore)
                roi_post_zscores.append(roi_post_zscore)

            trial_zscores.update({i : np.array(roi_zscores)})
            pre_trial_zscores.update({i : np.array(roi_f0_zscores)})
            post_trial_zscores.update({i : np.array(roi_post_zscores)})

        return trial_zscores, pre_trial_zscores, post_trial_zscores

    def compute_trial_averaged_zscores(self, trial_fluorescence, post_trial_fluorescence, average_baselines):

        trial_averaged_zscores, pre_trial_averaged_zscores, post_trial_averaged_zscores = {}, {}, {}

        for i in self.trial_fluorescence.keys() :
            average_baseline = average_baselines[i]
            average_trial = np.mean(trial_fluorescence[i], axis=1)
            average_post_trial = np.mean(post_trial_fluorescence[i], axis=1)

            trial_averaged_zscores.update({i : self.zscores(average_baseline, average_trial)})
            pre_trial_averaged_zscores.update({i : self.zscores(average_baseline, average_baseline)})
            post_trial_averaged_zscores.update({i : self.zscores(average_baseline, average_post_trial)})

        return trial_averaged_zscores, pre_trial_averaged_zscores, post_trial_averaged_zscores

    def find_responsive_rois(self, dt_min=0.2, auc_min=5):

        responsive = {}
        trial_response_bounds = {}
        nb_frames_min = dt_min * self.ca_img.fs

        for i in self.trial_fluorescence.keys():
            stim_dt = self.visual_stim.protocol_df['duration'][i]
            trial_response_bounds_roi = []
            responsive_roi = []

            for roi_idx in range(len(self.ca_img._list_ROIs_idx)):
                roi_trial = self.trial_averaged_zscores[i][roi_idx]
                max_val = np.max(roi_trial)

                if max_val > 1 : 
                    start_idx, end_idx = self.find_bounds(np.argmax(roi_trial), roi_trial >= 0)
                    if end_idx - start_idx + 1 >= nb_frames_min :
                        time = np.linspace(0, stim_dt, len(roi_trial))
                        auc = metrics.auc(time[start_idx:end_idx+1], roi_trial[start_idx:end_idx+1])
                        if auc >= auc_min :
                            responsive_roi.append(1)
                        else :
                            responsive_roi.append(0)
                    else :
                        responsive_roi.append(0)
                    trial_response_bounds_roi.append([start_idx, end_idx])
                else :
                    responsive_roi.append(0)
                    trial_response_bounds_roi.append([None, None])

            responsive.update({i : responsive_roi})
            trial_response_bounds.update({i : trial_response_bounds_roi})
        
        return trial_response_bounds, responsive

    def find_bounds(self, index, bool_array):
        n = len(bool_array)
        
        i = index - 1
        while i >= 0 and bool_array[i] == True :
            i -= 1
        start_idx = i + 1

        i = index + 1
        while i < n and bool_array[i] == True :
            i += 1
        end_idx = i - 1

        return start_idx, end_idx

    def trial_average_rasterplot(self, stimuli_id, savepath='', sort=True) :
        
        stim_dt = self.visual_stim.protocol_df['duration'][stimuli_id]
        stimuli_name = self.visual_stim.protocol_df['name'][stimuli_id]
        stimuli_onset = self.pre_trial_averaged_zscores[stimuli_id].shape[1]
        data = np.concatenate((self.pre_trial_averaged_zscores[stimuli_id], self.trial_averaged_zscores[stimuli_id], self.post_trial_averaged_zscores[stimuli_id]), axis=1)
        if sort :
            data = np.array([trace for _, trace in sorted(zip(np.mean(self.trial_averaged_zscores[stimuli_id], axis=1), data))])
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
        if self.dt_post_stim + self.dt_post_stim_plot > 0 :
            plt.axvline(stim_dt, color='black', linestyle='--')
        plt.xlabel('Time (s)')
        plt.ylabel('Neuron')
        plt.title(stimuli_name)

        if sort :
            fig.savefig(os.path.join(savepath, stimuli_name + "_trial_average_rasterplot_sorted.png"))
        else :
            fig.savefig(os.path.join(savepath, stimuli_name + "_trial_average_rasterplot.png"))
        plt.close(fig)

    def trial_rasterplot(self, trial_zscores, pre_trial_zscores, post_trial_zscores, stimuli_id, attr='fluorescence', savepath='') :
        
        stim_dt = self.visual_stim.protocol_df['duration'][stimuli_id]
        stimuli_name = self.visual_stim.protocol_df['name'][stimuli_id]
        stimuli_onset = pre_trial_zscores[stimuli_id].shape[2]
        data = np.concatenate((pre_trial_zscores[stimuli_id], trial_zscores[stimuli_id], post_trial_zscores[stimuli_id]), axis=2).reshape((-1, trial_zscores[stimuli_id].shape[2]+pre_trial_zscores[stimuli_id].shape[2]+post_trial_zscores[stimuli_id].shape[2]))
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
        if self.dt_post_stim + self.dt_post_stim_plot > 0 :
            plt.axvline(stim_dt, color='black', linestyle='--')
        if trial_zscores[stimuli_id].shape[1] > 1 :
            for i in range(1, trial_zscores[stimuli_id].shape[0]): 
                plt.axhline(i*trial_zscores[stimuli_id].shape[1]+0.5, color='gray', linestyle=':', linewidth=0.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Stimulus occurence per neuron')
        plt.title(stimuli_name)

        fig.savefig(os.path.join(savepath, stimuli_name + "_trial_rasterplot.png"))
        plt.close(fig)

    def plot_stim_response(self, stimuli_id, neuron_idx, save_dir, file_prefix='', show_per_rois=False):

        stim_dt = self.visual_stim.protocol_df['duration'][stimuli_id]
        stimuli_name = self.visual_stim.protocol_df['name'][stimuli_id]
        stimuli_onset = self.pre_trial_averaged_zscores[stimuli_id].shape[1]
        trial_averaged_zscore = np.array(self.trial_averaged_zscores[stimuli_id][neuron_idx])
        pre_trial_averaged_zscore = np.array(self.pre_trial_averaged_zscores[stimuli_id][neuron_idx])
        post_trial_averaged_zscore = np.array(self.post_trial_averaged_zscores[stimuli_id][neuron_idx])
        
        data_av = np.concatenate((pre_trial_averaged_zscore, trial_averaged_zscore, post_trial_averaged_zscore))
        time = (np.arange(data_av.shape[0]) - stimuli_onset) / self.ca_img.fs
        
        if show_per_rois :
            trial_rois_zscores = self.trial_zscores[stimuli_id][neuron_idx]
            pre_trial_rois_zscores = self.pre_trial_zscores[stimuli_id][neuron_idx]
            data = np.concatenate((pre_trial_rois_zscores, trial_rois_zscores), axis=1)
            colors, positions = ['midnightblue', 'paleturquoise'], [0, 1]
            cmap = LinearSegmentedColormap.from_list('my_colormap', list(zip(positions, colors)), N=trial_rois_zscores.shape[0])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axvline(x=0, color='orchid', linestyle='--', alpha=0.7, linewidth=2)
        if self.dt_post_stim + self.dt_post_stim_plot > 0 :
            ax.axvline(x=stim_dt, color='orchid', linestyle='--', alpha=0.7, linewidth=2)
        if show_per_rois :
            for i in range(trial_rois_zscores.shape[0]):
                ax.plot(time, data[i], color=cmap(i), linewidth=0.5, alpha=0.5)
        ax.plot(time, data_av, color='black', label='Mean', linewidth=2)
        if self.responsive[stimuli_id][neuron_idx] :
            start = self.trial_response_bounds[stimuli_id][neuron_idx][0] + stimuli_onset
            end = self.trial_response_bounds[stimuli_id][neuron_idx][1] + stimuli_onset + 1
            ax.plot(time[start:end], data_av[start:end], color='green', label='trial response', linewidth=2)
            ax.axvspan(0, stim_dt + self.dt_post_stim, color='skyblue', alpha=0.2, label='trial period')
        else :
            ax.axvspan(0, stim_dt + self.dt_post_stim, color='thistle', alpha=0.2, label='trial period')
        fig_name = stimuli_name + "_neuron_" + str(neuron_idx)
        ax.margins(x=0)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(self.ca_attr + " z-score")
        ax.set_title(stimuli_name + '\n' + fig_name)
        ax.legend(bbox_to_anchor=(0, 1), loc='upper left', frameon=False)
        
        foldername = "_".join(list(filter(None, [file_prefix, stimuli_name])))
        save_folder = os.path.join(save_dir, foldername)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        save_path = os.path.join(save_folder, fig_name)
        fig.savefig(save_path)
        plt.close(fig)

    def save_protocol_validity(self, save_dir, filename):
        protocol_validity = []
        for id in self.responsive.keys():
            d = {self.visual_stim.protocol_names[id] : self.responsive[id]}
            protocol_validity.append(d)
        np.savez(os.path.join(save_dir, filename + ".npz" ), **{key: value for d in protocol_validity for key, value in d.items()})