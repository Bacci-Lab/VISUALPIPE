import numpy as np
from Ca_imaging import CaImagingDataManager
from visual_stim import VisualStim
import matplotlib.pyplot as plt
import math
import os
import random
from scipy.ndimage import gaussian_filter1d
from statsmodels.stats.weightstats import ztest as ztest
from scipy.stats import pearsonr
from matplotlib.colors import LinearSegmentedColormap
from sklearn import metrics
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoLocator

class Trial(object):

    def __init__(self, ca_img: CaImagingDataManager, visual_stim: VisualStim, ca_onset_idxes, attr: str='fluorescence', dt_pre_stim:float=0.5, dt_post_stim:float=0, dt_post_stim_plot:float=None):

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

        self.trial_response_bounds, self.responsive, self.reliability = None, None, None
        
    def get_trials_trace(self, roi_id:int, stimulus_id:int, attr:str='fluorescence'):
        """
        For a fixed stimulus and neuron, get the calcium imaging baseline, trial trace and post-stimulus trace of each stimuli occurence.

        :param int roi_id: ROI index.
        :param int stimulus_id: Stimulus imdex.
        :param string attr: Attribute of calcium imaging ('raw_F', 'fluorescence' or 'dFoF0').

        :return baseline (ndarray): Array of baselines for a given stimulus and ROI id.
        :return trial_trace (ndarray): Array of trials fluorescence traces for a given stimulus and ROI id.
        :return post_trial_trace (ndarray): Array of post-trials fluorescence traces for a given stimulus and ROI id (for plotting purpose only).
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

            trial_trace_i = trace[roi_id][i : i + trial_nb_frames + 1]
            trial_trace_list.append(trial_trace_i)

            trial_trace_i = trace[roi_id][i + trial_nb_frames + 1 : i + trial_nb_frames + ptrial_nb_frames]
            ptrial_trace_list.append(trial_trace_i)
        
        return np.array(baseline_list), np.array(trial_trace_list), np.array(ptrial_trace_list)
    
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

    def find_responsive_rois(self, save_dir, folder_prefix, dt_min:float=0.2, auc_min:float=5):
        """
        Find responsive neurons using the method from C.G. Sweeney (2025) and T.D. Marks (2021) for the reliability metric.
        
        :param float dt_min: Time duration threshold
        :param float auc_min: Area under the curve threshold.
        
        :return trial_response_bounds (dict): Dictionnary with stimuli index as keys. It contains a list of boundaries defining the interval of the trace considered to compute the AUC.
        :return responsive (dict): Dictionnary with stimuli index as keys. It contains a list of integer (-1, 0, 1) indicating whether or not a ROI is responsive (1 if it is activated/prolonged, 0 if no and -1 if it is supressed).
        :return reliability (dict): Dictionnary with stimuli index as keys. It contains a list of tuple of format (r, p) corresponding to a neuron with r the reliability metrics and p the p-value of the two-tailed one-sample t-test.
        """

        responsive = {}
        trial_response_bounds = {}
        reliability = {}

        nb_frames_min = dt_min * self.ca_img.fs
        positive = None

        for i in self.trial_fluorescence.keys():

            responsive_roi = []
            trial_response_bounds_roi = []
            reliability_roi = []

            # For stimuli of duration under 2.5s, decrease auc threshold with a cross-product
            stim_dt = self.visual_stim.protocol_df['duration'][i] + self.dt_post_stim
            if stim_dt < 2.5 :
                auc_min = auc_min * stim_dt / 2.5

            for roi_idx in range(len(self.ca_img._list_ROIs_idx)):
                roi_trial_average = self.trial_averaged_zscores[i][roi_idx]
                roi_trial = self.trial_zscores[i][roi_idx]
                max_val, min_val = np.max(roi_trial_average), np.min(roi_trial_average)
                
                # Compute ROI reliability
                r, r_dist = self.compute_reliability(roi_trial, n_samples=1000)
                reliability_roi.append(r)

                if max_val > 1 and max_val > np.abs(min_val):
                    positive = True
                    start_idx, end_idx = self.find_bounds(np.argmax(roi_trial_average), roi_trial_average >= 0)
                elif min_val < -1 :
                    #In case the neuron is not activated/prolonged, check if it is supressed.
                    positive = False
                    start_idx, end_idx = self.find_bounds(np.argmin(roi_trial_average), roi_trial_average <= 0)
                else :
                    start_idx, end_idx = 0, 0

                trial_response_bounds_roi.append([start_idx, end_idx])
                
                if end_idx - start_idx + 1 >= nb_frames_min : #time duration constraint
                    time = np.linspace(0, stim_dt, len(roi_trial_average))
                    auc = metrics.auc(time[start_idx:end_idx+1], roi_trial_average[start_idx:end_idx+1])
                    if np.abs(auc) >= auc_min : #AUC constraint
                        r_null_distribution = self.generate_null_distribution(roi_trial)
                        perc_th = np.percentile(r_null_distribution, 99)
                        res = ztest(r_dist, value=perc_th, alternative='larger') #two-tailed one-sample t-test
                        if res[1] <= 0.001 :
                        #if perc_th <= r :
                            if positive :
                                responsive_roi.append((1, res[1]))
                            else :
                                responsive_roi.append((-1, res[1]))
                            self.plot_hist_reliability(r_dist, r, r_null_distribution, perc_th, res[1], 'skyblue', i, roi_idx, save_dir, folder_prefix)
                        else :
                            responsive_roi.append((0, res[1]))
                            self.plot_hist_reliability(r_dist, r, r_null_distribution, perc_th, res[1], 'thistle', i, roi_idx, save_dir, folder_prefix)
                    else :
                        responsive_roi.append((0, None))
                else :
                    responsive_roi.append((0, None))

            responsive.update({i : responsive_roi})
            trial_response_bounds.update({i : trial_response_bounds_roi})
            reliability.update({i : reliability_roi})
        
        self.trial_response_bounds = trial_response_bounds
        self.responsive = responsive
        self.reliability = reliability
        
        return trial_response_bounds, responsive, reliability
    
    def compute_reliability(self, roi_trials_traces, n_samples=1):
        """
        Compute the reliability using the method from T.D. Marks (2021). To compute reliability, the function splits the trials randomly in two halves, trial-averages the two groups and calculates the Pearson's correlation. The process is done n_samples times and averaged.

        :param list roi_trials_traces: List of trials traces of a specific ROI.
        :param int n_samples: Number of samples.

        :return r (float): Reliability R metric corresponding to the mean of the correlations.
        :return corr_list (list): List of correlations.
        """

        corr_list = []
        set_trials = list(range(len(roi_trials_traces)))

        for k in range(n_samples):
            # Divide randomly the trials in 2 groups
            random.shuffle(set_trials)
            group1 = set_trials[:len(set_trials)//2]
            group2 = set_trials[len(set_trials)//2:]

            averaged_group1 = np.mean(np.array(roi_trials_traces)[group1], axis=0)
            averaged_group2 = np.mean(np.array(roi_trials_traces)[group2], axis=0)

            corr = pearsonr(averaged_group1 , averaged_group2)[0]
            corr_list.append(corr)

        r = np.mean(corr_list)

        return r, corr_list
    
    """ def compute_reliability_2(self, roi_trials_traces):

        corr_list = []

        for k in range(len(roi_trials_traces)) :
            for j in range(k+1, len(roi_trials_traces)):
                corr = pearsonr(roi_trials_traces[k] , roi_trials_traces[j])[0]
                corr_list.append(corr)
        
        r = np.mean(corr_list)

        return r """

    def generate_null_distribution(self, roi_trials_traces, n_samples=1000):
        """
        Compute the null distribution of the reliability metric using the method from T.D. Marks (2021). The null distributio is generated by computing reliability on ciruclarly shuffled traces.

        :param list roi_trials_traces: List of trials traces of a specific ROI.
        :param int n_samples: Number of samples.

        :return r_null_distribution (list): List of reliability metrics R of the null distribution.
        """

        r_null_distribution = []

        for i in range(n_samples) :

            # Shuffle circularly
            time_shifts = np.random.choice(np.arange(0, roi_trials_traces.shape[1]), len(roi_trials_traces), replace=True)
            shifted_traces = np.array([np.roll(roi_trials_traces[j], dt) for j, dt in enumerate(time_shifts)])

            # Compute reliability
            r, _ = self.compute_reliability(shifted_traces)
            r_null_distribution.append(r)
        
        return r_null_distribution

    #--------------TOOL FUNCTIONS---------------
    def zscores(self, baselines, traces):
        return np.array([(traces[i, :] - np.mean(baselines, axis=1)[i]) / np.std(baselines, axis=1)[i] for i in range(traces.shape[0])])
    
    def zscores2(self, baselines, traces):
        return np.array([(traces[i, :] - np.mean(baselines)) / np.std(baselines) for i in range(traces.shape[0])])

    def find_bounds(self, index:int, bool_array:list):
        """
        Find the 'True' values interval boundaries surrounding the given index value.
        
        :param int index: Index to be considered.
        :param list bool_array: List of bool.
        
        :return start_idx (int): Starting index of the interval.
        :return end_idx (int): Ending index of the interval.
        """
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

    def get_period_states(self, real_time_states_sorted:list, t0:float, tf:float):
        """
        Get the behavioral states of the stimulus during a period of time (from t0 to tf).
        
        :param list real_time_states_sorted: List of tuples in the format ([t_start_state, t_end_state], state_name) ordered by time.
        :param float t0: Start time of the period to be considered.
        :param float tf: End time of the period to be considered.
        
        :return l (list): List of tuples in the format ([t_start_state, t_end_state], state_name) ordered by time during the period [t0, tf].
        """
        l_el, l_key = [], []

        k = 0
        while k < len(real_time_states_sorted) and \
            not (real_time_states_sorted[k][0][0] <=  t0 <= real_time_states_sorted[k][0][1]) and \
                np.abs(real_time_states_sorted[k][0][0] - t0) > np.abs(real_time_states_sorted[k+1][0][0] - t0) :
            k +=1
        idx_interval_1 = int(k)

        k = 0
        while k < len(real_time_states_sorted) and \
            not (real_time_states_sorted[k][0][0] <=  tf <= real_time_states_sorted[k][0][1]) and \
                np.abs(real_time_states_sorted[k][0][1] - tf) > np.abs(real_time_states_sorted[k+1][0][1] - tf) :
            k +=1
        idx_interval_2 = int(k)

        for i in range(idx_interval_1, idx_interval_2+1):
            l_el.append([np.max([real_time_states_sorted[i][0][0], t0]), 
                         np.min([real_time_states_sorted[i][0][1], tf])])
            l_key.append(real_time_states_sorted[i][1])

        return list(zip(l_el, l_key))

    #--------------PLOTS FUNCTIONS---------------
    def trial_average_rasterplot(self, stimuli_id:int, savepath:str='', sort:bool=True) :
        """
        Plot a rasterplot of all neurons trial-averaged response for a fixed stimulus.
        
        :param int stimuli_id: Index of the stimulus to consider.
        :param str savepath: Saving directory.
        :param bool sort: If True, shows traces sorted by mean fluorescence, if not show traces in ROIs index order.
        """

        stim_dt = self.visual_stim.protocol_df['duration'][stimuli_id]
        stimuli_name = self.visual_stim.protocol_df['name'][stimuli_id]
        stimuli_onset = self.pre_trial_averaged_zscores[stimuli_id].shape[1]
        data = np.concatenate((self.pre_trial_averaged_zscores[stimuli_id], self.trial_averaged_zscores[stimuli_id], self.post_trial_averaged_zscores[stimuli_id]), axis=1)
        if sort :
            idx_sorted = np.argsort(np.mean(self.trial_averaged_zscores[stimuli_id], axis=1))
            data = data[idx_sorted]
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

    def trial_rasterplot(self, trial_zscores, pre_trial_zscores, post_trial_zscores, stimuli_id:int, attr:str='fluorescence', savepath:str='') :
        """
        Plot a rasterplot of all neurons responses (not averaged) for a fixed stimulus.
        
        :param dict trial_zscores: Dictionnary of trial zscores with stimuli ids as keys. It contains every trace for each stimuli, ROIs and trials.
        :param dict pre_trial_zscores: Dictionnary of pre-trial (baselines) zscores with stimuli ids as keys. It contains every trace for each stimuli, ROIs and trials.
        :param dict post_trial_zscores: Dictionnary of post-trial zscores with stimuli ids as keys. It contains every trace for each stimuli, ROIs and trials.
        :param int stimuli_id: Index of the stimulus to consider.
        :param string attr: Attribute of calcium imaging ('raw_F', 'fluorescence' or 'dFoF0').
        :param str savepath: Saving directory.
        """
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

    def plot_stim_response(self, stimuli_id:int, neuron_idx:int, save_dir:str, folder_prefix:str='', show_per_rois:bool=False):
        """
        Plot a neuron trial-averaged response for a fixed stimulus.
        
        :param int stimuli_id: Index of the stimulus to consider.
        :param int neuron_idx: Index of the ROI to consider.
        :param str save_dir: Saving directory.
        :param str folder_prefix: Prefix of the created folder name.
        :param bool show_per_rois: If True, shows traces of trials, if not show only the average trace.
        """

        if self.trial_response_bounds is None :
            self.find_responsive_rois(save_dir, folder_prefix)
        
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
        if np.abs(self.responsive[stimuli_id][neuron_idx][0]) :
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
        
        foldername = "_".join(list(filter(None, [folder_prefix, stimuli_name])))
        save_folder = os.path.join(save_dir, foldername)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        save_path = os.path.join(save_folder, fig_name)
        fig.savefig(save_path)
        plt.close(fig)

    def plot_stim_occurence(self, stimuli_id:int, trial_zscores:dict, pre_trial_zscores:dict, real_time_states_sorted:list,
                            time_onset_aligned_on_ca_img:list, save_dir:str='', folder_prefix:str=''):
        """
        Plot all stimulus (defined by id) trials with the trace of all ROIs and the behavioral states.
        
        :param int stimuli_id: Index of the stimulus to consider.
        :param dict trial_zscores: Dictionnary of trial zscores with stimuli ids as keys. It contains every trace for each stimuli, ROIs and trials.
        :param dict pre_trial_zscores: Dictionnary of pre-trial (baselines) zscores with stimuli ids as keys. It contains every trace for each stimuli, ROIs and trials.
        :param list real_time_states_sorted: List of tuples in the format ([t_start_state, t_end_state], state_name) ordered by time.
        :param list time_onset_aligned_on_ca_img: Times of stimuli onset aligned with calcium imaging time.
        :param str save_dir: Saving directory.
        :param str folder_prefix: Prefix of the folder name.
        """

        stimuli_name = self.visual_stim.protocol_df['name'][stimuli_id]
        stim_dt = self.visual_stim.protocol_df['duration'][stimuli_id]
        colors, positions = ['darkred', 'lightgray'], [0, 1]
        cmap = LinearSegmentedColormap.from_list('my_colormap', list(zip(positions, colors)), N=3)
        color_dict = {'run': cmap(0), 'AS': cmap(1), 'rest': cmap(2)}

        for i in range(trial_zscores[stimuli_id].shape[1]) :
            
            stim_onset_idx = self.visual_stim.stimuli_idx[stimuli_id][i]
            stim_onset_time = np.array(time_onset_aligned_on_ca_img)[stim_onset_idx]
            time = (np.arange(pre_trial_zscores[stimuli_id].shape[2] + trial_zscores[stimuli_id].shape[2]) -  pre_trial_zscores[stimuli_id].shape[2]) / self.ca_img.fs + stim_onset_time
            
            stim_states = self.get_period_states(real_time_states_sorted, time[0], time[-1])

            fig = plt.figure(figsize=(24, 20))
            gs = fig.add_gridspec(2*trial_zscores[stimuli_id].shape[0], 1)

            for roi_id in range(trial_zscores[stimuli_id].shape[0]):
                data = np.concatenate((pre_trial_zscores[stimuli_id][roi_id, i, :], trial_zscores[stimuli_id][roi_id, i, :]))
                ax = fig.add_subplot(gs[2*roi_id:2*(roi_id+1), 0])
                ax.plot(time, data, color='black', linewidth=2)
                ax.axvline(x=stim_onset_time, color='red', linestyle='--', linewidth=2)
                if self.dt_post_stim > 0 :
                    ax.axvline(x=stim_onset_time + stim_dt, color='red', linestyle='--', linewidth=2)
                for k in range(len(stim_states)):
                    if stim_states[k][1] != 'undefined' :
                        ax.axvspan(stim_states[k][0][0], stim_states[k][0][1], color=color_dict[stim_states[k][1]], alpha=0.4)
                ax.set_xticks([])
                ax.margins(x=0)
                ax.set_ylabel(f'ROI {roi_id}')
            
            run_legend = Line2D([0], [0], color=color_dict['run'], linewidth=5)
            as_legend = Line2D([0], [0], color=color_dict['AS'], linewidth=5)
            rest_legend = Line2D([0], [0], color=color_dict['rest'], linewidth=5)
            ax.legend([run_legend, as_legend, rest_legend], ['run', 'AS', 'rest'], 
                      loc='lower left', bbox_to_anchor=(1.0, 0.0), prop={'size': 9})
            ax.set_xlabel("Time (s)")
            ax.xaxis.set_major_locator(AutoLocator())
            fig.suptitle(stimuli_name + '\n' + f'Occurence {i} of stimulus')

            fig_name = stimuli_name + "_occ_" + str(i)
            foldername = "_".join(list(filter(None, [folder_prefix, stimuli_name])))
            save_folder = os.path.join(save_dir, foldername, 'stimuli_occurence')
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
            save_path = os.path.join(save_folder, fig_name)
            fig.savefig(save_path)
            plt.close(fig)

    def plot_hist_reliability(self, r_dist, r, r_null, perc, p_value, color:str, stimuli_id:int, neuron_idx:int, save_dir:str, folder_prefix:str=''):
        stimuli_name = self.visual_stim.protocol_df['name'][stimuli_id]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(r_null, bins=20, alpha=0.4, edgecolor='white', color='grey', label='null distribution')
        ax.axvline(np.mean(r_null), color='grey', linestyle='--', label='mean of null dist')
        ax.hist(r_dist, bins=20, alpha=0.4, edgecolor='white', color=color, label='R distribution')
        ax.axvline(r, color=color, linestyle='--', label='r')
        ax.axvline(perc, color='black', linestyle='--', label='99th perc of null dist')
        ax.annotate(f'p-value = {p_value:.3f}', xy=(0.85, 0.98), xycoords='axes fraction', fontsize=9, va='top', ha='left')
        ax.set_xlabel('Reliability R')
        ax.set_ylabel('Count')
        ax.set_title(f'Neuron {neuron_idx} ({stimuli_name})')
        ax.legend(loc='upper left', prop={'size': 9})

        fig_name = "r_null_" + stimuli_name + "_neuron_" + str(neuron_idx)
        foldername = "_".join(list(filter(None, [folder_prefix, stimuli_name])))
        save_folder = os.path.join(save_dir, foldername)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        save_path = os.path.join(save_folder, fig_name)
        fig.savefig(save_path)
        plt.close(fig)

    #-------------SAVE FUNCTIONS---------------
    def save_protocol_validity(self, save_dir, filename):
        protocol_validity = []
        for id in self.responsive.keys():
            d = {self.visual_stim.protocol_names[id] : self.responsive[id]}
            protocol_validity.append(d)
        np.savez(os.path.join(save_dir, filename + ".npz" ), **{key: value for d in protocol_validity for key, value in d.items()})