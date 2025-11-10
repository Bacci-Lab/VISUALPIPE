import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

import visualpipe.utils.figures as figures

class VisualStim(object):
    
    def __init__(self, base_path):
        visual_stim = self.load_visual_stim(base_path)
        self.order: list = list(visual_stim['protocol_id'])
        self.duration = visual_stim['time_duration']
        self.interstim = visual_stim['interstim']
        self.time_onset = visual_stim['time_start']

        self.protocol_ids: list = []
        self.protocol_names: list = []
        self.protocol_df: pd.DataFrame = None
        self.real_time_onset: list = None
        self.dt_shift: list = None
        self.stimuli_idx: dict = {}
        self.stim_cat: list = []
        self.analyze_pupil: list = []

        self.divided_by_contrast = ['looming-stim-log', 'dimming-circle-log', 'black-sweeping-log', 'white-sweeping-log', 'looming-stim-lin', 'dimming-circle-lin', 'black-sweeping-lin', 'white-sweeping-lin', 'drifting-grating']
        self.divided_by_radius_and_contrast = ['center', 'size-tuning-contrast-log', 'size-tuning-contrast-lin']
        self.center_surround = ["center-surround", "center-surround_high_contrast", "center-surround_low_contrast"]

        self.get_protocol(base_path)
        self.build_df()
        self.add_subtype_stimuli(visual_stim)
        self.set_stimulus_order_idxes()
        self.set_stim_categories()

    def load_visual_stim(self, base_path):
        visual_stim_path = os.path.join(base_path, "visual-stim.npy")
        if os.path.exists(visual_stim_path):
            visual_stim = np.load(visual_stim_path, allow_pickle=True).item()
        else:
            raise Exception("No visual-stim.npy file exists in this directory")
        return visual_stim

    def get_protocol(self, base_path):
        # Load the JSON data
        protocol_path = os.path.join(base_path, "protocol.json")
        try :
            with open(protocol_path, "r", encoding="utf-8") as file:
                data = json.load(file)
        except :
            raise Exception("No protocol.json file exists in this directory")
        
        if data["Presentation"] == "multiprotocol" :
            for key, value in data.items():
                if key.startswith("Protocol-"):
                    # Extract the protocol number
                    protocol_number = int(key.split("-")[1])-1
                    if protocol_number not in self.protocol_ids :
                        self.protocol_ids.append(protocol_number)
                        protocol_name = value.split("/")[-1].replace(".json", "")
                        self.protocol_names.append(protocol_name)

        elif data["Presentation"] == "Stimuli-Sequence" :
            self.protocol_ids.append(0)
            self.protocol_names.append(data["Stimulus"])
        else :
            print(f"Protocol is neither multiprotocol or stimuli sequence: {data['Presentation']}")

    def build_df(self) :
        protocol_df = pd.DataFrame({"id" : self.protocol_ids, "name" : self.protocol_names})
        protocol_duration = pd.DataFrame({ "id" : self.order, "duration" : self.duration}).groupby(by="id").mean()
        self.protocol_df = protocol_df.join(protocol_duration, on="id", how="inner").set_index("id")

    def add_subtype_stimuli(self, visual_stim):

        contrast = False
        radius = False
        position = False
        angle = False
        angle_surround = False
        contrast_surround = False

        qsm_loc, qsm_names, id_qsm = self.get_qsm_loc(visual_stim)
        surround_type, surround_names, id_s = self.get_surround_mod_type(visual_stim)
        sp_angle, sp_names, id_sp = self.get_stat_patch_angle(visual_stim)
        cg_contrast_angle, cg_names, id_cg = self.get_cg_contrast_angle(visual_stim)
        
        for stim in self.divided_by_radius_and_contrast :
            stim_size_contrast, stim_names, id_stim = self.get_stim_by_radius_and_contrast(visual_stim, stim)
            
            if stim_size_contrast is not None :
                self.separate_subtype_stimuli(stim_size_contrast, stim_names, id_stim)
                radius = True
                contrast = True
                
        for stim in self.divided_by_contrast :
            stim_contrast, stim_names, id_stim = self.get_stim_by_contrast(visual_stim, stim)
            
            if stim_contrast is not None :
                self.separate_subtype_stimuli(stim_contrast, stim_names, id_stim)
                contrast = True

        for stim in self.center_surround :
            cs_type, cs_names, id_cs = self.get_cs_mod_type(visual_stim, stim)
            
            if cs_type is not None :
                self.separate_subtype_stimuli(cs_type, cs_names, id_cs)
                angle = True
                angle_surround = True
                contrast = True
                contrast_surround = True

        if qsm_loc is not None :
            self.separate_subtype_stimuli(qsm_loc, qsm_names, id_qsm)
            position = True
        
        if surround_type is not None :
            self.separate_subtype_stimuli(surround_type, surround_names, id_s)
            angle_surround = True
            contrast_surround = True
            radius = True
        
        if sp_angle is not None :
            self.separate_subtype_stimuli(sp_angle, sp_names, id_sp)
            angle = True
        
        if cg_contrast_angle is not None :
            self.separate_subtype_stimuli(cg_contrast_angle, cg_names, id_cg)
            contrast = True
            angle = True
        
        self.build_df()

        if position :
            self.add_column_df('x-center', visual_stim['x-center'])
            self.add_column_df('y-center', visual_stim['y-center'])

        if angle :
            self.add_column_df('angle', visual_stim['angle'])
        
        if angle_surround :
            self.add_column_df('angle-surround', visual_stim['angle-surround'])

        if contrast :
            self.add_column_df('contrast', visual_stim['contrast'])

        if contrast_surround:
            self.add_column_df('contrast-surround', visual_stim['contrast-surround'])

        if radius :
            self.add_column_df('radius', visual_stim['radius'])
        
    def get_qsm_loc(self, visual_stim) :
        if 'grating' in self.protocol_names or 'quick-spatial-mapping' in self.protocol_names:
            id_qsm = self.protocol_df.loc[(self.protocol_df['name'] == 'grating') | (self.protocol_df['name'] == 'quick-spatial-mapping')].index[0]
            qsm_names = ['center', 'right', 'left', 'up', 'down', 'up-right', 'up-left', 'down-right', 'down-left']
            map = {'0': 'center', 
                   '36' : 'right', 
                   '-36': 'left', 
                   '23': 'up', 
                   '-23': 'down', 
                   '59':'up-right', 
                   '-13': 'up-left', 
                   '13':'down-right', 
                   '-59':'down-left'}
            qsm_loc = [map[str(int(value))] if id == id_qsm else None for id, value in zip(visual_stim['protocol_id'], np.array(visual_stim['x-center'], dtype=float) + np.array(visual_stim['y-center'], dtype=float))]
        else :
            qsm_loc = None
            qsm_names = None
            id_qsm = None
        
        return qsm_loc, qsm_names, id_qsm

    def get_cs_mod_type(self, visual_stim, stimuli):
        if stimuli in self.protocol_names:
            cs_names = ["iso", "cross"]
            id_cs = self.protocol_df[self.protocol_df['name'] == stimuli].index[0]
            cs_type = ['iso' if angle_s == angle and id == id_cs 
                        else 'cross' if id == id_cs 
                        else None 
                        for id, angle, angle_s in zip(visual_stim['protocol_id'], visual_stim['angle'], visual_stim['angle-surround'])]
        else :
            cs_type = None
            cs_names = None
            id_cs = None

        return cs_type, cs_names, id_cs
    
    def get_surround_mod_type(self, visual_stim):
        if 'surround' in self.protocol_names :
            id_s = self.protocol_df[self.protocol_df['name'] == 'surround'].index[0]
            occ_stim_idx = np.where(np.array(self.order) == id_s )[0]

            if len(np.unique(visual_stim['contrast-surround'][occ_stim_idx])) > 1\
                and len(np.unique(visual_stim['radius'][occ_stim_idx])) > 1 :
                surround_type = [f'iso_ctrl-{round(radius)}-{round(cs*100)/100}' if angle_s == 0 and id == id_s 
                                 else f'cross_ctrl-{round(radius)}-{round(cs*100)/100}' if id == id_s 
                                 else None 
                                 for id, angle_s, cs, radius in zip(visual_stim['protocol_id'], visual_stim['angle-surround'], visual_stim['contrast-surround'], visual_stim['radius'])]
                s_names = list(set(surround_type))
                if None in s_names:
                    s_names.remove(None)

            elif len(np.unique(visual_stim['contrast-surround'][occ_stim_idx])) > 1 :
                surround_type = [f'iso_ctrl-{round(cs*100)/100}' if angle_s == 0 and id == id_s 
                                 else f'cross_ctrl-{round(cs*100)/100}' if id == id_s 
                                 else None 
                                 for id, angle_s, cs in zip(visual_stim['protocol_id'], visual_stim['angle-surround'], visual_stim['contrast-surround'])]
                s_names = list(set(surround_type))
                if None in s_names:
                    s_names.remove(None)
            
            else :
                s_names = ["iso_ctrl", "cross_ctrl"]
                surround_type = ['iso_ctrl' if angle_s == 0 and id == id_s 
                                 else 'cross_ctrl' if id == id_s 
                                 else None 
                                 for id, angle_s in zip(visual_stim['protocol_id'], visual_stim['angle-surround'])]
        else :
            surround_type = None
            s_names = None
            id_s = None

        return surround_type, s_names, id_s
    
    def get_stim_by_contrast(self, visual_stim, stimuli):
        if stimuli in self.protocol_names :
            id_stim = self.protocol_df[self.protocol_df['name'] == stimuli].index[0]
            stim_contrast = [str(round(contrast*10)/10) if id_stim == id else None for id, contrast in zip(visual_stim['protocol_id'], visual_stim['contrast'])]
            stim_names = list(set(stim_contrast))
            if None in stim_names:
                stim_names.remove(None)

            if len(stim_names) < 2 :
                stim_contrast = None
        else :
            stim_contrast = None
            stim_names = None
            id_stim = None

        return stim_contrast, stim_names, id_stim
    
    def get_stim_by_radius_and_contrast(self, visual_stim, stimuli):
        if stimuli in self.protocol_names :
            id_stim = self.protocol_df[self.protocol_df['name'] == stimuli].index[0]
            stim_size_contrast = [f"{round(radius)}-{round(contrast*100)/100}" if id == id_stim else None for id, radius, contrast in zip(visual_stim['protocol_id'], visual_stim['radius'], visual_stim['contrast'])]
            stim_names = list(set(stim_size_contrast))
            if None in stim_names:
                stim_names.remove(None)

            if len(stim_names) < 4 :
                stim_size_contrast = None
                self.divided_by_contrast.append(stimuli)
        else :
            stim_size_contrast = None
            stim_names = None
            id_stim = None

        return stim_size_contrast, stim_names, id_stim

    def get_stat_patch_angle(self, visual_stim):
        if 'static-patch' in self.protocol_names :
            id_sp = self.protocol_df[self.protocol_df['name'] == 'static-patch'].index[0]
            sp_angle = [str(int(angle)) if id == id_sp else None for id, angle in zip(visual_stim['protocol_id'], visual_stim['angle'])]
            sp_names = list(set(sp_angle))
            if None in sp_names:
                sp_names.remove(None)

            if len(sp_names) < 2 :
                sp_angle = None
        else :
            sp_angle = None
            sp_names = None
            id_sp = None

        return sp_angle, sp_names, id_sp

    def get_cg_contrast_angle(self, visual_stim):
        if 'center-grating' in self.protocol_names :
            id_cg = self.protocol_df[self.protocol_df['name'] == 'center-grating'].index[0]
            cg_contrast_angle = [f"{round(contrast*100)/100}-{angle}" if id == id_cg else None for id, contrast, angle in zip(visual_stim['protocol_id'], visual_stim['contrast'], visual_stim['angle'])]
            cg_names = list(set(cg_contrast_angle))
            if None in cg_names:
                cg_names.remove(None)

            if len(cg_names) < 2 :
                cg_contrast_angle = None
        else :
            cg_contrast_angle = None
            cg_names = None
            id_cg = None

        return cg_contrast_angle, cg_names, id_cg

    def separate_subtype_stimuli(self, subtype_stim_order, subtype_stim_names, id_stim) :
        protocol_name = self.protocol_names[id_stim]
        self.protocol_names[id_stim] = protocol_name + '-' + subtype_stim_names[0]
        for name in subtype_stim_names[1:] :
            i = len(self.protocol_ids)
            self.protocol_ids.append(i)
            self.protocol_names.append(protocol_name + '-' + name)
        
        for name in subtype_stim_names :
            indices = [i for i, val in enumerate(subtype_stim_order) if val == name]
            id_protocol = np.argwhere(np.array(self.protocol_names) == protocol_name + '-' + name).reshape(-1)[0]
            order_temp = np.array(self.order)
            order_temp[indices] = id_protocol
            self.order = order_temp.tolist()

    def set_stimulus_order_idxes(self):
        for protocol_id in self.protocol_ids :
            indices = [i for i, val in enumerate(self.order) if val == protocol_id]
            self.stimuli_idx.update({protocol_id : indices})

    def realign_from_photodiode(self, Psignal_time, Psignal, plot=False):
        """
        Adapted from yzerlaut : https://github.com/yzerlaut/physion/blob/main/src/physion/assembling/realign_from_photodiode.py
        Calculate the real time stamps of each stimuli onset from the photodiode signal.
        """
        acq_freq_photodiode = 1. / (Psignal_time[1] - Psignal_time[0])

        # compute signal boundaries to evaluate threshold crossing of photodiode signal
        H, bins = np.histogram(Psignal, bins=100)
        baseline = bins[np.argmax(H) + 1]
        threshold = (np.max(Psignal) - baseline) / 3.  # reaching 1/3 of peak level

        # extract time stamps where photodiode signal cross threshold ("peaks" time stamps)
        cond_thresh = (Psignal[1:] >= (baseline + threshold)) & (Psignal[:-1] < (baseline + threshold))
        peak_time_stamps = np.where(cond_thresh)[0] / acq_freq_photodiode

        # from the peaks, select only those at the beginning of each stimuli (time-delay is around 0.3s)
        stim_Time_start_realigned = []
        dt_shift = []
        for value in self.time_onset:
            index = np.argmin(np.abs(peak_time_stamps - value))
            stim_Time_start_realigned.append(peak_time_stamps[index])
            dt_shift.append(np.abs(value - peak_time_stamps[index]))
        stim_Time_start_realigned = [float(val) for val in stim_Time_start_realigned]
        
        self.real_time_onset = stim_Time_start_realigned
        self.dt_shift = dt_shift
        #print(np.mean(dt_shift), np.max(dt_shift), np.min(dt_shift))

        if plot :
            figures.Visualize_baseline(Psignal, baseline)

            plt.plot(Psignal_time, Psignal, label='photodiode signal')
            plt.scatter(stim_Time_start_realigned, np.ones(len(stim_Time_start_realigned))*(threshold+baseline), color='orange', marker='x', label='start of stimuli')
            plt.scatter(peak_time_stamps, np.ones(len(peak_time_stamps))*(threshold+baseline), color='green', marker='.', label='peak detection')
            plt.legend(loc="upper right")
            plt.show()

    def get_protocol_onset_index(self, chosen_protocols: list, F_stim_init_indexes, freq, tseries=None):
        """
        Get the start and end index of chosen protocol in a list.
        """
        protocol_nb_frames = []
        stimuli_ids_order = []

        idx_lim_protocol = []
        var_protocol = []

        for i in range(len(self.order)):
            
            if self.order[i] in chosen_protocols:
                stimuli_ids_order.append(self.order[i])
                start_spon_index = int(F_stim_init_indexes[i])
                protocol_nb_frames = int(self.protocol_df.loc[self.order[i]]["duration"] * freq)
                idx_lim_protocol.append([start_spon_index, start_spon_index + protocol_nb_frames])
                
                if tseries is not None :
                    temp = np.ones((len(tseries), protocol_nb_frames))
                    for neuron in range(len(tseries)):
                        F_spontaneous_i = tseries[neuron, start_spon_index: start_spon_index + protocol_nb_frames]
                        temp[neuron] = F_spontaneous_i
                    var_protocol.append(temp)
        
        if tseries is not None :
            return idx_lim_protocol, stimuli_ids_order, var_protocol
        else : 
            return idx_lim_protocol, stimuli_ids_order
        
    def set_stim_categories(self) :
        for stimuli_name in self.protocol_names :
            if 'grey' in stimuli_name or 'black' in stimuli_name and 'sweeping' not in stimuli_name:
                self.stim_cat.append(0)
                if 'black' in stimuli_name :
                    self.analyze_pupil.append(0)
                else :
                    self.analyze_pupil.append(1)
            else :
                self.stim_cat.append(1)
                self.analyze_pupil.append(1)
        self.add_column_df('visual_stim', self.stim_cat)
        self.add_column_df('analyze_pupil', self.analyze_pupil)

    def add_column_df(self, column_name, data:list):
        if len(data) == self.protocol_df.shape[0]:
            self.protocol_df[column_name] = data
        elif len(data) == len(self.order) :
            self.protocol_df.reset_index(inplace=True)
            new_df = pd.DataFrame({ "id" : self.order, column_name : data}).groupby(by="id").mean()
            self.protocol_df = self.protocol_df.join(new_df, on="id", how="inner").set_index("id")
        else : 
            raise Exception("Can't add new column. Size of data list incorrect.")

    def export_df_to_excel(self, save_folder, filename):
        save_path = os.path.join(save_folder, filename)
        self.protocol_df.to_excel(save_path)

if __name__ == "__main__":
    import visualpipe.analysis.photodiode as Photodiode
    import visualpipe.utils.general_functions as General_functions
    import visualpipe.analysis.ca_imaging as Ca_imaging

    base_path = "Y:/raw-imaging/TESTS/Mai-An/visual_test/16-00-59"

    visual_stim = VisualStim(base_path)
    print(visual_stim.protocol_df)
    NIdaq, acq_freq = Photodiode.load_and_data_extraction(base_path)
    Psignal_time, Psignal = General_functions.resample_signal(NIdaq['analog'][0],
                                                              original_freq=acq_freq,
                                                              new_freq=1000)
    visual_stim.realign_from_photodiode(Psignal_time, Psignal)

    ca_img_dm = Ca_imaging.CaImagingDataManager(base_path)
    F_Time_start_realigned, F_stim_init_indexes  = Photodiode.Find_F_stim_index(visual_stim.real_time_onset, ca_img_dm.time_stamps)
    idx_lim_protocol, stimuli_ids, F_spontaneous = visual_stim.get_protocol_onset_index([5], F_stim_init_indexes, ca_img_dm.fs, tseries=ca_img_dm.raw_F)

    print(len(idx_lim_protocol))
    print(len(F_spontaneous))
    print(stimuli_ids)