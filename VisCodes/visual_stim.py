import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import figures

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
        qsm_loc, qsm_names, id_qsm = self.get_qsm_loc(visual_stim)
        cs_type, surround_type, cs_names, surround_names, id_cs, id_s = self.get_surround_mod_type(visual_stim)

        if qsm_loc is not None :
            self.separate_subtype_stimuli(qsm_loc, qsm_names, id_qsm)

        if cs_type is not None :
            self.separate_subtype_stimuli(cs_type, cs_names, id_cs)
        
        if surround_type is not None :
            self.separate_subtype_stimuli(surround_type, surround_names, id_s)
        
        self.build_df()

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
            qsm_loc = [map[str(int(value))] if id == id_qsm else None for id, value in zip(visual_stim['protocol_id'], np.array(visual_stim['x-center']) + np.array(visual_stim['y-center']))]
        else :
            qsm_loc = None
            qsm_names = None
            id_qsm = None
        
        return qsm_loc, qsm_names, id_qsm

    def get_surround_mod_type(self, visual_stim):
        if 'center-surround' in self.protocol_names :
            cs_names = ["iso", "cross"]
            id_cs = self.protocol_df[self.protocol_df['name'] == 'center-surround'].index[0]
            cs_type = ['iso' if angle_s == angle and id == id_cs else 'cross' if id == id_cs else None for id, angle, angle_s in zip(visual_stim['protocol_id'], visual_stim['angle'], visual_stim['angle-surround'])]
        else :
            cs_type = None
            cs_names = None
            id_cs = None

        if 'surround' in self.protocol_names :
            s_names = ["iso_ctrl", "cross_ctrl"]
            id_s = self.protocol_df[self.protocol_df['name'] == 'surround'].index[0]
            surround_type = ['iso_ctrl' if angle_s == angle and id == id_s else 'cross_ctrl' if id == id_s else None for id, angle, angle_s in zip(visual_stim['protocol_id'], visual_stim['angle'], visual_stim['angle-surround'])]
        else :
            surround_type = None
            s_names = None
            id_s = None

        return cs_type, surround_type, cs_names, s_names, id_cs, id_s

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
        nb_stimuli = 0

        for id in chosen_protocols :
            protocol_duration = self.protocol_df['duration'][id]
            protocol_nb_frames.append(int(protocol_duration * freq))
            nb_stimuli += self.order.count(id)

        if nb_stimuli > 0 :
            idx_lim_protocol = []
            var_protocol = []
            stim = 0

            for i in range(len(self.order)):
                
                if self.order[i] in chosen_protocols:
                    stimuli_ids_order.append(self.order[i])
                    start_spon_index = int(F_stim_init_indexes[i])
                    idx_lim_protocol.append([start_spon_index, start_spon_index + protocol_nb_frames[stim]])
                    
                    if tseries is not None :
                        temp = np.ones((len(tseries), protocol_nb_frames[stim]))
                        for neuron in range(len(tseries)):
                            F_spontaneous_i = tseries[neuron, start_spon_index: start_spon_index + protocol_nb_frames[stim]]
                            temp[neuron] = F_spontaneous_i
                        var_protocol.append(temp)
                    stim += 1
        
        else :
            raise Exception("0 stimuli of the chosen protocol has been found")
        
        if tseries is not None :
            return idx_lim_protocol, stimuli_ids_order, var_protocol
        else : 
            return idx_lim_protocol, stimuli_ids_order
        
    def set_stim_categories(self) :
        for stimuli_name in self.protocol_names :
            if 'grey' in stimuli_name or 'black' in stimuli_name:
                self.stim_cat.append(0)
            else :
                self.stim_cat.append(1)

if __name__ == "__main__":
    import Photodiode
    import General_functions
    import Ca_imaging

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