import numpy as np
import os
import json
import h5py
import sys
sys.path.append("./src")
import visualpipe.analysis.photodiode as Photodiode
import visualpipe.utils.general_functions as General_functions

def process_session(session_path, analysis_id, pre_time=2.0, post_time=7.0):
    """
    Process a session to extract speed traces around all stimuli.

    Parameters
    ----------
    base_path : str
        Path to the session folder containing visual-stim.npy, protocol.json, and HDF5 files.
    pre_time : float
        Seconds before stimulus onset to include.
    post_time : float
        Seconds after stimulus onset to include.

    Returns
    -------
    pname_contrasts : list[str]
        Names for each protocol-contrast combination.
    protocol_traces : dict
        Dictionary mapping protocol-contrast name -> list of dicts with keys:
            'time' : relative time to stimulus onset
            'speed' : speed trace around stimulus
    """

    # ---------------------------
    # Load visual stim
    # ---------------------------
    visual_stim_path = os.path.join(session_path, "visual-stim.npy")
    visual_stim = np.load(visual_stim_path, allow_pickle=True).item()
    order = np.array(visual_stim['protocol_id'])
    onsets = np.array(visual_stim['time_start'])
    contrast_ids = np.array(visual_stim['contrast'])

    # ---------------------------
    # Load and resample photodiode
    # ---------------------------
    NIdaq, acq_freq = Photodiode.load_and_data_extraction(session_path)
    Psignal_time, Psignal = General_functions.resample_signal(
        NIdaq['analog'][0],
        original_freq=acq_freq,
        new_freq=1000
    )

    def realign_from_photodiode(onsets, Psignal_time, Psignal):
        acq_freq_photodiode = 1. / (Psignal_time[1] - Psignal_time[0])
        H, bins = np.histogram(Psignal, bins=100)
        baseline = bins[np.argmax(H) + 1]
        threshold = (np.max(Psignal) - baseline) / 3.
        cond_thresh = (Psignal[1:] >= (baseline + threshold)) & (Psignal[:-1] < (baseline + threshold))
        peak_time_stamps = np.where(cond_thresh)[0] / acq_freq_photodiode

        stim_Time_start_realigned = []
        for value in onsets:
            index = np.argmin(np.abs(peak_time_stamps - value))
            stim_Time_start_realigned.append(peak_time_stamps[index])

        return np.array(stim_Time_start_realigned)

    # Realign onsets
    onsets = realign_from_photodiode(onsets, Psignal_time, Psignal)

    # ---------------------------
    # Load protocol metadata
    # ---------------------------
    protocol_path = os.path.join(session_path, "protocol.json")
    with open(protocol_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    protocol_ids = []
    protocol_names = []
    if data["Presentation"] == "multiprotocol":
        for key, value in data.items():
            if key.startswith("Protocol-"):
                protocol_number = int(key.split("-")[1]) - 1
                if protocol_number not in protocol_ids:
                    protocol_ids.append(protocol_number)
                    protocol_name = value.split("/")[-1].replace(".json", "")
                    protocol_names.append(protocol_name)

    contrasts = np.unique(visual_stim['contrast'])

    # ---------------------------
    # Load speed data from HDF5
    # ---------------------------
    h5_path = os.path.join(session_path, analysis_id)
    h5_files = [f for f in os.listdir(h5_path) if f.endswith(".h5")]
    if len(h5_files) == 0:
        raise FileNotFoundError("No HDF5 file found in session folder")
    h5_path = os.path.join(h5_path, h5_files[0])
    with h5py.File(h5_path, 'r') as f:
        speed = f['Behavioral/Speed'][:]

    def extract_speed_traces(speed, onsets, order, contrast_ids, protocol_ids, protocol_names, contrasts,
                         pre_time, post_time):

        time = speed[0, :]
        speed_values = speed[1, :]

        pname_contrasts = []
        protocol_traces = {}

        order = np.array(order)
        contrast_ids = np.array(contrast_ids)
        onsets = np.array(onsets)

        for contrast in contrasts:
            contrast_rounded = str(round(contrast * 10) / 10)
            for pid, pname in zip(protocol_ids, protocol_names):
                name = f"{pname}-{contrast_rounded}"
                pname_contrasts.append(name)

                # Find which onsets match both protocol and contrast
                mask = (order == pid) & (contrast_ids == contrast)
                stim_onsets = onsets[mask]

                # Extract speed traces for each matching stimulus
                traces = []
                for onset in stim_onsets:
                    t_start = onset - pre_time
                    t_end = onset + post_time

                    idx = np.where((time >= t_start) & (time <= t_end))[0]
                    window_time = time[idx] - onset      # relative to stimulus
                    window_speed = speed_values[idx]

                    traces.append({
                        'time': window_time,
                        'speed': window_speed
                    })

                protocol_traces[name] = traces

        return pname_contrasts, protocol_traces

    # ---------------------------
    # Extract speed traces
    # ---------------------------
    pname_contrasts, protocol_traces = extract_speed_traces(
        speed=speed,
        onsets=onsets,
        order=order,
        contrast_ids=contrast_ids,
        protocol_ids=protocol_ids,
        protocol_names=protocol_names,
        contrasts=contrasts,
        pre_time=pre_time,
        post_time=post_time
    )

    return pname_contrasts, protocol_traces
