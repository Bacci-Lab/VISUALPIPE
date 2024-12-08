import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import interp1d
import os.path
import json
import matplotlib.pyplot as plt

# TODO  This section of code is adapted from the physion Pipeline.
# TODO  The purpose of this code is to process binary signals from a rotary encoder to compute the rotational position and speed over time
# TODO URL/Reference: https://github.com/yzerlaut/physion/blob/main/src/physion/behavior/locomotion.py

def load_and_data_extraction(base_path):
    # Load metadata file
    metadata_path = os.path.join(base_path, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as file:
            data = json.load(file)

        # Extract relevant data from metadata
        radius = data['rotating-disk']['radius-position-on-disk-cm']
        value_per_rotation = data['rotating-disk']['roto-encoder-value-per-rotation']
        Frequency = data['NIdaq-acquisition-frequency']
    else:
        raise Exception("No JSON metadata file exists in this directory")

    # Load NIdaq file
    NIdaq_path = os.path.join(base_path, "NIdaq.npy")
    if os.path.exists(NIdaq_path):
        NIdaq = np.load(NIdaq_path, allow_pickle=True).item()
        digital = NIdaq['digital']
        binary_signal = digital[0]
    else:
        raise Exception("No NIdaq.npy file exists in this directory")

    return binary_signal, radius, value_per_rotation, Frequency


def process_binary_signal(binary_signal):
    A = binary_signal % 2
    B = np.concatenate([A[1:], [0]])
    return A, B


def compute_position_from_binary_signals(A, B):
    '''
    Takes traces A and B and converts it to a trace that has the same number of
    points but with positions points.
    Algorithm based on the schematic of cases shown in the doc
    ---------------
    Input:
        A, B - traces to convert

    Output:
        Positions through time

    '''

    Delta_position = np.zeros(len(A) - 1, dtype=float)  # N-1 elements
    ################################
    ## positive_increment_cond #####
    ################################
    PIC = ((A[:-1] == 1) & (B[:-1] == 1) & (A[1:] == 0) & (B[1:] == 1)) | \
          ((A[:-1] == 0) & (B[:-1] == 1) & (A[1:] == 0) & (B[1:] == 0)) | \
          ((A[:-1] == 0) & (B[:-1] == 0) & (A[1:] == 1) & (B[1:] == 0)) | \
          ((A[:-1] == 1) & (B[:-1] == 0) & (A[1:] == 1) & (B[1:] == 1))
    Delta_position[PIC] = 1
    ################################
    ## negative_increment_cond #####
    ################################
    NIC = ((A[:-1] == 1) & (B[:-1] == 1) & (A[1:] == 1) & (B[1:] == 0)) | \
          ((A[:-1] == 1) & (B[:-1] == 0) & (A[1:] == 0) & (B[1:] == 0)) | \
          ((A[:-1] == 0) & (B[:-1] == 0) & (A[1:] == 0) & (B[1:] == 1)) | \
          ((A[:-1] == 0) & (B[:-1] == 1) & (A[1:] == 1) & (B[1:] == 1))
    Delta_position[NIC] = -1

    return np.cumsum(np.concatenate([[0], Delta_position]))

def get_alignment_index(Flourscnce_time,original_signal, original_freq):
    tlim_photodiode = np.linspace(1000, len(original_signal),len(original_signal) - 1000 )
    original_signal = original_signal[1000:]
    last_photodiode_stamp = tlim_photodiode[-1]
    Flourscnce_time2 = Flourscnce_time * original_freq
    index_flour = np.where(Flourscnce_time2 >= last_photodiode_stamp)[0][0]
    index_photodiod = np.where(tlim_photodiode >= Flourscnce_time[index_flour - 1] * original_freq)[0][0]
    original_signal = original_signal[:index_photodiod - 1]
    Interpolate_time = Flourscnce_time[:index_flour - 1]
    last_Flourscnce_index = index_flour - 1
    return Interpolate_time, last_Flourscnce_index, original_signal


def resample_running_signal(original_signal,
                    Interpolate_time,
                    original_freq=1e4,
                    new_freq=1e3,
                    pre_smoothing=0,
                    post_smoothing=0,
                    verbose=False):


    if verbose:
        print('resampling signal [...]')

    if (pre_smoothing * original_freq) > 1:
        if verbose:
            print(' - gaussian smoothing - pre')
        signal = gaussian_filter1d(original_signal, int(pre_smoothing * original_freq), mode='nearest')
    else:
        signal = original_signal

    t_sample = np.linspace(1000, len(original_signal)+1000, len(original_signal))/ original_freq

    print("this is speed time ", t_sample)

    if verbose:
        print(' - signal interpolation')
    func = interp1d(t_sample[np.isfinite(signal)], signal[np.isfinite(signal)],
                    fill_value='extrapolate')

    new_signal = func(Interpolate_time)

    if (post_smoothing * new_freq) > 1:
        if verbose:
            print(' - gaussian smoothing - post')
        new_signal = gaussian_filter1d(new_signal, int(post_smoothing * new_freq), mode='nearest')

    return new_signal

def compute_speed(base_path,new_freq,Flourscnce_time = None, position_smoothing = 10e-3, #s
                   with_raw_position=False):
    binary_signal, radius_position_on_disk, rotoencoder_value_per_rotation, acq_freq = load_and_data_extraction(base_path)
    A, B = process_binary_signal(binary_signal)

    position = compute_position_from_binary_signals(A,B) * 2. * np.pi * radius_position_on_disk / rotoencoder_value_per_rotation

    if position_smoothing > 0:
        speed = np.diff(gaussian_filter1d(position, int(position_smoothing * acq_freq), mode='nearest'))
        speed[:int(2 * position_smoothing * acq_freq)] = speed[int(2 * position_smoothing * acq_freq)]
        speed[-int(2 * position_smoothing * acq_freq):] = speed[-int(2 * position_smoothing * acq_freq)]
    else:
        speed = np.diff(position)

    speed *= acq_freq

    Interpolate_time, last_Flourscnce_index, speed = get_alignment_index(Flourscnce_time,speed, acq_freq)

    #sub sampling and filtering speed
    speed = resample_running_signal(speed, Interpolate_time,
                                    original_freq=acq_freq,
                                    new_freq = new_freq,
                                    pre_smoothing=0,
                                    post_smoothing=2. / 50.,
                                    verbose=True)

    if with_raw_position:
        return Interpolate_time, speed, position, last_Flourscnce_index
    else:
        return Interpolate_time, speed, last_Flourscnce_index

def resample_signal(original_signal,
                    original_freq=1e4,
                    t_sample=None,
                    new_freq=1e3,
                    pre_smoothing=0,
                    post_smoothing=0,
                    tlim=None,
                    verbose=False):
    if verbose:
        print('resampling signal [...]')

    if (pre_smoothing * original_freq) > 1:
        if verbose:
            print(' - gaussian smoothing - pre')
        signal = gaussian_filter1d(original_signal, int(pre_smoothing * original_freq), mode='nearest')
    else:
        signal = original_signal

    if t_sample is None:
        t_sample = np.arange(len(signal)) / original_freq

    if verbose:
        print(' - signal interpolation')

    func = interp1d(t_sample[np.isfinite(signal)], signal[np.isfinite(signal)],
                    fill_value='extrapolate')
    if tlim is None:
        tlim = [t_sample[0], t_sample[-1]]
    new_t = np.arange(int((tlim[1] - tlim[0]) * new_freq)) / new_freq + tlim[0]
    new_signal = func(new_t)

    if (post_smoothing * new_freq) > 1:
        if verbose:
            print(' - gaussian smoothing - post')
        new_signal = gaussian_filter1d(new_signal, int(post_smoothing * new_freq), mode='nearest')

    return new_t, new_signal
