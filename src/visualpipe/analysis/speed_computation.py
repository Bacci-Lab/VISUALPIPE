import numpy as np
from scipy.ndimage import gaussian_filter1d
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
        acq_freq = data['NIdaq-acquisition-frequency']
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

    return binary_signal, radius, value_per_rotation, acq_freq

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

def get_alignment_index(F_time_stamps, speed, original_freq):
    speedTimeStamps = np.arange(len(speed)) / original_freq + 1 / original_freq / 2 #shift from +dt/2 because we consider v(t_i + dt/2) = ( x(t_(i+1)) - x(t_i) ) / dt
    lastFIdx = np.argmin(np.abs(F_time_stamps - speedTimeStamps[-1]))
    timeReference = F_time_stamps[:lastFIdx+1]
    return lastFIdx, timeReference, speedTimeStamps

def resample_running_signal(original_signal,
                            t_sample,
                            interpTime,
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

    if verbose:
        print(' - signal interpolation')
    
    func = interp1d(t_sample[np.isfinite(signal)], signal[np.isfinite(signal)],
                    fill_value='extrapolate')
    new_signal = func(interpTime)

    if (post_smoothing * new_freq) > 1:
        if verbose:
            print(' - gaussian smoothing - post')
        new_signal = gaussian_filter1d(new_signal, int(post_smoothing * new_freq), mode='nearest')

    return new_signal

def compute_speed(base_path, position_smoothing=10e-3, #s
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

    speed_time_stamps = np.arange(len(speed)) / acq_freq + 1 / acq_freq / 2 #shift from +dt/2 because we consider v(t_i + dt/2) = ( x(t_(i+1)) - x(t_i) ) / dt

    """ lastFIdx, timeReference, speedTimeStamps = get_alignment_index(F_time_stamps, speed, acq_freq)

    #sub sampling and filtering speed
    speed = resample_running_signal(speed, speedTimeStamps,
                                    timeReference,
                                    original_freq=acq_freq,
                                    new_freq = new_freq,
                                    pre_smoothing=0,
                                    post_smoothing = 2. / 50.,
                                    verbose=False) """

    if with_raw_position:
        return speed, speed_time_stamps, position
    else:
        return speed, speed_time_stamps

if __name__ == "__main__":
    from visualpipe.analysis.ca_imaging import CaImagingDataManager
    
    starting_delay_2p = 0.1
    base_path = "Y:/raw-imaging/TESTS/Mai-An/visual_test/16-00-59"
    ca_img = CaImagingDataManager(base_path, starting_delay=starting_delay_2p)
    speed, speed_time_stamps = compute_speed(base_path)
    print(speed)