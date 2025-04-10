import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import scipy.stats
import matplotlib.pyplot as plt
import pandas as pd

def bootstrap(data, num_samples):
    indices = np.random.choice(data.shape[0], size=num_samples, replace=True)
    bootstrapped_data = data[indices]
    fith_perc_bootstraping = np.percentile(bootstrapped_data, 95, axis=1)
    return bootstrapped_data, fith_perc_bootstraping

def scale_trace(data):
    min_val = np.min(data)
    max_val = np.max(data)
    data = (data - min_val) / (max_val - min_val)
    data = data * 1
    return data

def resample_signal(original_signal,
                    original_freq=1e4,
                    t_sample=None,
                    new_freq=1e3,
                    pre_smoothing=0,
                    post_smoothing=0,
                    tlim=None,
                    interp_time=None,
                    verbose=False):
    '''
    Author: Yann Zerlaut
    from https://github.com/yzerlaut/physion/blob/main/src/physion/analysis/tools.py
    '''
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
    if interp_time is None :
        if tlim is None :
            tlim = [t_sample[0], t_sample[-1]]
        new_t = np.arange(int((tlim[1] - tlim[0]) * new_freq)) / new_freq + tlim[0]
    else : 
        new_t = interp_time 
    new_signal = func(new_t)

    if (post_smoothing * new_freq) > 1:
        if verbose:
            print(' - gaussian smoothing - post')
        new_signal = gaussian_filter1d(new_signal, int(post_smoothing * new_freq), mode='nearest')

    if interp_time is None :
        return new_t, new_signal
    else :
        return new_signal