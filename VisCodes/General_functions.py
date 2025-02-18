import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import interp1d

def bootstrap(data, num_samples):
    indices = np.random.choice(data.shape[0], size=num_samples, replace=True)
    bootstrapped_data = data[indices]
    fith_perc_bootstraping = np.percentile(bootstrapped_data, 95, axis=1)
    return bootstrapped_data, fith_perc_bootstraping

def scale_trace(data):
    min_val = np.min(data)
    max_val = np.max(data)
    data = (data - min_val) / (max_val - min_val)
    data = data * 5
    return data

def normalize_time_series(F, lower=0, upper=5):
    """
    Normalize each time series in F to the range [lower, upper].

    Args:
        F (numpy.ndarray): A 2D array where each row is a time series.
        lower (float): The lower bound of the normalization range.
        upper (float): The upper bound of the normalization range.

    Returns:
        numpy.ndarray: A normalized 2D array with values scaled to [lower, upper].
    """
    F_min = F.min(axis=1, keepdims=True)  # Min of each row
    F_max = F.max(axis=1, keepdims=True)  # Max of each row

    # Avoid division by zero for constant rows
    range_values = np.where(F_max - F_min == 0, 1, F_max - F_min)

    # Normalize to [0, 1]
    F_normalized = (F - F_min) / range_values

    # Scale to [lower, upper]
    F_scaled = F_normalized * (upper - lower) + lower
    return F_scaled

def resample_signal(original_signal,
                    original_freq=1e4,
                    t_sample=None,
                    new_freq=1e3,
                    pre_smoothing=0,
                    post_smoothing=0,
                    tlim=None,
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
    if tlim is None:
        tlim = [t_sample[0], t_sample[-1]]
    new_t = np.arange(int((tlim[1] - tlim[0]) * new_freq)) / new_freq + tlim[0]
    new_signal = func(new_t)

    if (post_smoothing * new_freq) > 1:
        if verbose:
            print(' - gaussian smoothing - post')
        new_signal = gaussian_filter1d(new_signal, int(post_smoothing * new_freq), mode='nearest')

    return new_t, new_signal

def create_H5_dataset(group, variable, variable_name):
    for name, value in zip(variable_name, variable):
        group.create_dataset(name, data=value)