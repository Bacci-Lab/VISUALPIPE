import numpy as np
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
