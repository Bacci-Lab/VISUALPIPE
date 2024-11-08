import numpy as np
def bootstrap(data, num_samples):
    indices = np.random.choice(data.shape[0], size=num_samples, replace=True)
    bootstrapped_data = data[indices]
    fith_perc_bootstraping = np.percentile(bootstrapped_data, 95, axis=1)
    return bootstrapped_data, fith_perc_bootstraping