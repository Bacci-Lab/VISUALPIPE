import matplotlib.pyplot as plt
import numpy as np
import os

def plot_traces(av_trial_traces:list[dict], stim:str, groups_id:dict, time:np.ndarray, nb_valid_neurons:list[dict], trace_type:str, suffix:str='', save_path:str='', show:bool=False):
    """
    Plot the average traces for a given stimulus per group to compare their responses.

    Parameters
    ----------
    av_trial_traces : list
        A list of dictionaries. Each dictionary contains the average traces for a group of neurons.
    stim : str
        The name of the stimulus.
    groups_id : dict
        A dictionary containing the group names as keys and the corresponding group IDs as values.
    time : array-like
        A time array.
    trace_type : str
        The traces type of the response to be plotted (dFoF0-baseline or z-scores).
    suffix : str
        A string that completes the filename.
    save_path : str
        The path to save the figure.
    show : bool, optional
        Whether to show the plot. Default is False.

    Returns
    -------
    None
    """

    fig = plt.figure(figsize=(6, 5))
    for group in groups_id.keys():
        nb_neurons = nb_valid_neurons[groups_id[group]][stim]
        plt.plot(time, av_trial_traces[groups_id[group]][stim], label=f"{group} ({nb_neurons} neurons)")
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel(f'Average response ({trace_type})')
    plt.title(f'Average response for {stim}')
    fig.savefig(os.path.join(save_path, f"{stim}_traces_{suffix}.png"), dpi=300)
    if show:
        plt.show()
    plt.close(fig)