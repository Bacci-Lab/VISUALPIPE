import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os.path
import math

def compute_synchrony_STTC(x, y, w_size):
    w_size = int(w_size / 2)
    tx = np.where(x > 0)[0]
    ty = np.where(y > 0)[0]

    mx, my = len(tx), len(ty)

    x_extend = np.zeros_like(x)
    y_extend = np.zeros_like(y)

    for i in tx:
        start_idx = max(0, i - w_size)
        end_idx = min(len(x), i + w_size + 1)
        x_extend[start_idx:end_idx] = 1

    for j in ty:
        start_idx = max(0, j - w_size)
        end_idx = min(len(y), j + w_size + 1)
        y_extend[start_idx:end_idx] = 1

    Ta, Tb = np.sum(x_extend) / len(x), np.sum(y_extend) / len(x)
    Pa = np.sum(x * y_extend) / mx
    Pb = np.sum(y * x_extend) / my

    STTC = ((Pa - Tb) / (1 - Pa * Tb) + (Pb - Ta) / (1 - Pb * Ta)) / 2

    return STTC

def synchrony(dataset, th, w_size=10):
    """ September 2019 - Rebola Lab - marie.fayolle@ens.fr
    ...........................................................................

    This function will compute the synchronicity inside a data set of n
    neurons, using the deconvolved trace (amplitude and not 0 and 1). The
    synchronicity is computed between each neuron so we get a matrix of
    correlation neuron per neuron.
    ...........................................................................

    - - - - - - - - - - - - - - - METHODS - - - - - - - - - - - - - - - - - - -
    'STTC'            Spike Time Tilling Coefficient

    - - - - - - - - - - - - - - - INPUT - - - - - - - - - - - - - - - - - - - -

    dataset           dataset of deconvolved firing trace (several ROIs) or dF
    th                threshold for each ROI
    method            'cross corr', 'ES constant', 'ES', 'ISI' or 'STTC'
                      ('constant' by default)
    w_size            limit number of frames to consider 2 events as
                      synchronous (1 by default)
    'derivative'      if you want to compute time-resolved synchrony

    - - - - - - - - - - - - - - - OUTPUT - - - - - - - - - - - - - - - - - - -

    synchro           matrix of synchronicity (neuron per neuron)

    ...........................................................................
    """

    NbOfROI = len(dataset)
    synchro = np.eye(NbOfROI)
    synchro[:] = np.nan
    for i in range(NbOfROI):
        for j in range(i + 1, NbOfROI):
            x, y = (dataset[i] > th[i]) * 1, (dataset[j] > th[j]) * 1
            Q = compute_synchrony_STTC(x, y, w_size)
            synchro[i, j], synchro[j, i] = Q, Q
    return (synchro)

def sumbre_threshold(dF, S):
    """
    ...........................................................................
    This function will calculate the threshold for each ROI based on its df and spike activity
    To calculate th we creat a numpy array contains only negative element of dF and their positive reflection(f_tot)
    then we divide spike range by df range and multiplying it by 3 times of "f_tot"'s standard deviation.
    - - - - - - - - - - - - - - - INPUT - - - - - - - - - - - - - - - - - - - -
    df                             df/f
    S                              Spikes

    - - - - - - - - - - - - - - - OUTPUT - - - - - - - - - - - - - - - - - - -

    th                             threshold for each ROI
     """
    f_bin = (dF < 0) * 1
    f_bin = f_bin.astype('float')
    f_bin[f_bin == 0] = 'nan'
    f_inf = dF * f_bin
    f_inf = f_inf[~np.isnan(f_inf)]
    f_tot = np.concatenate((f_inf, f_inf * (-1)))
    th = 3 * np.std(f_tot) * (max(S) - min(S)) / (max(dF) - min(dF))
    return th

def plot_matrix_synchro(synchro, save_path, name):
    vmin, vmax = np.nanmin(synchro), np.nanmax(synchro)
    if np.abs(vmin) > np.abs(vmax) :
        lim = math.floor(np.abs(vmin)*100) * 0.01
    else : 
        lim = math.floor(np.abs(vmax)*100) * 0.01
    save_path = os.path.join(save_path, name)
    plt.ioff()
    plt.matshow(synchro, cmap='RdBu_r', vmin=-lim, vmax=lim)
    plt.colorbar()
    plt.xlabel('ROI')
    plt.ylabel('ROI')
    plt.savefig(save_path)
    plt.close()

def randomizer(S):
    random_S = np.copy(S)
    for i in range(0,len(random_S)):
        np.random.shuffle(random_S[i])
    return random_S

def get_average_rand_synchronicity(spikes, N_iterations, thr_spikes, w_size=10):
    matrix_shuffled = np.array([synchrony(randomizer(spikes), thr_spikes, w_size=w_size) for n in tqdm(range(0, N_iterations), desc='Shuffling synchrony')])
    return matrix_shuffled

def plot_data(a, b, save_path, svg, ax=None):
    save_path2 = save_path + "\spike correlation.png"
    if ax is None:
        fig, ax = plt.subplots()
    else:
        ax = ax or plt.gcf()
    a_new = list(np.array(a)[~np.isnan(np.array(a))])
    b_new = list(np.array(b)[~np.isnan(np.array(b))])
    counts_a, bins_a = np.histogram(a_new, bins=100, density=True)
    center_a = (bins_a[:-1] + bins_a[1:]) / 2.
    counts_b, bins_b = np.histogram(b_new, bins=100, density=True)
    center_b = (bins_b[:-1] + bins_b[1:]) / 2.
    ax.plot(center_b, counts_b, 'black', label='Shuffled')
    ax.plot(center_a, counts_a, 'red', label='STTC')
    ax.axvline(x=0, linestyle='dotted', linewidth=1, color='grey')
    ax.set_xlabel('Spiking Correlation')
    ax.set_ylabel('Percentage (%)')
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.title('Distribution of STTC')
    if svg == True:
        svg_name = "spike correlation.svg"
        save_direction_svg = os.path.join(save_path, svg_name)
        plt.savefig(save_direction_svg,format = 'svg')
    plt.savefig(save_path2)