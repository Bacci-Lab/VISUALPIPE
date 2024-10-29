import numpy as np
import os.path
import xml_parser
import glob
from scipy.signal.windows import hamming
from scipy.signal import convolve
from sklearn.linear_model import LinearRegression
from scipy.ndimage import filters, gaussian_filter1d
import matplotlib.pyplot as plt


base_path = r"D:\Faezeh 2p2analyze\2024_09_03\16-00-59"
def find_Tseries(base_path):
    file = [f for f in os.listdir(base_path) if f.startswith("TSeries")]
    if len (file) > 1:
        raise Exception("There are multiple Tseries in this directory please keep one")
    else:
        directory = os.path.join(base_path, file[0])
    print(directory)
    return directory

def load_Suite2p(base_path):
    Tseries_directory = find_Tseries(base_path)
    Suite2p_path = os.path.join(Tseries_directory,"suite2p\plane0")
    F = np.load(os.path.join(Suite2p_path, "F.npy"), allow_pickle=True)
    Fneu_raw = np.load(os.path.join(Suite2p_path, "Fneu.npy"), allow_pickle=True)
    iscell = np.load(os.path.join(Suite2p_path, "iscell.npy"), allow_pickle=True)
    return F, Fneu_raw, iscell
def load_xml(base_path, metadate = False):
    Tseries_directory = find_Tseries(base_path)
    xml_direction = glob.glob(os.path.join(Tseries_directory, '*.xml'))[0]
    xml = xml_parser.bruker_xml_parser(xml_direction)
    if metadate == True:
        channel_number = xml['Nchannels']
        laserWavelength = xml['settings']['laserWavelength']
        objectiveLens = xml['settings']['objectiveLens']
        objectiveLensMag = xml['settings']['objectiveLensMag']
        opticalZoom = xml['settings']['opticalZoom']
        bitDepth = xml['settings']['bitDepth']
        dwellTime = xml['settings']['dwellTime']
        framePeriod = xml['settings']['framePeriod']
        micronsPerPixel = xml['settings']['micronsPerPixel']
        TwophotonLaserPower = xml['settings']['twophotonLaserPower']
        return xml, channel_number, laserWavelength, objectiveLens, objectiveLensMag, opticalZoom,\
        bitDepth, dwellTime, framePeriod, micronsPerPixel, TwophotonLaserPower
    else:
        return xml
def detect_cell(cell, F):
    removed_ROI = [i for i, c in enumerate(cell) if c[0] == 0]
    keeped_ROI = [j for j, i in enumerate(cell) if i[0] != 0]
    if len(F) != len(keeped_ROI):
        F = np.delete(F, removed_ROI, axis=0)
    return F, keeped_ROI
def detect_bad_neuropils(detected_roi,neuropil, F,iscell ,direction= None):
    neuropil_F = gaussian_filter1d(neuropil, 10)
    F_F = gaussian_filter1d(F,10)
    chosen_cell = [i for i in range(len(neuropil_F)) if np.std(F_F[i])/np.std(neuropil_F[i]) < 1.6]
    neuron_chosen = [i for i in chosen_cell if i in detected_roi]
    for i in chosen_cell:
        iscell[i][0] = 0
        if direction is not None:
            np.save(direction, iscell, allow_pickle=True)
        else:
            pass
    return iscell, neuron_chosen
def calculate_F0(F, fs, percentile, mode = 'sliding', win = 60, sig = 60):
    if mode == 'hamming':
        F0 = []
        window_duration = 0.5  # Duration of the Hamming window in seconds
        window_size = int(window_duration * fs)
        hamming_window = hamming(window_size)
        for i in range(len(F)):
            F_smooth = convolve(F[i], hamming_window, mode='same') / sum(hamming_window)
            roi_percentile = np.percentile(F_smooth, percentile)
            F_below_percentile = np.extract(F_smooth <= roi_percentile, F_smooth)
            f0 = np.mean(F_below_percentile)
            f0 = [f0]*len(F[i])
            F0.append(f0)
        F0 = np.array(F0)
    elif mode == 'sliding':
        F0 = filters.gaussian_filter(F, [0., sig])
        F0 = filters.minimum_filter1d(F0, win * fs, mode='wrap')
        F0 = filters.maximum_filter1d(F0, win * fs, mode='wrap')
    return F0
def deltaF_calculate(F, F0):
    normalized_F = np.copy(F)
    for i in range(0, np.size(F, 0)):
        normalized_F[i] = (F[i]-F0[i])/F0[i]
    return normalized_F


def calculate_alpha (F, Fneu):
    Slope = []
    per = np.arange(5,101,5)
    for k in range(len(F)):
        b = 0
        All_F, percentile_Fneu = [], []
        for i in per:
            percentile_before = np.percentile(Fneu[k], b)
            percentile_now = np.percentile(Fneu[k], i)
            index_percentile_i = np.where((percentile_before <= Fneu[k]) & (Fneu[k] < percentile_now))
            b = i
            F_percentile_i = F[k][index_percentile_i]
            perc_F_i = np.percentile(F_percentile_i, 5)
            percentile_Fneu.append(percentile_now)
            All_F.append(perc_F_i)

        #fitting a linear regression model
        x = np.array(percentile_Fneu).reshape(-1, 1)
        y = np.array(All_F)
        model = LinearRegression()
        model.fit(x, y)
        slope = model.coef_[0]
        Slope.append(slope)
    remove,alpha = [],[]
    for i in range(len(Slope)):
        if Slope[i] <= 0:
            remove.append(i)
        else:
            alpha.append(Slope[i])
    alpha = np.mean(alpha)
    return alpha, remove