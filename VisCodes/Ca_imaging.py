import numpy as np
import os.path
import xml_parser
import glob
import cv2
from scipy.signal.windows import hamming
from scipy.signal import convolve
from sklearn.linear_model import LinearRegression
from scipy.ndimage import minimum_filter1d, maximum_filter1d, gaussian_filter1d
import matplotlib.pyplot as plt

class CaImagingDataManager(object):
    __slots__ = ['_tseries_path', '_neuropil_if', '_f0_method', '_neuron_type', '_starting_delay',
                 'raw_F', 'raw_Fneu', 'fluorescence', 'f0', 'dFoF0',
                 'iscell', 'stat', 'xml', 'fs', 'time_stamps',
                 '_to_update_ROIs_list', '_to_update_frames_list', '_list_ROIs_idx']

    def __init__(self, base_path, neuropil_if=0.7, f0_method='sliding', neuron_type='PYR', starting_delay=0.1):
        
        self._tseries_path = self.find_Tseries_folder(base_path)
        self._neuropil_if = neuropil_if
        self._f0_method = f0_method
        self._neuron_type = neuron_type
        self._starting_delay = starting_delay

        self.raw_F, self.raw_Fneu, self.iscell, self.stat = self.load_suite2p()
        self._to_update_ROIs_list = ['raw_F', 'raw_Fneu', 'stat']
        bad_cells = [i for i, c in enumerate(self.iscell) if c[0] == 0]
        self.remove_ROIs(bad_cells) #remove cells that didn't pass manual curation in suite2p
        
        self._list_ROIs_idx = np.array(self.get_list_of_ROIs_from_iscell())

        self.xml = self.load_xml()
        time_stamps = self.xml['Green']['relativeTime']
        self.fs = (len(time_stamps) - 1) / time_stamps[-1]
        self.time_stamps = time_stamps + self._starting_delay

        self._to_update_frames_list = ['time_stamps', 'raw_F', 'raw_Fneu']
        
        self.fluorescence = None
        self.f0 = None
        self.dFoF0 = None
    
    def __str__(self) :
        list_attr = ['_tseries_path', '_neuropil_if', '_f0_method', '_neuron_type', '_starting_delay',
                     'time_stamps', 'raw_F', 'raw_Fneu', 'fluorescence', 'f0', 'dFoF0', 
                     'iscell', 'stat', 'xml', 'fs','_list_ROIs_idx']
        return f"Available attributes : {list_attr}"

    def find_Tseries_folder(self, base_path):
        file = [f for f in os.listdir(base_path) if f.startswith("TSeries")]
        if len(file) > 1:
            raise Exception("There are multiple Tseries in this directory please keep one")
        elif len(file) == 0 :
            raise Exception("No Tseries found in the base directory.")
        else:
            directory = os.path.join(base_path, file[0])
        return directory

    def load_suite2p(self):
        suite2p_path = os.path.join(self._tseries_path, "suite2p", "plane0")
        raw_F = np.load(os.path.join(suite2p_path, "F.npy"), allow_pickle=True)
        raw_Fneu = np.load(os.path.join(suite2p_path, "Fneu.npy"), allow_pickle=True)
        iscell = np.load(os.path.join(suite2p_path, "iscell.npy"), allow_pickle=True)
        stat = np.load((os.path.join(suite2p_path, "stat.npy")), allow_pickle=True)
        return raw_F, raw_Fneu, iscell, stat

    def load_xml(self, metadata=False):
        xml_direction = glob.glob(os.path.join(self._tseries_path, '*.xml'))[0]
        xml = xml_parser.bruker_xml_parser(xml_direction)
        if metadata :
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
    
    def save_mean_image(self, save_directory):
        save_image_dir = os.path.join(save_directory, "Mean_image_grayscale.png")
        if not os.path.exists(save_image_dir) :
            suite2p_path = os.path.join(self._tseries_path, "suite2p", "plane0")
            ops = np.load((os.path.join(suite2p_path, "ops.npy")), allow_pickle=True).item()
            mean_image = ((ops['meanImg']))
            normalized_image = (mean_image - mean_image.min()) / (mean_image.max() - mean_image.min()) * 255
            normalized_image = normalized_image.astype(np.uint8)
            cv2.imwrite(save_image_dir, normalized_image)
        else :
            print("Mean_image_grayscale.png already exist.")

    def get_list_of_ROIs_from_iscell(self):
        kept_ROI = [j for j, i in enumerate(self.iscell) if i[0] != 0]
        return kept_ROI
    
    def cut_frames(self, start_index=0, last_index=None):
        for el in self._to_update_frames_list :
            if hasattr(self, el):
                attribute = getattr(self, el)
                if len(attribute.shape) == 1  :
                    if last_index is None :
                        setattr(self, el, attribute[start_index:])
                    else : 
                        setattr(self, el, attribute[start_index:last_index+1])
                else :
                    if last_index is None :
                        setattr(self, el, attribute[:, start_index:])
                    else : 
                        setattr(self, el, attribute[:, start_index:last_index+1])
                    
            else :
                raise Exception("Object has no attribute " + el)

    def remove_ROIs(self, bad_cells):
        for el in self._to_update_ROIs_list :
            if hasattr(self, el):
                attribute = getattr(self, el)
                setattr(self, el, np.delete(attribute, bad_cells, axis=0))
            else :
                raise Exception("Object has no attribute " + el)

    def get_corresponding_idx(self, bad_cells):
        return self._list_ROIs_idx[bad_cells]
    
    def update_iscell(self, bad_cells, save_directory=None):
        bad_cells_2p_index = self.get_corresponding_idx(bad_cells)
        for i in bad_cells_2p_index:
            self.iscell[i][0] = 0
            if save_directory is not None:
                save_path = os.path.join(save_directory, 'iscell_post')
                np.save(save_path, self.iscell, allow_pickle=True)
        
        self._list_ROIs_idx = np.delete(self._list_ROIs_idx, bad_cells)
    
    def detect_bad_neuropils(self, save_directory=None):

        raw_Fneu_filtered = gaussian_filter1d(self.raw_Fneu, 10)
        raw_F_filtered = gaussian_filter1d(self.raw_F, 10)

        bad_cells = [i for i in range(len(raw_F_filtered)) if np.std(raw_F_filtered[i])/np.std(raw_Fneu_filtered[i]) < 1.6]
        self.update_iscell(bad_cells, save_directory)
        self.remove_ROIs(bad_cells)

    def calculate_alpha(self, save_directory=None):

        slopes_list = []
        per = np.arange(5,101,5)
        
        for k in range(len(self.raw_F)):
            b = 0
            All_F, percentile_Fneu = [], []
            for i in per:
                percentile_before = np.percentile(self.raw_Fneu[k], b)
                percentile_now = np.percentile(self.raw_Fneu[k], i)
                index_percentile_i = np.where((percentile_before <= self.raw_Fneu[k]) & (self.raw_Fneu[k] < percentile_now))
                b = i
                F_percentile_i = self.raw_F[k][index_percentile_i]
                perc_F_i = np.percentile(F_percentile_i, 5)
                percentile_Fneu.append(percentile_now)
                All_F.append(perc_F_i)

            #fitting a linear regression model
            x = np.array(percentile_Fneu).reshape(-1, 1)
            y = np.array(All_F)
            model = LinearRegression()
            model.fit(x, y)
            slope = model.coef_[0]
            slopes_list.append(slope)
        
        # Only keep positive alphas
        bad_cells, alpha = [],[]
        for i in range(len(slopes_list)):
            if slopes_list[i] <= 0:
                bad_cells.append(i)
            else:
                alpha.append(slopes_list[i])
        
        self._neuropil_if = np.mean(alpha)
        self.update_iscell(bad_cells, save_directory)
        self.remove_ROIs(bad_cells)

    def compute_F(self) :
        
        # Calculate neuropil impact factor if neurons are pyramidals
        if self._neuron_type == "PYR" :
            self.calculate_alpha()
        
        self.fluorescence = self.raw_F - (self._neuropil_if * self.raw_Fneu)
        
        if 'fluorescence' not in self._to_update_frames_list :
            self._to_update_frames_list.append('fluorescence')
        if 'fluorescence' not in self._to_update_ROIs_list :
            self._to_update_ROIs_list.append('fluorescence')

    def compute_F0(self, percentile=10, win=60, sig=60, save_directory=None):

        if self._f0_method == 'hamming':
            f0_list = []
            window_duration = 0.5  # Duration of the Hamming window in seconds
            window_size = round(window_duration * self.fs)
            hamming_window = hamming(window_size)

            for i in range(len(self.fluorescence)):
                F_smooth = convolve(self.fluorescence[i], hamming_window, mode='same') / sum(hamming_window)
                roi_percentile = np.percentile(F_smooth, percentile)
                F_below_percentile = np.extract(F_smooth <= roi_percentile, F_smooth)
                f0 = np.mean(F_below_percentile)
                f0 = [f0]*len(self.fluorescence[i])
                f0_list.append(f0)
            self.f0 = np.array(f0_list)
        
        elif self._f0_method  == 'sliding':
            f0 = gaussian_filter1d(self.fluorescence, sig)
            f0 = minimum_filter1d(f0, round(win * self.fs), mode='wrap')
            self.f0 = maximum_filter1d(f0, round(win * self.fs), mode='wrap')
        
        else :
            raise Exception(f"Invalid f0 calculation method selected : {self._f0_method}")
        
        if 'f0' not in self._to_update_frames_list :
            self._to_update_frames_list.append('f0')
        if 'f0' not in self._to_update_ROIs_list :
            self._to_update_ROIs_list.append('f0')
        
        #Remove Neurons with F0 less than 1 at any point in time
        invalid_f0_cells = [i for i, val in enumerate(self.f0) if np.any(val < 1)]

        #Update metrics and iscell
        self.update_iscell(invalid_f0_cells, save_directory)
        self.remove_ROIs(invalid_f0_cells)     

    def compute_dFoF0(self):
        normalized_F = np.copy(self.fluorescence)
        for i in range(0, np.size(self.fluorescence, 0)):
            normalized_F[i] = (self.fluorescence[i] - self.f0[i]) / self.f0[i]
        
        self.dFoF0 = normalized_F
        if 'dFoF0' not in self._to_update_frames_list :
            self._to_update_frames_list.append('dFoF0')
        if 'dFoF0' not in self._to_update_ROIs_list :
            self._to_update_ROIs_list.append('dFoF0')

    def normalize_time_series(self, attr='dFoF0', lower=0, upper=5):
        """
        Normalize each time series to the range [lower, upper].

        Args:
            lower (float): The lower bound of the normalization range.
            upper (float): The upper bound of the normalization range.

        Returns:
            numpy.ndarray: A normalized 2D array with values scaled to [lower, upper].
        """
        
        if hasattr(self, attr) : 
            trace = getattr(self, attr)
        else : 
            raise Exception("Object has no attribute " + attr)

        trace_min = trace.min(axis=1, keepdims=True)  # Min of each row
        trace_max = trace.max(axis=1, keepdims=True)  # Max of each row

        # Avoid division by zero for constant rows
        range_values = np.where(trace_max - trace_min == 0, 1, trace_max - trace_min)

        # Normalize to [0, 1]
        trace_normalized = (trace - trace_min) / range_values

        # Scale to [lower, upper]
        trace_scaled = trace_normalized * (upper - lower) + lower
        
        return trace_scaled

if __name__ == "__main__":
    starting_delay_2p = 0.1
    base_path = "Y:/raw-imaging/TESTS/Mai-An/visual_test/16-00-59"
    ca_img = CaImagingDataManager(base_path, neuron_type='Other', starting_delay=starting_delay_2p)
    detected_roi = ca_img._list_ROIs_idx
    print('Original number of neurons :', len(detected_roi))
    
    #---------------------------------- Detect ROIs with bad neuropils ------------------
    ca_img.detect_bad_neuropils()
    kept2p_ROI = ca_img._list_ROIs_idx
    print('After removing bad neuropil neurons, nb of neurons :', len(kept2p_ROI))

    #---------------------------------- Compute Fluorescence ------------------
    ca_img.compute_F()
    kept_ROI_alpha = ca_img._list_ROIs_idx
    print('Number of remaining neurons after alpha calculation :', len(kept_ROI_alpha))

    #---------------------------------- Calculation of F0 ----------------------
    ca_img.compute_F0(percentile=10, win=60)
    kept_ROI_F0 = ca_img._list_ROIs_idx
    print('Number of remaining neurons after F0 calculation  :', len(kept_ROI_F0))

    #---------------------------------- Calculation of dF over F0 ----------------------
    ca_img.compute_dFoF0()
    computed_F_norm = ca_img.normalize_time_series("dFoF0", lower=0, upper=5)

    print(ca_img.dFoF0.shape)
    print(ca_img.fluorescence.shape)
    print(ca_img.f0.shape)
    print(ca_img.raw_F.shape)
    print(ca_img.raw_Fneu.shape)
