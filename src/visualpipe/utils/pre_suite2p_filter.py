import sys
sys.path.append("./src/visualpipe")
from analysis.ca_imaging import CaImagingDataManager

def pre_suite2p_filter(base_path):
    ca_img_dm = CaImagingDataManager(base_path)
    
    detected_roi = ca_img_dm._list_ROIs_idx
    print('Sampling frequency:', ca_img_dm.fs)
    print('Original number of neurons :', len(detected_roi))

    #---------------------------------- Detect ROIs with bad neuropils ------------------
    ca_img_dm.detect_bad_neuropils(save_directory=ca_img_dm._suite2p_path)
    kept2p_ROI = ca_img_dm._list_ROIs_idx
    print('After removing bad neuropil neurons, nb of neurons :', len(kept2p_ROI))


if __name__ == "__main__":
    
    base_path = r"C:\Users\mai-an.nguyen\Documents\t_series_compression_test\log8bit-02172025-108-003"

    pre_suite2p_filter(base_path)