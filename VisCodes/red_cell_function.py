import matplotlib.pyplot as plt
import os.path
import cv2
import numpy as np
import os
from numpy.fft import fft2, ifft2, fftshift
import glob

def image_shift(suite2p_image, red_image):
    f1 = fft2(suite2p_image)
    f2 = fft2(red_image)

    # Compute the cross-correlation in the frequency domain
    cross_corr = ifft2(f1 * np.conj(f2))

    # Shift the zero-frequency component to the center of the spectrum
    cross_corr = fftshift(cross_corr)

    # Find the peak in cross-correlation
    y_shift, x_shift = np.unravel_index(np.argmax(np.abs(cross_corr)), cross_corr.shape)

    # Correct for the shift (same as above)
    y_shift -= suite2p_image.shape[0] // 2
    x_shift -= suite2p_image.shape[1] // 2
    if abs(y_shift) > 10 and abs(x_shift)> 10:
        x_shift = 0
        y_shift = 0
    M = np.float32([[1, 0, -x_shift], [0, 1, -y_shift]])

    # Apply translation to image2
    shifted_image = cv2.warpAffine(red_image, M, (red_image.shape[1], red_image.shape[0]))

    # shifted_image should now be aligned with image1

    return shifted_image

def detect_REDROI(thresh,image, min_area, max_area):
    opening_kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, opening_kernel)
    dilation_kernel = np.ones((5,5), np.uint8)
    dilation = cv2.dilate(opening, dilation_kernel, iterations = 1)
    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_contours = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    blank3 = np.zeros((512, 512),  dtype='uint8')
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            cv2.drawContours(blank3, [contour], -1, 2, -1)
            cv2.drawContours(image_contours, [contour], -1, (255, 0, 0), 1)
    return blank3, image_contours

def get_red_channel(base_path):
    list_dir_red = glob.glob(os.path.join(base_path, "SingleImage-*red*"))

    if len(list_dir_red) > 0 :
        if len(list_dir_red) > 1 : 
            print(f"Many possible directory found : {list_dir_red}.\nUsing the first one:{list_dir_red[0]}")
        red_channel_path = list_dir_red[0]
        list_dir_red_image = glob.glob(os.path.join(red_channel_path, "*_Ch2_*.ome.tif"))
        if len(list_dir_red_image) >= 1 :
            red_image_path = list_dir_red_image[0]
            if len(list_dir_red_image) != 1 :
                print(f"More than one red channel image found. First image used: {red_image_path}")
                print(f"Other found red channel image: {list_dir_red_image[1:]}")
        
        else :
            print("No red channel image found.")
            red_image_path = None
    else :
        print("No red channel directory found.")
        red_channel_path, red_image_path = None, None
    
    return red_channel_path, red_image_path

def loadred(Base_path):
    suite2p_path = os.path.join(Base_path, "suite2p", "plane0")
    ops = np.load((os.path.join(suite2p_path, "ops.npy")), allow_pickle=True).item()
    Mean_image = ((ops['meanImg']))
    stat = np.load((os.path.join(suite2p_path, "stat.npy")), allow_pickle=True)
    single_red = cv2.imread((os.path.join(Base_path, 'red.tif')))
    return suite2p_path, ops, Mean_image, stat, single_red

def single_mask(ops, cell_info):
    separated_masks = []
    for i in range(0, len(cell_info)):
        roi_mask = np.zeros((ops['Ly'], ops['Lx']))
        roi_mask[cell_info[i]['ypix'], cell_info[i]['xpix']] = 2
        separated_masks.append(roi_mask)
    return separated_masks

def select_mask(red_masks_dir, separated_masks, cell_true=2, save=True, save_red_results='', with_masks=False):
    red_cell_masks =  np.load(red_masks_dir, allow_pickle=True)
    overlap_masks, overlap_cells = [], []
    only_green_masks, only_green_cells = [], []

    for i in range(len(separated_masks)):
        masks_sum = red_cell_masks + separated_masks[i]
        if (cell_true + 2) in masks_sum: #if there are any overlap in the two masks, it is considered as the same cell
            overlap_cells.append(i)
            overlap_masks.append(separated_masks[i])
        else:
            only_green_cells.append(i)
            only_green_masks.append(separated_masks[i])
    
    if save :
        save_direction1 = os.path.join(save_red_results, 'red_green_cells.npy')
        save_direction2 = os.path.join(save_red_results, 'only_green.npy')
        np.save(save_direction1, overlap_cells, allow_pickle=True)
        np.save(save_direction2, only_green_cells, allow_pickle=True)

    if with_masks :
        return only_green_cells, overlap_cells, only_green_masks, overlap_masks
    else :
        return  only_green_cells, overlap_cells
