import matplotlib.pyplot as plt
import os.path
import cv2
import numpy as np
import os
from numpy.fft import fft2, ifft2, fftshift

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

def loadred(Base_path):
    suite2p_path = os.path.join(Base_path, "suite2p", "plane0")
    ops = np.load((os.path.join(suite2p_path, "ops.npy")), allow_pickle=True).item()
    Mean_image = ((ops['meanImg']))
    stat = np.load((os.path.join(suite2p_path, "stat.npy")), allow_pickle=True)
    single_red = cv2.imread((os.path.join(Base_path, 'red.tif')))
    return suite2p_path, ops, Mean_image, stat, single_red
def single_mask(ops, cell_info):
    separete_masks = []
    for i in range(0, len(cell_info)):
        neumask1 = np.zeros((ops['Ly'], ops['Lx']))
        neumask1[cell_info[i]['ypix'], cell_info[i]['xpix']] = 2
        separete_masks.append(neumask1)
    return separete_masks


def select_mask2(save_red_results, thresh2, separete_masks, cell_true = 2):
    KeepMask = []
    comen_cell = []
    only_green_mask = []
    only_green_cell = []

    for i in range(len(separete_masks)):
        blank2 = thresh2 + separete_masks[i]
        if (cell_true + 2) in blank2:
            comen_cell.append(i)
            KeepMask.append(separete_masks[i])
        else:
            only_green_cell.append(i)
            only_green_mask.append(separete_masks[i])
    save_direction1 = os.path.join(save_red_results, 'red_green_cells.npy')
    save_direction2 = os.path.join(save_red_results, 'only_green.npy')
    np.save(save_direction1, comen_cell, allow_pickle=True)
    np.save(save_direction2, only_green_cell, allow_pickle=True)
    return only_green_mask, only_green_cell, comen_cell, KeepMask, blank2
############################ Check ####################################
def select_mask(save_red_results, thresh_dir, separete_masks, cell_true = 2):
    thrsh =  np.load(thresh_dir, allow_pickle=True)
    KeepMask = []
    comen_cell = []
    only_green_mask = []
    only_green_cell = []

    for i in range(len(separete_masks)):
        blank2 = thrsh + separete_masks[i]
        if (cell_true + 2) in blank2:
            comen_cell.append(i)
            KeepMask.append(separete_masks[i])
        else:
            only_green_cell.append(i)
            only_green_mask.append(separete_masks[i])
    save_direction1 = os.path.join(save_red_results, 'red_green_cells.npy')
    save_direction2 = os.path.join(save_red_results, 'only_green.npy')
    np.save(save_direction1, comen_cell, allow_pickle=True)
    np.save(save_direction2, only_green_cell, allow_pickle=True)
    return only_green_mask, only_green_cell, comen_cell, KeepMask, blank2
