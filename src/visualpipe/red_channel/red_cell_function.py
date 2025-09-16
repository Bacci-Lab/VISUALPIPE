import os.path
import cv2
import numpy as np
import os
import glob
from scipy.ndimage import shift as ndi_shift
from scipy.ndimage import label as ndi_label
from skimage.feature import blob_log
from skimage.measure   import regionprops
from skimage.registration import phase_cross_correlation
from PIL import Image

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

# ----------------------------------- Vessels functions ---------------------------------------
def preprocess(image: np.ndarray, blur_kernel: int):
    """
    Convert to grayscale and apply Gaussian blur.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    blur = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), sigmaX=1)
    return gray, blur

def segment_vessels(gray: np.ndarray,
                    thresh: int = 40,
                    open_size: int = 5,
                    close_size: int = 5,
                    dilation_radius: int = 10):
    """
    Identify dark 'vessels' and build a neighbor mask,
    but *never* include any pixel where gray == 0.
    Returns vessel_mask, vessel_mask_area.
    """
    # 1) raw threshold (everything < thresh becomes vessel)
    _, vessel_mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)

    # 2) force any originally-zero pixel to background
    vessel_mask[gray == 0] = 0

    # 3) clean up
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    vessel_mask = cv2.morphologyEx(vessel_mask, cv2.MORPH_OPEN, kernel_open)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    vessel_mask = cv2.morphologyEx(vessel_mask, cv2.MORPH_CLOSE, kernel_close)

    # 4) dilate to get neighbor zone
    dil_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * dilation_radius + 1, 2 * dilation_radius + 1)
    )
    vessel_mask_area = cv2.dilate(vessel_mask, dil_kernel)

    # 5) again mask out zeros so the dilation never “grows” into black border
    vessel_mask_area[gray == 0] = 0

    return vessel_mask, vessel_mask_area

def plot_vessel(image, vessel_area_thr, blur_kernel):
    """
    Draw *all* vessel-area boundaries in red on top of `image` (gray).
    """
    gray, blur = preprocess(image, blur_kernel)
    vessel_mask, vessel_mask_area = segment_vessels(blur,thresh=vessel_area_thr )

    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # binarize mask and find all external contours
    mask_bin = (vessel_mask_area > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        mask_bin,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    # draw every contour
    cv2.drawContours(
        overlay,
        contours,
        contourIdx=-1,
        color=(0, 0, 255),
        thickness=2
    )

    return overlay

# ----------------------------------- Images functions ---------------------------------------
def compute_intensity_bounds(image, intensity_bound):
    p_low, p_high = np.percentile(image, [intensity_bound, 100 - intensity_bound])
    image = (
            (image - p_low)  # shift so p_low is zero
            / (p_high - p_low)  # scale so p_high − p_low is one
            * 255.0  # map to 0–255 range
    ).clip(0, 255).astype(np.uint8)  # clamp outliers and cast to 8-bit
    return image

def percentile_contrast_stretch(image):
    p_low, p_high = np.percentile(image, [0.35, 100 - 0.35])
    image = (
            (image - p_low)  # shift so p_low is zero
            / (p_high - p_low)  # scale so p_high − p_low is one
            * 255.0  # map to 0–255 range
    ).clip(0, 255).astype(np.uint8)  # clamp outliers and cast to 8-bit
    return image

def image_shift(suite2p_meanImage, red,
                upsample_factor: int = 10,
                fill_mode: str = 'constant',
                fill_value: float = 0,
                savepath: str = ''):
    """
    Compute and apply the translation that aligns `red` to `suite2p_meanImage`.

    Returns:
      - shifted: the red image after applying the estimated shift
      - shift: (dy, dx) tuple in pixels
      - magnitude: sqrt(dy**2 + dx**2)
      - angle: direction of shift in degrees (0° = right, 90° = up)
      - direction: human string like "4.2px up, 12.7px right"
    """
    if suite2p_meanImage.shape[0] > red.shape[0]:
        og_width = suite2p_meanImage.shape[0]
        og_height = suite2p_meanImage.shape[1]
        new_width = red.shape[0]
        wpercent = (new_width / float(og_width))
        new_height = int((float(og_height) * float(wpercent)))
        suite2p_meanImage = Image.fromarray(suite2p_meanImage)
        suite2p_meanImage = suite2p_meanImage.resize((new_width, new_height), Image.Resampling.LANCZOS)
        suite2p_meanImage = np.asarray(suite2p_meanImage)
        cv2.imwrite(os.path.join(savepath, 'suite2p_meanImage_resized.png'), suite2p_meanImage)

    elif suite2p_meanImage.shape[0] < red.shape[0]:
        og_width = red.shape[0]
        og_height = red.shape[1]
        new_width = suite2p_meanImage.shape[0]
        wpercent = (new_width / float(og_width))
        new_height = int((float(og_height) * float(wpercent)))
        red = Image.fromarray(red)
        red = red.resize((new_width, new_height), Image.Resampling.LANCZOS)
        red = np.asarray(red)
        cv2.imwrite(os.path.join(savepath, 'red_resized.png'), red)

    # 1) Estimate subpixel shift that moves `red` → `suite2p_meanImage`
    (dy, dx), error, diffphase = phase_cross_correlation(
        suite2p_meanImage, red, upsample_factor=upsample_factor
    )

    # 2) Compute magnitude & angle
    magnitude = np.hypot(dy, dx)
    # note: arctan2 uses (y, x) but we want 0°→right, 90°→up
    angle = np.degrees(np.arctan2(-dy, dx))

    # 3) vertical & horizontal components
    vert = f"{abs(dy):.1f}px {'down' if dy>0 else 'up'}"
    horz = f"{abs(dx):.1f}px {'right' if dx>0 else 'left'}"
    direction = f"{vert}, {horz}"

    # 4) Apply the shift to `red`
    shifted = ndi_shift(
        red,
        shift=(dy, dx),
        mode=fill_mode,
        cval=fill_value
    )

    return shifted, (dy, dx), magnitude, angle, direction

def save_as_gray_png(shifted: np.ndarray,
                             path: str) -> None:
    """
    Normalize a 2D array to 8-bit, replicate it across R,G,B and save as a PNG.

    Args:
      shifted: 2D image array.
      path:    filename to write, e.g. "shifted_gray.png".
    """
    # 1) replace NaNs with zero
    arr = np.nan_to_num(shifted, nan=0.0)

    # 2) normalize to 0–255
    lo, hi = float(arr.min()), float(arr.max())
    if hi > lo:
        norm = (arr - lo) / (hi - lo)
    else:
        norm = np.zeros_like(arr)
    img8 = (norm * 255).astype(np.uint8)

    # 3) replicate into BGR
    h, w = img8.shape
    gray_bgr = np.stack([img8, img8, img8], axis=-1)

    # 4) write out
    if not cv2.imwrite(path, gray_bgr):
        raise IOError(f"Failed to write image to {path}")
    
# ----------------------------------- Detect red ROIs functions ---------------------------------------
def detect_REDROI(image: np.ndarray,
                  min_area, max_area,
                  vessel_threshold, vessel_area_thr,
                  min_sigma, threshold_rel,
                  blur_kernel, overlap,
                  max_sigma=20, num_sigma=20,
                  ring_width=1, dilate_radius=1,
                  exclude_border=True):
    """
    Same as before, but assumes any 0 in `image` is 'invalid' and will be ignored.
    """
    # 0) mask of "real" pixels (nonzero)
    valid_mask = image > 0

    # 1) grayscale & blur
    gray, blur = preprocess(image, blur_kernel)

    # 2) Vessel segmentation
    vessel_mask, vessel_mask_area = segment_vessels(gray, thresh=vessel_area_thr)
    vessel_mask[~valid_mask] = 0
    vessel_mask_area[~valid_mask] = 0

    # 3) LoG blob detection
    blobs = blob_log(blur,
                     min_sigma=min_sigma,
                     max_sigma=max_sigma,
                     num_sigma=num_sigma,
                     threshold=threshold_rel,
                     overlap=overlap,
                     exclude_border=exclude_border)
    blobs[:, 2] *= np.sqrt(2)  # sigma→radius

    # 4) local contrast filtering, skipping any blob overlapping zeros
    h, w = gray.shape
    ring_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2*ring_width+1, 2*ring_width+1)
    )
    kept_mask = np.zeros_like(gray, dtype=np.uint8)

    for y, x, r in blobs:
        cy, cx, radius = int(round(y)), int(round(x)), int(round(r))
        # bounding‐box check
        if cy-radius<0 or cy+radius>=h or cx-radius<0 or cx+radius>=w:
            continue

        # interior mask
        interior = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(interior, (cx, cy), radius, 255, -1)

        # drop if any interior pixel was zero
        if np.any((interior>0) & (~valid_mask)):
            continue

        # standard contrast test
        dil = cv2.dilate(interior, ring_kernel)
        ring = cv2.subtract(dil, interior)
        mean_in  = cv2.mean(gray, mask=interior)[0]
        mean_out = cv2.mean(gray, mask=ring)[0]
        is_vessel = np.any((interior>0) & (vessel_mask_area>0))
        factor = vessel_threshold if is_vessel else 1.0
        if mean_in > mean_out * factor:
            kept_mask[interior>0] = 2

    # 5) dilate
    if dilate_radius>0:
        dk = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2*dilate_radius+1, 2*dilate_radius+1)
        )
        kept_mask = cv2.dilate(kept_mask, dk)

    # 6) morphology filter
    labels = ndi_label(kept_mask>0)[0]
    filtered = np.zeros_like(kept_mask)
    for region in regionprops(labels):
        if min_area <= region.area <= max_area:
            filtered[labels==region.label] = 2
    kept_mask = filtered

    # 7) draw cell contours
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(
        kept_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for cnt in contours:
        cv2.drawContours(overlay, [cnt], -1, (0,255,0), 1)
    # plt.imshow(overlay)
    # plt.show()

    return kept_mask, overlay, gray

# ----------------------------------- Categorize ROIs functions ---------------------------------------
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

def get_GreenMask(save_dir, ops, cell_info):

    separated_masks = single_mask(ops, cell_info)

    red_masks_dir = os.path.join(save_dir, "red_mask.npy")
    if os.path.exists(red_masks_dir) :
        _, overlap_cells = select_mask(red_masks_dir, separated_masks, save=False, save_red_results=save_dir)
        only_green_cells = np.ones(len(cell_info))
        only_green_cells[overlap_cells] = 0
        return only_green_cells
    else :
        raise Exception(f"{red_masks_dir} does not exist.")
