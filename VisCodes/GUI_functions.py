from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import os.path
from PyQt5.QtGui import QPixmap

def initialize_attributes(obj):
    obj.intensity = 1
    obj.brightness = 0
    obj.min_area = 25
    obj.max_area = 450
    obj.threshold = 0
    obj.blur_kernel = 1
    obj.detect_ROI = None
    obj.hresholded_im = None
    obj.image_contours = None
    obj.adjust_image_exist = False
    
def setup_sliders(parent, location,size,min,max,set_value, orientation, function):
    Slider = QtWidgets.QSlider(parent)
    Slider.setGeometry(QtCore.QRect(location[0], location[1], size[0], size[1]))
    if orientation == "vertical":
        Slider.setOrientation(QtCore.Qt.Vertical)
    elif orientation == "horizontal":
        Slider.setOrientation(QtCore.Qt.Horizontal)
    Slider.setMinimum(min)
    Slider.setMaximum(max)
    Slider.setValue(set_value)
    Slider.valueChanged.connect(function)

    return Slider
def setup_labels(parent,location,size,color, set_text = False):
    label = QtWidgets.QLabel(parent)
    label.setGeometry(QtCore.QRect(location[0], location[1], size[0], size[1]))
    label.setAlignment(QtCore.Qt.AlignCenter)
    label.setStyleSheet(color)
    if set_text:
        label.setText(str(set_text))
    return label
def setup_pushButton(parent, location, size, color, function):
    PushButton = QtWidgets.QPushButton(parent)
    PushButton.setGeometry(QtCore.QRect(location[0], location[1], size[0], size[1]))
    PushButton.setStyleSheet(color)
    PushButton.clicked.connect(function)
    return PushButton

def cv2_to_pixmap(cv_image):
    height, width = cv_image.shape[:2]
    bytes_per_line = width
    q_image = QtGui.QImage(cv_image.data.tobytes(), width, height, bytes_per_line, QtGui.QImage.Format_Indexed8)
    pixmap = QtGui.QPixmap.fromImage(q_image)
    return pixmap

def update_image(scene, image, intensity,blur_kernel, brightness):
    scene.clear()
    adjusted_image = cv2.convertScaleAbs(image, alpha=intensity, beta=brightness)
    if isinstance(blur_kernel, tuple):
        blur_kernel = blur_kernel
    else:
        blur_kernel = (blur_kernel, blur_kernel)
    if blur_kernel[0] > 0 :
        blurred = cv2.GaussianBlur(adjusted_image, blur_kernel, 0)
    else :
        blurred = adjusted_image
    pixmap = cv2_to_pixmap(blurred)
    item = QtWidgets.QGraphicsPixmapItem(pixmap)
    scene.addItem(item)
    return blurred

def load_init_image(scene, image_path):
    if image_path is not None and os.path.exists(image_path) :
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        ## check if you wantn clahe
        # clahe = cv2.createCLAHE(clipLimit=9.0, tileGridSize=(15,15))
        # image = clahe.apply(image)

        pixmap = cv2_to_pixmap(normalized_image)
        scene.clear()
        item = QtWidgets.QGraphicsPixmapItem(pixmap)
        scene.addItem(item)
        return normalized_image
    
def load_mask_image(scene,image_contours):
    height, width, channel = image_contours.shape
    bytes_per_line = 3 * width
    q_image = QtGui.QImage(image_contours.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(q_image)
    scene.clear()
    item = QtWidgets.QGraphicsPixmapItem(pixmap)
    scene.addItem(item)

def thresholding(image,intensity, brightness, th_value, blur_kernel, scene):
    if isinstance(blur_kernel, tuple):
        blur_kernel = blur_kernel
    else:
        blur_kernel = (blur_kernel, blur_kernel)
    #print("blur_kernel", blur_kernel)
    adjusted_image = cv2.convertScaleAbs(image, alpha=intensity, beta=brightness)
    if blur_kernel[0] > 0 :
        blurred = cv2.GaussianBlur(adjusted_image, blur_kernel, 0)
    else :
        blurred = adjusted_image
    _, thresh = cv2.threshold(blurred, th_value, 255, cv2.THRESH_BINARY)
    scene.clear()
    pixmap= cv2_to_pixmap(thresh)
    item = QtWidgets.QGraphicsPixmapItem(pixmap)
    scene.addItem(item)
    return thresh, blurred