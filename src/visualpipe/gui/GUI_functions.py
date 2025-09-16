from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
from PyQt5.QtGui import QPixmap, QImage
from typing import Tuple
import numpy as np

def show_warning_popup(widget, message):
    msg = QtWidgets.QMessageBox(widget)
    msg.setIcon(QtWidgets.QMessageBox.Warning)
    msg.setText(message)
    msg.setWindowTitle("Warning")
    msg.exec_()

# ----------------------------------- setup UI functions ---------------------------------------
def add_slider_row(group: QtWidgets.QGroupBox,
                   vbox: QtWidgets.QVBoxLayout,
                   text: str,
                   minimum: float,
                   maximum: float,
                   initial: float,
                   step: float,
                   on_value_changed,
                   enabled:bool) -> Tuple[QtWidgets.QSlider, QtWidgets.QLabel]:
    """
    Adds a horizontal row into `vbox` (which should belong to `group`) containing:
      [ QLabel(text) | QSlider | QLabel(current_value) ]
    Maps the float range [minimum, maximum] with step `step` onto the slider.
    Hooks slider.valueChanged to update the right‐hand QLabel
    and call `on_value_changed(new_float_value)`.
    Returns (slider, value_label).
    """
    # compute integer slider range
    n_steps = int(round((maximum - minimum) / step))
    # create layout
    hbox = QtWidgets.QHBoxLayout()
    hbox.setContentsMargins(0, 0, 0, 0)

    # left‐side text
    lbl = QtWidgets.QLabel(text, group)
    lbl.setStyleSheet("color: white;")
    hbox.addWidget(lbl)

    # slider
    slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, group)
    slider.setRange(0, n_steps)
    init_val = int(round((initial - minimum) / step))
    slider.setValue(init_val)
    hbox.addWidget(slider)

    # display label
    val_lbl = QtWidgets.QLabel(f"{initial:.3g}", group)
    val_lbl.setStyleSheet("color: white;")
    hbox.addWidget(val_lbl)

    # when slider moves, compute float, update label & call callback
    def _on_val(v: int):
        fv = minimum + v * step
        # format with sensible precision
        val_lbl.setText(f"{fv:.3g}")
        on_value_changed(fv)

    slider.valueChanged.connect(_on_val)

    slider.setEnabled(enabled)

    vbox.addLayout(hbox)
    return slider, val_lbl

def setup_labels(parent, location, size, color, set_text=False):
    label = QtWidgets.QLabel(parent)
    label.setGeometry(QtCore.QRect(location[0], location[1], size[0], size[1]))
    label.setAlignment(QtCore.Qt.AlignCenter)
    label.setStyleSheet(color)
    if set_text:
        label.setText(str(set_text))
    return label

def setup_pushButton(parent, location, size, color, function, enabled):
    PushButton = QtWidgets.QPushButton(parent)
    PushButton.setGeometry(QtCore.QRect(location[0], location[1], size[0], size[1]))
    PushButton.setStyleSheet(color)
    PushButton.clicked.connect(function)
    PushButton.setEnabled(enabled)
    return PushButton

def update_slider(slider, label, new_value, minimum, maximum, step, format_spec):
    new_val = max(minimum, min(maximum, new_value))
    int_tick = int(round((new_val - minimum) / step))
    slider.blockSignals(True)
    slider.setValue(int_tick)
    formatted_val = format_spec.format(new_val)
    label.setText(f"{formatted_val}")
    slider.blockSignals(False)

def update_slider_blur(slider, label, new_value, minimum, maximum, step, format_spec):
    value_slider = (float(new_value) - 1.0) / 2.0
    new_val = max(minimum, min(maximum, new_value))
    int_tick = int(round((value_slider - minimum) / step))
    slider.blockSignals(True)
    slider.setValue(int_tick)
    formatted_val = format_spec.format(new_val)
    label.setText(f"{formatted_val}")
    slider.blockSignals(False)

def add_radio_button_group(parent:QtWidgets.QWidget, labels:list[str], layout:QtWidgets.QLayout, function, init_id=0):
    radio_group = QtWidgets.QButtonGroup(parent)
    for i, label in enumerate(labels):
        radio_button = QtWidgets.QRadioButton(label, parent)
        radio_button.setStyleSheet("color: white;")
        radio_group.addButton(radio_button, id=i)
        layout.addWidget(radio_button)
    radio_group.button(init_id).setChecked(True)
    radio_group.buttonClicked[int].connect(function)
    return radio_button

# ----------------------------------- Images functions ---------------------------------------
def cv2_to_pixmap(cv_image):
    height, width = cv_image.shape[:2]
    bytes_per_line = width
    q_image = QtGui.QImage(cv_image.data.tobytes(), width, height, bytes_per_line, QtGui.QImage.Format_Indexed8)
    pixmap = QtGui.QPixmap.fromImage(q_image)
    return pixmap

def cv2_to_qpixmap(cv_img: np.ndarray) -> QPixmap:
    """
    Convert an OpenCV BGR uint8 image to QPixmap for display.
    """
    # 1) Convert BGR → RGB
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w

    # 2) Wrap in QImage
    qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

    # 3) Convert to QPixmap
    return QPixmap.fromImage(qt_image)