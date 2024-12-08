from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QColor, QPen, QBrush
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QGraphicsPixmapItem
import cv2
import red_cell_function
import numpy as np
import os.path
import json
import GUI_functions
from PyQt5 import QtWidgets, QtCore
import sys

class CustomGraphicsView_red(QGraphicsView):
    objectClicked = pyqtSignal(int)
    def __init__(self, cell_info, Green_Cell, background_image_path=None):
        super().__init__()
        self.setScene(QGraphicsScene())
        self.Green_Cell = Green_Cell
        self.cell_info = cell_info
        self.background_image_path = background_image_path
        self.setBackgroundImage()
        self.drawObjects()

    def setBackgroundImage(self):
        if self.background_image_path:
            pixmap = QPixmap(self.background_image_path)
            if not pixmap.isNull():
                pixmap_item = QGraphicsPixmapItem(pixmap)
                pixmap_item.setZValue(-100)
                self.scene().addItem(pixmap_item)
            else:
                print("Failed to load background image:", self.background_image_path)
        self.drawObjects()

    def drawObjects(self):
        scene = self.scene()
        for i, cell in enumerate(self.cell_info):
            if self.Green_Cell[i] == 0:
                color = QColor(Qt.red)
                color.setAlpha(5)
                for x, y in zip(cell['xpix'], cell['ypix']):
                    ellipse = scene.addEllipse(x, y, 1, 1, QPen(color), QBrush(color))
                    ellipse.setData(0, i)

    def mousePressEvent(self, event):
        items = self.items(event.pos())
        for item in items:
            if isinstance(item, QGraphicsEllipseItem):
                object_index = item.data(0)
                self.objectClicked.emit(object_index)
                return
        super().mousePressEvent(event)


class CustomGraphicsView_green(QGraphicsView):
    objectClicked = pyqtSignal(int)

    def __init__(self, cell_info, Green_Cell, background_image_path=None):
        super().__init__()
        self.setScene(QGraphicsScene())
        self.Green_Cell = Green_Cell
        self.cell_info = cell_info
        self.background_image_path = background_image_path
        self.setBackgroundImage()
        self.drawObjects()

    def setBackgroundImage(self):
        if self.background_image_path:
            pixmap = QPixmap(self.background_image_path)
            if not pixmap.isNull():
                pixmap_item = QGraphicsPixmapItem(pixmap)
                pixmap_item.setZValue(-100)
                self.scene().addItem(pixmap_item)
            else:
                print("Failed to load background image:", self.background_image_path)

    def drawObjects(self):
        scene = self.scene()
        for i, cell in enumerate(self.cell_info):
            if self.Green_Cell[i] == 1:
                color = QColor(Qt.green)
                color.setAlpha(5)
                for x, y in zip(cell['xpix'], cell['ypix']):
                    ellipse = scene.addEllipse(x, y, 1, 1, QPen(color), QBrush(color))
                    ellipse.setData(0, i)

    def mousePressEvent(self, event):
        items = self.items(event.pos())
        for item in items:
            if isinstance(item, QGraphicsEllipseItem):
                object_index = item.data(0)
                self.objectClicked.emit(object_index)
                return
        super().mousePressEvent(event)


class SelectCell(QtWidgets.QMainWindow):
    def __init__(self, cell_info, Green_Cell, background_image_path):
        super().__init__()
        self.setupUi(cell_info, Green_Cell, background_image_path)

    def setupUi(self, cell_info, Green_Cell, background_image_path):
        self.setObjectName("MainWindow")
        self.setWindowTitle("You can transfer masks between channels")
        self.setStyleSheet("""
            background-color: rgb(27, 27, 27);
            gridline-color: rgb(213, 213, 213);
            border-top-color: rgb(197, 197, 197);
        """)
        self.red_cell_num  = 200
        self.Green_cell_num= 500
        self.setGeometry(100, 100, 1100, 641)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_3 = QtWidgets.QLabel(self.frame)
        self.label_3.setAccessibleName("")
        self.label_3.setStyleSheet("color: rgb(167, 167, 167);")
        self.label_3.setTextFormat(QtCore.Qt.RichText)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_2.addWidget(self.label_3)
        self.lineEdit_Red = QtWidgets.QLineEdit(self.frame)
        self.lineEdit_Red.setEnabled(True)
        self.lineEdit_Red.setObjectName("lineEdit_Red")
        self.verticalLayout_2.addWidget(self.lineEdit_Red)
        self.Red_view = CustomGraphicsView_red(cell_info, Green_Cell, background_image_path)
        self.Green_view = CustomGraphicsView_green(cell_info, Green_Cell, background_image_path)
        self.verticalLayout_2.addWidget(self.Red_view)
        self.horizontalLayout_2.addWidget(self.frame)
        self.horizontalLayout_4.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setEnabled(True)
        self.frame_2.setToolTip("")
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame_2)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_4 = QtWidgets.QLabel(self.frame_2)
        self.label_4.setAccessibleName("")
        self.label_4.setStyleSheet("color: rgb(167, 167, 167);")
        self.label_4.setTextFormat(QtCore.Qt.RichText)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.verticalLayout.addWidget(self.label_4)
        self.lineEdit_Green = QtWidgets.QLineEdit(self.frame_2)
        self.lineEdit_Green.setObjectName("lineEdit_Green")
        self.verticalLayout.addWidget(self.lineEdit_Green)

        self.verticalLayout.addWidget(self.Green_view)
        self.horizontalLayout_5.addWidget(self.frame_2)
        self.horizontalLayout_4.addLayout(self.horizontalLayout_5)
        self.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)
        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 954, 21))
        self.menubar.setObjectName("menubar")
        self.menubar.setStyleSheet("color: white;")
        self.menuOpen = QtWidgets.QMenu(self.menubar)
        self.menuOpen.setObjectName("menuOpen")
        self.menuOpen.setStyleSheet("""
                  QMenu {background-color: rgb(200, 200, 200);
                      color: rgb(20, 20, 20);}
              """)
        self.setMenuBar(self.menubar)
        self.actionload_proccesd_file = QtWidgets.QAction(self)
        self.actionload_proccesd_file.setObjectName("actionload_proccesd_file")
        self.menuOpen.addAction(self.actionload_proccesd_file)
        self.menubar.addAction(self.menuOpen.menuAction())
        ####################################################
        # Connecting signals
        self.Red_view.objectClicked.connect(lambda idx: self.toggleItem(idx, 1))
        self.Green_view.objectClicked.connect(lambda idx: self.toggleItem(idx, 0))

        # Initialize tracking lists
        self.currentRedObjects = []
        self.currentGreenObjects = []

        # tracking update
        self.updateObjectTracking()
        self.retranslateUi()

    def toggleItem(self, object_index, target_view):
        # Update the Green_Cell to reflect the new view state
        self.Red_view.Green_Cell[object_index] = target_view
        self.Green_view.Green_Cell[object_index] = target_view
        self.Red_view.scene().clear()
        self.Green_view.scene().clear()
        self.Red_view.drawObjects()
        self.Green_view.drawObjects()
        self.Red_view.setBackgroundImage()
        self.Green_view.setBackgroundImage()
        self.updateObjectTracking()
        self.lineEdit_Red.setText(str(self.red_cell_num))
        self.lineEdit_Green.setText(str(self.Green_cell_num))

    def updateObjectTracking(self):
        self.currentRedObjects.clear()
        self.currentGreenObjects.clear()



        # Re-populate the lists based on current state in Green_Cell
        for idx, state in enumerate(self.Red_view.Green_Cell):
            if state == 0:
                self.currentRedObjects.append(idx)

            else:
                self.currentGreenObjects.append(idx)
            self.red_cell_num = len(self.currentRedObjects)
            self.Green_cell_num = len(self.currentGreenObjects)
        return self.currentRedObjects, self.currentGreenObjects




    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "You can transfer masks between channels"))
        self.label_3.setText(_translate("MainWindow", "Red/Green"))
        self.label_4.setText(_translate("MainWindow", "Only Green"))
        self.menuOpen.setTitle(_translate("MainWindow", "Open"))
        self.lineEdit_Green.setStyleSheet("color: white")
        self.lineEdit_Green.setText(_translate("MainWindow", f"{self.Green_cell_num}"))
        self.lineEdit_Red.setStyleSheet("color: white")
        self.lineEdit_Red.setText(_translate("MainWindow", f"{self.red_cell_num}"))
        self.actionload_proccesd_file.setText(_translate("MainWindow", "load proccesd file"))

class Red_IMAGE_Adgustment(object):

    def setupUi(self, SelectCell,  SavePath, red_frame_path):
        GUI_functions.initialize_attributes(self)
        self.SavePath = SavePath
        self.SelectCell = SelectCell
        label_color = "color: rgb(255, 255, 255);"

        SelectCell.setObjectName("MainWindow")
        SelectCell.resize(1300, 581)
        SelectCell.setStyleSheet("background-color: rgb(27, 27, 27);")
        self.centralwidget = QtWidgets.QWidget(SelectCell)
        self.centralwidget.setObjectName("centralwidget")
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(20, 30, 514, 514))
        self.graphicsView.setObjectName("graphicsView")
        #-------------------second graphic view -------------------
        self.mask_graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.mask_graphicsView.setGeometry(QtCore.QRect(766, 30, 514, 514))
        self.mask_graphicsView.setObjectName("mask_graphicsView")
        self.scene_mask = QtWidgets.QGraphicsScene()
        self.mask_graphicsView.setScene(self.scene_mask)
        #------------------------------------
        self.scene = QtWidgets.QGraphicsScene()
        self.graphicsView.setScene(self.scene)
        self.image = GUI_functions.load_init_image(self.scene, red_frame_path)
        # ---------------------------------------- Setup Sliders ------------------------------------
        self.verticalSlider_brightness = GUI_functions.setup_sliders(self.centralwidget,(630,70), (22,131), -100, 100, 0,"vertical", self.brightness_value)
        self.verticalSlider_threshold = GUI_functions.setup_sliders(self.centralwidget, (700, 70), (22, 131), 0, 255, 0,"vertical", self.set_threshold)
        self.verticalSlider_intensity = GUI_functions.setup_sliders(self.centralwidget,(560, 70),(22, 131),0,30,1,"vertical",self.intensity_value)
        self.Slider_min = GUI_functions.setup_sliders(self.centralwidget,(620, 250),(100, 25),0,500,self.min_area, "horizontal", self.set_min)
        self.Slider_max = GUI_functions.setup_sliders(self.centralwidget,(620, 280),(100, 25),0,500,self.max_area, "horizontal", self.set_max)
        self.Slider_blur = GUI_functions.setup_sliders(self.centralwidget, (620, 310), (100, 25), 0, 7, 0,"horizontal", self.blur_values)
        #-----------------------------------------SetUp Labels----------------------------------------
        self.label_threshold_value  = GUI_functions.setup_labels(self.centralwidget,(700, 210),( 20, 10), label_color, self.verticalSlider_threshold.value())
        self.label_intensity_value  = GUI_functions.setup_labels(self.centralwidget,(560, 210),( 20, 10),  label_color, self.verticalSlider_intensity.value())
        self.label_blur_value       = GUI_functions.setup_labels(self.centralwidget,(730, 310),( 20, 25),  label_color, self.Slider_blur.value())
        self.label_brightness_value = GUI_functions.setup_labels(self.centralwidget,(630, 210),( 20, 10), label_color, self.verticalSlider_brightness.value())
        self.label_min_value        = GUI_functions.setup_labels(self.centralwidget, (730, 250), (20, 25), label_color, self.Slider_min.value())
        self.label_max_value        = GUI_functions.setup_labels(self.centralwidget,(730, 280),( 20, 25), label_color, self.Slider_max.value())
        self.label_Brightness = GUI_functions.setup_labels(self.centralwidget, (610, 30),(51,21), label_color)
        self.label_intensity = GUI_functions.setup_labels(self.centralwidget, (540, 30), (51, 21), label_color)
        self.label_min_area = GUI_functions.setup_labels(self.centralwidget, (540, 250), (65, 25), label_color)
        self.label_max_area = GUI_functions.setup_labels(self.centralwidget,(540, 280),( 65, 25), label_color)
        self.label_blur_kernel = GUI_functions.setup_labels(self.centralwidget, (540, 310), (65, 25), label_color)
        self.label_Thresholding = GUI_functions.setup_labels(self.centralwidget,(680, 30),( 71, 21), label_color)
        self.label_red_image = GUI_functions.setup_labels(self.centralwidget,(247, 3),(110, 23),  label_color)
        self.label_detected_mask = GUI_functions.setup_labels(self.centralwidget,(987, 3),(110, 23), label_color)
        #--------------------------------------------- PushButton -------------------------------------
        self.pushSave = GUI_functions.setup_pushButton(self.centralwidget, (650, 520), (95, 23), label_color, lambda: self.save_image(self.SavePath))
        self.pushmask = GUI_functions.setup_pushButton(self.centralwidget, (550, 520), (95, 23), label_color, self.show_mask)
        self.pushsaveParameter = GUI_functions.setup_pushButton(self.centralwidget, (650, 490), (95, 23),  label_color, lambda: self.save_para(self.SavePath))
        self.pushloadParameter = GUI_functions.setup_pushButton(self.centralwidget, (550, 490), (95, 23),  label_color, lambda: self.load_parameters(self.SavePath))
        #--------------------------------------------------
        SelectCell.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(SelectCell)
        SelectCell.setStatusBar(self.statusbar)
        self.retranslateUi(SelectCell)
        QtCore.QMetaObject.connectSlotsByName(SelectCell)

    def show_warning_popup(self, message):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setText(message)
        msg.setWindowTitle("Warning")
        msg.exec_()

    def set_max(self, value):
        self.max_area = value
        self.label_max_value.setText(str(self.max_area))
    def blur_values(self, value):
        value = (value*2) + 1
        self.blur_kernel = value
        self.label_blur_value.setText(str(self.blur_kernel))
        self.updated_image = GUI_functions.update_image(self.scene, self.image, self.intensity, self.blur_kernel,
                                                        self.brightness)

    def set_min(self, value):
        self.min_area = value
        self.label_min_value.setText(str(self.min_area))

    def intensity_value(self, value):
        self.intensity = value
        self.label_intensity_value.setText(str(self.intensity))
        self.updated_image = GUI_functions.update_image(self.scene, self.image, self.intensity,self.blur_kernel, self.brightness)
    def brightness_value(self, bightness_val):
        self.brightness = bightness_val
        self.label_brightness_value.setText(str(self.brightness))
        self.updated_image = GUI_functions.update_image(self.scene, self.image, self.intensity,self.blur_kernel, self.brightness)
    def set_threshold(self, threshold_value):
        self.threshold = threshold_value
        self.label_threshold_value.setText(str(self.threshold))
        self.hresholded_im, self.updated_image = GUI_functions.thresholding(self.image, self.intensity, self.brightness, self.threshold,self.blur_kernel, self.scene)

    def save_para(self, SavePath):
        parameters = {'min_area': self.min_area,
                      'max_area': self.max_area,
                      'brightness': self.brightness,
                      'threshold': self.threshold,
                      'intensity': self.intensity,
                      'blur kernel': self.blur_kernel}

        SaveMetadata = os.path.join(SavePath, 'red_image_parameters.txt')
        with open(SaveMetadata, 'w') as file:
            file.write(json.dumps(parameters, indent = 4))
        print("parameter saved")

    def adjust_image_exist(self):
        self.adjust_image_exist = True
        return self.adjust_image_exist


    def save_image(self, SavePath):
        save_background_path = os.path.join(SavePath, "adjusted_image.jpg")
        save_mask_path = os.path.join(SavePath, "red_mask.npy")
        save_mask_path_image = os.path.join(SavePath, "red_mask.jpg")

        if self.detect_ROI is not None:
            cv2.imwrite(save_background_path,  self.updated_image)
            np.save(save_mask_path,  self.detect_ROI)
            cv2.imwrite(save_mask_path_image, self.image_contours)
            print("saving function works")

        else:
            self.show_warning_popup("Please first detect masks")




    def show_mask(self):
        if self.hresholded_im is not None:
            self.hresholded_im, self.updated_image = GUI_functions.thresholding(self.image, self.intensity, self.brightness,
                                                                  self.threshold,self.blur_kernel, self.scene)
            self.detect_ROI, self.image_contours = red_cell_function.detect_REDROI(self.hresholded_im,self.updated_image, self.min_area, self.max_area )
            GUI_functions.load_mask_image(self.scene_mask, self.image_contours)
        else:
            self.show_warning_popup("Please first adjust thresholding")


    def load_parameters(self, SavePath):
        parameter_directory = os.path.join(SavePath, "red_image_parameters.txt")
        with open(parameter_directory, 'r') as file:
            data = file.read()
        json_data = json.loads(data)
        self.min_area = json_data["min_area"]
        self.max_area = json_data["max_area"]
        self.brightness = json_data["brightness"]
        self.threshold = json_data["threshold"]
        self.intensity = json_data["intensity"]
        self.blur_kernel = json_data["blur kernel"]
        self.verticalSlider_threshold.setValue(self.threshold)
        self.verticalSlider_intensity.setValue(self.intensity)
        self.verticalSlider_brightness.setValue(self.brightness)
        self.Slider_max.setValue(self.max_area)
        self.Slider_min.setValue(self.min_area)
        self.blur_kernel = int((self.blur_kernel -1)/2)
        self.Slider_blur.setValue(self.blur_kernel)



    def retranslateUi(self, SelectCell):
        _translate = QtCore.QCoreApplication.translate
        SelectCell.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_intensity.setText(_translate("MainWindow", "intensity"))
        self.label_Brightness.setText(_translate("MainWindow", "Brightness"))
        self.label_Thresholding.setText(_translate("MainWindow", "Thresholding"))
        self.label_min_area.setText(_translate("MainWindow", "Min ROI size"))
        self.label_max_area.setText(_translate("MainWindow", "Max ROI size"))
        self.label_blur_kernel.setText(_translate("MainWindow", "Blurring Kernel"))
        self.pushSave.setText(_translate("MainWindow", "Save & close"))
        self.pushmask.setText(_translate("MainWindow", "Detect mask"))
        self.pushsaveParameter.setText(_translate("MainWindow", "save Para"))
        self.pushloadParameter.setText(_translate("MainWindow", "load Para"))
        self.label_red_image.setText(_translate("MainWindow", "Red Image"))
        self.label_detected_mask.setText(_translate("MainWindow", "Detected Masks"))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    SelectCell = QtWidgets.QMainWindow()
    # Pass the necessary paths (replace with actual paths if needed)
    SavePath = r"C:\Users\faezeh.rabbani\Desktop\2024_09_03\16-00-59"
    red_frame_path = r"C:\Users\faezeh.rabbani\Desktop\2024_09_03\16-00-59"

    ui = Red_IMAGE_Adgustment()
    ui.setupUi(SelectCell, SavePath, red_frame_path)
    SelectCell.show()
    sys.exit(app.exec_())
