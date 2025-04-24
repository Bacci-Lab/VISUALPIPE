import sys, os
from pathlib import Path
import json
import cv2
from PyQt5 import QtWidgets, QtCore
import numpy as np
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QColor, QPen, QBrush
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QGraphicsPixmapItem, QFileDialog

import GUI_functions
import red_cell_function
import utils.file as file

class RedGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__(flags=Qt.WindowStaysOnTopHint)
        self.save_folder = QFileDialog.getExistingDirectory(self, caption='Select folder containing output data')
        self.setupUi()

    def setupUi(self):

        #------------------------- Main settings -------------------------
        self.setObjectName("MainWindow")
        self.setGeometry(500, 50, 1341, 750)
        self.setStyleSheet("""
            background-color: rgb(104, 104, 104);
            gridline-color: rgb(213, 213, 213);
            border-top-color: rgb(197, 197, 197);
        """)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.setCentralWidget(self.centralwidget)

        #------------------------- Status Bar -------------------------
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        #------------------------- Menu Bar -------------------------
        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(20, 20, 1301, 20))
        self.menubar.setObjectName("menubar")
        self.menubar.setStyleSheet("color: white;")

        self.button_load_data = QtWidgets.QAction(self)
        self.button_load_data.triggered.connect(self.menu_button_clicked)
        self.menubar.addAction(self.button_load_data)

        self.setMenuBar(self.menubar)
        
        #------------------------- Tabs -------------------------
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setTabShape(QtWidgets.QTabWidget.Triangular)
        self.tabWidget.setTabPosition(QtWidgets.QTabWidget.North)
        self.tabWidget.setDocumentMode(True)
        self.tabWidget.setObjectName("tabWidget")
        self.tabWidget.setGeometry(20, 30, 1301, 680)
        self.tabWidget.setStyleSheet("background-color: rgb(104, 104, 104);\ncolor: white;")
        
        self.categorize_cells_tab = QtWidgets.QWidget(self.centralwidget)
        self.adjust_red_img_tab = QtWidgets.QWidget(self.centralwidget)

        self.tabWidget.addTab(self.categorize_cells_tab, 'Categorize cells')
        self.tabWidget.addTab(self.adjust_red_img_tab, 'Adjust Red Channel Image')
                              
        #------------------------- Tab 1 -------------------------
        self.categorize_ui = CategorizeCells(self.categorize_cells_tab, self.save_folder)

        #------------------------- Tab 2 -------------------------
        self.red_img_adjust_ui = RedImageAdjust(self.adjust_red_img_tab, self.save_folder)

        #--------------------------------------------------
        self.retranslateUi()
    
    def menu_button_clicked(self):
        
        save_dir = QFileDialog.getExistingDirectory(self, caption='Select folder containing output data')
        if save_dir is not None :
            self.save_folder = save_dir
            self.categorize_ui.save_folder = save_dir
            self.red_img_adjust_ui.save_folder = save_dir

            out = self.categorize_ui.set_cell_info()
            if out == -1 :
                raise Exception("No valid stat.npy file.")
            
            out = self.categorize_ui.set_ops()
            if out == -1 :
                raise Exception("No valid ops.npy file.")
            
            # Load data in GUI
            out = self.categorize_ui.load_data_in_GUI()
            if out == -1:
                self.tabWidget.setCurrentIndex(1)
            
            self.red_img_adjust_ui.reset_UI()
            return 0
        
        else :
            return -1
        
    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "Red-Green Channels"))
        self.button_load_data.setText(_translate("MainWindow", "Load Data Folder"))

class RedGreenView(QGraphicsView):

    objectClicked = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene())
        self.only_green_cell = None
        self.cell_info = None
        self.background_image_path = None

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
            if self.only_green_cell[i] == 0:
                color = QColor(Qt.red)
                color.setAlpha(10)
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

class GreenView(QGraphicsView):
    objectClicked = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene())
        self.only_green_cell = None
        self.cell_info = None
        self.background_image_path = None

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
            if self.only_green_cell[i] == 1:
                color = QColor(Qt.green)
                color.setAlpha(10)
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

class CategorizeCells(object):
    def __init__(self, centralwidget, save_folder, cell_info=None, ops=None):

        self.centralwidget = centralwidget
        self.save_folder = save_folder
        self.cell_info = cell_info
        self.ops = ops

        # Initialize tracking lists
        self.currentRedObjects = []
        self.currentGreenObjects = []
        self.red_cell_num = 0
        self.green_cell_num = 0

        if self.cell_info is None :
            out = self.set_cell_info()
            if out == -1 :
                raise Exception("No valid stat.npy file.")
        if self.ops is None :
            out = self.set_ops()
            if out == -1 :
                raise Exception("No valid ops.npy file.")
        self.setupUi()
        self.load_data_in_GUI()

    def setupUi(self):
        
        #------------------------- LAYOUT -------------------------
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")

        self.load_pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.load_pushButton.setObjectName("pushButton")
        self.load_pushButton.clicked.connect(self.load_data_in_GUI)
        self.verticalLayout.addWidget(self.load_pushButton)

        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout.addLayout(self.horizontalLayout)

        #_____________________________________________________
        self.left_frame = QtWidgets.QFrame(self.centralwidget)
        self.left_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.left_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.left_frame.setObjectName("frame")
        self.verticalLayout_red_green = QtWidgets.QVBoxLayout(self.left_frame)
        self.verticalLayout_red_green.setObjectName("verticalLayout_red_green")

        self.red_green_label = QtWidgets.QLabel(self.left_frame)
        self.red_green_label.setStyleSheet("color: rgb(167, 167, 167);")
        self.red_green_label.setTextFormat(QtCore.Qt.RichText)
        self.red_green_label.setAlignment(QtCore.Qt.AlignCenter)
        self.red_green_label.setObjectName("red_green_label")
        self.verticalLayout_red_green.addWidget(self.red_green_label)
        
        self.lineEdit_Red = QtWidgets.QLineEdit(self.left_frame)
        self.lineEdit_Red.setEnabled(False)
        self.lineEdit_Red.setObjectName("lineEdit_Red")
        self.verticalLayout_red_green.addWidget(self.lineEdit_Red)

        self.red_view = RedGreenView()
        self.verticalLayout_red_green.addWidget(self.red_view)

        self.horizontalLayout.addWidget(self.left_frame)

        #_____________________________________________________
        self.right_frame = QtWidgets.QFrame(self.centralwidget)
        self.right_frame.setEnabled(True)
        self.right_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.right_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.right_frame.setObjectName("frame_2")
        self.verticalLayout_only_green = QtWidgets.QVBoxLayout(self.right_frame)
        self.verticalLayout_only_green.setObjectName("verticalLayout_only_green")

        self.green_label = QtWidgets.QLabel(self.right_frame)
        self.green_label.setStyleSheet("color: rgb(167, 167, 167);")
        self.green_label.setTextFormat(QtCore.Qt.RichText)
        self.green_label.setAlignment(QtCore.Qt.AlignCenter)
        self.green_label.setObjectName("green_label")
        self.verticalLayout_only_green.addWidget(self.green_label)

        self.lineEdit_Green = QtWidgets.QLineEdit(self.right_frame)
        self.lineEdit_Green.setEnabled(False)
        self.lineEdit_Green.setObjectName("lineEdit_Green")
        self.verticalLayout_only_green.addWidget(self.lineEdit_Green)

        self.green_view = GreenView()
        self.verticalLayout_only_green.addWidget(self.green_view)

        self.horizontalLayout.addWidget(self.right_frame)
        
        #_____________________________________________________
        self.save_pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.save_pushButton.setObjectName("pushButton")
        self.save_pushButton.clicked.connect(self.save_npy_files)
        self.verticalLayout.addWidget(self.save_pushButton)

        #--------------------------------------------------
        # Connecting signals
        self.red_view.objectClicked.connect(lambda idx: self.toggleItem(idx, 1))
        self.green_view.objectClicked.connect(lambda idx: self.toggleItem(idx, 0))

        self.retranslateUi()

    def set_ops(self):

        path = Path(self.save_folder)
        base_path = path.parent.absolute()

        # Load ops file
        tseries = [f for f in os.listdir(base_path) if f.startswith("TSeries")]
        if len(tseries) == 0 :
            print("No Tseries folder found in the base directory.")
            directory = base_path
        else:
            directory = os.path.join(base_path, tseries[0])
        
        suite2p_path = os.path.join(directory, "suite2p", "plane0")
        if os.path.exists(suite2p_path) :
            self.ops = np.load(os.path.join(suite2p_path, "ops.npy"), allow_pickle=True).item()
        else :
            print(f"{suite2p_path} doesn't exists. Select op.npy file.")
            ops_path = QFileDialog.getOpenFileName(self.centralwidget, caption='Select ops.npy file', filter="npy(*.npy)")[0]
            if 'ops' in ops_path: 
                self.ops = np.load(ops_path, allow_pickle=True).item()
            else :
                print(f"Selected file not valid ({ops_path}).")
                return -1

        return 0
    
    def set_cell_info(self):

        path = Path(self.save_folder)
        base_path = path.parent.absolute()
        unique_id, _, _, _ = file.get_metadata(base_path)
        id_version = self.save_folder.split('_')[5]

        # Load stat file
        filename = "_".join([unique_id, id_version, 'stat.npy'])
        stat_path = os.path.join(self.save_folder, filename)
        if os.path.exists(stat_path) :
            self.cell_info = np.load(stat_path, allow_pickle=True)
        else :
            print(f"{stat_path} doesn't exists. Select stat.npy file.")
            stat_path = QFileDialog.getOpenFileName(self.centralwidget, caption='Select stat.npy file', filter="npy(*.npy)")[0]
            if 'stat' in stat_path: 
                self.cell_info = np.load(stat_path, allow_pickle=True)
            else :
                print(f"Selected file not valid ({stat_path}).")
                return -1

        return 0
        
    def toggleItem(self, object_index, target_view):
        # Update the Green_Cell to reflect the new view state
        self.red_view.only_green_cell[object_index] = target_view
        self.green_view.only_green_cell[object_index] = target_view
        self.red_view.scene().clear()
        self.green_view.scene().clear()
        self.red_view.drawObjects()
        self.green_view.drawObjects()
        self.red_view.setBackgroundImage()
        self.green_view.setBackgroundImage()
        self.updateObjectTracking()
        self.lineEdit_Red.setText(str(self.red_cell_num))
        self.lineEdit_Green.setText(str(self.green_cell_num))

    def updateObjectTracking(self):
        self.currentRedObjects.clear()
        self.currentGreenObjects.clear()

        # Re-populate the lists based on current state in Green_Cell
        for idx, state in enumerate(self.red_view.only_green_cell):
            if state == 0:
                self.currentRedObjects.append(idx)
            else:
                self.currentGreenObjects.append(idx)
            self.red_cell_num = len(self.currentRedObjects)
            self.green_cell_num = len(self.currentGreenObjects)
        return self.currentRedObjects, self.currentGreenObjects

    def save_npy_files(self):
        save_direction1 = os.path.join(self.save_folder, 'red_green_cells.npy')
        save_direction2 = os.path.join(self.save_folder, 'only_green.npy')
        np.save(save_direction1, self.currentRedObjects, allow_pickle=True)
        np.save(save_direction2, self.currentGreenObjects, allow_pickle=True)

    def load_data_in_GUI(self):
        
        # Load red_green_cells.npy file
        red_green_filepath = os.path.join(self.save_folder, 'red_green_cells.npy')
        if os.path.exists(red_green_filepath) :
            data = np.load(red_green_filepath, allow_pickle=True)
            only_green_cells = np.ones(len(self.cell_info))
            only_green_cells[data] = 0
        else :
            red_masks_dir = os.path.join(self.save_folder, "red_mask.npy")
            if os.path.exists(red_masks_dir) :
                only_green_cells = red_cell_function.get_GreenMask(self.save_folder, self.ops, self.cell_info)
            else : 
                print("red_mask.npy doesn't exist. Compute masks first.")
                return -1
        
        # Check existence of adjusted_image.jpg file
        background_image_path = os.path.join(self.save_folder, 'adjusted_image.jpg')
        if not os.path.exists(background_image_path):
            print("adjusted_image.jpg doesn't exist. Compute masks first.")
            return -1
            
        self.red_view.background_image_path = background_image_path
        self.red_view.only_green_cell = only_green_cells
        self.red_view.cell_info = self.cell_info
        self.green_view.background_image_path = background_image_path
        self.green_view.only_green_cell = only_green_cells
        self.green_view.cell_info = self.cell_info

        self.red_view.scene().clear()
        self.green_view.scene().clear()

        self.red_view.setBackgroundImage()
        self.red_view.drawObjects()
        self.green_view.setBackgroundImage()
        self.green_view.drawObjects()
        self.updateObjectTracking()
        self.lineEdit_Red.setText(str(self.red_cell_num))
        self.lineEdit_Green.setText(str(self.green_cell_num))

        return 0

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.red_green_label.setText(_translate("MainWindow", "Red/Green"))
        self.green_label.setText(_translate("MainWindow", "Only Green"))
        self.lineEdit_Green.setStyleSheet("color: white")
        self.lineEdit_Green.setText(_translate("MainWindow", f"{self.green_cell_num}"))
        self.lineEdit_Red.setStyleSheet("color: white")
        self.lineEdit_Red.setText(_translate("MainWindow", f"{self.red_cell_num}"))
        self.load_pushButton.setText(_translate("MainWindow", "Load Data in GUI"))
        self.load_pushButton.setStyleSheet("color: white")
        self.save_pushButton.setText(_translate("MainWindow", "Save"))
        self.save_pushButton.setStyleSheet("color: white")

class RedImageAdjust(object):

    def __init__(self, centralwidget, save_folder, red_frame_path=None):
        self.centralwidget = centralwidget
        self.save_folder = save_folder
        GUI_functions.initialize_attributes(self)

        self.image = None
        self.red_frame_path = red_frame_path

        self.setupUi()

        if self.red_frame_path is not None :
            self.image = GUI_functions.load_init_image(self.scene, self.red_frame_path)

    def setupUi(self):
        
        #------------------------- Main settings -------------------------
        label_color = "color: rgb(255, 255, 255);"

        #------------------------- LAYOUT -------------------------
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")

        self.load_pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.load_pushButton.setObjectName("loadButton")
        self.load_pushButton.clicked.connect(self.load_red_tif)
        self.verticalLayout.addWidget(self.load_pushButton)

        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout.addWidget(self.frame)

        #------------------------- First graphic view -------------------------

        self.graphicsView = QtWidgets.QGraphicsView(self.frame)
        self.graphicsView.setGeometry(QtCore.QRect(0, 30, 520, 520))
        self.graphicsView.setObjectName("graphicsView")
        self.scene = QtWidgets.QGraphicsScene()
        self.graphicsView.setScene(self.scene)

        #------------------- Second graphic view -------------------
        self.mask_graphicsView = QtWidgets.QGraphicsView(self.frame)
        self.mask_graphicsView.setGeometry(QtCore.QRect(761, 30, 520, 520))
        self.mask_graphicsView.setObjectName("mask_graphicsView")
        self.scene_mask = QtWidgets.QGraphicsScene()
        self.mask_graphicsView.setScene(self.scene_mask)

        # ---------------------------------------- Setup Sliders ------------------------------------
        self.verticalSlider_brightness = GUI_functions.setup_sliders(self.frame,(625,70), (22,131), -100, 100, 0,"vertical", self.brightness_value)
        self.verticalSlider_threshold = GUI_functions.setup_sliders(self.frame, (695, 70), (22, 131), 0, 255, 0,"vertical", self.set_threshold)
        self.verticalSlider_intensity = GUI_functions.setup_sliders(self.frame,(555, 70),(22, 131),0,30,1,"vertical",self.intensity_value)
        self.Slider_min = GUI_functions.setup_sliders(self.frame,(615, 250),(100, 25),0,500,self.min_area, "horizontal", self.set_min)
        self.Slider_max = GUI_functions.setup_sliders(self.frame,(615, 280),(100, 25),0,500,self.max_area, "horizontal", self.set_max)
        self.Slider_blur = GUI_functions.setup_sliders(self.frame, (615, 310), (100, 25), 0, 7, 0,"horizontal", self.blur_values)

        #-----------------------------------------SetUp Labels----------------------------------------
        self.label_threshold_value  = GUI_functions.setup_labels(self.frame,(695, 210),( 20, 10), label_color, self.verticalSlider_threshold.value())
        self.label_intensity_value  = GUI_functions.setup_labels(self.frame,(555, 210),( 20, 10),  label_color, self.verticalSlider_intensity.value())
        self.label_blur_value       = GUI_functions.setup_labels(self.frame,(725, 310),( 20, 25),  label_color, self.Slider_blur.value())
        self.label_brightness_value = GUI_functions.setup_labels(self.frame,(625, 210),( 20, 10), label_color, self.verticalSlider_brightness.value())
        self.label_min_value        = GUI_functions.setup_labels(self.frame, (725, 250), (20, 25), label_color, self.Slider_min.value())
        self.label_max_value        = GUI_functions.setup_labels(self.frame,(725, 280),( 20, 25), label_color, self.Slider_max.value())
        self.label_Brightness = GUI_functions.setup_labels(self.frame, (605, 30),(51,21), label_color)
        self.label_intensity = GUI_functions.setup_labels(self.frame, (535, 30), (51, 21), label_color)
        self.label_min_area = GUI_functions.setup_labels(self.frame, (535, 250), (65, 25), label_color)
        self.label_max_area = GUI_functions.setup_labels(self.frame,(535, 280),( 65, 25), label_color)
        self.label_blur_kernel = GUI_functions.setup_labels(self.frame, (535, 310), (70, 25), label_color)
        self.label_Thresholding = GUI_functions.setup_labels(self.frame,(675, 30),( 71, 21), label_color)
        self.label_red_image = GUI_functions.setup_labels(self.frame,(205, 3),(110, 23),  label_color)
        self.label_detected_mask = GUI_functions.setup_labels(self.frame,(966, 3),(110, 23), label_color)
        
        #--------------------------------------------- PushButton -------------------------------------
        self.pushSave = GUI_functions.setup_pushButton(self.frame, (630, 520), (95, 23), label_color, self.save_image)
        self.pushmask = GUI_functions.setup_pushButton(self.frame, (530, 520), (95, 23), label_color, self.show_mask)
        self.pushsaveParameter = GUI_functions.setup_pushButton(self.frame, (630, 490), (95, 23),  label_color, self.save_para)
        self.pushloadParameter = GUI_functions.setup_pushButton(self.frame, (530, 490), (95, 23),  label_color, self.load_parameters)

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self.centralwidget)

    def show_warning_popup(self, message):
        msg = QtWidgets.QMessageBox(self.centralwidget)
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setText(message)
        msg.setWindowTitle("Warning")
        msg.exec_()

    def set_max(self, value):
        if self.image is None :
            self.show_warning_popup("Please first load Red TIF")
        else :
            self.max_area = value
            self.label_max_value.setText(str(self.max_area))
    
    def blur_values(self, value):
        if self.image is None :
            self.show_warning_popup("Please first load Red TIF")
        else :
            value = (value*2) + 1
            self.blur_kernel = value
            self.label_blur_value.setText(str(self.blur_kernel))
            self.updated_image = GUI_functions.update_image(self.scene, self.image, self.intensity, self.blur_kernel,
                                                            self.brightness)

    def set_min(self, value):
        if self.image is None :
            self.show_warning_popup("Please first load Red TIF")
        else :
            self.min_area = value
            self.label_min_value.setText(str(self.min_area))

    def intensity_value(self, value):
        if self.image is None :
            self.show_warning_popup("Please first load Red TIF")
        else :
            self.intensity = value
            self.label_intensity_value.setText(str(self.intensity))
            self.updated_image = GUI_functions.update_image(self.scene, self.image, self.intensity,self.blur_kernel, self.brightness)
    
    def brightness_value(self, bightness_val):
        if self.image is None :
            self.show_warning_popup("Please first load Red TIF")
        else :
            self.brightness = bightness_val
            self.label_brightness_value.setText(str(self.brightness))
            self.updated_image = GUI_functions.update_image(self.scene, self.image, self.intensity,self.blur_kernel, self.brightness)
    
    def set_threshold(self, threshold_value):
        if self.image is None :
            self.show_warning_popup("Please first load Red TIF")
        else :
            self.threshold = threshold_value
            self.label_threshold_value.setText(str(self.threshold))
            self.thresholded_im, self.updated_image = GUI_functions.thresholding(self.image, self.intensity, self.brightness, self.threshold,self.blur_kernel, self.scene)

    def save_para(self):
        parameters = {'min_area': self.min_area,
                      'max_area': self.max_area,
                      'brightness': self.brightness,
                      'threshold': self.threshold,
                      'intensity': self.intensity,
                      'blur kernel': self.blur_kernel}

        SaveMetadata = os.path.join(self.save_folder, 'red_image_parameters.txt')
        with open(SaveMetadata, 'w') as file:
            file.write(json.dumps(parameters, indent = 4))
        print("parameter saved")

    def adjust_image_exist(self):
        self.adjust_image_exist = True
        return self.adjust_image_exist

    def save_image(self):
        save_background_path = os.path.join(self.save_folder, "adjusted_image.jpg")
        save_mask_path = os.path.join(self.save_folder, "red_mask.npy")
        save_mask_path_image = os.path.join(self.save_folder, "red_mask.jpg")

        if self.detect_ROI is not None:
            cv2.imwrite(save_background_path,  self.updated_image)
            np.save(save_mask_path,  self.detect_ROI)
            cv2.imwrite(save_mask_path_image, self.image_contours)
            print("saving function works")
        else:
            self.show_warning_popup("Please first detect masks")

    def show_mask(self):
        if self.image is None :
            self.show_warning_popup("Please first load Red TIF")
        elif self.thresholded_im is not None:
            self.thresholded_im, self.updated_image = GUI_functions.thresholding(self.image, self.intensity, self.brightness,
                                                                  self.threshold,self.blur_kernel, self.scene)
            self.detect_ROI, self.image_contours = red_cell_function.detect_REDROI(self.thresholded_im,self.updated_image, self.min_area, self.max_area )
            GUI_functions.load_mask_image(self.scene_mask, self.image_contours)
        else:
            self.show_warning_popup("Please first adjust thresholding")

    def load_parameters(self):
        parameter_directory = os.path.join(self.save_folder, "red_image_parameters.txt")
        if not os.path.exists(parameter_directory) :
            parameter_directory = QFileDialog.getOpenFileName(self.centralwidget, caption='Select red_image_parameters.txt file', filter="Text files (*.txt)")[0]
        try : 
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
        except :
            print('No parameters file selected.')
    
    def load_red_tif(self):
        red_frame_path = QFileDialog.getOpenFileName(self.centralwidget, caption='Select red channel tif', filter="Images(*.tif *.tiff)")[0]
        
        if red_frame_path != '' :
            self.red_frame_path = red_frame_path
            self.image = GUI_functions.load_init_image(self.scene, self.red_frame_path)
    
    def reset_UI(self):
        self.image = GUI_functions.load_init_image(self.scene, self.red_frame_path)
        self.scene_mask.clear()

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.centralwidget.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.load_pushButton.setText(_translate("MainWindow", "Load Red TIF"))
        self.load_pushButton.setStyleSheet("color: white")
        self.label_intensity.setText(_translate("MainWindow", "intensity"))
        self.label_Brightness.setText(_translate("MainWindow", "Brightness"))
        self.label_Thresholding.setText(_translate("MainWindow", "Thresholding"))
        self.label_min_area.setText(_translate("MainWindow", "Min ROI size"))
        self.label_max_area.setText(_translate("MainWindow", "Max ROI size"))
        self.label_blur_kernel.setText(_translate("MainWindow", "Blurring Kernel"))
        self.pushSave.setText(_translate("MainWindow", "Save"))
        self.pushmask.setText(_translate("MainWindow", "Detect mask"))
        self.pushsaveParameter.setText(_translate("MainWindow", "Save Param"))
        self.pushloadParameter.setText(_translate("MainWindow", "Load Param"))
        self.label_red_image.setText(_translate("MainWindow", "Red Image"))
        self.label_detected_mask.setText(_translate("MainWindow", "Detected Masks"))

if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    main_window = RedGUI()
    main_window.show()
    sys.exit(app.exec_())