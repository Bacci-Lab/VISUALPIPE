import sys, os
from pathlib import Path
import json
import cv2
from PyQt5 import QtWidgets, QtCore
import numpy as np
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QColor, QPen, QBrush, QImage
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QGraphicsPixmapItem, QFileDialog
import tifffile

if __name__ == "__main__":
    sys.path.append("./src")
import visualpipe.gui.GUI_functions as GUI_functions
import visualpipe.red_channel.red_cell_function as red_cell_function
import visualpipe.utils.file as file

class RedGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__(flags=Qt.WindowStaysOnTopHint)
        self.setupUi()

    def setupUi(self):

        #------------------------- Main settings -------------------------
        self.setObjectName("MainWindow")
        self.setGeometry(500, 50, 1341, 770)
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
        self.tabWidget.setGeometry(20, 30, 1301, 700)
        self.tabWidget.setStyleSheet("background-color: rgb(104, 104, 104);\ncolor: white;")
        
        self.categorize_cells_tab = QtWidgets.QWidget(self.centralwidget)
        self.adjust_red_img_tab = QtWidgets.QWidget(self.centralwidget)

        self.tabWidget.addTab(self.categorize_cells_tab, 'Categorize cells')
        self.tabWidget.addTab(self.adjust_red_img_tab, 'Adjust Red Channel Image')
                              
        #------------------------- Tab 1 -------------------------
        self.categorize_ui = CategorizeCells(self.categorize_cells_tab)

        #------------------------- Tab 2 -------------------------
        self.red_img_adjust_ui = RedImageAdjust(self.adjust_red_img_tab)

        #--------------------------------------------------
        self.retranslateUi()
    
    def menu_button_clicked(self):
        
        output_dir = QFileDialog.getExistingDirectory(self, caption='Select folder containing output data')
        if output_dir != '':
            self.categorize_ui.output_dir = output_dir
            self.red_img_adjust_ui.save_folder = os.path.join(output_dir, 'red_channel')

            out = self.categorize_ui.update_UI()
            if out == -1:
                self.tabWidget.setCurrentIndex(1)
            
            self.red_img_adjust_ui.reset_UI()
            return 0
        
        else :
            print("No output folder selected")
            return -1
        
    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "Red-Green Channels"))
        self.button_load_data.setText(_translate("MainWindow", "Load Data Folder"))

class RedGreenView(QGraphicsView):

    objectClicked = pyqtSignal(int)

    def __init__(self, cell_info=None, only_green_cell=None, background_image_path=None):
        super().__init__()
        self.setScene(QGraphicsScene())
        self.only_green_cell = only_green_cell
        self.cell_info = cell_info
        self.background_image_path = background_image_path
        if background_image_path is not None:
            self.setBackgroundImage()
        if cell_info is not None and only_green_cell is not None: 
            self.drawObjects()

    def setBackgroundImage(self):
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

    def __init__(self, cell_info=None, only_green_cell=None, background_image_path=None):
        super().__init__()
        self.setScene(QGraphicsScene())
        self.only_green_cell = only_green_cell
        self.cell_info = cell_info
        self.background_image_path = background_image_path
        if background_image_path is not None:
            self.setBackgroundImage()
        if cell_info is not None and only_green_cell is not None: 
            self.drawObjects()

    def setBackgroundImage(self):
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
    def __init__(self, centralwidget, output_folder=None, cell_info=None, ops=None):

        self.centralwidget = centralwidget
        self.output_dir = output_folder
        self.cell_info = cell_info
        self.ops = ops

        self.save_dir = None
        self.currentBackgroundImage_left  = None
        self.currentBackgroundImage_right = None
        # Initialize tracking lists
        self.currentRedObjects = []
        self.currentGreenObjects = []
        self.red_cell_num = 0
        self.green_cell_num = 0

        self.setupUi()

        if self.output_dir is not None:
            self.update_UI()    

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

        self.radio_hbox_left = QtWidgets.QHBoxLayout()
        self.verticalLayout_red_green.addLayout(self.radio_hbox_left)
        self.radio_group_left = GUI_functions.add_radio_button_group(self.left_frame, ["Adjusted Red Channel", "Red Channel", "Green Channel"], self.radio_hbox_left, self.change_background_left, 0)

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

        self.radio_hbox_right = QtWidgets.QHBoxLayout()
        self.verticalLayout_only_green.addLayout(self.radio_hbox_right)
        self.radio_group_right = GUI_functions.add_radio_button_group(self.right_frame, ["Adjusted Red Channel", "Red Channel", "Green Channel"], self.radio_hbox_right, self.change_background_right, 0)

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

    def update_UI(self):
        self.save_dir = os.path.join(self.output_dir, 'red_channel')
        self.currentBackgroundImage_left  = os.path.join(self.save_dir, "adjusted_image.png")
        self.currentBackgroundImage_right = os.path.join(self.save_dir, "adjusted_image.png")

        out = self.set_cell_info()
        if out == -1 :
            raise Exception("No valid stat.npy file.")
        
        out = self.set_ops()
        if out == -1 :
            raise Exception("No valid ops.npy file.")
        
        # Load data in GUI
        out = self.load_data_in_GUI()

        return out
    
    def set_ops(self):

        path = Path(self.output_dir)
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
            print(f"{suite2p_path} doesn't exists. Select ops.npy file.")
            ops_path = QFileDialog.getOpenFileName(self.centralwidget, caption='Select ops.npy file', filter="npy(*.npy)")[0]
            if 'ops' in ops_path: 
                self.ops = np.load(ops_path, allow_pickle=True).item()
            else :
                print(f"Selected file not valid ({ops_path}).")
                return -1

        return 0
    
    def set_cell_info(self):

        path = Path(self.output_dir)
        base_path = path.parent.absolute()
        unique_id, _, _, _ = file.get_metadata(base_path)
        foldername = os.path.basename(self.output_dir)
        id_version = foldername.split('_')[5]
        
        # Load stat file
        filename = "_".join([unique_id, id_version, 'stat.npy'])
        stat_path = os.path.join(self.output_dir, filename)
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
        if not os.path.exists(self.save_dir) : os.mkdir(self.save_dir) # Create save_dir
        save_direction1 = os.path.join(self.save_dir, 'red_green_cells.npy')
        save_direction2 = os.path.join(self.save_dir, 'only_green.npy')
        np.save(save_direction1, self.currentRedObjects, allow_pickle=True)
        np.save(save_direction2, self.currentGreenObjects, allow_pickle=True)

    def load_data_in_GUI(self):

        if self.save_dir is None :
            GUI_functions.show_warning_popup(self.centralwidget, "Please first select output folder.")
            return -2
        
        # Load red_green_cells.npy file
        red_green_filepath = os.path.join(self.save_dir, 'red_green_cells.npy')
        if os.path.exists(red_green_filepath) :
            data = np.load(red_green_filepath, allow_pickle=True)
            only_green_cells = np.ones(len(self.cell_info))
            only_green_cells[data] = 0
        else :
            red_masks_dir = os.path.join(self.save_dir, "red_mask.npy")
            if os.path.exists(red_masks_dir) :
                only_green_cells = red_cell_function.get_GreenMask(self.save_dir, self.ops, self.cell_info)
            else : 
                print("red_mask.npy doesn't exist. Compute masks first.")
                return -1
        
        # Check existence of adjusted_image.png file
        background_image_path = os.path.join(self.save_dir, 'adjusted_image.png')
        if not os.path.exists(background_image_path):
            print("adjusted_image.png doesn't exist. Compute masks first.")
            return -1
            
        self.red_view.background_image_path = self.currentBackgroundImage_left
        self.red_view.only_green_cell = only_green_cells
        self.red_view.cell_info = self.cell_info
        self.green_view.background_image_path = self.currentBackgroundImage_right
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

    def change_background_right(self, idx):
        if idx == 0:
            self.currentBackgroundImage_right = os.path.join(self.save_dir, "adjusted_image.png")
        elif idx == 1:
            self.currentBackgroundImage_right = os.path.join(self.save_dir, "red_image.png")
        elif idx == 2:
            self.currentBackgroundImage_right = os.path.join(self.save_dir, "Suite2pMeanImage.png")

        if os.path.exists(self.currentBackgroundImage_right):
            self.green_view.background_image_path = self.currentBackgroundImage_right
            self.green_view.setBackgroundImage()
        else :
            self.show_warning_popup(f"{self.currentBackgroundImage_right} not found.")

    def change_background_left(self, idx):
        if idx == 0:
            self.currentBackgroundImage_left = os.path.join(self.save_dir, "adjusted_image.png")
        elif idx == 1:
            self.currentBackgroundImage_left = os.path.join(self.save_dir, "red_image.png")
        elif idx == 2:
            self.currentBackgroundImage_left = os.path.join(self.save_dir, "Suite2pMeanImage.png")

        if os.path.exists(self.currentBackgroundImage_left):
            self.red_view.background_image_path = self.currentBackgroundImage_left
            self.red_view.setBackgroundImage()
        else :
            self.show_warning_popup(f"{self.currentBackgroundImage_left} not found.")

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

    def __init__(self, centralwidget, save_folder=None, red_frame_path=None, green_image=None, center=False):
        self.centralwidget = centralwidget
        self.save_folder = save_folder
        self.red_frame_path = red_frame_path
        self.green_image = green_image

        self.initialize_attributes()
        self.image = None
        self.present_image = None
        self.enabled = False

        self.setupUi(center)
        
        if save_folder is not None :
            if self.red_frame_path is not None :
                self.image = self.load_init_image()
                self.present_image = self.image
                self.enabled = True

            if green_image is None :
                self.set_green_image()

    def setupUi(self, center):
        
        #------------------------- Main settings -------------------------
        label_color = "color: rgb(255, 255, 255);"
        main_window = self.centralwidget
        if center :
            while main_window.objectName() != "MainWindow" :
                main_window = main_window.parent()
            aw = 1280
            parent_w = main_window.width()
            margin = (parent_w-aw)//2
        else :
            margin = 0
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
        self.graphicsView.setGeometry(QtCore.QRect(margin, 30, 520, 520))
        self.graphicsView.setObjectName("graphicsView")
        self.scene = QtWidgets.QGraphicsScene()
        self.graphicsView.setScene(self.scene)
        self.label_red_image = GUI_functions.setup_labels(self.frame,(margin, 3),(520, 23), label_color)

        #------------------------- Second graphic view -------------------------
        self.mask_graphicsView = QtWidgets.QGraphicsView(self.frame)
        self.mask_graphicsView.setGeometry(QtCore.QRect(margin + 761, 30, 520, 520))
        self.mask_graphicsView.setObjectName("mask_graphicsView")
        self.scene_mask = QtWidgets.QGraphicsScene()
        self.mask_graphicsView.setScene(self.scene_mask)
        self.label_detected_mask = GUI_functions.setup_labels(self.frame,(margin + 761, 3),(520, 23), label_color)

        #------------------------------------- Vessel settings -----------------------------------
        self.group_vessel = QtWidgets.QGroupBox("Vessel settings", self.frame)
        self.group_vessel.setStyleSheet("color: white;")
        self.group_vessel.setGeometry(margin +545, 30, 190, 100)

        # 2) give it a vertical layout
        vbox = QtWidgets.QVBoxLayout(self.group_vessel)
        vbox.setContentsMargins(10, 10, 10, 10)

        self.vessel_projection = QtWidgets.QCheckBox("Show vessels", self.group_vessel)
        self.vessel_projection.toggled.connect(self.on_toggle_vessel)
        self.vessel_projection.setStyleSheet("color: white;")
        self.vessel_projection.setEnabled(self.enabled)
        vbox.addWidget(self.vessel_projection)

        self.slider_vessel_area_thr, self.label_vessel_area_thr_value = GUI_functions.add_slider_row(
            group=self.group_vessel,
            vbox=vbox,
            text="Area thr:",
            minimum=0.0,
            maximum=255.0,
            initial=self.vessel_area_thr_value,
            step= 1.0,
            on_value_changed=self.vessel_area_thr,
            enabled=self.enabled
        )

        self.slider_vessel_th_ratio, self.label_vessel_value = GUI_functions.add_slider_row(
            group=self.group_vessel,
            vbox=vbox,
            text="Vessel th",
            minimum     = 1.0,
            maximum     = 3.0,
            initial    = self.vessel_threshold,
            step   = 0.1,
            on_value_changed = self.vessel_threshold_value,
            enabled=self.enabled
        )

        vbox.addStretch()

        # -------------------------------------- Image settings ------------------------------------
        self.group_Image = QtWidgets.QGroupBox("Image settings", self.frame)
        self.group_Image.setStyleSheet("color: white;")
        self.group_Image.setGeometry(margin + 545, 140, 190, 100)

        # 2) give it a vertical layout
        vbox2 = QtWidgets.QVBoxLayout(self.group_Image)
        vbox2.setContentsMargins(10, 20, 10, 10)

        self.slider_intensity, self.label_intensity_value = GUI_functions.add_slider_row(
            group=self.group_Image,
            vbox=vbox2,
            text="Intensity:",
            minimum= 0.30,
            maximum= 1.00,
            initial=self.intensity,
            step= 0.01,
            on_value_changed = self.intensity_value,
            enabled=self.enabled
        )

        self.slider_blur, self.label_blur_value = GUI_functions.add_slider_row(
            group=self.group_Image,
            vbox=vbox2,
            text="Blur kernel",
            minimum     = 0,
            maximum     = 7,
            initial    = (self.blur_kernel-1)//2,
            step   = 1,
            on_value_changed = self.blur_values,
            enabled=self.enabled
        )

        vbox2.addStretch()

        # --------------------------------------- ROI settings --------------------------------------
        self.group_ROI = QtWidgets.QGroupBox("ROI settings", self.frame)
        self.group_ROI.setStyleSheet("color: white;")
        self.group_ROI.setGeometry(margin + 545, 250, 190, 180)

        # 2) give it a vertical layout
        vbox3 = QtWidgets.QVBoxLayout(self.group_ROI)
        vbox3.setContentsMargins(10, 20, 10, 10)

        self.slider_thresh, self.label_threshold_value = GUI_functions.add_slider_row(
            group=self.group_ROI,
            vbox=vbox3,
            text="Threshold:",
            minimum= 0.01,
            maximum= 0.30,
            initial=self.general_threshold,
            step= 0.01,
            on_value_changed = self.set_threshold,
            enabled=self.enabled
        )

        self.slider_min_area, self.label_min_value = GUI_functions.add_slider_row(
            group=self.group_ROI,
            vbox=vbox3,
            text="Min ROI",
            minimum     = 1,
            maximum     = 300,
            initial    = self.min_area,
            step   = 1,
            on_value_changed = self.set_min,
            enabled=self.enabled
        )

        self.slider_max_area, self.label_max_value = GUI_functions.add_slider_row(
            group=self.group_ROI,
            vbox=vbox3,
            text="Max ROI",
            minimum     = 1,
            maximum     = 1000,
            initial    = self.max_area,
            step   = 1,
            on_value_changed = self.set_max,
            enabled=self.enabled
        )

        self.slider_overlap, self.label_overlap_value = GUI_functions.add_slider_row(
            group=self.group_ROI,
            vbox=vbox3,
            text="ROI Overlap",
            minimum     = 0.1,
            maximum     = 1.0,
            initial    = self.overlap_value,
            step   = 0.1,
            on_value_changed = self.overlap,
            enabled=self.enabled
        )

        self.slider_min_sigma, self.label_min_sigma_value = GUI_functions.add_slider_row(
            group=self.group_ROI,
            vbox=vbox3,
            text="sigma",
            minimum     = 1.0,
            maximum     = 20.0,
            initial    = 5.0,
            step   = 1,
            on_value_changed = self.min_sigma_value,
            enabled=self.enabled
        )

        vbox3.addStretch()
        
        #--------------------------------------------- PushButton -------------------------------------
        self.registration_pushButton = GUI_functions.setup_pushButton(self.frame, (margin+540, 460), (95, 23), label_color, self.registration, self.enabled)
        self.undo_reg_pushButton = GUI_functions.setup_pushButton(self.frame, (margin+645, 460), (95, 23), label_color, self.undo_registration, self.enabled)
        self.pushloadParameter = GUI_functions.setup_pushButton(self.frame, (margin+540, 490), (95, 23),  label_color, self.load_parameters, self.enabled)
        self.pushsaveParameter = GUI_functions.setup_pushButton(self.frame, (margin+645, 490), (95, 23),  label_color, self.save_parameters, self.enabled)
        self.pushmask = GUI_functions.setup_pushButton(self.frame, (margin+540, 520), (95, 23), label_color, self.show_mask, self.enabled)
        self.pushSave = GUI_functions.setup_pushButton(self.frame, (margin+645, 520), (95, 23), label_color, self.save_image, self.enabled)

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self.centralwidget)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.centralwidget.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.load_pushButton.setText(_translate("MainWindow", "Load Red TIF"))
        self.load_pushButton.setStyleSheet("color: white")
        self.registration_pushButton.setText(_translate("MainWindow", "Registration"))
        self.undo_reg_pushButton.setText(_translate("MainWindow", "Undo Registration"))
        self.pushSave.setText(_translate("MainWindow", "Save"))
        self.pushmask.setText(_translate("MainWindow", "Detect mask"))
        self.pushsaveParameter.setText(_translate("MainWindow", "Save Param"))
        self.pushloadParameter.setText(_translate("MainWindow", "Load Param"))
        self.label_red_image.setText(_translate("MainWindow", "Red Image"))
        self.label_detected_mask.setText(_translate("MainWindow", "Detected Masks"))

    def initialize_attributes(self):
        self.intensity = 0.35
        self.vessel_threshold = 1.3
        self.min_sigma = 5.0
        self.min_area = 20
        self.max_area = 500
        self.general_threshold = 0.07
        self.blur_kernel = 3
        self.vessel_area_thr_value = 30
        self.overlap_value = 0.7
        self.detect_ROI = None
        self.image_contours = None
        self.adjust_image_exist = False
        self.registration_flag = False

    def set_green_image(self, img_type='meanImg'):

        path = Path(self.save_folder)
        base_path = path.parent.absolute().parent.absolute()
        mean_image_path = os.path.join(base_path, 'Mean_image_grayscale.png')

        if os.path.exists(mean_image_path) :
            mean_img = cv2.imread(mean_image_path, cv2.IMREAD_GRAYSCALE)
        else :
            # Load ops file
            print('Searching for ops file to load green channel mean image.')
            tseries = [f for f in os.listdir(base_path) if f.startswith("TSeries")]
            if len(tseries) == 0 :
                print("No Tseries folder found in the base directory.")
                directory = base_path
            else:
                directory = os.path.join(base_path, tseries[0])
            
            suite2p_path = os.path.join(directory, "suite2p", "plane0")
            if os.path.exists(suite2p_path) :
                ops = np.load(os.path.join(suite2p_path, "ops.npy"), allow_pickle=True).item()
            else :
                print(f"{suite2p_path} doesn't exists. Select ops.npy file.")
                ops_path = QFileDialog.getOpenFileName(self.centralwidget, caption='Select ops.npy file', filter="npy(*.npy)")[0]
                if 'ops' in ops_path: 
                    ops = np.load(ops_path, allow_pickle=True).item()
                else :
                    print(f"Selected file not valid ({ops_path}).")
                    return -1
            
            mean_img = ops.get(img_type, None)

        print('Green channel mean image loaded.')

        self.green_image = red_cell_function.percentile_contrast_stretch(mean_img)
        save_green_image_path = os.path.join(self.save_folder, "Suite2pMeanImage.png")
        red_cell_function.save_as_gray_png(self.green_image, save_green_image_path)

        return 0
    
    def set_enabled_elements(self, enabled:bool=True):
        self.vessel_projection.setEnabled(enabled)
        self.slider_vessel_area_thr.setEnabled(enabled)
        self.slider_vessel_th_ratio.setEnabled(enabled)
        self.slider_intensity.setEnabled(enabled)
        self.slider_blur.setEnabled(enabled)
        self.slider_thresh.setEnabled(enabled)
        self.slider_min_area.setEnabled(enabled)
        self.slider_max_area.setEnabled(enabled)
        self.slider_min_sigma.setEnabled(enabled)
        self.slider_overlap.setEnabled(enabled)
        self.registration_pushButton.setEnabled(enabled)
        self.undo_reg_pushButton.setEnabled(enabled)
        self.pushloadParameter.setEnabled(enabled)
        self.pushsaveParameter.setEnabled(enabled)
        self.pushmask.setEnabled(enabled)
        self.pushSave.setEnabled(enabled)

    def load_init_image(self):
        """
        Load and normalize a TIFF image, display it in the given Qt graphics scene,
        and return the processed 8-bit image array.
        """

        if self.red_frame_path is not None and os.path.exists(self.red_frame_path) :

            # Read the TIFF file into a NumPy array (could be 16-bit or higher)
            image = tifffile.imread(self.red_frame_path)
            if len(image.shape) == 3:
                image = image[:, :, 0]
            image = red_cell_function.compute_intensity_bounds(image, self.intensity)

            # Convert the NumPy array into a QPixmap for Qt display
            pixmap = GUI_functions.cv2_to_pixmap(image)

            # Clear any existing items in the QGraphicsScene
            self.scene.clear()

            # Create a new pixmap item and add it to the scene
            item = QtWidgets.QGraphicsPixmapItem(pixmap)
            self.scene.addItem(item)

            # Return the normalized 8-bit image for further processing
            return image
        else : 
            self.scene.clear()
            return None
    
    def update_image(self):
        self.scene.clear()
        adjusted_image = red_cell_function.compute_intensity_bounds(self.image, self.intensity)
        if isinstance(self.blur_kernel, tuple):
            blur_kernel = self.blur_kernel
        else:
            blur_kernel = (self.blur_kernel, self.blur_kernel)
        blurred = cv2.GaussianBlur(adjusted_image, blur_kernel, 0)
        pixmap = GUI_functions.cv2_to_pixmap(blurred)
        item = QtWidgets.QGraphicsPixmapItem(pixmap)
        self.scene.addItem(item)

        return blurred

    def load_mask_image(self):
        height, width, channel = self.image_contours.shape
        bytes_per_line = 3 * width
        q_image = QImage(self.image_contours.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.scene_mask.clear()
        item = QtWidgets.QGraphicsPixmapItem(pixmap)
        self.scene_mask.addItem(item)

    def reset_UI(self):
        self.red_frame_path = None
        self.image = self.load_init_image()
        self.present_image = self.image
        self.scene_mask.clear()
        self.set_green_image()

        self.set_enabled_elements(False)
    
    #--------------------------------------------- Signal handlers ---------------------------------------------

    def load_red_tif(self):

        if self.save_folder is None :
            GUI_functions.show_warning_popup(self.centralwidget, "Please first select output folder.")
            return -1
        
        red_frame_path = QFileDialog.getOpenFileName(self.centralwidget, caption='Select red channel tif', filter="Images(*.tif *.tiff)")[0]
        
        if red_frame_path != '' :
            self.red_frame_path = red_frame_path
            self.image = self.load_init_image()
            self.present_image = self.image

        self.set_enabled_elements()

        return 0
        
    def on_toggle_vessel(self, enabled: bool):
        """
        Called when the "Show vessels" checkbox changes.
        If enabled, overlay the vessel_mask_area onto self.scene_mask.
        Otherwise clear it (or redraw your base image).
        """
        if enabled:
            self.vessel_boundary = red_cell_function.plot_vessel(self.present_image, self.vessel_area_thr_value,
                                                                 self.blur_kernel)
            pixmap = GUI_functions.cv2_to_qpixmap(self.vessel_boundary)
            self.scene.clear()
            item = QGraphicsPixmapItem(pixmap)
            self.scene.addItem(item)
        else:
            self.scene.clear()
            self.update_image()

    def vessel_area_thr(self, value):
        self.vessel_area_thr_value = int(value)
        self.label_vessel_area_thr_value.setText(str(self.vessel_area_thr_value))

    def vessel_threshold_value(self, value):
        self.vessel_threshold = float(value)
        self.label_vessel_value.setText(f"{self.vessel_threshold:.1f}")

    def intensity_value(self, value):
        self.intensity = float(value)
        self.label_intensity_value.setText(f"{self.intensity:.1f}")
        self.present_image = self.update_image()
    
    def blur_values(self, value):
        value = (value*2) + 1
        self.blur_kernel = value
        self.label_blur_value.setText(str(self.blur_kernel))
        self.present_image = self.update_image()

    def set_threshold(self, threshold_value):
        self.general_threshold = float(threshold_value)
        self.label_threshold_value.setText(f"{self.general_threshold :.2f}")

    def set_min(self, value):
        self.min_area = int(value)
        self.label_min_value.setText(str(self.min_area))

    def set_max(self, value):
        self.max_area = int(value)
        self.label_max_value.setText(str(self.max_area))
    
    def overlap(self, v):
        self.overlap_value = float(v)
        self.label_overlap_value.setText(f"{self.overlap_value:.2f}")

    def min_sigma_value(self, v):
        self.min_sigma = int(v)
        self.label_min_sigma_value.setText(str(self.min_sigma))

    def registration(self):
        self.registration_flag = True
        shifted, _, _, _, _ = red_cell_function.image_shift(self.green_image, self.present_image, savepath=self.save_folder)
        
        save_red_shifted_path = os.path.join(self.save_folder, "shifted_red.png")
        red_cell_function.save_as_gray_png(shifted, save_red_shifted_path)
        
        self.registration_pushButton.setStyleSheet(
            "background-color: rgb(159, 181, 189); color: white;"
        )
        print("image motion is corrected")
        self.update_image()
        self.no_shift_image = self.present_image
        self.present_image = shifted

    def undo_registration(self):
        if self.registration_flag == True:
            self.registration_flag = False
            self.present_image = self.no_shift_image
            self.registration_pushButton.setStyleSheet(
                "background-color: rgb(27, 27, 27); color: white;"
            )
            self.update_image()

    def load_parameters(self):
        parameter_directory = os.path.join(self.save_folder, "red_image_parameters.txt")

        if not os.path.isfile(parameter_directory):
            GUI_functions.show_warning_popup(self.centralwidget, f"Cannot find parameter file:\n{parameter_directory}")
            return -1

        with open(parameter_directory, 'r') as file:
            data = file.read()
        json_data = json.loads(data)

        # 1) Read from JSON into self.<fields>
        self.min_area            = json_data["min_area"]
        self.max_area            = json_data["max_area"]
        self.vessel_threshold    = json_data["vessel ROI threshold"]
        self.general_threshold   = json_data["Global threshold"]
        self.vessel_area_thr_value = json_data["vessel area threshold"]
        self.intensity           = json_data["intensity"]
        self.blur_kernel         = json_data["blur kernel"]
        self.overlap_value       = json_data["ROI Overlap"]
        self.min_sigma           = json_data["sigma"]
        self.registration_flag   = json_data["Registration"]

        # 2) Now push each into its slider/label via the update methods:
        GUI_functions.update_slider(self.slider_vessel_area_thr, self.label_vessel_area_thr_value, float(self.vessel_area_thr_value), 0.0, 255.0, 1.0, "{:.0f}")
        GUI_functions.update_slider(self.slider_vessel_th_ratio, self.label_vessel_value, self.vessel_threshold, 1.0, 3.0, 0.1, '{:.1f}')

        GUI_functions.update_slider(self.slider_intensity, self.label_intensity_value, self.intensity, 0.30, 1.00, 0.01, '{:.2f}')
        GUI_functions.update_slider_blur(self.slider_blur, self.label_blur_value, float(self.blur_kernel), 0.0, 7.0, 1.0, "{:.0f}")

        GUI_functions.update_slider(self.slider_thresh, self.label_threshold_value, self.general_threshold, 0.01, 0.30, 0.01, '{:.2f}')
        GUI_functions.update_slider(self.slider_min_area, self.label_min_value, float(self.min_area), 1.0, 100.0, 1.0, "{:.0f}")
        GUI_functions.update_slider(self.slider_max_area, self.label_max_value, float(self.max_area), 1.0, 1000.0, 1.0, "{:.0f}")
        GUI_functions.update_slider(self.slider_overlap, self.label_overlap_value, self.overlap_value, 0.1, 1.0, 0.1, '{:.1f}')
        GUI_functions.update_slider(self.slider_min_sigma, self.label_min_sigma_value, float(self.min_sigma), 1.0, 20.0, 1.0, "{:.0f}")
        
        if self.registration_flag == True:
            self.registration()

    def save_parameters(self):
        parameters = {'min_area': self.min_area,
                      'max_area': self.max_area,
                      'vessel ROI threshold': self.vessel_threshold,
                      'vessel area threshold': self.vessel_area_thr_value,
                      'Global threshold': self.general_threshold,
                      'intensity': self.intensity,
                      'blur kernel': self.blur_kernel,
                      'ROI Overlap': self.overlap_value,
                      'sigma': self.min_sigma,
                      'Registration':self.registration_flag}

        if not os.path.exists(self.save_folder) : os.mkdir(self.save_folder) # Create save_folder
        savepath_parameter = os.path.join(self.save_folder, 'red_image_parameters.txt')
        with open(savepath_parameter, 'w') as file:
            file.write(json.dumps(parameters, indent = 4))
        print("parameters saved")

    def show_mask(self):
        self.detect_ROI, self.image_contours, self.updated_image\
            = red_cell_function.detect_REDROI(self.present_image, self.min_area, self.max_area,
                                              self.vessel_threshold, self.vessel_area_thr_value, 
                                              self.min_sigma, self.general_threshold,
                                              self.blur_kernel, self.overlap_value)
        self.load_mask_image()
    
    def save_image(self):
        if not os.path.exists(self.save_folder) : os.mkdir(self.save_folder) # Create save_folder
        save_background_path = os.path.join(self.save_folder, "adjusted_image.png")
        save_mask_path = os.path.join(self.save_folder, "red_mask.npy")
        save_mask_path_image = os.path.join(self.save_folder, "red_mask.png")
        save_red_image_path = os.path.join(self.save_folder, "red_image.png")

        if self.detect_ROI is not None:
            cv2.imwrite(save_background_path, self.updated_image)
            np.save(save_mask_path,  self.detect_ROI)
            cv2.imwrite(save_mask_path_image, self.image_contours)
            red_cell_function.save_as_gray_png(self.image, save_red_image_path)
            print("red masks saved")
        else:
            GUI_functions.show_warning_popup("Please first detect masks")

if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    main_window = RedGUI()
    main_window.show()
    sys.exit(app.exec_())