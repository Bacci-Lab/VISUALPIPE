from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QPen, QBrush
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QGraphicsPixmapItem, QFileDialog
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QColor
import os
import numpy as np
import h5py
import glob
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import cm, colors, colormaps

from red_Image_GUI import RedImageAdjust, CategorizeCells
from Time_series_GUI import TimeSeriesUI  # Import your previous class
import GUI_functions

class CorrelationView(QGraphicsView):
    objectClicked = pyqtSignal(int)

    def __init__(self, cell_info, speed_corr, facemotion_corr, pupil_corr, background_image_path=None):
        super().__init__()
        self.speed_corr = speed_corr
        self.facemotion_corr = facemotion_corr
        self.pupil_corr = pupil_corr
        self.cmap_extremum = 1
        self.cell_info = cell_info
        self.background_image_path = background_image_path
        self.setScene(QGraphicsScene())
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

    def drawObjects(self, behavioral_corr:str="speed"):

        scene = self.scene()

        if behavioral_corr == "speed":
            var_scale = self.speed_corr
        elif behavioral_corr == "facemotion":
            var_scale = self.facemotion_corr
        elif behavioral_corr == "pupil":
            var_scale = self.pupil_corr

        # Normalize the values to [0, 1] for colormap
        if np.abs(min(var_scale)) < np.abs(max(var_scale)) :
            self.cmap_extremum = np.abs(round(max(var_scale)* 100) /100)
        else :
            self.cmap_extremum = np.abs(round(min(var_scale)* 100) /100)
        norm = colors.Normalize(vmin=-self.cmap_extremum, vmax=self.cmap_extremum)
        #norm = colors.Normalize(vmin=-1, vmax=1)

        # Choose a colormap (e.g., "viridis")
        colormap = colormaps['RdBu_r']

        # Iterate through cells and draw them with color-mapped values
        for i, cell in enumerate(self.cell_info):
            # Get RGBA color from colormap
            rgba = colormap(norm(var_scale[i]))
            # Convert RGBA to QColor
            color = QColor.fromRgbF(rgba[0], rgba[1], rgba[2], rgba[3])

            # Draw the object
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

class ResponsiveView(QGraphicsView):
    objectClicked = pyqtSignal(int)

    def __init__(self, cell_info, selected_protocol, background_image_path=None):
        super().__init__()
        self.setScene(QGraphicsScene())
        self.selected_protocol = selected_protocol
        self.cell_info = cell_info
        self.background_image_path = background_image_path
        self.setBackgroundImage()
        self.drawObjects(self.selected_protocol)

    def setBackgroundImage(self):
        if self.background_image_path:
            pixmap = QPixmap(self.background_image_path)
            if not pixmap.isNull():
                pixmap_item = QGraphicsPixmapItem(pixmap)
                pixmap_item.setZValue(-100)
                self.scene().addItem(pixmap_item)
            else:
                print("Failed to load background image:", self.background_image_path)

    def drawObjects(self, Chosen_Protocol):
        """
        Clear non-background items and draw objects based on the chosen protocol.

        Args:
            Chosen_Protocol (list): List indicating the protocol state for each cell.
        """
        scene = self.scene()

        # Remove all non-background items
        for item in scene.items():
            if item.zValue() >= 0:  # Only remove items with non-negative Z-values
                scene.removeItem(item)

        # Draw new objects based on the chosen protocol
        for i, cell in enumerate(self.cell_info):
            if Chosen_Protocol[i] == 1:
                color = QColor("coral")
                # color.setAlpha(5)
                for x, y in zip(cell['xpix'], cell['ypix']):
                    ellipse = scene.addEllipse(x, y, 1, 1, QPen(color), QBrush(color))
                    ellipse.setData(0, i)
            elif Chosen_Protocol[i] == -1:
                color = QColor("mediumaquamarine")
                # color.setAlpha(5)
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

class MainVisUI(object):
    def __init__(self, centralwidget, cell_info, background_image_path,
                 protocol_validity, speed_corr, facemotion_corr, pupil_corr):

        self.centralwidget = centralwidget

        self.cell_info = cell_info
        self.background_image_path = background_image_path

        self.protocol_validity = protocol_validity
        self.stimuliNames = protocol_validity.files
        self.selectedProtocol = [self.protocol_validity[self.stimuliNames[0]][i][0] for i in range(len(self.protocol_validity[self.stimuliNames[0]]))]
        
        self.speed_corr = speed_corr
        self.facemotion_corr = facemotion_corr
        self.pupil_corr = pupil_corr

        self.setupUi()

    @classmethod
    def from_file(cls, centralwidget, stat_filepath, background_image_path, protocol_validity_filepath, h5_filepath):
        with h5py.File(h5_filepath, "r") as f:
            speed_corr = f['Behavioral']['Correlation']['speed_corr'][()]
            facemotion_corr = f['Behavioral']['Correlation']['facemotion_corr'][()]
            pupil_corr = f['Behavioral']['Correlation']['pupil_corr'][()]
        
        cell_info = np.load(stat_filepath, allow_pickle=True)
        protocol_validity = np.load(protocol_validity_filepath, allow_pickle=True)
        
        return cls(centralwidget, cell_info, background_image_path,
                   protocol_validity, speed_corr, facemotion_corr, pupil_corr)

    def setupUi(self):
        self.setupLayouts()
        self.setupFrame1()
        self.setupFrame2()
        self.setup_colorbar()
        self.retranslateUi()
    
    def retranslateUi(self):
        """Translate UI elements."""
        _translate = QtCore.QCoreApplication.translate

    def setupLayouts(self):
        """Set up primary layouts."""
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)

    def setupFrame1(self):

        self.frame_1 = self.createFrame(self.centralwidget)
        self.verticalLayout_1 = self.createVerticalLayout(self.frame_1)

        self.label_protocol = self.createLabel("Active in Protocol", self.frame_1)
        self.verticalLayout_1.addWidget(self.label_protocol)

        self.stim_combobox = self.createComboBox(self.frame_1, self.stimuliNames, self.change_protocol)
        self.verticalLayout_1.addWidget(self.stim_combobox)

        self.stim_view = ResponsiveView(self.cell_info, self.selectedProtocol, self.background_image_path)
        self.verticalLayout_1.addWidget(self.stim_view)

        self.lineEdit_protocol = self.createLineEdit(self.frame_1, True)
        self.lineEdit_protocol.setEnabled(False)
        self.lineEdit_protocol.setText(str(int(np.sum(np.abs(self.selectedProtocol)))))
        self.lineEdit_protocol.setStyleSheet("color: white")
        self.verticalLayout_1.addWidget(self.lineEdit_protocol)

        self.horizontalLayout.addWidget(self.frame_1)

    def setupFrame2(self):
        """Set up the first frame with red view."""
        self.frame_2 = self.createFrame(self.centralwidget)
        self.verticalLayout_2 = self.createVerticalLayout(self.frame_2)

        self.label_corr = self.createLabel("Behavioral Correlation", self.frame_2)
        self.verticalLayout_2.addWidget(self.label_corr)

        behavioral_types = [
            "Running Correlation",
            "Facemotion Correlation",
            "Pupil dilation Correlation"
        ]
        self.corr_combobox = self.createComboBox(self.frame_2, behavioral_types, self.change_correlation)
        self.verticalLayout_2.addWidget(self.corr_combobox)

        self.corr_view = CorrelationView(self.cell_info, self.speed_corr, self.facemotion_corr, self.pupil_corr, self.background_image_path)
        corr_cell_num = len(np.extract(self.corr_view.speed_corr > 0.2, self.corr_view.speed_corr))
        self.verticalLayout_2.addWidget(self.corr_view)

        self.lineEdit_corr = self.createLineEdit(self.frame_2, True)
        self.lineEdit_corr.setEnabled(False)
        self.lineEdit_corr.setStyleSheet("color: white")
        self.lineEdit_corr.setText(str(corr_cell_num) + " (corr > 0.20)")
        self.verticalLayout_2.addWidget(self.lineEdit_corr)

        self.horizontalLayout.addWidget(self.frame_2)

    def setup_colorbar(self):
        # Create a frame to hold the widgets
        self.frame_colorbar = self.createFrame(self.centralwidget)

        # Create a grid layout for the frame
        self.gridLayout = self.createGridlLayout(self.frame_colorbar)

        self.addColormapToLayout(
            grid_layout=self.gridLayout,
            colormap_name="RdBu_r",
            min_val=-self.corr_view.cmap_extremum,
            max_val=self.corr_view.cmap_extremum,
            row=0,
            col=0,
            colspan=1
        )

        # Add the frame containing the grid layout to the main horizontal layout
        self.horizontalLayout.addWidget(self.frame_colorbar)
    
    def addColormapToLayout(self, grid_layout, colormap_name, min_val, max_val, row, col, colspan=1):
        """
        Add a colormap legend to the layout.

        Args:
            grid_layout (QGridLayout): The layout to which the colormap is added.
            colormap_name (str): Name of the colormap to use (e.g., "viridis").
            min_val (float): Minimum value for the colormap.
            max_val (float): Maximum value for the colormap.
            row (int): The row position in the grid layout.
            col (int): The column position in the grid layout.
            colspan (int): The number of columns to span.
        """
        # Create a Matplotlib figure and axis
        fig = Figure(figsize=(1, 0.05))  # Reduce the height here
        ax = fig.add_axes([0.25, 0.15, 0.075, 0.7])
        fig.patch.set_alpha(0)

        # Create the colormap
        colormap = colormaps[colormap_name]
        norm = colors.Normalize(vmin=min_val, vmax=max_val)
        colorbar = cm.ScalarMappable(norm=norm, cmap=colormap)

        # Add the colorbar to the axis
        cbar = fig.colorbar(colorbar, cax=ax, orientation="vertical")
        cbar.set_ticks([-max_val, -max_val/2, 0, max_val/2, max_val])
        #ax.set_title("Correlation", fontsize=8, color="white")

        # Create a canvas for the Matplotlib figure
        canvas = FigureCanvas(fig)

        # Add the canvas to the grid layout
        grid_layout.addWidget(canvas, row, col, 1, colspan)

    def reset_data_in_GUI(self):
        self.stim_view.cell_info = self.cell_info
        self.stim_view.selected_protocol = self.selectedProtocol
        self.stim_view.background_image_path = self.background_image_path
        self.stim_view.setBackgroundImage()
        self.stim_view.drawObjects(self.stim_view.selected_protocol)
        self.stim_combobox.setCurrentText(self.stimuliNames[0])
        self.lineEdit_protocol.setText(str(int(np.sum(np.abs(self.selectedProtocol)))))
        
        self.corr_view.cell_info = self.cell_info
        self.corr_view.speed_corr = self.speed_corr
        self.corr_view.facemotion_corr = self.facemotion_corr
        self.corr_view.pupil_corr = self.pupil_corr
        self.corr_view.background_image_path = self.background_image_path
        self.corr_view.setBackgroundImage()
        self.corr_view.drawObjects()
        corr_cell_num = len(np.extract(self.corr_view.speed_corr > 0.2, self.corr_view.speed_corr))
        self.lineEdit_corr.setText(str(corr_cell_num) + " (corr > 0.20)")
        self.addColormapToLayout(
            grid_layout=self.gridLayout,
            colormap_name="RdBu_r",
            min_val=-self.corr_view.cmap_extremum,
            max_val=self.corr_view.cmap_extremum,
            row=0,
            col=0,
            colspan=1
        )

    #Slots
    def change_protocol(self):
        """
        Slot to handle the current index change in the combo box.

        Args:
            combo_box (QComboBox): The combo box whose value changed.
        """
        protocol = self.stim_combobox.currentText()

        # Check if protocolValidity exists and process it
        if hasattr(self, "protocol_validity") and self.protocol_validity :
            if protocol in self.stimuliNames:  # Ensure the key exists in the dictionary
                self.selectedProtocol = [self.protocol_validity[protocol][i][0] for i in range(len(self.protocol_validity[protocol]))]
            self.lineEdit_protocol.setText(str(int(np.sum(np.abs(self.selectedProtocol)))))
            self.stim_view.drawObjects(self.selectedProtocol)
        else:
            print("protocolValidity attr is not set or empty.")

    def change_correlation(self):
        corr = self.corr_combobox.currentText()
        if corr == "Running Correlation":
            self.corr_view.drawObjects("speed")
            corr_cell_num = len(np.extract(self.corr_view.speed_corr > 0.2, self.corr_view.speed_corr))
        elif corr == "Facemotion Correlation":
            self.corr_view.drawObjects("facemotion")
            corr_cell_num = len(np.extract(self.corr_view.facemotion_corr > 0.2, self.corr_view.facemotion_corr))
        elif corr == "Pupil dilation Correlation":
            self.corr_view.drawObjects("pupil")
            corr_cell_num = len(np.extract(self.corr_view.pupil_corr > 0.2, self.corr_view.pupil_corr))
        else :
            raise Exception("Text doesn't match with combo box.")
        
        self.lineEdit_corr.setText(str(corr_cell_num) + " (corr > 0.20)")
        self.addColormapToLayout(
            grid_layout=self.gridLayout,
            colormap_name="RdBu_r",
            min_val=-self.corr_view.cmap_extremum,
            max_val=self.corr_view.cmap_extremum,
            row=0,
            col=0,
            colspan=1
        )

    #Tool function for UI layouts and widgets
    def createFrame(self, tab):
        """Create a styled frame."""
        frame = QtWidgets.QFrame(tab)
        frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        frame.setFrameShadow(QtWidgets.QFrame.Raised)
        frame.setObjectName("frame")
        return frame

    def createVerticalLayout(self, parent, name=None):
        """Create a vertical layout."""
        layout = QtWidgets.QVBoxLayout(parent)
        if name is not None :
            layout.setObjectName(name)
        return layout
    
    def createGridlLayout(self, parent):
        """Create a vertical layout."""
        layout = QtWidgets.QGridLayout(parent)
        layout.setObjectName("GridLayout")
        return layout

    def createLabel(self, text, parent):
        """Create a styled label."""
        label = QtWidgets.QLabel(parent)
        label.setAccessibleName("")
        label.setText(text)
        label.setStyleSheet("color: rgb(167, 167, 167);")
        label.setTextFormat(QtCore.Qt.RichText)
        label.setAlignment(QtCore.Qt.AlignCenter)
        return label

    def createLineEdit(self, parent, enabled=True):
        """Create a line edit."""
        line_edit = QtWidgets.QLineEdit(parent)
        line_edit.setEnabled(enabled)
        return line_edit

    def createComboBox(self, parent, combo_box_items, func):
        combo_box = QtWidgets.QComboBox(parent)
        combo_box.addItems(combo_box_items)
        combo_box.setStyleSheet("color: white; ")
        combo_box.currentIndexChanged.connect(func)
        return combo_box
    
class VisualizationGUI(QtWidgets.QMainWindow):
    def __init__(self, save_folder,
                 cell_info, ops, background_image_path, 
                 protocol_validity, 
                 speed_corr, facemotion_corr, pupil_corr, 
                 fluorescence, time, speed, facemotion, pupil, photodiode, stimuli_intervals,
                 red_tif_path=None):
        
        super().__init__(flags=Qt.WindowStaysOnTopHint)

        if speed_corr is None :
            speed_corr = np.zeros(len(fluorescence))
        if facemotion_corr is None:
            facemotion_corr = np.zeros(len(fluorescence))
        if pupil_corr is None:
            pupil_corr = np.zeros(len(fluorescence))

        self.save_folder = save_folder
        self.cell_info = cell_info
        self.ops = ops
        self.background_image_path = background_image_path 
        self.protocol_validity = protocol_validity
        self.fluorescence = fluorescence
        self.time = time
        self.speed = speed
        self.facemotion = facemotion
        self.pupil = pupil
        self.photodiode = photodiode
        self.stimuli_intervals = stimuli_intervals
        self.speed_corr = speed_corr
        self.facemotion_corr = facemotion_corr
        self.pupil_corr = pupil_corr
        self.red_tif_path = red_tif_path

        self.setupUi()

    @classmethod
    def from_file(cls, save_folder, stat_filepath, ops_filepath, background_image_path, protocol_validity_filepath, h5_filepath, visual_stim_filepath, red_tif_path):

        cell_info = np.load(stat_filepath, allow_pickle=True)
        ops = np.load(ops_filepath, allow_pickle=True).item()
        protocol_validity = np.load(protocol_validity_filepath, allow_pickle=True)

        with h5py.File(h5_filepath, "r") as f:
            time_stamps = f['Ca_imaging']['Time'][()]
            dFoF0 = f['Ca_imaging']['full_trace']['dFoF0'][()]

            speed = f['Behavioral']['Speed'][()]
            facemotion = f['Behavioral']['FaceMotion'][()]
            pupil = f['Behavioral']['Pupil'][()]
            photodiode = f['Behavioral']['Photodiode'][()]

            time_onset = f['Stimuli']['time_onset'][()]

            speed_corr = f['Behavioral']['Correlation']['speed_corr'][()]
            facemotion_corr = f['Behavioral']['Correlation']['facemotion_corr'][()]
            pupil_corr = f['Behavioral']['Correlation']['pupil_corr'][()]
        
        if np.sum(np.isnan(speed_corr)) == len(dFoF0):
            speed_corr = np.zeros(len(dFoF0))
        if np.sum(np.isnan(facemotion_corr)) == len(dFoF0):
            facemotion_corr = np.zeros(len(dFoF0))
        if np.sum(np.isnan(pupil_corr)) == len(dFoF0):
            pupil_corr = np.zeros(len(dFoF0))

        print(facemotion_corr)
        # Load visual stimuli data
        visual_stim = np.load(visual_stim_filepath, allow_pickle=True).item()
        duration = visual_stim['time_duration']
        stim_time_period = [time_onset, list(time_onset + duration)]

        dFoF0_norm = General_functions.scale_trace(dFoF0, axis=1)
        
        return cls(save_folder, cell_info, ops, background_image_path, 
                   protocol_validity,
                   speed_corr, facemotion_corr, pupil_corr, 
                   dFoF0_norm, time_stamps, speed, facemotion, pupil, photodiode, stim_time_period,
                   red_tif_path)
    
    def setupUi(self):
        """Set up main window properties and the tab widget."""
        
        #------------------------- Set window properties -------------------------
        self.setWindowTitle("Visualization GUI")
        self.setStyleSheet("""
            background-color: rgb(85, 85, 85);
            gridline-color: rgb(213, 213, 213);
            border-top-color: rgb(197, 197, 197);
        """)
        self.aw, self.ah = 1500, 750
        self.x_margins = 50
        self.y_margins = 50
        h = 20
        self.y_spacing = 30
        self.setGeometry(500, 50, self.aw, self.ah)
        
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.setCentralWidget(self.centralwidget)

        self.setupMenuBar(self, h)

        #------------------------- Tabs -------------------------
        # Initialize the tab widget
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.setCentralWidget(self.tabWidget)
        self.tabWidget.setGeometry(self.x_margins, 
                                   self.y_margins + h + self.y_spacing, 
                                   self.aw - 2*self.x_margins, 
                                   self.ah - 2*self.y_margins - h - self.y_spacing)

        # Create tabs
        self.first_tab = QtWidgets.QWidget(self.centralwidget)
        self.tabWidget.addTab(self.first_tab, "Main GUI")

        self.second_tab = QtWidgets.QWidget(self.centralwidget)
        self.tabWidget.addTab(self.second_tab, "Red Image Adjustment")

        self.third_tab = QtWidgets.QWidget(self.centralwidget)
        self.tabWidget.addTab(self.third_tab, "Categorize Cells")

        self.fourth_tab = QtWidgets.QWidget(self.centralwidget)
        self.tabWidget.addTab(self.fourth_tab, "Data Visualisation")

        self.main_vis_ui = MainVisUI(self.first_tab, self.cell_info, self.background_image_path,
                                     self.protocol_validity, self.speed_corr, self.facemotion_corr, self.pupil_corr)
        self.red_img_adjust_ui = RedImageAdjust(self.second_tab, self.save_folder, self.red_tif_path)
        self.categorize_cells_ui = CategorizeCells(self.third_tab, self.save_folder, self.cell_info, self.ops)
        self.data_vis_ui = TimeSeriesUI(self.fourth_tab, self.fluorescence, self.time, self.speed, self.facemotion, self.pupil, self.photodiode, self.stimuli_intervals)

        self.retranslateUi()

    def setupMenuBar(self, parent, h):
        """Set up the menu bar and actions."""
        self.menubar = QtWidgets.QMenuBar(parent)
        self.menubar.setGeometry(QtCore.QRect(self.x_margins, self.y_margins, self.aw - 2*self.x_margins, h))
        self.menubar.setObjectName("menubar")
        self.menubar.setStyleSheet("color: white;")
        self.setMenuBar(self.menubar)

        self.button_load_data = QtWidgets.QAction(self)
        self.button_load_data.triggered.connect(self.menu_button_clicked)
        self.menubar.addAction(self.button_load_data)

    #Slots
    def menu_button_clicked(self):
        
        save_folder = QFileDialog.getExistingDirectory(self, caption='Select folder containing output data')

        if save_folder is not None and save_folder!='':
            
            save_folder, stat_filepath, ops_filepath, background_image_path, protocol_validity_filepath, h5_filepath, visual_stim_filepath, red_tif_path = self.get_filepaths(save_folder)
            
            self.set_attr_from_path(save_folder, stat_filepath, ops_filepath, background_image_path, protocol_validity_filepath, h5_filepath, visual_stim_filepath, red_tif_path)

            self.update_tabs()

            return 0
        
        else :
            return -1

    def get_filepaths(self, save_folder) :

        path = Path(save_folder)
        base_path = path.parent.absolute()

        unique_id, _, _, _ = file.get_metadata(base_path)
        id_version = save_folder.split('_')[5]
        
        # Load stat file
        filename = "_".join([unique_id, id_version, 'stat.npy'])
        stat_filepath = os.path.join(save_folder, filename)
        if not os.path.exists(stat_filepath) :
            stat_filepath = QFileDialog.getOpenFileName(self.centralwidget, caption='Select stat.npy file', filter="npy(*.npy)")[0]
        
        # Load ops file
        tseries = [f for f in os.listdir(base_path) if f.startswith("TSeries")]
        if len(tseries) == 0 :
            print("No Tseries folder found in the base directory.")
            directory = base_path
        else:
            directory = os.path.join(base_path, tseries[0])
        ops_filepath = os.path.join(directory, "suite2p", "plane0", "ops.npy")
        if not os.path.exists(ops_filepath) :
            ops_filepath = QFileDialog.getOpenFileName(self.centralwidget, caption='Select ops.npy file', filter="npy(*.npy)")[0]
        
        # Green channel background image filepath
        background_image_path = os.path.join(base_path, "Mean_image_grayscale.png")
        if not os.path.exists(background_image_path) :
            background_image_path = QFileDialog.getOpenFileName(self.centralwidget, caption='Select Mean_image_grayscale.png', filter="Images (*.png)")[0]

        # Protocol validity filepath
        filename_protocol = "_".join([unique_id, id_version, 'protocol_validity']) + ".npz"
        protocol_validity_filepath = os.path.join(save_folder, filename_protocol)
        if not os.path.exists(protocol_validity_filepath) :
            protocol_validity_filepath = QFileDialog.getOpenFileName(self.centralwidget, caption='Select protocol validity file', filter="npz(*.npz)")[0]
        
        # HDF5 filepath
        filename_h5 = "_".join([unique_id, id_version, 'postprocessing']) + ".h5"
        h5_filepath = os.path.join(save_folder, filename_h5)
        if not os.path.exists(h5_filepath) :
            h5_filepath = QFileDialog.getOpenFileName(self.centralwidget, caption='Select HDF5 file', filter="HDF5(*.h5)")[0]

        # Visual stimuli filepath
        visual_stim_filepath = os.path.join(base_path, "visual-stim.npy")
        if not os.path.exists(visual_stim_filepath) :
            visual_stim_filepath = QFileDialog.getOpenFileName(self.centralwidget, caption='Select visual-stim.npy file', filter="npy(*.npy)")[0]
        
        # Red TIF filepath
        red_tif_path = self.get_red_channel_path(base_path)

        return save_folder, stat_filepath, ops_filepath, background_image_path, protocol_validity_filepath, h5_filepath, visual_stim_filepath, red_tif_path

    def set_attr_from_path(self, save_folder, stat_filepath, ops_filepath, background_image_path, protocol_validity_filepath, h5_filepath, visual_stim_filepath, red_tif_path):

        cell_info = np.load(stat_filepath, allow_pickle=True)
        ops = np.load(ops_filepath, allow_pickle=True).item()
        protocol_validity = np.load(protocol_validity_filepath, allow_pickle=True)

        with h5py.File(h5_filepath, "r") as f:
            time_stamps = f['Ca_imaging']['Time'][()]
            dFoF0 = f['Ca_imaging']['full_trace']['dFoF0'][()]

            speed = f['Behavioral']['Speed'][()]
            facemotion = f['Behavioral']['FaceMotion'][()]
            pupil = f['Behavioral']['Pupil'][()]
            photodiode = f['Behavioral']['Photodiode'][()]

            time_onset = f['Stimuli']['time_onset'][()]

            speed_corr = f['Behavioral']['Correlation']['speed_corr'][()]
            facemotion_corr = f['Behavioral']['Correlation']['facemotion_corr'][()]
            pupil_corr = f['Behavioral']['Correlation']['pupil_corr'][()]
        
        # Load visual stimuli data
        visual_stim = np.load(visual_stim_filepath, allow_pickle=True).item()
        duration = visual_stim['time_duration']
        stimuli_intervals = [time_onset, list(time_onset + duration)]

        dFoF0_norm = General_functions.scale_trace(dFoF0, axis=1)
        
        self.save_folder = save_folder
        self.cell_info = cell_info
        self.ops = ops
        self.background_image_path = background_image_path 
        self.protocol_validity = protocol_validity
        self.fluorescence = dFoF0_norm
        self.time = time_stamps
        self.speed = speed
        self.facemotion = facemotion
        self.pupil = pupil
        self.photodiode = photodiode
        self.stimuli_intervals = stimuli_intervals
        self.speed_corr = speed_corr
        self.facemotion_corr = facemotion_corr
        self.pupil_corr = pupil_corr
        self.red_tif_path = red_tif_path
    
    def update_tabs(self) :
        self.main_vis_ui.cell_info = self.cell_info
        self.main_vis_ui.background_image_path = self.background_image_path
        self.main_vis_ui.protocol_validity = self.protocol_validity
        self.main_vis_ui.stimuliNames = self.protocol_validity.files
        self.main_vis_ui.selectedProtocol = [self.protocol_validity[self.main_vis_ui.stimuliNames[0]][i][0] for i in range(len(self.protocol_validity[self.main_vis_ui.stimuliNames[0]]))]
        self.main_vis_ui.speed_corr = self.speed_corr
        self.main_vis_ui.facemotion_corr = self.facemotion_corr
        self.main_vis_ui.pupil_corr = self.pupil_corr
        self.main_vis_ui.reset_data_in_GUI()

        self.red_img_adjust_ui.save_folder = self.save_folder
        self.red_img_adjust_ui.red_frame_path = self.red_tif_path
        self.red_img_adjust_ui.reset_UI()

        self.categorize_cells_ui.save_folder = self.save_folder
        self.categorize_cells_ui.cell_info = self.cell_info
        self.categorize_cells_ui.ops = self.ops
        self.categorize_cells_ui.load_data_in_GUI()

        self.data_vis_ui.fluorescence = self.fluorescence
        self.data_vis_ui.time = self.time
        self.data_vis_ui.speed = self.speed
        self.data_vis_ui.facemotion = self.facemotion
        self.data_vis_ui.pupil = self.pupil
        self.data_vis_ui.photodiode = self.photodiode
        self.data_vis_ui.stimuli = self.stimuli_intervals
        self.data_vis_ui._clear_graphics_view(self.data_vis_ui.graphicsView)

    def get_red_channel_path(self, base_path):
        list_dir_red = glob.glob(os.path.join(base_path, "SingleImage-*red*"))

        if len(list_dir_red) == 1 :
            red_channel_path = list_dir_red[0]
            list_dir_red_image = glob.glob(os.path.join(red_channel_path, "*_Ch2_*.ome.tif"))
            if len(list_dir_red_image) == 1 :
                red_image_path = list_dir_red_image[0]
            else :
                red_image_path = None
        else :
            red_image_path = None
        
        return  red_image_path

    #Translate UI elements
    def retranslateUi(self):
        """Translate UI elements."""
        _translate = QtCore.QCoreApplication.translate
        self.button_load_data.setText(_translate("MainWindow", "Load Data Folder"))

if __name__ == "__main__":
    import sys
    import easygui
    from pathlib import Path

    import utils.file as file
    import General_functions

    save_folder = easygui.diropenbox(title='Select folder containing output data')
    path = Path(save_folder)
    base_path = path.parent.absolute()

    unique_id, global_protocol, experimenter, subject_id = file.get_metadata(base_path)
    id_version = save_folder.split('_')[5]
    
    # Load stat file
    filename = "_".join([unique_id, id_version, 'stat.npy'])
    stat_filepath = os.path.join(save_folder, filename)
    
    # Load ops file
    tseries = [f for f in os.listdir(base_path) if f.startswith("TSeries")]
    if len(tseries) == 0 :
        print("No Tseries folder found in the base directory.")
        directory = base_path
    else:
        directory = os.path.join(base_path, tseries[0])
    ops_filepath = os.path.join(directory, "suite2p", "plane0", "ops.npy")
    
    # Green channel background image filepath
    background_image_path = os.path.join(base_path, "Mean_image_grayscale.png")

    # Protocol validity filepath
    filename_protocol = "_".join([unique_id, id_version, 'protocol_validity_2']) + ".npz"
    protocol_validity_filepath = os.path.join(save_folder, filename_protocol)
    
    # HDF5 filepath
    filename_h5 = "_".join([unique_id, id_version, 'postprocessing']) + ".h5"
    h5_filepath = os.path.join(save_folder, filename_h5)

    # Visual stimuli filepath
    visual_stim_filepath = os.path.join(base_path, "visual-stim.npy")
    
    # Red TIF filepath
    list_dir_red = glob.glob(os.path.join(base_path, "SingleImage-*red*"))
    if len(list_dir_red) == 1 :
        red_channel_path = list_dir_red[0]
        list_dir_red_image = glob.glob(os.path.join(red_channel_path, "*_Ch2_*.ome.tif"))
        if len(list_dir_red_image) == 1 :
            red_tif_path = list_dir_red_image[0]
        else :
            red_tif_path = None
    else :
        red_tif_path = None

    # Launch App
    app = QtWidgets.QApplication(sys.argv)
    main_window = VisualizationGUI.from_file(save_folder, stat_filepath, ops_filepath, background_image_path, protocol_validity_filepath, h5_filepath, visual_stim_filepath, red_tif_path)
    main_window.show()
    app.exec_()