from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QPen, QBrush
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QGraphicsPixmapItem
from PyQt5 import QtCore, QtWidgets
from red_Image_GUI import Red_IMAGE_Adgustment, SelectCell
from PyQt5.QtGui import QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import cm, colors
import numpy as np
from Time_series_GUI import Ui_MainWindow as TimeseriesUI  # Import your previous class
import os
# Add the previous code tab


class CustomGraphicsView_red(QGraphicsView):
    objectClicked = pyqtSignal(int)
    def __init__(self, cell_info, Chosen_Protocol, All_protocols, background_image_path,
                 corr_running, F, Time, Run,FaceMotion, Pupil, Photodiode, stimulus):
        super().__init__()
        self.corr_running = corr_running
        self.setScene(QGraphicsScene())
        self.Chosen_Protocol = Chosen_Protocol
        self.All_protocols = All_protocols
        self.cell_info = cell_info
        self.background_image_path = background_image_path
        self.setBackgroundImage()
        self.drawObjects()
        self.F = F
        self.Time = Time
        self.Run = Run
        self.FaceMotion = FaceMotion
        self.Pupil = Pupil
        self.Photodiode = Photodiode

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

        # Normalize the values to [0, 1] for colormap
        norm = colors.Normalize(vmin=min(self.corr_running), vmax=max(self.corr_running))

        # Choose a colormap (e.g., "viridis")
        colormap = cm.get_cmap('viridis')

        # Iterate through cells and draw them with color-mapped values
        for i, cell in enumerate(self.cell_info):
            # Get RGBA color from colormap
            rgba = colormap(norm(self.corr_running[i]))
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


class CustomGraphicsView_protocol(QGraphicsView):
    objectClicked = pyqtSignal(int)

    def __init__(self, cell_info, Chosen_Protocol, All_protocols, background_image_path=None):
        super().__init__()
        self.setScene(QGraphicsScene())
        self.Chosen_Protocol = Chosen_Protocol
        self.cell_info = cell_info
        self.background_image_path = background_image_path
        self.setBackgroundImage()
        self.drawObjects(self.Chosen_Protocol)

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
                color = QColor(Qt.blue)
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


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, cell_info, protocol_validity_npz, corr_running, F, Time, Run, FaceMotion, Pupil, Photodiode, stimulus, red_frame_path, save_dir):
        super().__init__()
        self.protocolValidity = protocol_validity_npz
        self.stimuliNames = protocol_validity_npz.files
        self.selectedProtocol = protocol_validity_npz[self.stimuliNames[0]]
        self.background_image_path = os.path.join(save_dir, "Mean_image_grayscale.png")
        self.computed_F = F
        self.Time = Time
        self.Run = Run
        self.FaceMotion = FaceMotion
        self.Pupil = Pupil
        self.Photodiode = Photodiode
        self.stimulus = stimulus
        
        self.setupFirstTab(cell_info, corr_running)
        self.setupSecondTab(red_frame_path, save_dir)
        Green_Cell = protocol_validity_npz["static-patch"]
        self.SetUpThirdTab(cell_info, Green_Cell, red_frame_path)

    def setupMainWindow(self):
        """Set up main window properties and the tab widget."""
        # Initialize the tab widget
        self.tabWidget = QtWidgets.QTabWidget(self)
        self.setCentralWidget(self.tabWidget)

        # Create tabs
        self.first_tab = QtWidgets.QWidget()
        self.tabWidget.addTab(self.first_tab, "Main GUI")

        self.second_tab = QtWidgets.QWidget()
        self.tabWidget.addTab(self.second_tab, "Red Image adjustment")

        self.third_tab = QtWidgets.QWidget()
        self.tabWidget.addTab(self.third_tab, "Red cell control")

        self.Data_visualisationTab()

        # Set window properties
        self.setWindowTitle("You can transfer masks between channels")
        self.setStyleSheet("""
            background-color: rgb(85, 85, 85);
            gridline-color: rgb(213, 213, 213);
            border-top-color: rgb(197, 197, 197);
        """)
        self.setGeometry(100, 100, 1500, 641)

    def setupFirstTab(self, cell_info, corr_running):
        self.setupMainWindow()
        self.setupLayouts()
        self.setupFrame1(cell_info, corr_running)
        self.setupFrame2(cell_info)
        self.setupMenuBar()
        self.setup_buttons()
        self.retranslateUi()
  
    def setupSecondTab(self, red_frame_path, save_dir):
        """Set up the second (Red Image Adjustment) tab."""
        layout = QtWidgets.QVBoxLayout(self.second_tab)

        # Create a new QMainWindow instance for Red_IMAGE_Adgustment
        self.red_image_adjustment_window = QtWidgets.QMainWindow()

        # Create an instance of Red_IMAGE_Adgustment and set it up
        self.ui_red_image_adjustment = Red_IMAGE_Adgustment()

        # Set up the UI in the QMainWindow
        self.ui_red_image_adjustment.setupUi(self.red_image_adjustment_window, save_dir, red_frame_path)

        # Embed the QMainWindow into the second tab by adding it as a widget
        layout.addWidget(self.red_image_adjustment_window.centralWidget())

    def SetUpThirdTab(self, cell_info, Green_Cell, redtif_path):
        layout_third = QtWidgets.QVBoxLayout(self.third_tab)
        self.Sellect_cell_window = QtWidgets.QMainWindow()

        self.select_cell_ui = SelectCell(cell_info, Green_Cell, redtif_path + "/red.tif")

        # Add SelectCell's central widget to the third tab layout
        layout_third.addWidget(self.select_cell_ui.centralWidget())
    
    def Data_visualisationTab(self):
        """
        Set up a new tab to include the previous GUI functionality.

        Args:
            datasets: The primary datasets for plotting.
            dataset2: The secondary datasets for plotting.
        """
        # Create a new QWidget for the tab
        self.previous_code_tab = QtWidgets.QWidget()

        # Add a layout to the tab
        layout = QtWidgets.QVBoxLayout(self.previous_code_tab)

        # Import and initialize the previous UI class

        self.previous_ui = TimeseriesUI()

        # Create a container widget for the previous code
        self.previous_code_window = QtWidgets.QMainWindow()
        self.previous_ui.setupUi(self.previous_code_window, self.computed_F, self.Time, self.Run, self.FaceMotion, self.Pupil, self.Photodiode, self.stimulus)

        # Embed the previous code into the tab
        layout.addWidget(self.previous_code_window.centralWidget())

        # Add the tab to the tab widget
        self.tabWidget.addTab(self.previous_code_tab, "Data Visualisation")
      
    def setupLayouts(self):
        """Set up primary layouts."""
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.first_tab)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_4.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_4.addLayout(self.horizontalLayout_6)

    def setupFrame1(self, cell_info, corr_running):
        """Set up the first frame with red view."""
        self.frame = self.createFrame()
        self.verticalLayout_2 = self.createVerticalLayout(self.frame)

        self.label_3 = self.createLabel("Behavioral Correlation", self.frame)
        self.verticalLayout_2.addWidget(self.label_3)

        self.lineEdit_Red = self.createLineEdit( self.frame, True)
        self.verticalLayout_2.addWidget(self.lineEdit_Red)

        self.Red_view = CustomGraphicsView_red(cell_info, self.selectedProtocol, self.protocolValidity, self.background_image_path, corr_running, self.computed_F, self.Time, self.Run, self.FaceMotion, self.Pupil, self.Photodiode, self.stimulus)
        self.verticalLayout_2.addWidget(self.Red_view)
        self.horizontalLayout_2.addWidget(self.frame)

    def setupFrame2(self, cell_info):

        self.frame_2 = self.createFrame()
        self.verticalLayout = self.createVerticalLayout(self.frame_2)

        self.label_4 = self.createLabel("Active in Protocol", self.frame_2)
        self.verticalLayout.addWidget(self.label_4)

        self.lineEdit_protocol = self.createLineEdit(self.frame_2, True)
        self.verticalLayout.addWidget(self.lineEdit_protocol)

        self.Protocol_view = CustomGraphicsView_protocol(cell_info, self.selectedProtocol, self.protocolValidity, self.background_image_path)
        self.verticalLayout.addWidget(self.Protocol_view)
        self.horizontalLayout_5.addWidget(self.frame_2)

    def setup_buttons(self):
        """
        Set up buttons and combo boxes for stimulus type selection.

        Creates two labeled sections, each with a QComboBox for selecting
        stimulus types (left and right views). Adds them to a grid layout
        within a frame, which is added to the main horizontal layout.
        """
        # Create a frame to hold the widgets
        self.frame_setting = self.createFrame()

        # Create a grid layout for the frame
        self.gridLayout = self.createGridlLayout(self.frame_setting)

        # Define common stimulus types
        stimulus_types = self.stimuliNames

        behavioral_types = [
            "Running Correlation",
            "whisking Correlation",
            "Pupil dilation Correlation"
        ]

        # Set up the left view combo box with its label
        self.addStimulusSelector(
            grid_layout=self.gridLayout,
            label_text="Stim type",
            combo_box_items=stimulus_types,
            row=0,
            col=0
        )

        # Set up the right view combo box with its label
        self.addStimulusSelector_behavioral(
            grid_layout=self.gridLayout,
            label_text="Correlation with behavioral",
            combo_box_items=behavioral_types,
            row=0,
            col=1
        )

        self.addColormapToLayout(
            grid_layout=self.gridLayout,
            colormap_name="viridis",
            min_val=min(self.Red_view.corr_running),
            max_val=max(self.Red_view.corr_running),
            row=2,
            col=0,
            colspan=1
        )

        # Add the frame containing the grid layout to the main horizontal layout
        self.horizontalLayout_4.addWidget(self.frame_setting)

    def setupMenuBar(self):
        """Set up the menu bar and actions."""
        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 954, 21))
        self.menubar.setObjectName("menubar")
        self.menubar.setStyleSheet("color: white;")
        self.setMenuBar(self.menubar)

        self.menuOpen = QtWidgets.QMenu(self.menubar)
        self.menuOpen.setObjectName("menuOpen")
        self.menuOpen.setStyleSheet("""
            QMenu {
                background-color: rgb(200, 200, 200);
                color: rgb(20, 20, 20);
            }
        """)

        self.menubar.addAction(self.menuOpen.menuAction())
        self.actionload_proccesd_file = QtWidgets.QAction(self)
        self.actionload_proccesd_file.setObjectName("actionload_proccesd_file")
        self.menuOpen.addAction(self.actionload_proccesd_file)

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
        ax = fig.add_axes([0.05, 0.7, 0.9, 0.1])
        fig.patch.set_alpha(0)

        # Create the colormap
        colormap = cm.get_cmap(colormap_name)
        norm = colors.Normalize(vmin=min_val, vmax=max_val)
        colorbar = cm.ScalarMappable(norm=norm, cmap=colormap)

        # Add the colorbar to the axis
        fig.colorbar(colorbar, cax=ax, orientation="horizontal")
        ax.set_title("Correlation", fontsize=7, color="white")

        # Create a canvas for the Matplotlib figure
        canvas = FigureCanvas(fig)

        # Add the canvas to the grid layout
        grid_layout.addWidget(canvas, row, col, 1, colspan)

    def addStimulusSelector(self, grid_layout, label_text, combo_box_items, row, col):
        """
        Helper method to add a labeled QComboBox to a grid layout.

        Args:
            grid_layout (QGridLayout): The grid layout to which the widgets are added.
            label_text (str): The text for the label.
            combo_box_items (list): A list of items to populate the QComboBox.
            row (int): The row position in the grid layout.
            col (int): The column position in the grid layout.
        """
        # Create and add a label to the grid layout
        label = self.createLabel(label_text, self.frame_setting)
        grid_layout.addWidget(label, row, col)

        # Create and add a QComboBox to the grid layout
        combo_box = QtWidgets.QComboBox(self.frame_setting)
        combo_box.addItems(combo_box_items)
        combo_box.setStyleSheet("color: white; ")
        grid_layout.addWidget(combo_box, row + 1, col)
        combo_box.currentIndexChanged.connect(lambda: self.SelectedItem(combo_box))

    def addStimulusSelector_behavioral(self, grid_layout, label_text, combo_box_items, row, col):
        """
        Helper method to add a labeled QComboBox to a grid layout.

        Args:
            grid_layout (QGridLayout): The grid layout to which the widgets are added.
            label_text (str): The text for the label.
            combo_box_items (list): A list of items to populate the QComboBox.
            row (int): The row position in the grid layout.
            col (int): The column position in the grid layout.
        """
        # Create and add a label to the grid layout
        label = self.createLabel(label_text, self.frame_setting)
        grid_layout.addWidget(label, row, col)

        # Create and add a QComboBox to the grid layout
        combo_box = QtWidgets.QComboBox(self.frame_setting)
        combo_box.addItems(combo_box_items)
        combo_box.setStyleSheet("color: white; ")
        grid_layout.addWidget(combo_box, row + 1, col)
        # combo_box.currentIndexChanged.connect(lambda: self.SelectedItem(combo_box))

    #Slots
    def SelectedItem(self, combo_box):
        """
        Slot to handle the current index change in the combo box.

        Args:
            combo_box (QComboBox): The combo box whose value changed.
        """
        print(f"Selected item: {combo_box.currentText()}")
        protocol = combo_box.currentText()

        # Check if protocolValidity exists and process it
        if hasattr(self, "protocolValidity") and self.protocolValidity :
            if protocol in self.stimuliNames:  # Ensure the key exists in the dictionary
                    self.selectedProtocol = self.protocolValidity[protocol]
            self.Protocol_view.drawObjects(self.selectedProtocol)
            # self.Red_view.drawObjects()  # Ensure this calls the correct instance
        else:
            print("protocolValidity attr is not set or empty.")

    #Tool function for UI layouts and widgets
    def createFrame(self):
        """Create a styled frame."""
        frame = QtWidgets.QFrame(self.first_tab)
        frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        frame.setFrameShadow(QtWidgets.QFrame.Raised)
        frame.setObjectName("frame")
        return frame

    def createVerticalLayout(self, parent):
        """Create a vertical layout."""
        layout = QtWidgets.QVBoxLayout(parent)
        layout.setObjectName("verticalLayout")
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

    #Translate UI elements
    def retranslateUi(self):
        """Translate UI elements."""
        _translate = QtCore.QCoreApplication.translate
        self.lineEdit_Red.setStyleSheet("color: white")
        self.lineEdit_Red.setText(_translate("MainWindow",""))
        self.setWindowTitle(_translate("MainWindow", "VISGUI"))
        self.menuOpen.setTitle(_translate("MainWindow", "Open"))
        self.lineEdit_protocol.setStyleSheet("color: white")
        self.lineEdit_protocol.setText(_translate("MainWindow",""))
        self.actionload_proccesd_file.setText(_translate("MainWindow", "load proccesd file"))

