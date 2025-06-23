from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QGridLayout, QComboBox, QLabel, QLineEdit, QPushButton, QListView, QStatusBar, QMenuBar, QFileDialog
from PyQt5 import QtCore, QtWidgets
import sys
from PyQt5.QtGui import QIntValidator, QDoubleValidator 
import json
from PyQt5.QtCore import QStringListModel
from pathlib import Path
from red_cell_function import get_red_channel

class InputWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

    def get_inputs(self):
        """Retrieve user inputs from the first GUI."""
        return {
            "base_path": self.ui.data_directory,
            "compile_dir": self.ui.compile_directory,
            "red_image_path": self.ui.red_image_path,
            "neuropil_impact_factor": self.ui.neuropil_if,
            "bootstrap_nb_samples" : self.ui.nb_samples,
            "F0_method": self.ui.f0_method,
            "neuron_type": self.ui.neural_type,
            "starting_delay_2p": self.ui.starting_delay,
            "speed_th": self.ui.speed_th,
            "facemotion_th": self.ui.motion_th,
            "pupil_th": self.ui.pupil_th,
            "pupil_th_type" : self.ui.pupil_th_type,
            "min_run_window": self.ui.min_run_window,
            "min_as_window": self.ui.min_as_window,
            "min_rest_window": self.ui.min_rest_window,
            "protocol_ids": self.ui.protocol_numbers if hasattr(self.ui, 'protocol_numbers') else [],
            "protocol_names": self.ui.protocol_names if hasattr(self.ui, 'protocol_names') else [],
        }

class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        MainWindow.resize(457, 600)
        MainWindow.setStyleSheet("background-color: rgb(165, 165, 165);")

        self.centralwidget = QWidget(MainWindow)
        self.main_layout = QVBoxLayout(self.centralwidget)

        # First Grid Layout
        self.setup_first_grid()
        self.main_layout.addLayout(self.first_grid)

        # Third Grid Layout
        self.setup_third_grid()
        self.main_layout.addLayout(self.third_grid)

        # Second Grid Layout
        self.setup_second_grid()
        self.main_layout.addLayout(self.second_grid)

        # List View
        self.listView = QListView(self.centralwidget)
        self.main_layout.addWidget(self.listView)

        # Load Protocols Button
        self.pushButton_load_protocol = QPushButton("Load Protocols", self.centralwidget)
        self.pushButton_load_protocol.clicked.connect(self.loadprotocol)
        self.main_layout.addWidget(self.pushButton_load_protocol)

        self.pushButton_save_changes = QPushButton("Save", self.centralwidget)
        self.pushButton_save_changes.clicked.connect(self.save_changes)
        self.main_layout.addWidget(self.pushButton_save_changes)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.statusbar)
        MainWindow.setMenuBar(QMenuBar(MainWindow))

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.protocolLoaded = False

    def setup_first_grid(self):
        self.first_grid = QGridLayout()

        self.label_2p_delay = self.create_label("2 photon starting delay (s)", self.first_grid, 0, 0)
        self.lineEdit_starting_delay = self.create_line_edit(self.first_grid, 0, 1)
        self.lineEdit_starting_delay.setValidator(QDoubleValidator())
        self.lineEdit_starting_delay.setText("0.100")

        self.label_Neuropil_IF = self.create_label("Neuropil IF", self.first_grid, 1, 0)
        self.spinbox_Neuropil_IF = self.create_spin_box(0.7, self.first_grid, 1, 1, double=True)
        self.spinbox_Neuropil_IF.setMaximum(1.)
        self.spinbox_Neuropil_IF.setMinimum(0.)

        self.label_num_samples = self.create_label("Boostrapping nb samples", self.first_grid, 2, 0)
        self.lineEdit_num_samples = self.create_line_edit(self.first_grid, 2, 1)
        self.lineEdit_num_samples.setValidator(QIntValidator())
        self.lineEdit_num_samples.setText("1000")

        self.label_3 = self.create_label("Neural type", self.first_grid, 3, 0)
        self.comboBox_neural_type = self.create_combo_box(["PYR", "Other"], self.first_grid, 3, 1)
        
        self.label_4 = self.create_label("F0 method", self.first_grid, 4, 0)
        self.comboBox_F0_method = self.create_combo_box(["sliding", "hamming"], self.first_grid, 4, 1)

        verticalSpacer = QtWidgets.QSpacerItem(0, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.first_grid.addItem(verticalSpacer, 5, 0, columnSpan=2)

    def setup_third_grid(self):
        self.third_grid = QGridLayout()

        self.speed_th_label = self.create_label("Speed threshold (cm/s)", self.third_grid, 0, 0, colspan=2)
        self.speed_th_lineEdit = self.create_line_edit(self.third_grid, 1, 0, colspan=2)
        self.speed_th_lineEdit.setValidator(QDoubleValidator())
        self.speed_th_lineEdit.setText("0.5")

        self.motion_th_label = self.create_label("Facemotion threshold (std)", self.third_grid, 0, 2, colspan=2)
        self.motion_th_lineEdit = self.create_line_edit(self.third_grid, 1, 2, colspan=2)
        self.motion_th_lineEdit.setValidator(QDoubleValidator())
        self.motion_th_lineEdit.setText("2.0")

        self.pupil_th_label = self.create_label("Pupil threshold", self.third_grid, 0, 4, colspan=2)
        self.pupil_th_spinbox = self.create_spin_box(0.5, self.third_grid, 1, 4, double=True)
        self.pupil_th_spinbox.setMaximum(1.)
        self.pupil_th_spinbox.setMinimum(0.)
        self.comboBox_pupil_th_type = self.create_combo_box(["quantile", "std"], self.third_grid, 1, 5)
        self.comboBox_pupil_th_type.currentTextChanged.connect(self.on_change_pupil_th_type)

        verticalSpacer = QtWidgets.QSpacerItem(0, 5, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.third_grid.addItem(verticalSpacer, 2, 0, columnSpan=6)

        self.min_run_window_label = self.create_label("Run min duration (s)", self.third_grid, 3, 0, colspan=2)
        self.min_run_window_spinbox = self.create_spin_box(3.5, self.third_grid, 4, 0, double=True, colspan=2)

        self.min_as_window_label = self.create_label("AS min duration (s)", self.third_grid, 3, 2, colspan=2)
        self.min_as_window_spinbox = self.create_spin_box(2.5, self.third_grid, 4, 2, double=True, colspan=2)

        self.min_rest_window_label = self.create_label("Rest min duration (s)", self.third_grid, 3, 4, colspan=2)
        self.min_rest_window_spinbox = self.create_spin_box(1.5, self.third_grid, 4, 4, double=True, colspan=2)

        verticalSpacer = QtWidgets.QSpacerItem(0, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.third_grid.addItem(verticalSpacer, 5, 0, columnSpan=6)

    def setup_second_grid(self):
        self.second_grid = QGridLayout()

        self.lineEdit_data_directory = self.create_line_edit(self.second_grid, 0, 1)
        self.lineEdit_compile_directory = self.create_line_edit(self.second_grid, 1, 1)
        self.lineEdit_red_tif_path = self.create_line_edit(self.second_grid, 2, 1)

        self.pushButton_data = QPushButton("Data directory", self.centralwidget)
        self.pushButton_data.clicked.connect(self.open_folder_dialog)
        self.second_grid.addWidget(self.pushButton_data, 0, 0)

        self.pushButton_save = QPushButton("Compile directory", self.centralwidget)
        self.pushButton_save.clicked.connect(self.open_save_folder)
        self.second_grid.addWidget(self.pushButton_save, 1, 0)

        self.pushButton_red = QPushButton("Red channel", self.centralwidget)
        self.pushButton_red.clicked.connect(self.get_redfile_path)
        self.second_grid.addWidget(self.pushButton_red, 2, 0)

    def save_changes(self):
        # Retrieve inputs from combo boxes
        self.neural_type = str(self.comboBox_neural_type.currentText())
        self.f0_method = str(self.comboBox_F0_method.currentText())

        self.starting_delay = float(self.lineEdit_starting_delay.text())
        self.neuropil_if = self.spinbox_Neuropil_IF.value()
        self.nb_samples = int(self.lineEdit_num_samples.text())

        self.speed_th = float(self.speed_th_lineEdit.text())
        self.motion_th = float(self.motion_th_lineEdit.text())
        self.pupil_th = self.pupil_th_spinbox.value()
        self.pupil_th_type = str(self.comboBox_pupil_th_type.currentText())
        self.min_run_window = self.min_run_window_spinbox.value()
        self.min_as_window = self.min_as_window_spinbox.value()
        self.min_rest_window = self.min_rest_window_spinbox.value()

        self.data_directory = str(self.lineEdit_data_directory.text())
        self.compile_directory = str(self.lineEdit_compile_directory.text())
        text = str(self.lineEdit_red_tif_path.text())
        self.red_image_path = text if text != '' else None
        
        if not self.compile_directory.strip():
            self.show_warning_popup("Do you want to analyze data without final compiling?", self.handle_warning_response)

        if self.compile_directory.strip():
            if not self.data_directory.strip():
                self.show_error_popup('No data folder selected. Please select a folder')
            elif not self.protocolLoaded :
                self.show_error_popup('Protocol not loaded. Please select a file')
            else : 
                self.statusbar.showMessage('Information saved. Close the window.', 0)

    def show_error_popup(self, message):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setText(message)
        msg.setWindowTitle("Error")
        msg.exec_()
    
    def show_warning_popup(self, message, fct):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setText(message)
        msg.setWindowTitle("Warning")
        msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        msg.buttonClicked.connect(fct)
        msg.exec_()

    def handle_warning_response(self, button):
        if button.text() == "&Yes":
            self.lineEdit_compile_directory.setText("No compilation")
            self.compile_directory = "no_compile"
        elif  button.text() == "&No":
            self.statusbar.showMessage('', 0)
    
    def open_folder_dialog(self):
        folder_path = QFileDialog.getExistingDirectory(None, "Select folder containing data")
        if folder_path:
            self.lineEdit_data_directory.setText(folder_path)
            protocol_filepath = Path(folder_path + "/protocol.json")
            if protocol_filepath.is_file():
                self.get_protocol(protocol_filepath)
                self.protocolLoaded = True
            _, red_image_path = get_red_channel(folder_path)
            if red_image_path is not None :
                self.lineEdit_red_tif_path.setText(red_image_path)

    def open_save_folder(self):
        save_folder = QFileDialog.getExistingDirectory(None, "Select Saving directory")
        if save_folder:
            self.lineEdit_compile_directory.setText(save_folder)

    def get_redfile_path(self):
        red_path = QFileDialog.getOpenFileName(None, "Select red channel TIF file", filter="Images (*.tif *.tiff )")
        self.lineEdit_red_tif_path.setText(red_path[0])

    def on_change_pupil_th_type(self, text):
        if text == 'quantile' :
            self.pupil_th_spinbox.setMaximum(1.)
            self.pupil_th_spinbox.setValue(0.5)
        elif text =='std' :
            self.pupil_th_spinbox.setMaximum(99.)
            self.pupil_th_spinbox.setValue(2.)
        else :
            raise Exception(f"Pupil threshold type incorrect: {text}. Should be quantile or std.")
        
    def create_label(self, text, layout, row, col, rowspan=1, colspan=1):
        label = QLabel(text, self.centralwidget)
        layout.addWidget(label, row, col, rowspan, colspan)
        return label

    def create_line_edit(self, layout, row, col, rowspan=1, colspan=1):
        line_edit = QLineEdit(self.centralwidget)
        layout.addWidget(line_edit, row, col, rowspan, colspan)
        return line_edit

    def create_combo_box(self, items, layout, row, col, rowspan=1, colspan=1):
        combo_box = QComboBox(self.centralwidget)
        combo_box.addItems(items)
        layout.addWidget(combo_box, row, col, rowspan, colspan)
        return combo_box
    
    def create_spacer(self, items, layout, row, col, rowspan=1, colspan=1):
        combo_box = QComboBox(self.centralwidget)
        combo_box.addItems(items)
        layout.addWidget(combo_box, row, col, rowspan, colspan)
        return combo_box
    
    def create_spin_box(self, value, layout, row, col, rowspan=1, colspan=1, double=False):
        if double :
            spinbox = QtWidgets.QDoubleSpinBox(self.centralwidget)
            spinbox.setSingleStep(0.10)
        else :
            spinbox = QtWidgets.QSpinBox(self.centralwidget)
        spinbox.setValue(value)
        layout.addWidget(spinbox, row, col, rowspan, colspan)
        return spinbox
    
    def loadprotocol(self):
        protocol_path, _ = QFileDialog.getOpenFileName(None, "Select the protocol File", "", "All Files (*.*);;JSON Files (*.json)")
        if protocol_path:
            self.get_protocol(protocol_path)

    def get_protocol(self, protocol_path):
        # Load the JSON data
        with open(protocol_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        
        self.protocol_items = []

        if data["Presentation"] == "multiprotocol" :
            self.protocol_numbers = []
            self.protocol_names = []
            for key, value in data.items():
                if key.startswith("Protocol-"):
                    # Extract the protocol number
                    self.protocol_number = int(key.split("-")[1])-1
                    if self.protocol_number not in self.protocol_numbers :
                        self.protocol_numbers.append(self.protocol_number)
                        self.protocol_name = value.split("/")[-1].replace(".json", "")
                        self.protocol_names.append(self.protocol_name)
                        self.protocol_items.append(f"{self.protocol_number}: {self.protocol_name}")

        elif data["Presentation"] == "Stimuli-Sequence" :
            self.protocol_items.append(f"{0}: {data['Stimulus']}")
        else :
            print(f"Protocol is neither multiprotocol or stimuli sequence: {data['Presentation']}")

        # Display the protocols in the QListView
        model = QStringListModel()
        model.setStringList(self.protocol_items)
        self.listView.setModel(model)

        self.statusbar.showMessage('Protocol was loaded', 4000)
        self.protocolLoaded = True

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle("InputWindow")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    input_window = InputWindow()
    input_window.show()
    sys.exit(app.exec())
