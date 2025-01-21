from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QGridLayout, QComboBox, QLabel, QLineEdit, QPushButton, QListView, QStatusBar, QMenuBar, QFileDialog
from PyQt5 import QtCore
import sys
from PyQt5.QtGui import QIntValidator
import json
from PyQt5.QtCore import QStringListModel

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.resize(457, 563)
        MainWindow.setStyleSheet("background-color: rgb(85, 85, 85);")

        self.centralwidget = QWidget(MainWindow)
        self.main_layout = QVBoxLayout(self.centralwidget)

        # First Grid Layout
        self.setup_first_grid()
        self.main_layout.addLayout(self.first_grid)

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
        MainWindow.setStatusBar(QStatusBar(MainWindow))
        MainWindow.setMenuBar(QMenuBar(MainWindow))

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def setup_first_grid(self):
        self.first_grid = QGridLayout()

        self.comboBox_neural_type = self.create_combo_box(["PYR", "VIP"], self.first_grid, 4, 0)
        self.label_3 = self.create_label("Neural type", self.first_grid, 3, 0)

        self.comboBox_F0_method = self.create_combo_box(["sliding", "Hamming"], self.first_grid, 4, 1)
        self.label_4 = self.create_label("F0 method", self.first_grid, 3, 1)

        self.label_2p_fr = self.create_label("2 photon Frequency", self.first_grid, 0, 0)
        self.lineEdit_Fre = self.create_line_edit(self.first_grid, 0, 1)
        self.lineEdit_Fre.setValidator(QIntValidator())
        self.lineEdit_Fre.setText("29.7597")

        self.label_2p_delay = self.create_label("2 photon starting delay(ms)", self.first_grid, 1, 0)
        self.lineEdit_starting_delay = self.create_line_edit(self.first_grid, 1, 1)
        self.lineEdit_starting_delay.setValidator(QIntValidator())
        self.lineEdit_starting_delay.setText("0.100")

        self.label_5 = self.create_label("Neuropil IF", self.first_grid, 2, 0)
        self.lineEdit_Neuropil_IF = self.create_line_edit(self.first_grid, 2, 1)
        self.lineEdit_Neuropil_IF.setValidator(QIntValidator())
        self.lineEdit_Neuropil_IF.setText("0.7")

    def setup_second_grid(self):
        self.second_grid = QGridLayout()

        self.lineEdit_data_directory = self.create_line_edit(self.second_grid, 0, 1)
        self.lineEdit_save_directory = self.create_line_edit(self.second_grid, 1, 1)

        self.pushButton_data = QPushButton("Data directory", self.centralwidget)
        self.pushButton_data.clicked.connect(self.open_folder_dialog)
        self.second_grid.addWidget(self.pushButton_data, 0, 0)


        self.pushButton_save = QPushButton("Save directory", self.centralwidget)
        self.pushButton_save.clicked.connect(self.open_save_folder)
        self.second_grid.addWidget(self.pushButton_save, 1, 0)

    def save_changes(self):
        # Retrieve inputs from combo boxes
        self.neural_type = str(self.comboBox_neural_type.currentText())
        self.f0_method = str(self.comboBox_F0_method.currentText())

        self.photon_frequency = float(self.lineEdit_Fre.text())
        self.starting_delay = float(self.lineEdit_starting_delay.text())
        self.neuropil_if = float(self.lineEdit_Neuropil_IF.text())
        self.data_directory = str(self.lineEdit_data_directory.text())
        self.save_directory = str(self.lineEdit_save_directory.text())
    def open_folder_dialog(self):
        folder_path = QFileDialog.getExistingDirectory(None, "Select Folder contains data")
        if folder_path:
            self.lineEdit_data_directory.setText(folder_path)

    def open_save_folder(self):
        save_folder = QFileDialog.getExistingDirectory(None, "Select Saving directory")
        if save_folder:
            self.lineEdit_save_directory.setText(save_folder)

    def create_label(self, text, layout, row, col):
        label = QLabel(text, self.centralwidget)
        layout.addWidget(label, row, col)
        return label

    def create_line_edit(self, layout, row, col):
        line_edit = QLineEdit(self.centralwidget)
        layout.addWidget(line_edit, row, col)
        return line_edit

    def create_combo_box(self, items, layout, row, col):
        combo_box = QComboBox(self.centralwidget)
        combo_box.addItems(items)
        layout.addWidget(combo_box, row, col)
        return combo_box
    def loadprotocol(self):
        protocol_path, _ = QFileDialog.getOpenFileName(None, "Select the protocol File", "", "All Files (*.*);;JSON Files (*.json)")
        if protocol_path:
            self.get_protocol(protocol_path)

    def get_protocol(self, protocol_path):
        # Load the JSON data
        with open(protocol_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        self.protocol_numbers = []
        self.protocol_names = []
        self.protocol_items = []
        for key, value in data.items():
            if key.startswith("Protocol-"):
                # Extract the protocol number
                self.protocol_number = int(key.split("-")[1])-1
                self.protocol_numbers.append(self.protocol_number)
                self.protocol_name = value.split("/")[-1].replace(".json", "")
                self.protocol_names.append(self.protocol_name)
                self.protocol_items.append(f"{self.protocol_number}: {self.protocol_name}")

            # Display the protocols in the QListView
            model = QStringListModel()
            model.setStringList(self.protocol_items)
            self.listView.setModel(model)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle("MainWindow")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
