import matplotlib.pyplot as plt
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtGui import QIntValidator
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class TimeSeriesUI(object):
    
    def __init__(self, centralwidget, fluorescence, time, speed, facemotion, pupil, photodiode, stimuli):
        self.centralwidget = centralwidget

        self.fluorescence = fluorescence
        self.time = time
        self.speed = speed
        self.facemotion = facemotion
        self.pupil = pupil
        self.photodiode = photodiode
        self.stimuli = stimuli

        self.setupUi()

    def setupUi(self):
        """
        Sets up the main window UI and initializes the layout, input fields, and graphics view.
        """

        # Create the central widget and layout
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)

        # Create a horizontal layout for the checkboxes, line edit, and refresh button
        self.topLayout = QtWidgets.QHBoxLayout()

        stylesheet = "color: lightgray; font-family: 'Calibri'; font-size: 15px"

        # Add checkboxes
        self.checkboxRun = QtWidgets.QCheckBox("Run", self.centralwidget)
        self.checkboxRun.setChecked(True)
        self.checkboxRun.setStyleSheet(stylesheet)
        self.checkboxFaceMotion = QtWidgets.QCheckBox("FaceMotion", self.centralwidget)
        self.checkboxFaceMotion.setChecked(True)
        self.checkboxFaceMotion.setStyleSheet(stylesheet)
        self.checkboxPupil = QtWidgets.QCheckBox("Pupil", self.centralwidget)
        self.checkboxPupil.setChecked(True)
        self.checkboxPupil.setStyleSheet(stylesheet)
        self.checkboxPhotodiode = QtWidgets.QCheckBox("Photodiode", self.centralwidget)
        self.checkboxPhotodiode.setChecked(True)
        self.checkboxPhotodiode.setStyleSheet(stylesheet)
        self.checkboxstimulus = QtWidgets.QCheckBox("Stimulus", self.centralwidget)
        self.checkboxstimulus.setChecked(True)
        self.checkboxstimulus.setStyleSheet(stylesheet)

        # Add checkboxes to the horizontal layout
        self.topLayout.addWidget(self.checkboxRun)
        self.topLayout.addWidget(self.checkboxFaceMotion)
        self.topLayout.addWidget(self.checkboxPupil)
        self.topLayout.addWidget(self.checkboxPhotodiode)
        self.topLayout.addWidget(self.checkboxstimulus)

        # Add a line edit for user input and a refresh button
        self.line_edit = QtWidgets.QLineEdit(self.centralwidget)
        self.line_edit.setPlaceholderText("Enter number of series (n)...")
        self.line_edit.setValidator(QIntValidator())
        self.refresh_button = QtWidgets.QPushButton("Refresh", self.centralwidget)
        self.refresh_button.setStyleSheet("color: white")

        # Add the line edit and button to the horizontal layout
        self.topLayout.addWidget(self.line_edit)
        self.topLayout.addWidget(self.refresh_button)

        # Add the top layout to the main grid layout
        self.gridLayout.addLayout(self.topLayout, 0, 0, 1, 2)

        # Add a graphics view for plotting
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.gridLayout.addWidget(self.graphicsView, 1, 0, 1, 2)

        # Connect the refresh button to the plotting function
        self.refresh_button.clicked.connect(
            lambda: self.plot_n_time_series(self.graphicsView)
        )

    def plot_n_time_series(self, graphics_view):
        """
        Plots multiple time series on the given graphics view.

        Args:
            F: A 2D numpy array where each row represents a time series.
            Time: A 1D numpy array representing the time points.
            graphics_view: The graphics view where the plot will be displayed.
        """
        # Get the number of time series to plot from the line edit
        if self.line_edit.text() == '' :
            self.line_edit.setText('0')
        
        n = int(self.line_edit.text())
        if n > self.fluorescence.shape[0]:
            n = self.fluorescence.shape[0]  # Limit to the number of available series in F
            self.line_edit.setText(str(n))

        # Clear the existing content in the graphics view
        self._clear_graphics_view(graphics_view)

        # Create a new Matplotlib figure and axes
        fig, ax = plt.subplots()
        fig.tight_layout(pad=0)  # Remove padding
        fig.patch.set_facecolor("#3d4242")  # Set figure background color to gray

        # Plot the time series with vertical offsets
        for i in range(n):
            y = self.fluorescence[i, :]  # Get the i-th time series
            offset = i * (max(y) - min(y) + 1)  # Calculate vertical offset
            y_offset = y + offset  # Add the offset to the y-values
            ax.plot(self.time, y_offset, linewidth=0.5, color="forestgreen")  # Plot with the time array as x-axis
            # Highlight specified intervals
        
        if self.checkboxstimulus.isChecked():
            if len(self.stimuli) == 2:  # Ensure stimulus contains two lists
                for start, end in zip(self.stimuli[0], self.stimuli[1]):  # Pair start and end times
                    ax.axvspan(start, end, color="white", alpha=0.2, zorder=0)

        self._plot_secondary_series(ax)

        # Customize the plot appearance
        ax.set_facecolor("#3d4242")  # Set axes background to gray
        ax.set_xlim(self.time[0], self.time[-1])  # Set x-axis limits to match Time
        ax.set_yticks([])  # Remove y-axis ticks
        ax.tick_params(labelleft=False)  # Remove y-axis labels
        ax.grid(False)  # Disable the grid
        for spine in ax.spines.values():  # Remove all spines (box lines around the plot)
            spine.set_visible(False)

        # Add zoom and pan interactions
        self._setup_interaction_events(fig, ax)

        # Integrate the Matplotlib canvas into the graphics view
        self._integrate_canvas_into_view(graphics_view, fig)

    def _plot_secondary_series(self, ax):
        """
        Plots the secondary datasets (e.g., Run, FaceMotion, Pupil, Photodiode)
        on the lower part of the graph when corresponding checkboxes are checked.
        """
        # Define specific colors for each dataset type
        color_map = {
            "Run": "goldenrod",
            "FaceMotion": "lightgray",
            "Pupil": "black",
            "Photodiode": "thistle",
        }

        secondary_data = []
        if self.checkboxRun.isChecked():
            secondary_data.append(("Run", self.speed))
        if self.checkboxFaceMotion.isChecked():
            secondary_data.append(("FaceMotion", self.facemotion))
        if self.checkboxPupil.isChecked():
            secondary_data.append(("Pupil", self.pupil))
        if self.checkboxPhotodiode.isChecked():
            secondary_data.append(("Photodiode", self.photodiode))

        offset = -1
        # Plot secondary datasets with offsets to avoid overlap
        for i, (label, (x, y)) in enumerate(secondary_data):
            offset -= (max(y) - min(y))
            y_offset = np.array(y) + offset
            ax.plot(
                x, y_offset, label=label, linewidth=0.5, linestyle="-", color=color_map.get(label, "gray")
            )  # Use the specific color for each label, default to gray
            offset -= 1

        # Add the legend to the plot
        leg = ax.legend(
            loc="upper right",  # Position of the legend
            fontsize=10,  # Font size
            frameon=True,  # Add a box around the legend
            labelcolor='lightgray',
            facecolor="#3d4242",  # Legend background color to match the plot
        )
        for line in leg.get_lines():
            line.set_linewidth(2)

    def _setup_interaction_events(self, fig, ax):
        panning = {"active": False, "start_xlim": None, "start_ylim": None, "start_pos": None}

        def on_press(event):
            if event.button == 1 and event.inaxes:  # Left mouse button
                panning["active"] = True
                panning["start_xlim"] = ax.get_xlim()
                panning["start_ylim"] = ax.get_ylim()
                panning["start_pos"] = event.xdata, event.ydata

        def on_release(event):
            if event.button == 1:  # Left mouse button
                panning["active"] = False
                panning["start_xlim"] = None
                panning["start_ylim"] = None
                panning["start_pos"] = None

        def on_motion(event):
            if panning["active"] and event.inaxes:
                dx = event.xdata - panning["start_pos"][0]
                dy = event.ydata - panning["start_pos"][1]
                ax.set_xlim(panning["start_xlim"][0] - dx, panning["start_xlim"][1] - dx)
                ax.set_ylim(panning["start_ylim"][0] - dy, panning["start_ylim"][1] - dy)
                fig.canvas.draw_idle()

        def on_scroll(event):
            if event.inaxes:
                current_xlim = ax.get_xlim()
                current_ylim = ax.get_ylim()
                xdata, ydata = event.xdata, event.ydata
                zoom_factor = 0.9 if event.button == "up" else 1.1
                new_xlim = [
                    xdata - (xdata - current_xlim[0]) * zoom_factor,
                    xdata + (current_xlim[1] - xdata) * zoom_factor,
                ]
                new_ylim = [
                    ydata - (ydata - current_ylim[0]) * zoom_factor,
                    ydata + (current_ylim[1] - ydata) * zoom_factor,
                ]
                ax.set_xlim(new_xlim)
                ax.set_ylim(new_ylim)
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect("button_press_event", on_press)
        fig.canvas.mpl_connect("button_release_event", on_release)
        fig.canvas.mpl_connect("motion_notify_event", on_motion)
        fig.canvas.mpl_connect("scroll_event", on_scroll)

    def _clear_graphics_view(self, graphics_view):
        """
        Clears the contents of the graphics view.

        Args:
            graphics_view: The graphics view to clear.
        """
        layout = graphics_view.layout()
        if layout:
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            QtWidgets.QWidget().setLayout(layout)

    def _integrate_canvas_into_view(self, graphics_view, fig):
        """
        Integrates the Matplotlib canvas into the graphics view.

        Args:
            graphics_view: The graphics view where the canvas will be embedded.
            fig: The Matplotlib figure.
        """
        canvas = FigureCanvas(fig)

        # Add a vertical layout with the toolbar and canvas
        layout = QtWidgets.QVBoxLayout(graphics_view)
        layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        layout.setSpacing(0)  # No spacing

        # Create a toolbar for panning and zooming
        toolbar = NavigationToolbar(canvas, graphics_view)
        layout.addWidget(toolbar)  # Add the toolbar to the layout
        layout.addWidget(canvas)  # Add the canvas to the layout
        graphics_view.setLayout(layout)

if __name__ == "__main__":
    from PyQt5.QtCore import Qt
    import easygui
    import sys
    import os
    import numpy as np
    import h5py
    from pathlib import Path

    import utils.file as file
    import General_functions

    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self, raw_F, time_stamps, speed, FaceMotion, Pupil, photodiode, stim_time_period):
            super().__init__(flags=Qt.WindowStaysOnTopHint)
            self.setWindowTitle("Visualization GUI")
            self.setStyleSheet("""
                background-color: rgb(85, 85, 85);
                gridline-color: rgb(213, 213, 213);
                border-top-color: rgb(197, 197, 197);
            """)
            self.aw, self.ah = 1500, 700
            self.setGeometry(500, 50, self.aw, self.ah)
            self.centralwidget = QtWidgets.QWidget(self)
            self.setCentralWidget(self.centralwidget)

            self.ui = TimeSeriesUI(self.centralwidget, raw_F, time_stamps, speed, FaceMotion, Pupil, photodiode, stim_time_period)

    # Generate some sample datasets
    save_dir = easygui.diropenbox(title='Select folder containing output data')
    path = Path(save_dir)
    base_path = path.parent.absolute()

    unique_id, global_protocol, experimenter, subject_id = file.get_metadata(base_path)
    id_version = save_dir.split('_')[5]
    
    # Load HDF5 file data
    filename_h5 = "_".join([unique_id, id_version, 'postprocessing']) + ".h5"
    with h5py.File(os.path.join(save_dir, filename_h5), "r") as f:

        time_stamps = f['Ca_imaging']['Time'][()]
        dFoF0 = f['Ca_imaging']['full_trace']['dFoF0'][()]
        
        speed_corr = f['Behavioral']['Correlation']['speed_corr'][()]
        facemotion_corr = f['Behavioral']['Correlation']['facemotion_corr'][()]
        pupil_corr = f['Behavioral']['Correlation']['pupil_corr'][()]

        speed = f['Behavioral']['Speed'][()]
        facemotion = f['Behavioral']['FaceMotion'][()]
        pupil = f['Behavioral']['Pupil'][()]
        photodiode = f['Behavioral']['Photodiode'][()]

        time_onset = f['Stimuli']['time_onset'][()]

    # Load visual stimuli data
    visual_stim_path = os.path.join(base_path, "visual-stim.npy")
    visual_stim = np.load(visual_stim_path, allow_pickle=True).item()
    duration = visual_stim['time_duration']
    stim_time_period = [time_onset, list(time_onset + duration)]

    dFoF0_norm = General_functions.scale_trace(dFoF0, axis=1)

    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow(dFoF0_norm, time_stamps, speed, facemotion, pupil, photodiode, stim_time_period)
    main_window.show()
    sys.exit(app.exec_())
