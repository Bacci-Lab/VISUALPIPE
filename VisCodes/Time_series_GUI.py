import matplotlib.pyplot as plt
import numpy as np
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class Ui_MainWindow(object):
    def setupUi(self, MainWindow, F, Time, Run,FaceMotion, Pupil, Photodiode, stimulus):
        """
        Sets up the main window UI and initializes the layout, input fields, and graphics view.

        Args:
            MainWindow: The main application window.
            datasets: A list of tuples (x, y) representing time series data.
        """
        # Configure the main window
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(945, 842)
        MainWindow.setStyleSheet("background-color: rgb(85, 85, 85);")

        # Create the central widget and layout
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)

        # Create a horizontal layout for the checkboxes, line edit, and refresh button
        self.topLayout = QtWidgets.QHBoxLayout()
        self.Run = Run
        self.FaceMotion = FaceMotion
        self.Pupil = Pupil
        self.Photodiode = Photodiode


        # Add checkboxes
        self.checkboxRun = QtWidgets.QCheckBox("Run", self.centralwidget)
        self.checkboxFaceMotion = QtWidgets.QCheckBox("FaceMotion", self.centralwidget)
        self.checkboxPupil = QtWidgets.QCheckBox("Pupil", self.centralwidget)
        self.checkboxPhotodiode = QtWidgets.QCheckBox("Photodiode", self.centralwidget)
        self.checkboxstimulus = QtWidgets.QCheckBox("stimulus", self.centralwidget)

        # Add checkboxes to the horizontal layout
        self.topLayout.addWidget(self.checkboxRun)
        self.topLayout.addWidget(self.checkboxFaceMotion)
        self.topLayout.addWidget(self.checkboxPupil)
        self.topLayout.addWidget(self.checkboxPhotodiode)
        self.topLayout.addWidget(self.checkboxstimulus)

        # Add a line edit for user input and a refresh button
        self.line_edit = QtWidgets.QLineEdit(self.centralwidget)
        self.line_edit.setPlaceholderText("Enter number of series (n)...")
        self.refresh_button = QtWidgets.QPushButton("Refresh", self.centralwidget)

        # Add the line edit and button to the horizontal layout
        self.topLayout.addWidget(self.line_edit)
        self.topLayout.addWidget(self.refresh_button)

        # Add the top layout to the main grid layout
        self.gridLayout.addLayout(self.topLayout, 0, 0, 1, 2)

        # Add a graphics view for plotting
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.gridLayout.addWidget(self.graphicsView, 1, 0, 1, 2)

        # Set the central widget
        MainWindow.setCentralWidget(self.centralwidget)

        # Connect the refresh button to the plotting function
        self.refresh_button.clicked.connect(
            lambda: self.plot_n_time_series(F, Time, self.graphicsView, stimulus)
        )

    def plot_n_time_series(self, F, Time, graphics_view, stimulus):
        """
        Plots multiple time series on the given graphics view.

        Args:
            F: A 2D numpy array where each row represents a time series.
            Time: A 1D numpy array representing the time points.
            graphics_view: The graphics view where the plot will be displayed.
        """
        try:
            # Get the number of time series to plot from the line edit
            n = int(self.line_edit.text())
            if n > F.shape[0]:
                n = F.shape[0]  # Limit to the number of available series in F

            # Clear the existing content in the graphics view
            self._clear_graphics_view(graphics_view)

            # Create a new Matplotlib figure and axes
            fig, ax = plt.subplots()
            fig.tight_layout(pad=0)  # Remove padding
            fig.patch.set_facecolor("#3d4242")  # Set figure background color to gray

            # Plot the time series with vertical offsets
            for i in range(n):
                y = F[i, :]  # Get the i-th time series
                offset = i * (max(y) - min(y) + 1) * 1.2  # Calculate vertical offset
                y_offset = y + offset  # Add the offset to the y-values
                ax.plot(Time, y_offset, color="green")  # Plot with the time array as x-axis
                # Highlight specified intervals
            if self.checkboxstimulus.isChecked():
                if len(stimulus) == 2:  # Ensure stimulus contains two lists
                    for start, end in zip(stimulus[0], stimulus[1]):  # Pair start and end times
                        ax.axvspan(start, end, color="white", alpha=0.3, zorder=0)

            self._plot_secondary_series(ax)

            # Customize the plot appearance
            ax.set_facecolor("#3d4242")  # Set axes background to gray
            ax.set_xlim(Time[0], Time[-1])  # Set x-axis limits to match Time
            ax.set_yticks([])  # Remove y-axis ticks
            ax.tick_params(labelleft=False)  # Remove y-axis labels
            ax.grid(False)  # Disable the grid
            for spine in ax.spines.values():  # Remove all spines (box lines around the plot)
                spine.set_visible(False)

            # Add zoom and pan interactions
            self._setup_interaction_events(fig, ax)

            # Integrate the Matplotlib canvas into the graphics view
            self._integrate_canvas_into_view(graphics_view, fig)

        except ValueError:
            # Show a warning message box if the input is invalid
            QtWidgets.QMessageBox.warning(None, "Input Error", "Please enter a valid integer.")

    def _plot_secondary_series(self, ax):
        """
        Plots the secondary datasets (e.g., Run, FaceMotion, Pupil, Photodiode)
        on the lower part of the graph when corresponding checkboxes are checked.
        """
        # Define specific colors for each dataset type
        color_map = {
            "Run": "red",
            "FaceMotion": "blue",
            "Pupil": "plum",
            "Photodiode": "orange",
        }

        secondary_data = []
        if self.checkboxRun.isChecked():
            secondary_data.append(("Run", self.Run))
        if self.checkboxFaceMotion.isChecked():
            secondary_data.append(("FaceMotion", self.FaceMotion))
        if self.checkboxPupil.isChecked():
            secondary_data.append(("Pupil", self.Pupil))
        if self.checkboxPhotodiode.isChecked():
            secondary_data.append(("Photodiode", self.Photodiode))


        # Plot secondary datasets with offsets to avoid overlap
        for i, (label, (x, y)) in enumerate(secondary_data):
            offset = -1 * (i + 1) * (max(max(y) - min(y), 1) * 1.2)  # Negative vertical offset
            y_offset = np.array(y) + offset
            ax.plot(
                x, y_offset, label=label, linestyle="--", color=color_map.get(label, "gray")
            )  # Use the specific color for each label, default to gray

        # Add the legend to the plot
        ax.legend(
            loc="upper right",  # Position of the legend
            fontsize=10,  # Font size
            frameon=True,  # Add a box around the legend
            facecolor="#3d4242",  # Legend background color to match the plot
            edgecolor="white",  # Edge color of the legend box
        )

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
    import sys
    import os
    import Photodiode
    import numpy as np
    import Running_computation
    import Ca_imaging
    import General_functions

    # Generate some sample datasets
    base_path = "Y:/raw-imaging/TESTS/Mai-An/visual_test/16-00-59"
    starting_delay_2p = 0.1
    ca_img = Ca_imaging.CaImagingDataManager(base_path, starting_delay=starting_delay_2p)

    face_camera = np.load(os.path.join(base_path,"FaceCamera-summary.npy"), allow_pickle=True)
    fvideo_time = face_camera.item().get('times')
    faceitOutput = np.load(os.path.join(base_path, "FaceIt", "FaceIt.npz"), allow_pickle=True)
    pupil = (faceitOutput['pupil_dilation'])
    facemotion = (faceitOutput['motion_energy'])

    speed, speed_time_stamps, last_F_index = Running_computation.compute_speed(base_path, ca_img.fs, ca_img.time_stamps)
    speed = (speed_time_stamps, speed)
    ca_img.cut_frames(last_index=last_F_index)

    stim_Time_start_realigned, Psignal, Psignal_time = Photodiode.realign_from_photodiode(base_path)
    visual_stim_path = os.path.join(base_path, "visual-stim.npy")
    visual_stim = np.load(visual_stim_path, allow_pickle=True).item()
    stim_time_durations = visual_stim['time_duration']
    stim_time_period = [stim_Time_start_realigned, stim_Time_start_realigned + stim_time_durations]

    raw_F = ca_img.normalize_time_series("raw_F", lower=0, upper=5)
    Psignal = General_functions.scale_trace(Psignal)
    pupil = General_functions.scale_trace(pupil)
    facemotion = General_functions.scale_trace(facemotion)

    FaceMotion = (fvideo_time, facemotion)
    Pupil = (fvideo_time, pupil)
    photodiode = (Psignal_time, Psignal)

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow, raw_F, ca_img.time_stamps, speed, FaceMotion, Pupil, photodiode, stim_time_period)
    MainWindow.show()
    sys.exit(app.exec_())
