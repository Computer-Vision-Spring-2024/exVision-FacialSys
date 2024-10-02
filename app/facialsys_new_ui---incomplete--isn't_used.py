import sys

import matplotlib.pyplot as plt

# in CMD: pip install qdarkstyle -> pip install pyqtdarktheme
import qdarktheme
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtCore, QtGui, QtWidgets


class FacialSys_Ui(object):
    def setupUi(self, FacialSys):
        FacialSys.setObjectName("FacialSys")
        FacialSys.resize(1280, 720)
        font = QtGui.QFont()
        font.setFamily("Montserrat")
        font.setPointSize(12)
        FacialSys.setFont(font)

        self.central_widget = QtWidgets.QWidget(FacialSys)
        self.central_widget.setObjectName("central_widget")
        FacialSys.setCentralWidget(self.central_widget)
        self.main_layout = QtWidgets.QVBoxLayout(self.central_widget)
        self.main_layout.setObjectName("main_layout")

        ## "Organization Logo" & "Toggle App Mode: Recognition or Detection"
        ### Layout
        self.logo_toggle_layout = QtWidgets.QHBoxLayout()
        self.logo_toggle_layout.setObjectName("logo_toggle_layout")
        self.main_layout.addLayout(self.logo_toggle_layout)
        ### Organization Logo
        self.organization_logo = QtWidgets.QLabel()
        self.organization_logo.setObjectName("organization_logo")
        self.logo_toggle_layout.addWidget(self.organization_logo)
        ### Load and set the organization logo
        logo_pixmap = QtGui.QPixmap("app/assets/exVision_logo.jpg")
        scaled_pixmap = logo_pixmap.scaled(
            100, 100, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        self.organization_logo.setPixmap(scaled_pixmap)
        self.organization_logo.setScaledContents(True)

        ### Toggle App Mode Button: Recognition or Detection
        #### Create buttons for the segmented control
        self.recognition_mode_btn = QtWidgets.QPushButton(
            "Face Recognition", self.central_widget
        )
        self.detection_mode_btn = QtWidgets.QPushButton(
            "Face Detection", self.central_widget
        )
        #### Create a button group and add buttons to it
        self.mode_segmented_button = QtWidgets.QButtonGroup(self.central_widget)
        self.mode_segmented_button.addButton(self.recognition_mode_btn, 1)
        self.mode_segmented_button.addButton(self.detection_mode_btn, 2)
        #### Make buttons checkable (behave like toggle buttons)
        self.recognition_mode_btn.setCheckable(True)
        self.detection_mode_btn.setCheckable(True)
        #### Set the first button as initially checked
        self.recognition_mode_btn.setChecked(True)
        #### Connect the button group's signal to a slot
        self.mode_segmented_button.buttonClicked[int].connect(self.on_button_clicked)
        #### Add buttons to the layout
        self.logo_toggle_layout.addWidget(self.recognition_mode_btn)
        self.logo_toggle_layout.addWidget(self.detection_mode_btn)

        ## Detection Extra Widget: Last Stage Threshold
        self.last_stage_threshold_label = QtWidgets.QLabel(self.central_widget)
        self.last_stage_threshold_label.setObjectName("last_stage_threshold_label")

        self.last_stage_threshold_spinbox = QtWidgets.QDoubleSpinBox(
            self.central_widget
        )
        self.last_stage_threshold_spinbox.setObjectName("last_stage_threshold_label")
        self.last_stage_threshold_spinbox.setValue(1)
        self.last_stage_threshold_spinbox.setSingleStep(0.1)
        self.last_stage_threshold_spinbox.setMinimum(0)
        self.last_stage_threshold_spinbox.setMaximum(9)

        self.logo_toggle_layout.addWidget(self.last_stage_threshold_label)
        self.logo_toggle_layout.addWidget(self.last_stage_threshold_spinbox)

        ## Layout that contains the canvases on the right and the controls on the left
        self.mode_main_hlayout = QtWidgets.QHBoxLayout()
        self.mode_main_hlayout.setObjectName("mode_main_hlayout")
        self.main_layout.addLayout(self.mode_main_hlayout)

        ## controls widget
        self.controls_widget = QtWidgets.QWidget(self.central_widget)
        self.controls_widget.setObjectName("controls_widget")
        self.controls_layout = QtWidgets.QVBoxLayout(self.controls_widget)
        self.controls_layout.setObjectName("controls_layout")
        self.mode_main_hlayout.addWidget(self.controls_widget)
        ### Apply Button
        self.apply = QtWidgets.QPushButton("Apply", self.controls_widget)
        self.apply.setIcon(QtGui.QIcon("app/assets/icons/apply_button.png"))
        self.apply.setIconSize(QtCore.QSize(64, 64))
        self.apply.setStyleSheet(
            "QtWidgets.QPushButton { text-align: center; }"
            "QtWidgets.QPushButton::icon { position: top; }"
        )
        self.controls_layout.addWidget(self.apply)
        ### Toggle Query Button
        self.toggle_query = QtWidgets.QPushButton("Toggle Query", self.controls_widget)
        self.toggle_query.setIcon(QtGui.QIcon("app/assets/icons/toggle_query.png"))
        self.toggle_query.setIconSize(QtCore.QSize(64, 64))
        self.toggle_query.setStyleSheet(
            "QtWidgets.QPushButton { text-align: center; }"
            "QtWidgets.QPushButton::icon { position: top; }"
        )
        self.controls_layout.addWidget(self.toggle_query)
        ### Import Image Button
        self.import_img = QtWidgets.QPushButton("Import", self.controls_widget)
        self.import_img.setIcon(QtGui.QIcon("app/assets/icons/import_image.png"))
        self.import_img.setIconSize(QtCore.QSize(64, 64))
        self.import_img.setStyleSheet(
            "QtWidgets.QPushButton { text-align: center; }"
            "QtWidgets.QPushButton::icon { position: top; }"
        )
        self.controls_layout.addWidget(self.import_img)

        ## Canvases
        ### Canvases outline
        self.Canvases_hlayout = QtWidgets.QHBoxLayout()
        self.Canvases_hlayout.setObjectName("Canvases_hlayout")
        self.input_canvas_frame = QtWidgets.QFrame()
        self.input_canvas_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.input_canvas_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.input_canvas_frame.setObjectName("input_canvas_frame")
        self.Canvases_hlayout.addWidget(self.input_canvas_frame)
        self.output_canvas_frame = QtWidgets.QFrame()
        self.output_canvas_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.output_canvas_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.output_canvas_frame.setObjectName("output_canvas_frame")
        self.Canvases_hlayout.addWidget(self.output_canvas_frame)
        self.mode_main_hlayout.addLayout(self.Canvases_hlayout)
        #### Input Canvas
        self.input_canvas_hlayout = QtWidgets.QHBoxLayout(self.input_canvas_frame)
        self.input_canvas_hlayout.setObjectName("input_canvas_hlayout")
        self.input_figure = plt.figure()
        self.input_figure_canvas = FigureCanvas(self.input_figure)
        self.input_canvas_hlayout.addWidget(self.input_figure_canvas)
        #### Output Canvas
        self.output_canvas_hlayout = QtWidgets.QHBoxLayout(self.output_canvas_frame)
        self.output_canvas_hlayout.setObjectName("output_canvas_hlayout")
        self.output_figure = plt.figure()
        self.output_figure_canvas = FigureCanvas(self.output_figure)
        self.output_canvas_hlayout.addWidget(self.output_figure_canvas)

        self.retranslateUi(FacialSys)
        QtCore.QMetaObject.connectSlotsByName(FacialSys)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("FacialSys", "FacialSys"))

        font_global_thresholds_label = QtGui.QFont()
        font_global_thresholds_label.setPointSize(14)
        self.last_stage_threshold_label.setText(
            _translate("FacialSys", "Last Stage Threshold")
        )

    def exit_application(self):
        sys.exit()

    def on_button_clicked(self, button):
        if button == 1:
            print("Recognition Mode")
        else:
            print("Detection Mode")


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = FacialSys_Ui()
    ui.setupUi(MainWindow)
    MainWindow.show()
    qdarktheme.setup_theme("dark")
    sys.exit(app.exec_())
