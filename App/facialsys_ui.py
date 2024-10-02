import sys

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, FacialSys):
        FacialSys.setObjectName("MainWindow")
        FacialSys.resize(1101, 732)
        font = QtGui.QFont()
        font.setFamily("Montserrat")
        font.setPointSize(12)
        FacialSys.setFont(font)
        self.centralwidget = QtWidgets.QWidget(FacialSys)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")

        self.verticalLayout_2.addWidget(self.tabWidget)

        # PCA
        self.pca_tab = QtWidgets.QWidget()
        self.pca_tab.setObjectName("tab_10")
        self.verticalLayout_14 = QtWidgets.QVBoxLayout(self.pca_tab)
        self.verticalLayout_14.setObjectName("verticalLayout_14")
        self.horizontalLayout_37 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_37.setObjectName("horizontalLayout_37")
        self.PCA_input = QtWidgets.QFrame(self.pca_tab)
        self.PCA_input.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.PCA_input.setFrameShadow(QtWidgets.QFrame.Raised)
        self.PCA_input.setObjectName("PCA_input")
        self.horizontalLayout_37.addWidget(self.PCA_input)
        self.PCA_output = QtWidgets.QFrame(self.pca_tab)
        self.PCA_output.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.PCA_output.setFrameShadow(QtWidgets.QFrame.Raised)
        self.PCA_output.setObjectName("PCA_output")
        self.horizontalLayout_37.addWidget(self.PCA_output)
        self.verticalLayout_14.addLayout(self.horizontalLayout_37)
        self.horizontalLayout_40 = QtWidgets.QHBoxLayout()
        self.toggle = QtWidgets.QPushButton(self.pca_tab)
        self.toggle.setObjectName("toggle")
        self.horizontalLayout_40.addWidget(self.toggle)
        spacerItem5 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.horizontalLayout_40.addItem(spacerItem5)

        self.apply_PCA = QtWidgets.QPushButton(self.pca_tab)
        self.apply_PCA.setObjectName("apply_PCA")
        self.horizontalLayout_40.addWidget(self.apply_PCA)
        self.verticalLayout_14.addLayout(self.horizontalLayout_40)
        self.tabWidget.addTab(self.pca_tab, "")

        # Detection
        self.tab_12 = QtWidgets.QWidget()
        self.tab_12.setObjectName("tab_12")
        self.verticalLayout_15 = QtWidgets.QVBoxLayout(self.tab_12)
        self.verticalLayout_15.setObjectName("verticalLayout_15")
        self.detection_HLayout = QtWidgets.QHBoxLayout()
        self.detection_HLayout.setObjectName("detection_HLayout")
        self.detection_input_frame = QtWidgets.QFrame(self.tab_12)
        self.detection_input_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.detection_input_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.detection_input_frame.setObjectName("detection_input_frame")
        self.detection_HLayout.addWidget(self.detection_input_frame)
        self.detection_output_frame = QtWidgets.QFrame(self.tab_12)
        self.detection_output_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.detection_output_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.detection_output_frame.setObjectName("PCA_output")
        self.detection_HLayout.addWidget(self.detection_output_frame)
        self.verticalLayout_15.addLayout(self.detection_HLayout)
        self.detection_inputs_HLayout = QtWidgets.QHBoxLayout()

        self.last_stage_threshold_label = QtWidgets.QLabel(self.tab_12)
        self.last_stage_threshold_label.setObjectName("last_stage_threshold_label")
        self.detection_inputs_HLayout.addWidget(self.last_stage_threshold_label)

        self.last_stage_threshold_spinbox = QtWidgets.QDoubleSpinBox(self.tab_12)
        self.last_stage_threshold_spinbox.setObjectName("last_stage_threshold_label")
        self.last_stage_threshold_spinbox.setValue(1)
        self.last_stage_threshold_spinbox.setSingleStep(0.1)
        self.last_stage_threshold_spinbox.setMinimum(0)
        self.last_stage_threshold_spinbox.setMaximum(9)
        self.detection_inputs_HLayout.addWidget(self.last_stage_threshold_spinbox)

        self.apply_detection = QtWidgets.QPushButton(self.tab_12)
        self.apply_detection.setObjectName("apply_detection")
        self.detection_inputs_HLayout.addWidget(self.apply_detection)
        self.verticalLayout_15.addLayout(self.detection_inputs_HLayout)
        self.tabWidget.addTab(self.tab_12, "")

        FacialSys.setCentralWidget(self.centralwidget)

        ## Menu Bar
        self.menubar = QtWidgets.QMenuBar(FacialSys)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1101, 28))
        self.menubar.setObjectName("menubar")
        FacialSys.setMenuBar(self.menubar)

        ## Status Bar
        self.statusbar = QtWidgets.QStatusBar(FacialSys)
        self.statusbar.setObjectName("statusbar")
        FacialSys.setStatusBar(self.statusbar)

        ### File Menu
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menubar.addAction(self.menuFile.menuAction())
        #### Load Image
        self.actionImport_Image = QtWidgets.QAction(FacialSys)
        self.actionImport_Image.setShortcut("Ctrl+I")
        self.actionImport_Image.setObjectName("actionLoad_Image")
        self.menuFile.addAction(self.actionImport_Image)
        #### Exit app
        self.actionExit = QtWidgets.QAction(FacialSys)
        self.actionExit.setObjectName("actionExit")
        self.actionExit.setShortcut("Ctrl+Q")
        self.actionExit.triggered.connect(self.exit_application)
        self.menuFile.addAction(self.actionExit)

        ## PCA Input
        self.PCA_input_vlayout = QtWidgets.QHBoxLayout(self.PCA_input)
        self.PCA_input_vlayout.setObjectName("PCA_input_hlayout")
        self.PCA_input_figure = plt.figure()
        self.PCA_input_figure_canvas = FigureCanvas(self.PCA_input_figure)
        self.PCA_input_vlayout.addWidget(self.PCA_input_figure_canvas)
        ## PCA Output
        self.PCA_output_vlayout = QtWidgets.QHBoxLayout(self.PCA_output)
        self.PCA_output_vlayout.setObjectName("PCA_output_hlayout")
        self.PCA_output_figure = plt.figure()
        self.PCA_output_figure_canvas = FigureCanvas(self.PCA_output_figure)
        self.PCA_output_vlayout.addWidget(self.PCA_output_figure_canvas)
        ## End of PCA

        ## Detection Input
        self.detection_input_vlayout = QtWidgets.QHBoxLayout(self.detection_input_frame)
        self.detection_input_vlayout.setObjectName("detection_input_vlayout")
        self.detection_input_figure = plt.figure()
        self.detection_input_figure_canvas = FigureCanvas(self.detection_input_figure)
        self.detection_input_figure_canvas.figure.subplots_adjust(
            left=0, right=1, bottom=0.05, top=0.95
        )
        self.detection_input_vlayout.addWidget(self.detection_input_figure_canvas)
        ## Detection Output
        self.detection_output_vlayout = QtWidgets.QHBoxLayout(
            self.detection_output_frame
        )
        self.detection_output_vlayout.setObjectName("detection_output_vlayout")
        self.detection_output_figure = plt.figure()
        self.detection_output_figure_canvas = FigureCanvas(self.detection_output_figure)
        self.detection_output_figure_canvas.figure.subplots_adjust(
            left=0, right=1, bottom=0, top=1
        )
        self.detection_output_vlayout.addWidget(self.detection_output_figure_canvas)
        ## End of Detection

        self.retranslateUi(FacialSys)
        self.tabWidget.setCurrentIndex(11)
        QtCore.QMetaObject.connectSlotsByName(FacialSys)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))

        font_global_thresholds_label = QtGui.QFont()
        font_global_thresholds_label.setPointSize(14)
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.pca_tab),
            _translate("MainWindow", "PCA"),
        )
        self.toggle.setText(_translate("MainWindow", "Toggle Query"))
        self.apply_PCA.setText(_translate("MainWindow", "Apply"))
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.tab_12),
            _translate("MainWindow", "Face Detection"),
        )
        self.last_stage_threshold_label.setText(
            _translate("MainWindow", "Last Stage Threshold")
        )
        self.apply_detection.setText(_translate("MainWindow", "Apply"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionImport_Image.setText(_translate("MainWindow", "Import Image"))
        self.actionExit.setText(_translate("MainWindow", "Exit app"))

    def exit_application(self):
        sys.exit()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
