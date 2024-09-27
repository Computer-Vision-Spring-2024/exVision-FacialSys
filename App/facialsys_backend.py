import os

import numpy as np

# To prevent conflicts with pyqt6
os.environ["QT_API"] = "PyQt5"
# To solve the problem of the icons with relative path
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import os
from math import cos, sin
from typing import *

import cv2
import numpy as np

# in CMD: pip install qdarkstyle -> pip install pyqtdarktheme
import qdarktheme
from facialsys_ui import Ui_MainWindow
from features import *
from PIL import Image
from PyQt5 import QtGui

# imports
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox
from scipy.signal import convolve2d
from skimage.transform import rescale, resize
from utils.detection_utils import *
from utils.helper_functions import *
from utils.recognition_utils import *


class BackendClass(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        ### ==== PCA ==== ###
        self.PCA_test_image_index = 30
        self.face_recognition_threshold = 2900
        # Configured by the user
        self.structure_number = "one"  # Dataset folder, containing subfolders named after subjects, each containing a minimum of 5 images, with extra images limited to the quantity of the smallest subject folder.
        self.dataset_dir = "../face_recognition_dataset"
        self.faces_train, self.faces_test, self.first_image_size = (
            store_dataset_method_one(self.dataset_dir)
        )
        (
            self.train_faces_matrix,
            self.train_faces_labels,
            self.PCA_eigen_faces,
            self.PCA_weights,
        ) = train_pca(self.faces_train)
        self.test_faces_list, self.test_labels_list = test_faces_and_labels(
            self.faces_test
        )
        self.PCA_test_img = self.test_faces_list[self.PCA_test_image_index]
        self.display_image(
            self.test_faces_list[self.PCA_test_image_index],
            self.ui.PCA_input_figure_canvas,
            "Query",
            True,
        )

        # Test size is 20% by default
        # PCA cumulativa variance is 90% by default

        # PCA Buttons
        self.ui.toggle.clicked.connect(self.toggle_PCA_test_image)
        self.ui.apply_PCA.clicked.connect(self.apply_PCA)

        ### ==== Detection ==== ###
        self.detection_original_image = None
        self.detection_thumbnail_image = None
        self.detection_original_float = None
        self.detection_grayscale_image = None
        self.detection_integral_image = None
        self.ui.apply_detection.setEnabled(False)
        self.features_per_window = get_number_of_features_per_window()
        self.detection_models = upload_cascade_adaboost("../15x15_window_size_model")
        self.weak_classifiers = self.detection_models["1st"]
        self.weak_classifiers_2 = self.detection_models["2nd"]
        self.weak_classifiers_3 = self.detection_models["3rd"]
        self.last_stage_threshold = 0
        self.ui.apply_detection.clicked.connect(self.apply_face_detection)
        self.ui.last_stage_threshold_spinbox.valueChanged.connect(
            self.get_face_detection_parameters
        )
        self.last_stage_info = None
        self.detection_output_image = None

        ### ==== General ==== ###
        # Connect menu action to load_image
        self.ui.actionImport_Image.triggered.connect(self.load_image)

        # Change the icon and title of the app
        self.change_the_icon()

    def change_the_icon(self):
        self.setWindowIcon(QtGui.QIcon("assets/app_icon.png"))
        self.setWindowTitle("FacialSys")

    def load_image(self):
        # Open file dialog if file_path is not provided
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "Images",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.ppm *.pgm)",
        )

        if file_path and isinstance(file_path, str):
            # Read the matrix, convert to rgb
            img = cv2.imread(file_path, 1)
            img = convert_BGR_to_RGB(img)
            current_tab = self.ui.tabWidget.currentIndex()

            if current_tab == 0:
                self.PCA_test_img = convert_to_gray(img)
                self.display_image(
                    self.PCA_test_img,
                    self.ui.PCA_input_figure_canvas,
                    "Query",
                    True,
                )
                self.apply_PCA()
            elif current_tab == 1:
                self.detection_original_image = Image.open(file_path)
                self.detection_thumbnail_image = resize_image_object(
                    self.detection_original_image, (384, 288)
                )
                self.detection_original_float = to_float_array(
                    self.detection_thumbnail_image
                )
                self.detection_grayscale_image = gleam_converion(
                    self.detection_original_float
                )
                self.detection_integral_image = integrate_image(
                    self.detection_grayscale_image
                )

                self.display_image(
                    self.detection_original_float,
                    self.ui.detection_input_figure_canvas,
                    "Input Image",
                    False,
                )
                self.ui.apply_detection.setEnabled(True)

    def display_image(
        self, image, canvas, title, grey, hist_or_not=False, axis_disabled="off"
    ):
        """ "
        Description:
            - Plots the given (image) in the specified (canvas)
        """
        canvas.figure.clear()
        ax = canvas.figure.add_subplot(111)
        if not hist_or_not:
            if not grey:
                ax.imshow(image)
            elif grey:
                ax.imshow(image, cmap="gray")
        else:
            self.ui.histogram_global_thresholds_label.setText(" ")
            if grey:
                ax.hist(image.flatten(), bins=256, range=(0, 256), alpha=0.75)
                for thresh in self.global_thresholds[0]:
                    ax.axvline(x=thresh, color="r")
                    thresh = int(thresh)
                    # Convert the threshold to string with 3 decimal places and add it to the label text
                    current_text = self.ui.histogram_global_thresholds_label.text()
                    self.ui.histogram_global_thresholds_label.setText(
                        current_text + " " + str(thresh)
                    )
            else:
                image = convert_to_gray(image)
                ax.hist(image.flatten(), bins=256, range=(0, 256), alpha=0.75)
                for thresh in self.global_thresholds[0]:
                    ax.axvline(x=thresh, color="r")
                    thresh = int(thresh)
                    # Convert the threshold to string with 3 decimal places and add it to the label text
                    current_text = self.ui.histogram_global_thresholds_label.text()
                    self.ui.histogram_global_thresholds_label.setText(
                        current_text + " " + str(thresh)
                    )

        ax.axis(axis_disabled)
        ax.set_title(title)
        canvas.figure.subplots_adjust(left=0.1, right=0.90, bottom=0.08, top=0.95)
        canvas.draw()

    # @staticmethod
    def display_selection_dialog(self, image):
        """
        Description:
            - Shows a message dialog box to determine whether this is the a template or the target image in SIFT

        Args:
            - image: The image to be displayed.
        """
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Question)
        msgBox.setText("Select an Image")
        msgBox.setWindowTitle("Image Selection")
        msgBox.setMinimumWidth(150)

        # Set custom button text
        msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msgBox.button(QMessageBox.Yes).setText("Target Image")
        msgBox.button(QMessageBox.No).setText("Template")

        # Executing the message box
        response = msgBox.exec()
        if response == QMessageBox.Rejected:
            return
        else:
            if response == QMessageBox.Yes:
                self.sift_target_image = image
                self.display_image(
                    image,
                    self.ui.input_1_figure_canvas,
                    "Target Image",
                    False,
                )
            elif response == QMessageBox.No:
                self.sift_template_image = image
                self.display_image(
                    image,
                    self.ui.input_2_figure_canvas,
                    "Template Image",
                    False,
                )

    ## ============== PCA ============== ##

    def apply_PCA(self):
        self.ui.PCA_output_figure.clear()
        test_image = self.PCA_test_img.copy()

        best_match_subject, best_match_subject_distance, best_match_indx = (
            recognise_face(
                test_image,
                self.first_image_size,
                self.train_faces_matrix,
                self.train_faces_labels,
                self.PCA_weights,
                self.PCA_eigen_faces,
                self.face_recognition_threshold,
            )
        )
        if best_match_subject_distance < self.face_recognition_threshold:
            # Visualize
            self.display_image(
                self.train_faces_matrix[best_match_indx].reshape(self.first_image_size),
                self.ui.PCA_output_figure_canvas,
                f"Best match:{best_match_subject}",
                True,
            )
        else:
            self.display_image(
                np.full_like(
                    self.train_faces_matrix[0].reshape(self.first_image_size),
                    255,
                    dtype=np.uint8,
                ),
                self.ui.PCA_output_figure_canvas,
                "No matching subject",
                True,
            )
        self.ui.PCA_output_figure_canvas.draw()

    def toggle_PCA_test_image(self):
        self.ui.PCA_output_figure.clear()
        self.PCA_test_image_index += 1
        test_labels_list = self.test_labels_list.copy()
        self.PCA_test_image_index = self.PCA_test_image_index % len(test_labels_list)
        self.PCA_test_img = self.test_faces_list[self.PCA_test_image_index]
        self.display_image(
            self.PCA_test_img,
            self.ui.PCA_input_figure_canvas,
            "Query",
            True,
        )

    ## ============== Detection ============== ##
    def get_face_detection_parameters(self):
        self.last_stage_threshold = self.ui.last_stage_threshold_spinbox.value()

    def apply_face_detection(self):
        self.get_face_detection_parameters()
        rows, cols = self.detection_integral_image.shape[:2]
        HALF_WINDOW = WINDOW_SIZE // 2

        face_positions_1 = list()
        face_positions_2 = list()
        face_positions_3 = list()
        face_positions_3_strength = list()

        normalized_integral = integrate_image(
            normalize(self.detection_grayscale_image)
        )  # to reduce lighting variance

        for row in range(HALF_WINDOW + 1, rows - HALF_WINDOW):
            for col in range(HALF_WINDOW + 1, cols - HALF_WINDOW):
                curr_window = normalized_integral[
                    row - HALF_WINDOW - 1 : row + HALF_WINDOW + 1,
                    col - HALF_WINDOW - 1 : col + HALF_WINDOW + 1,
                ]

                # First cascade stage
                probably_face, _ = strong_classifier(curr_window, self.weak_classifiers)
                if probably_face < 0.5:
                    continue
                face_positions_1.append((row, col))

                probably_face, strength = strong_classifier(
                    curr_window, self.weak_classifiers_2
                )
                if probably_face < 0.5:
                    continue
                face_positions_2.append((row, col))

                probably_face, strength = strong_classifier(
                    curr_window, self.weak_classifiers_3
                )
                if probably_face < 0.5:
                    continue
                face_positions_3.append((row, col))
                face_positions_3_strength.append(strength)

        self.last_stage_info = (face_positions_3, face_positions_3_strength)
        self.truncate_candidates()

    def render_candidates(self, image: Image.Image, candidates: List[Tuple[int, int]]):
        HALF_WINDOW = WINDOW_SIZE // 2
        canvas = to_float_array(image.copy())
        for row, col in candidates:
            canvas[
                row - HALF_WINDOW - 1 : row + HALF_WINDOW, col - HALF_WINDOW - 1, :
            ] = [1.0, 0.0, 0.0]
            canvas[
                row - HALF_WINDOW - 1 : row + HALF_WINDOW, col + HALF_WINDOW - 1, :
            ] = [1.0, 0.0, 0.0]
            canvas[
                row - HALF_WINDOW - 1, col - HALF_WINDOW - 1 : col + HALF_WINDOW, :
            ] = [1.0, 0.0, 0.0]
            canvas[
                row + HALF_WINDOW - 1, col - HALF_WINDOW - 1 : col + HALF_WINDOW, :
            ] = [1.0, 0.0, 0.0]

        self.detection_output_image = canvas
        self.display_image(
            self.detection_output_image,
            self.ui.detection_output_figure_canvas,
            "Output Image",
            False,
        )

    def truncate_candidates(self):
        filtered_faces = list()
        expected_faces = np.argwhere(
            np.array(self.last_stage_info[1]) > self.last_stage_threshold
        )
        for i in range(len(self.last_stage_info[0])):
            if [i] in expected_faces:
                filtered_faces.append(self.last_stage_info[0][i])

        self.render_candidates(self.detection_thumbnail_image, filtered_faces)


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    MainWindow = BackendClass()
    MainWindow.show()
    qdarktheme.setup_theme("dark")
    sys.exit(app.exec_())
