import os

import cv2
import numpy as np


class PCA_class:
    """
    Principal Component Analysis (PCA) class.

    Parameters
    ----------
    n_components : int, optional
        Number of components to keep.
    svd_solver : str, optional
        Solver to use for the decomposition. Currently not used.
    """

    def __init__(self, n_components=None, svd_solver="full"):
        self.n_components = n_components
        self.svd_solver = svd_solver
        self.mean = None
        self.components = None
        self.explained_variance_ratio_ = None

    def fit(self, X, method="svd"):
        """
        Fit the model with X using the specified method.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        method : str, optional
            Method to use for the decomposition ('svd' or 'eigen').
        """
        # Mean centering
        self.mean = np.mean(X, axis=0)  # Compute the mean of X
        X = X - self.mean  # Subtract the mean from X

        # Handle number of components
        if self.n_components is None:
            self.n_components = min(X.shape) - 1

        if method == "svd":
            # Compute SVD
            U, S, Vt = np.linalg.svd(X, full_matrices=False)  # Perform SVD on X

            # Compute explained variance ratio
            explained_variance_ = (S**2) / (
                X.shape[0] - 1
            )  # Compute the explained variance
            total_variance = explained_variance_.sum()  # Compute the total variance
            explained_variance_ratio_ = (
                explained_variance_ / total_variance
            )  # Compute the explained variance ratio

            self.components = Vt[
                : self.n_components
            ]  # Keep the first n_components components
            self.explained_variance_ratio_ = explained_variance_ratio_[
                : self.n_components
            ]  # Keep the explained variance ratio for the first n_components

        elif method == "eigen":
            # Compute covariance matrix
            covariance_matrix = np.dot(X.T, X)  # Compute the covariance matrix of X

            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(
                covariance_matrix
            )  # Compute the eigenvalues and eigenvectors of the covariance matrix

            # Sort eigenvalues and eigenvectors by decreasing eigenvalues
            idx = eigenvalues.argsort()[
                ::-1
            ]  # Get the indices that would sort the eigenvalues in decreasing order
            eigenvalues = eigenvalues[idx]  # Sort the eigenvalues
            eigenvectors = eigenvectors[:, idx]  # Sort the eigenvectors accordingly

            # Compute explained variance ratio
            total_variance = eigenvalues.sum()  # Compute the total variance
            explained_variance_ratio_ = (
                eigenvalues / total_variance
            )  # Compute the explained variance ratio

            self.components = eigenvectors[
                :, : self.n_components
            ].T  # Keep the first n_components components
            self.explained_variance_ratio_ = explained_variance_ratio_[
                : self.n_components
            ]  # Keep the explained variance ratio for the first n_components

        else:
            raise ValueError("Invalid method. Expected 'svd' or 'eigen'.")
        return self

    def project(self, X):
        """
        Apply dimensionality reduction to X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            Transformed values.
        """
        X = X - self.mean  # Mean centering
        return np.dot(X, self.components.T)  # Project X onto the principal components

    def fit_transform(self, X, method="svd"):
        """
        Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        method : str, optional
            Method to use for the decomposition ('svd' or 'eigen').

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            Transformed values.
        """
        self.fit(X, method)  # Fit the model with X
        return self.transform(X)  # Apply the dimensionality reduction on X


def store_dataset_method_one(dataset_dir):
    faces_train = dict()
    faces_test = dict()

    # Initialize a variable to store the size of the first image
    first_image_size = None

    for subject in os.listdir(dataset_dir):
        images = []
        if subject == "no match":
            # Add to self.faces_test['no match']
            subject_dir = os.path.join(dataset_dir, subject)
            faces_test[subject] = [
                cv2.imread(os.path.join(subject_dir, filename), cv2.IMREAD_GRAYSCALE)
                for filename in sorted(os.listdir(subject_dir))
            ]
            continue

        # if subjcet is not 'no match'
        subject_dir = os.path.join(dataset_dir, subject)

        for filename in sorted(os.listdir(subject_dir)):
            image = cv2.imread(
                os.path.join(subject_dir, filename), cv2.IMREAD_GRAYSCALE
            )

            # If first_image_size is None, this is the first image
            # So, store its size and don't resize it
            if first_image_size is None:
                first_image_size = image.shape

            images.append(image)

        # Warning for the user that the minimum number of faces per subject is 5
        if len(images) >= 5:
            # Split the data: 80% for training, 20% for testing
            split_index = int(len(images) * 0.8)
            faces_train[subject] = images[:split_index]
            faces_test[subject] = images[split_index:]

    # Resize images of 'no match' to match the size of the first image
    if "no match" in faces_test:
        for i, image in enumerate(faces_test["no match"]):
            faces_test["no match"][i] = cv2.resize(
                image, (first_image_size[1], first_image_size[0])
            )

    return faces_train, faces_test, first_image_size


def train_pca(faces_train):
    faces_train_pca = faces_train.copy()
    # Use list comprehension to flatten images and create labels
    train_faces_matrix = []
    train_faces_labels = []
    for subject, images in faces_train_pca.items():
        train_faces_matrix.extend(img.flatten() for img in images)
        train_faces_labels.extend([subject] * len(images))
    train_faces_matrix = np.array(train_faces_matrix)
    # Create instance of the class
    pca = PCA_class().fit(train_faces_matrix, "svd")
    sorted_eigen_values = np.sort(pca.explained_variance_ratio_)[::-1]
    cumulative_variance = np.cumsum(sorted_eigen_values)
    # let's assume that we will consider just 90 % of variance in the data, so will consider just first 101 principal components
    upto_index = np.where(cumulative_variance < 0.9)[0][-1]  # the last one
    no_principal_components = upto_index + 1
    PCA_eigen_faces = pca.components[:no_principal_components]
    PCA_weights = (
        PCA_eigen_faces
        @ (train_faces_matrix - np.mean(train_faces_matrix, axis=0)).transpose()
    )
    return train_faces_matrix, train_faces_labels, PCA_weights, PCA_eigen_faces


def recognise_face(
    test_face,
    first_image_size,
    train_faces_matrix,
    train_faces_labels,
    PCA_weights,
    PCA_eigen_faces,
    face_recognition_threshold,
):

    test_face_to_recognise = test_face.copy()
    if test_face_to_recognise.shape != first_image_size:
        test_face_to_recognise = cv2.resize(test_face_to_recognise, first_image_size)
    test_face_to_recognise = test_face_to_recognise.reshape(1, -1)
    # print(PCA_eigen_faces.shape)
    # print(test_face_to_recognise.shape)
    # print(np.mean(train_faces_matrix, axis=0,keepdims=True).shape)
    test_face_weights = (
        PCA_eigen_faces
        @ (test_face_to_recognise - np.mean(train_faces_matrix, axis=0)).transpose()
    )
    # test_face_weights = (PCA_eigen_faces @ (test_face_to_recognise - np.mean(train_faces_matrix, axis=0,keepdims=True)).transpose())
    distances = np.linalg.norm(
        PCA_weights - test_face_weights, axis=0
    )  # compare row wise
    best_match = np.argmin(distances)
    best_match_subject, best_match_subject_distance = (
        train_faces_labels[best_match],
        distances[best_match],
    )
    if best_match_subject_distance > face_recognition_threshold:
        best_match_subject = "no match"
    else:
        best_match_subject = best_match_subject

    return best_match_subject, best_match_subject_distance, best_match


def test_faces_and_labels(test_faces_dict) -> (list, list):  # type: ignore
    test_faces_dictionary = test_faces_dict.copy()
    # Flatten the test faces and create corresponding labels
    test_faces = []
    test_labels = []
    for subject, faces in test_faces_dictionary.items():
        for face in faces:
            test_faces.append(face)
            # Encode the labels, no match -> 0, otherwise -> 1
            label = 0 if subject == "no match" else 1
            test_labels.append(label)
    return test_faces, test_labels
