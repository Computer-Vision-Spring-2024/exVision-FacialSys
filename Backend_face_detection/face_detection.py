import numpy as np
from PIL import Image, ImageOps
from Features import *
from typing import *  
import pickle
import os


WINDOW_SIZE = 15

# size and location NamedTuple objects 
Size = NamedTuple("Size", [("height", int), ("width", int)])
Location = NamedTuple("Location", [("top", int), ("left", int)])

def resize_image_object(img, target_size):
    thumbnail_image = img.copy()
    thumbnail_image.thumbnail(target_size, Image.LANCZOS) # anti-alising-resize
    return thumbnail_image

def to_float_array(img): 
    return np.array(img).astype(np.float32) / 255.0 # float division 

def to_image(arr):
    return Image.fromarray(np.uint8(arr * 255.0))


def gamma(channel, coeff = 2.2):
    return channel**(1./coeff)

def gleam_converion(img):
    return np.sum(gamma(img), axis=2) / img.shape[2] # divide by 3 


def integrate_image(img):
    """The padding compensates for the loss that might happen in differentiation"""
    integral = np.cumsum(np.cumsum(img, axis= 0), axis=1) # 2d integral
    return np.pad(integral, (1, 1), 'constant', constant_values=(0, 0))[:-1,:-1] 

def possible_combinations(size, window_size = WINDOW_SIZE):
    return range(0, window_size - size + 1) # size can be height or width  

def possible_locations(base_size: Size, window_size = WINDOW_SIZE):
    return (Location(left=x, top=y)
            for x in possible_combinations(base_size.width, window_size)
            for y in possible_combinations(base_size.height, window_size))

def possible_feature_shapes(base_size: Size, window_size = WINDOW_SIZE):
    base_height = base_size.height
    base_width = base_size.width
    return (Size(height=height, width=width)
            for width in range(base_width, window_size+1, base_width)
            for height in range(base_height, window_size+ 1, base_height))


# this is helper types 
ThresholdPolarity = NamedTuple('ThresholdPolarity', [('threshold', float), ('polarity', float)])

ClassifierResult = NamedTuple('ClassifierResult', [('threshold', float), ('polarity', int), 
                                                   ('classification_error', float),
                                                   ('classifier', Callable[[np.ndarray], float])])

WeakClassifier = NamedTuple('WeakClassifier', [('threshold', float), ('polarity', int), 
                                               ('alpha', float), 
                                               ('classifier', Callable[[np.ndarray], float])])



def weak_classifier(window: np.ndarray, feature: Feature, polarity: float, theta: float):
    return (np.sign((polarity * theta) - (polarity* feature(window))) + 1) // 2
    # computational optimization

def run_weak_classifier(window: np.ndarray, weak_classier: WeakClassifier): 
    return weak_classifier(window, weak_classier.classifier, weak_classier.polarity, weak_classier.threshold)



def strong_classifier(window: np.ndarray, weak_classifiers: List[WeakClassifier]):
    sum_hypotheses = 0.
    sum_alpha = 0.
    for cl in weak_classifiers:
        sum_hypotheses += cl.alpha * run_weak_classifier(window, cl)
        sum_alpha += cl.alpha
    vote = 1 if (sum_hypotheses >= 0.5 * sum_alpha) else 0
    how_strong = sum_hypotheses -  0.5 * sum_alpha
    return (vote, how_strong)

def normalize(im) :
    return (im - im.mean()) / im.std()


# -------------------------------------------------------------------------------------------------------------------------

original_image = Image.open("images/solvay-conference.jpg")
thumbnail_image = resize_image_object(original_image, (384, 288)) # this image is displayed as input image 
original_float = to_float_array(thumbnail_image)
grayscale_image = gleam_converion(original_float) # the image upon which we apply the algorithm
integral_image =  integrate_image(grayscale_image)

#---------------------------------------------------------------------------------------------------------------------------

def get_number_of_features_per_window():

    feature2h = list(Feature2h(location.left, location.top, shape.width, shape.height)
                    for shape in possible_feature_shapes(Size(1,2), WINDOW_SIZE)
                    for location in possible_locations(shape, WINDOW_SIZE))
    #--------------------------------------------------------------
    feature2v = list(Feature2v(location.left, location.top, shape.width, shape.height)
                    for shape in possible_feature_shapes(Size(2,1), WINDOW_SIZE)
                    for location in possible_locations(shape, WINDOW_SIZE))
    #--------------------------------------------------------------
    feature3h = list(Feature3h(location.left, location.top, shape.width, shape.height)
                    for shape in possible_feature_shapes(Size(1,3), WINDOW_SIZE)
                    for location in possible_locations(shape, WINDOW_SIZE))
    #--------------------------------------------------------------
    feature3v = list(Feature3v(location.left, location.top, shape.width, shape.height)
                    for shape in possible_feature_shapes(Size(3,1), WINDOW_SIZE)
                    for location in possible_locations(shape, WINDOW_SIZE))
    #---------------------------------------------------------------
    feature4 = list(Feature4(location.left, location.top, shape.width, shape.height)
                    for shape in possible_feature_shapes(Size(2,2), WINDOW_SIZE)
                    for location in possible_locations(shape, WINDOW_SIZE))

    features_per_window =  feature2h + feature2v + feature3h + feature3v + feature4

    return features_per_window

# --------------------------------------------------- upload the model ---------------------------------

folder_path = "new_model_15_window" 
def upload_cascade_optimization(dir = folder_path):
    models = {
        "1st": list(),
        "2nd": list(),
        "3rd": list()
    } 

    for filename in os.listdir(dir):
        if filename.endswith('.pickle'):  # Check if the file is a pickle file
            file_path = os.path.join(dir, filename)
            with open(file_path, 'rb') as file:
                loaded_objects = pickle.load(file)
                models[filename[:3]].append(loaded_objects)
    return models

models = upload_cascade_optimization()
weak_classifiers = models["1st"]
weak_classifiers_2 = models["2nd"]
weak_classifiers_3 = models["3rd"]

# -------------------------------------------------------------------------------------------------------


rows, cols = integral_image.shape[:2]
HALF_WINDOW = WINDOW_SIZE // 2

face_positions_1 = list()
face_positions_2 = list()
face_positions_3 = list()
face_positions_3_strenght = list()

normalized_integral = integrate_image(normalize(grayscale_image)) # to reduce lighting variance 

for row in range(HALF_WINDOW+1, rows - HALF_WINDOW):
    for col in range(HALF_WINDOW + 1, cols - HALF_WINDOW):
        curr_window = normalized_integral[row-HALF_WINDOW-1: row+HALF_WINDOW+1, col - HALF_WINDOW-1: col + HALF_WINDOW+1 ]

        # First cascade stage 
        probably_face, _ = strong_classifier(curr_window, weak_classifiers)
        if probably_face < 0.5:
            continue
        face_positions_1.append((row, col))

        probably_face, strength= strong_classifier(curr_window, weak_classifiers_2)
        if probably_face < 0.5:
            continue
        face_positions_2.append((row, col))
        
        probably_face, strength = strong_classifier(curr_window, weak_classifiers_3)
        if probably_face < 0.5:
            continue
        face_positions_3.append((row, col))
        face_positions_3_strenght.append(strength)

#---------------------------------------------------------------------------------
def render_candidates(image: Image.Image, candidates: List[Tuple[int, int]]):
    canvas = to_float_array(image.copy())
    for row, col in candidates:
        canvas[row-HALF_WINDOW-1:row+HALF_WINDOW, col-HALF_WINDOW-1, :] = [1., 0., 0.]
        canvas[row-HALF_WINDOW-1:row+HALF_WINDOW, col+HALF_WINDOW-1, :] = [1., 0., 0.]
        canvas[row-HALF_WINDOW-1, col-HALF_WINDOW-1:col+HALF_WINDOW, :] = [1., 0., 0.]
        canvas[row+HALF_WINDOW-1, col-HALF_WINDOW-1:col+HALF_WINDOW, :] = [1., 0., 0.]
    return to_image(canvas)

# -----------------------------------------------------------------------------

def truncate_candidates(threshold_value):
    filtered_faces = list()
    expected_faces = np.argwhere(np.array(face_positions_3_strenght) > threshold_value )
    for i in range(len(face_positions_3)):
        if [i] in expected_faces:
            filtered_faces.append(face_positions_3[i]) 
    return filtered_faces 

threshold_value = ...
filtered_faces = truncate_candidates(threshold_value)
render_candidates(thumbnail_image, filtered_faces)


