import numpy as np
from napari_convpaint.conv_paint_utils import (Hookmodel, filter_image_multichannels, get_features_current_layers, predict_image)
from sklearn.ensemble import RandomForestClassifier
from time import time

def selfpred_convpaint(image, labels, layer_list=[0], scalings=[1,2], model="vgg16", random_state=None):
    '''
    Predict full semantic segmentation of an image using labels for this same image with ConvPaint and VGG16 as feature extractor (train and predict on same image).
    INPUT:
        image (np.ndarray): image to predict on. Shape (H, W, C) or (C, H, W)
        labels (np.ndarray): labels for the image. Shape (H, W), same dimensions as image
        layer_list (list of int): list of layer indices to use for feature extraction with vgg16
        scalings (list of int): list of scalings to use for feature extraction with vgg16
        model (str): model to use for feature extraction
        random_state (int): random state for the random forest classifier
    OUTPUTS:
        predicted (np.ndarray): predicted segmentation. Shape (H, W)
    '''
    # Ensure (H, W, C) - expected by ConvPaint
    if image.ndim == 3 and image.shape[2] < 4:
        image = np.moveaxis(image, 2, 0)

    # Get the features, targets and model
    features_annot, targets, model = get_features_targets_model(
        image, labels, layer_list=layer_list, scalings=scalings, model_name=model)

    # Train the classifier
    features_train, labels_train = features_annot, targets
    random_forest = RandomForestClassifier(random_state=random_state)
    random_forest.fit(features_train, labels_train)

    # Predict on the image
    predicted = predict_image(
        image, model, random_forest, scalings=scalings,
        order=1, use_min_features=False, image_downsample=1)
    return predicted

def get_features_targets_model(image, labels, layer_list=[0], scalings=[1,2], model_name="vgg16"):
    '''
    Extract features of annotated pixels (not entire image) from an image using ConvPaint and VGG16 as feature extractor.
    Return the extracted features together with their targets and the model used for feature extraction.
    INPUT:
        image (np.ndarray): image to extract features from. Shape (C, H, W)
        labels (np.ndarray): labels for the image. Shape (H, W), same dimensions as image
        layer_list (list of int): list of layer indices to use for feature extraction with vgg16
        scalings (list of int): list of scalings to use for feature extraction with vgg16
        model (str): model to use for feature extraction
    OUTPUTS:
        features (np.ndarray): extracted features. Shape (H, W, n_features)
    '''
    # Define the model
    model = Hookmodel(model_name=model_name)
    # Ensure the layers are given as a list
    if isinstance(layer_list, int):
        layer_list = [layer_list]
    # Read out the layer names
    all_layers = [key for key in model.module_dict.keys()]
    layers = [all_layers[i] for i in layer_list]
    # Register the hooks for the selected layers
    model.register_hooks(selected_layers=layers)
    # Get the features and targets
    features_annot, targets = get_features_current_layers(
        model=model, image=image, annotations=labels, scalings=scalings,
        order=1, use_min_features=False, image_downsample=1)
    return features_annot, targets, model


def generate_convpaint_tag(layer_list, scalings, model="vgg16"):
    '''
    Generate a tag for a ConvPaint prediction based on the model, the layer list and the scalings
    '''
    model_pref = f'_{model}' if model != 'vgg16' else ''
    layer_pref = f'_l-{str(layer_list)[1:-1].replace(", ", "-")}'# if layer_list != [0] else ''
    scalings_pref = f'_s-{str(scalings)[1:-1].replace(", ", "-")}'# if scalings != [1,2] else ''
    pred_tag = f"convpaint{model_pref}{layer_pref}{scalings_pref}"
    return pred_tag


def extract_convpaint_features(image, layer_list=[0], scalings=[1,2], model_name="vgg16", order=0):
    # Define the model
    model = Hookmodel(model_name=model_name)
    # Ensure the layers are given as a list
    if isinstance(layer_list, int):
        layer_list = [layer_list]
    # Read out the layer names
    all_layers = [key for key in model.module_dict.keys()]
    layers = [all_layers[i] for i in layer_list]
    # Register the hooks for the selected layers
    model.register_hooks(selected_layers=layers)
    features = filter_image_multichannels(image, model, scalings=scalings, order=order, image_downsample=1)
    return features

def time_convpaint(image, labels=None, layer_list=[0], scalings=[1,2], model_name="vgg16", order=0, random_state=None):
    """
    Time different steps of slefprediction using Convpaint.
    """
    # Load the model
    t_start_load = time()
    # Ensure (H, W, C) - expected by ConvPaint
    if image.ndim == 3 and image.shape[2] < 4:
        image = np.moveaxis(image, 2, 0)
    # Define the model
    model = Hookmodel(model_name=model_name)
    # Ensure the layers are given as a list
    if isinstance(layer_list, int):
        layer_list = [layer_list]
    # Read out the layer names
    all_layers = [key for key in model.module_dict.keys()]
    layers = [all_layers[i] for i in layer_list]
    # Register the hooks for the selected layers
    model.register_hooks(selected_layers=layers)
    t_load = time() - t_start_load

    # Extract features for the full image
    t_start_features_full = time()
    features = filter_image_multichannels(image, model, scalings=scalings, order=order, image_downsample=1)
    t_features_full = time() - t_start_features_full
    if labels is None:
        return t_load, t_features_full, None, None, None, None

    # Do the full self-prediction (extracting only annot features for training) if labels are given
    t_start_features_train = time()
    features_annot, targets = get_features_current_layers(
        model=model, image=image, annotations=labels, scalings=scalings,
        order=1, use_min_features=False, image_downsample=1)
    t_features_train = time() - t_start_features_train
    # Train the classifier
    t_start_train = time()
    features_train, labels_train = features_annot, targets
    random_forest = RandomForestClassifier(random_state=random_state)
    random_forest.fit(features_train, labels_train)
    t_train = time() - t_start_train
    # Predict on the image
    t_start_pred = time()
    predicted = predict_image(
        image, model, random_forest, scalings=scalings,
        order=1, use_min_features=False, image_downsample=1)
    t_pred = time() - t_start_pred
    # Also calculate the total time for the self-prediction
    t_selfpred = time() - t_start_features_train
    return t_load, t_features_full, t_features_train, t_train, t_pred, t_selfpred
