from ilastik.napari.filters import (FilterSet,
                                    Gaussian,
                                    LaplacianOfGaussian,
                                    GaussianGradientMagnitude,
                                    DifferenceOfGaussians,
                                    StructureTensorEigenvalues,
                                    HessianOfGaussianEigenvalues)
import itertools
from sparse import COO
from ilastik.napari.classifier import NDSparseClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from time import time


# Define the filter set and scales
FILTER_LIST = (Gaussian,
               LaplacianOfGaussian,
               GaussianGradientMagnitude,
               DifferenceOfGaussians,
               StructureTensorEigenvalues,
               HessianOfGaussianEigenvalues)
SCALE_LIST = (0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10.0)
# Generate all combinations of FILTER_LIST and SCALE_LIST
ALL_FILTER_SCALING_COMBOS = list(itertools.product(range(len(FILTER_LIST)), range(len(SCALE_LIST))))
# Create a FilterSet with all combinations
FILTERS = tuple(FILTER_LIST[row](SCALE_LIST[col]) for row, col in sorted(ALL_FILTER_SCALING_COMBOS))
FILTER_SET = FilterSet(filters=FILTERS)

def extract_ila_features_multichannel(image, filter_set=FILTER_SET):
    """
    Feature Extraction with Ilastik for multichannel images. Concatenates the feature maps of each channel.
    INPUT:
        image (np.ndarray): image to predict on; shape (C, H, W) or (H, W, C)
        filter_set (FilterSet from ilastik.napari.filters): filter set to use for feature extraction
    OUTPUT:
        features (np.ndarray): feature map (H, W, C) with C being the number of features per pixel
    """
    # Ensure (H, W, C) - expected by Ilastik
    if len(image.shape) == 3 and image.shape[0] < 4:
        image = np.moveaxis(image, 0, -1)
    # Loop over channels, extract features and concatenate them
    for c in range(image.data.shape[2]):
        channel_feature_map = filter_set.transform(np.asarray(image[:,:,c]))
        if c == 0:
            feature_map = channel_feature_map
        else:
            feature_map = np.concatenate((feature_map, channel_feature_map), axis=2)
    return feature_map

def extract_ilastik_features(image, filter_set=FILTER_SET):
    """
    Feature Extraction with Ilastik for single- or multi-channel images.
    INPUT:
        image (np.ndarray): image to predict on; shape (C, H, W) or (H, W, C) or (H, W)
        filter_set (FilterSet from ilastik.napari.filters): filter set to use for feature extraction
    OUTPUT:
        features (np.ndarray): feature map (H, W, F) with F being the number of features per pixel
    """
    # Extract features (depending on the number of channels)
    if image.ndim > 2:
        features = extract_ila_features_multichannel(image)
    else:
        features = filter_set.transform(image)
    return features

def ila_self_pred_from_features(feature_map, labels, random_state=None):
    '''
    Predicts all pixel classes with Ilastik from a feature map and sparse annotation (labels).
    INPUT:
        feature_map (np.ndarray): feature map; shape (H, W, F) with F being the number of features per pixel
        labels (np.ndarray): labels for the image; shape (H, W), same dimensions as image
        random_state (int): random state to use for the random forest classifier
    OUTPUT:
        labels_predicted (np.ndarray): predicted labels; shape (H, W)
    '''
    # TRAIN
    sparse_labels = COO.from_numpy(labels) # convert to sparse format (incl. coordinates)
    clf = NDSparseClassifier(RandomForestClassifier(random_state=random_state))
    clf.fit(feature_map, sparse_labels)
    # PREDICT
    # Get the class probabilities
    proba = clf.predict_proba(feature_map)
    prediction = np.moveaxis(proba, -1, 0)
    # Assign the class with the highest probability to each pixel
    labels_predicted = np.zeros_like(prediction[0].astype(np.uint8))
    max_probs = np.zeros_like(prediction[0])
    for class_label in range(0, prediction.shape[0]):
        # Where the probability for the current class is higher than the previous maximum, assign the class label
        labels_predicted[prediction[class_label] > max_probs] = class_label+1
        # Update the maximum probability
        max_probs = np.maximum(prediction[class_label], max_probs)
    return labels_predicted


def selfpred_ilastik(image, labels, random_state=None, filter_set=FILTER_SET):
    '''
    Pixel classification with Ilastik.
    Chooses between simple feature extraction (transform with filterset) and extract_ila_features_multichannel() based on the number of channels in the image.
    Uses the default filter set and scales of ilastik.
    INPUT:
        image (np.ndarray): image to predict on; shape (C, H, W) or (H, W, C) or (H, W)
        labels (np.ndarray): labels for the image; shape (H, W), same dimensions as image
        random_state (int): random state for the random forest classifier
        filter_set (FilterSet from ilastik.napari.filters): filter set to use for feature extraction
    OUTPUT:
        labels_predicted (np.ndarray): predicted labels; shape (H, W)
    '''
    # Extract features (depending on the number of channels)
    features = extract_ilastik_features(image, filter_set=filter_set)
    # Fit the classifier and predict
    prediction = ila_self_pred_from_features(features, labels, random_state=random_state)
    return prediction


def time_ilastik(image, labels=None, filter_set=FILTER_SET, random_state=None):
    """
    Time different steps of slefprediction using Ilastik.
    """
    # For ilastik, there is no model to load, so we only time the feature extraction and prediction
    t_load = None
    
    # Extract features
    t_start_features_full = time()
    features = extract_ilastik_features(image, filter_set=filter_set)
    t_features_full = time() - t_start_features_full
    if labels is None:
        return t_load, t_features_full, None, None, None, None
    
    # We always extract the features of the full image for the training
    t_features_train = t_features_full
    t_start_train = time()
    # TRAIN
    sparse_labels = COO.from_numpy(labels) # convert to sparse format (incl. coordinates)
    clf = NDSparseClassifier(RandomForestClassifier(random_state=random_state))
    clf.fit(features, sparse_labels)
    t_train = time() - t_start_train
    # PREDICT
    t_start_pred = time()
    # Get the class probabilities
    proba = clf.predict_proba(features)
    prediction = np.moveaxis(proba, -1, 0)
    # Assign the class with the highest probability to each pixel
    labels_predicted = np.zeros_like(prediction[0].astype(np.uint8))
    max_probs = np.zeros_like(prediction[0])
    for class_label in range(0, prediction.shape[0]):
        # Where the probability for the current class is higher than the previous maximum, assign the class label
        labels_predicted[prediction[class_label] > max_probs] = class_label+1
        # Update the maximum probability
        max_probs = np.maximum(prediction[class_label], max_probs)
    t_pred = time() - t_start_pred
    # Also calculate the total time for the self-prediction
    t_selfpred = time() - t_start_features_full
    return t_load, t_features_full, t_features_train, t_train, t_pred, t_selfpred