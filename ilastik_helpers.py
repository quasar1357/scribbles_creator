from ilastik.napari.filters import (FilterSet,
                                    Gaussian,
                                    LaplacianOfGaussian,
                                    GaussianGradientMagnitude,
                                    DifferenceOfGaussians,
                                    StructureTensorEigenvalues,
                                    HessianOfGaussianEigenvalues)
from ilastik.napari.plugin import _pixel_classification
import itertools
from sparse import COO
from ilastik.napari.classifier import NDSparseClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

filter_list = (Gaussian,
               LaplacianOfGaussian,
               GaussianGradientMagnitude,
               DifferenceOfGaussians,
               StructureTensorEigenvalues,
               HessianOfGaussianEigenvalues)
scale_list = (0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10.0)

# Generate all combinations of filter_list and scale_list
all_combinations = list(itertools.product(range(len(filter_list)), range(len(scale_list))))

filter_set = FilterSet(
    filters=tuple(
        filter_list[row](scale_list[col])
        for row, col in sorted(all_combinations)
        )
    )


def pixel_classification_ilastik(image, labels, filter_set=filter_set, random_state=None):
    feature_map = filter_set.transform(np.asarray(image.data))
    sparse_labels = COO.from_numpy(np.asarray(labels.data))
    clf = NDSparseClassifier(RandomForestClassifier(random_state = random_state))
    clf.fit(feature_map, sparse_labels)
    prediction = np.moveaxis(clf.predict_proba(feature_map), -1, 0)
    labels_predicted = np.zeros_like(prediction[0].astype(np.uint8))
    for class_label in range(0, prediction.shape[0]):
        labels_predicted[prediction[class_label] > 0.5] = class_label+1
    return labels_predicted


def pixel_classification_ilastik_multichannel(image, labels, filter_set=filter_set, random_state=None):
    """Pixel classification for multichannel images."""
    if len(image.shape) == 3 and image.shape[0] < 4:
        image = np.moveaxis(image, 0, -1) # Ilastik expects (H, W, C)
    for c in range(image.data.shape[2]):
        feature_map = filter_set.transform(np.asarray(image[:,:,c]))
        if c == 0:
            feature_map_all = feature_map
        else:
            feature_map_all = np.concatenate((feature_map_all, feature_map), axis=2)
    # print(feature_map_all.shape)
    sparse_labels = COO.from_numpy(labels)
    clf = NDSparseClassifier(RandomForestClassifier(random_state = random_state))
    clf.fit(feature_map_all, sparse_labels)
    proba = clf.predict_proba(feature_map_all)
    prediction = np.moveaxis(proba, -1, 0)
    labels_predicted = np.zeros_like(prediction[0].astype(np.uint8))
    for class_label in range(0, prediction.shape[0]):
        labels_predicted[prediction[class_label] > 0.5] = class_label+1
    return labels_predicted


def selfpred_ilastik(image, labels, random_state=None):
    if image.ndim > 2:
        prediction = pixel_classification_ilastik_multichannel(image, labels, random_state = random_state)
    else:
        prediction = pixel_classification_ilastik(image, labels, random_state = random_state)
    return prediction