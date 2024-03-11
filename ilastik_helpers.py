from ilastik.napari.filters import FilterSet
from ilastik.napari import filters
from ilastik.napari.plugin import _pixel_classification
import itertools
import sparse
from ilastik.napari.classifier import NDSparseClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

filter_list = (
    filters.Gaussian,
    filters.LaplacianOfGaussian,
    filters.GaussianGradientMagnitude,
    filters.DifferenceOfGaussians,
    filters.StructureTensorEigenvalues,
    filters.HessianOfGaussianEigenvalues,
)
scale_list = (0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10.0)

# Generate all combinations of filter_list and scale_list
all_combinations = list(itertools.product(range(len(filter_list)), range(len(scale_list))))

filter_set = FilterSet(
    filters=tuple(
        filter_list[row](scale_list[col])
        for row, col in sorted(all_combinations)
    )
)

def pixel_classification_ilastik(image, labels, filter_set=filter_set):
    feature_map = filter_set.transform(np.asarray(image.data))
    sparse_labels = sparse.COO.from_numpy(np.asarray(labels.data))
    clf = NDSparseClassifier(RandomForestClassifier())
    clf.fit(feature_map, sparse_labels)
    prediction = np.moveaxis(clf.predict_proba(feature_map), -1, 0)
    labels_predicted = np.zeros_like(prediction[0].astype(np.uint8))
    for class_label in range(0, prediction.shape[0]):
        labels_predicted[prediction[class_label] > 0.5] = class_label+1
    return labels_predicted