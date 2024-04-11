import numpy as np
from dino_paint_utils import extract_feature_space, predict_space_to_image
from napari_convpaint.conv_paint_utils import train_test_split, extract_annotated_pixels, train_classifier
from sklearn.ensemble import RandomForestClassifier


def selfpred_dino(image, labels, dinov2_model='s', dinov2_layers=(), dinov2_scales=(), upscale_order=1, pad_mode='reflect', extra_pads=(), vgg16_layers=None, vgg16_scales=(), append_image_as_feature=False, random_state=None):
    feature_space = extract_feature_space(image, dinov2_model, dinov2_layers, dinov2_scales, upscale_order, pad_mode, extra_pads, vgg16_layers, vgg16_scales, append_image_as_feature)
    # Extract annotated pixels
    features_annot, targets = extract_annotated_pixels(feature_space, labels, full_annotation=False)
    features = features_annot
    # Train the classifier
    if random_state is None:
        random_forest = train_classifier(features, targets)
    else:
        # Do RF training manually (copied from convpaint utils) to have access to seed/random_state:
        X, X_test, y, y_test = train_test_split(features, targets,
                                                test_size=0.2,
                                                random_state=42)
        random_forest = RandomForestClassifier(n_estimators=100, random_state = random_state)
        random_forest.fit(X, y)
    predicted_labels = predict_space_to_image(feature_space, random_forest)
    return predicted_labels