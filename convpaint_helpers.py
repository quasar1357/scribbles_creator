import numpy as np
from napari_convpaint.conv_paint_utils import (Hookmodel, filter_image_multioutputs, get_features_current_layers, get_multiscale_features, train_classifier, predict_image, train_test_split)
from sklearn.ensemble import RandomForestClassifier

def selfpred_convpaint(image, labels, layer_list, scalings, model="vgg16", random_state=None):
    if len(image.shape) == 3 and image.shape[2] < 4:
        image = np.moveaxis(image, 2, 0) # Convpaint expects (C, H, W)
    # Define the model
    model = Hookmodel(model_name=model)
    # Ensure the layers are given as a list
    if isinstance(layer_list, int):
        layer_list = [layer_list]
    # Read out the layer names
    all_layers = [key for key in model.module_dict.keys()]
    layers = [all_layers[i] for i in layer_list]
    # Register the hooks for the selected layers
    model.register_hooks(selected_layers=layers)
    # Get the features and targets
    features, targets = get_features_current_layers(
        model=model, image=image, annotations=labels, scalings=scalings,
        order=1, use_min_features=False, image_downsample=1)

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

    # Predict on the image
    predicted = predict_image(
        image, model, random_forest, scalings=scalings,
        order=1, use_min_features=False, image_downsample=1)
    return predicted