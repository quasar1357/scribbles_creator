import numpy as np
from napari_convpaint.conv_paint_utils import (Hookmodel, filter_image_multioutputs, get_features_current_layers, get_multiscale_features, train_classifier, predict_image)

def selfpred_convpaint(image, labels, layer_list, scalings, model="vgg16"):
    # Ensure right shape and dimension order
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = np.moveaxis(image, -1, 0) # ConvPaint expects (C, H, W)
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
    random_forest = train_classifier(features, targets)
    # Predict on the image
    predicted = predict_image(
        image, model, random_forest, scalings=scalings,
        order=1, use_min_features=False, image_downsample=1)
    return predicted