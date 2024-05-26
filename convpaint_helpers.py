import numpy as np
from napari_convpaint.conv_paint_utils import (Hookmodel, filter_image_multioutputs, get_features_current_layers, get_multiscale_features, train_classifier, predict_image, train_test_split)
from sklearn.ensemble import RandomForestClassifier

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
    if image.ndim == 3 and image.shape[2] < 4:
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
    features_annot, targets = get_features_current_layers(
        model=model, image=image, annotations=labels, scalings=scalings,
        order=1, use_min_features=False, image_downsample=1)

    # Train the classifier
    # split_dataset = train_test_split(features_annot, targets, test_size=0.2, random_state=42)
    # features_train, features_test, labels_train, labels_test = split_dataset
    features_train, labels_train = features_annot, targets
    random_forest = RandomForestClassifier(n_estimators=100, random_state=random_state)
    random_forest.fit(features_train, labels_train)

    # Predict on the image
    predicted = predict_image(
        image, model, random_forest, scalings=scalings,
        order=1, use_min_features=False, image_downsample=1)
    return predicted


def generate_convpaint_tag(layer_list, scalings, model="vgg16"):
    '''
    Generate a tag for a ConvPaint prediction based on the model, the layer list and the scalings
    '''
    model_pref = f'_{model}' if model != 'vgg16' else ''
    layer_pref = f'_l-{str(layer_list)[1:-1].replace(", ", "-")}'# if layer_list != [0] else ''
    scalings_pref = f'_s-{str(scalings)[1:-1].replace(", ", "-")}'# if scalings != [1,2] else ''
    pred_tag = f"convpaint{model_pref}{layer_pref}{scalings_pref}"
    return pred_tag