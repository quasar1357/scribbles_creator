import numpy as np
from dino_paint_utils import extract_feature_space, predict_space_to_image
from napari_convpaint.conv_paint_utils import train_test_split, extract_annotated_pixels, train_classifier
from sklearn.ensemble import RandomForestClassifier
from dino_forest import selfpredict_dino_forest


def selfpred_dino(image, labels, dinov2_model='s', upscale_order=1, pad_mode='reflect', random_state=None):
    '''
    Predict full semantic segmentation of an image using labels for this same image with ConvPaint and DINOv2 (from dino_paint) as feature extractor (train and predict on same image).
    INPUT:
        image (np.ndarray): image to predict on; shape (H, W, C) or (C, H, W)
        labels (np.ndarray): labels for the image; shape (H, W), same dimensions as image
        dinov2_model (str), dinov2_layers (tuple of int), dinov2_scales (tuple of int), upscale_order (int), pad_mode (str), extra_pads (tuple): DINOv2 parameters to use for feature extraction
        vgg16_layers (list of int), vgg16_scales (tuple of int): VGG16 parameters to use for feature extraction
        append_image_as_feature (bool): whether to append the image as a feature
        random_state (int): random state for the random forest classifier
    OUTPUTS:
        predicted (np.ndarray): predicted segmentation. Shape (H, W)
    '''    
    if len(image.shape) == 3 and image.shape[0] < 4:
        image = np.moveaxis(image, 0, -1) # DINOv2 expects (H, W, C)    
    pred = selfpredict_dino_forest(image, labels, dinov2_model=dinov2_model, pad_mode=pad_mode, random_state=random_state, rgb=True, interpolate_features=upscale_order)
    return pred