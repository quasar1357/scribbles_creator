import numpy as np
from dino_forest import *
from time import time
import torch
from torchvision.transforms import ToTensor

loaded_dinov2_models = {}

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
    # Ensure (H, W, C) - expected by DINOv2
    if len(image.shape) == 3 and image.shape[0] < 4:
        image = np.moveaxis(image, 0, -1)
    # Predict with DINOv2
    pred = selfpredict_dino_forest(image, labels, dinov2_model=dinov2_model, pad_mode=pad_mode, random_state=random_state, rgb=True, interpolate_features=upscale_order)
    return pred


def extract_dino_features(image, dinov2_model='s', pad_mode='reflect'):
    if len(image.shape) == 3 and image.shape[0] < 4:
        image = np.moveaxis(image, 0, -1) # DINOv2 expects (H, W, C)
    padded_image = pad_to_patch(image, "bottom", "right", pad_mode=pad_mode, patch_size=(14,14))
    patch_features_flat = extract_features(padded_image, dinov2_model, rgb=True)
    return patch_features_flat

def time_dino(image, labels=None, dinov2_model='s', pad_mode='reflect', random_state=None, interpolate_features=False):
    # Ensure (H, W, C) - expected by DINOv2
    if len(image.shape) == 3 and image.shape[0] < 4:
        image = np.moveaxis(image, 0, -1)
    # Pad the image to a multiple of the patch size
    padded_image = pad_to_patch(image, "bottom", "right", pad_mode=pad_mode, patch_size=(14,14))
    
    # We could use the direct feature extraction function, but this would also include the time to load the model
    # t_start = time()
    # patch_features_flat = extract_features(padded_image, dinov2_model, rgb=True)
    # t_elapsed = time() - t_start

    # Define and load model
    models = {'s': 'dinov2_vits14',
            'b': 'dinov2_vitb14',
            'l': 'dinov2_vitl14',
            'g': 'dinov2_vitg14',
            's_r': 'dinov2_vits14_reg',
            'b_r': 'dinov2_vitb14_reg',
            'l_r': 'dinov2_vitl14_reg',
            'g_r': 'dinov2_vitg14_reg'}
    dinov2_name = models[dinov2_model]
    if dinov2_name not in loaded_dinov2_models:
        loaded_dinov2_models[dinov2_name] = torch.hub.load('facebookresearch/dinov2', dinov2_name, pretrained=True, verbose=False)
    model = loaded_dinov2_models[dinov2_name]
    model.eval()

    # Extract features
    t_start = time()
    trainset_mean, trainset_sd = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    # If the image is RGB, extract features from the RGB channels
    if len(padded_image.shape) == 3 and padded_image.shape[2] == 3:
        # Normalize the image
        padded_image = normalize_np_array(padded_image, trainset_mean, trainset_sd, axis = (0,1))
        # Convert to tensor and add batch dimension
        image_tensor = ToTensor()(padded_image).float()
        image_batch = image_tensor.unsqueeze(0)
        # Extract features
        with torch.no_grad():
            features_dict = model.forward_features(image_batch)
            features = features_dict['x_norm_patchtokens']
        # Convert to numpy array
        features = features.numpy()
        # Remove batch dimension
        features = features[0]
    # If the image is not RGB, extract features from each channel as RGB, and concatenate them
    else:
        # If image is single channel, stack it to have 3 dimensions (the last one being the single channel)
        if len(padded_image.shape) == 2:
            padded_image = np.stack((padded_image,), axis=-1)
        # Extract features for each channel and concatenate them as separate features
        features_list = []
        for channel in range(padded_image.shape[2]):
            # Use channel as r, g and b channels for feature extraction
            image_channel = padded_image[:,:,channel]
            img_channel = np.stack((image_channel,)*3, axis=-1)
            # Normalize the channel
            img_channel = normalize_np_array(img_channel, trainset_mean, trainset_sd, axis = (0,1))
            # Convert to tensor and add batch dimension
            channel_tensor = ToTensor()(img_channel).float()
            channel_batch = channel_tensor.unsqueeze(0)        # Extract features
            with torch.no_grad():
                features_dict = model.forward_features(channel_batch)
                channel_features = features_dict['x_norm_patchtokens']
            # Convert to numpy array
            channel_features = channel_features.numpy()
            # Remove batch dimension
            channel_features = channel_features[0]

            features_list.append(channel_features)
        features = np.concatenate(features_list, axis=1)
    t_features = time() - t_start
    if labels is None:
        return t_features

    # Do training and prediction
    t_start_pred = time()
    padded_labels = pad_to_patch(labels, "bottom", "right", pad_mode="constant", patch_size=(14,14))
    patch_features_flat = features
    num_features = patch_features_flat.shape[1]
    features_annot, targets = get_annot_features_and_targets(patch_features_flat, padded_labels, interpolate_features=interpolate_features)
    # TRAIN
    features_train, labels_train = features_annot, targets
    random_forest = RandomForestClassifier(n_estimators=100, random_state=random_state)
    random_forest.fit(features_train, labels_train)
    # PREDICT
    # If we want interpolated features, we reshape them to the image size (with interpolation), and then reshape them back to flat features
    if interpolate_features:
        feature_space = reshape_patches_to_img(patch_features_flat, padded_image.shape[:2], patch_size=(14,14), interpolation_order=interpolate_features)
        features = np.reshape(feature_space, (padded_image.shape[0]*padded_image.shape[1], num_features))
    else:
        features = patch_features_flat
    predicted_labels = random_forest.predict(features)
    # If we are not using interpolated per pixel features, we reshape the predicted labels to the image size considering the patches
    if not interpolate_features:
        pred_img = reshape_patches_to_img(predicted_labels, padded_image.shape[:2], interpolation_order=0)
    # Otherwise the features are already per pixel and can be reshaped directly
    else:
        pred_img = np.reshape(predicted_labels, padded_image.shape[:2])
    pred_img_recrop = pred_img[:image.shape[0], :image.shape[1]]
    t_pred = time() - t_start_pred
    t_tot = time() - t_start
    return t_features, t_pred, t_tot