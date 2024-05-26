import numpy as np
from sklearn.ensemble import RandomForestClassifier
from skimage.transform import resize
import torch
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split

loaded_dinov2_models = {}

### FEATURE EXTRACTION ###

def extract_features_rgb(image, dinov2_model='s', pad_mode='reflect'):
    trainset_mean, trainset_sd = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    image = normalize_np_array(image, trainset_mean, trainset_sd, axis = (0,1))
    # Convert to tensor and add batch dimension
    image_tensor = ToTensor()(image).float()
    image_batch = image_tensor.unsqueeze(0)
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
    with torch.no_grad():
        features_dict = model.forward_features(image_batch)
        features = features_dict['x_norm_patchtokens']
    # Convert to numpy array
    features = features.numpy()
    # Remove batch dimension
    features = features[0]
    return features

def extract_features_multichannel(image, dinov2_model='s', pad_mode='reflect'):
    # If image is single channel, stack it to have 3 dimensions (the last one being the single channel)
    if len(image.shape) == 2:
        image = np.stack((image,), axis=-1)
    # Extract features for each channel and concatenate them as separate features
    features_list = []
    for channel in range(image.shape[2]):
        # Use channel as r, g and b channels for feature extraction
        image_channel = image[:,:,channel]
        img_channel = np.stack((image_channel,)*3, axis=-1)
        channel_features = extract_features_rgb(img_channel, dinov2_model, pad_mode)
        features_list.append(channel_features)
    features = np.concatenate(features_list, axis=1)
    return features

def extract_features(image, dinov2_model='s', pad_mode='reflect', rgb=True):
    # If the image has 3 channels and RGB is chosen, extract features as usual
    if len(image.shape) == 3 and image.shape[2] == 3 and rgb:
        dinov2_features = extract_features_rgb(image, dinov2_model, pad_mode)
    # If the image does not have 3 channels and/or RGB is not chosen,
    # extract features for each channel and concatenate them
    else:
        dinov2_features = extract_features_multichannel(image, dinov2_model, pad_mode)
    return dinov2_features

def get_annot_features_and_targets(patch_features_flat, labels, patch_size = (14,14), interpolate_features=True):
    num_features = patch_features_flat.shape[1]
    labels_flat = labels.flatten()
    labels_mask = labels_flat > 0
    targets = labels_flat[labels_mask]
    feature_space = reshape_patches_to_img(patch_features_flat, labels.shape, patch_size, interpolation_order=interpolate_features)
    # Flatten the spatial dimensions (keeping the features in the last dimension)
    features_flat = np.reshape(feature_space, (len(labels_flat), num_features))
    # Extract only the annotated pixels
    features_annot = features_flat[labels_mask]
    return features_annot, targets

### MAIN FUNCTIONS ###

def train_dino_forest(image_batch, labels_batch, dinov2_model='s', pad_mode='reflect', random_state=None, rgb=True, interpolate_features=True):
    if not len(image_batch) == len(labels_batch):
        raise ValueError('Image and label batch must have the same length')
    # if not all([image.shape[:2] == labels.shape for image, labels in zip(image_batch, labels_batch)]):
    #     raise ValueError('Each image and label in the batch must have the same spatial dimensions')
    if not all([len(image.shape) == len(image_batch[0].shape) for image in image_batch]):
        raise ValueError('All images in the batch must have the same number of channels')
    if not all([image.shape[2] == image_batch[0].shape[2] for image in image_batch]):
        raise ValueError('All images in the batch must have the same number of channels')
    features_list = []
    targets_list = []
    for image, labels in zip(image_batch, labels_batch):
        padded_image = pad_to_patch(image, "bottom", "right", pad_mode=pad_mode, patch_size=(14,14))
        padded_labels = pad_to_patch(labels, "bottom", "right", pad_mode="constant", patch_size=(14,14))
        patch_features_flat = extract_features(padded_image, dinov2_model, pad_mode, rgb)
        features_annot, targets = get_annot_features_and_targets(patch_features_flat, padded_labels, interpolate_features=interpolate_features)
        features_list.append(features_annot)
        targets_list.append(targets)
    features_annot = np.concatenate(features_list)
    targets = np.concatenate(targets_list)
    # split_dataset = train_test_split(features, targets, test_size=0.2, random_state=42)
    # features_train, features_test, labels_train, labels_test = split_dataset
    features_train, labels_train = features_annot, targets
    random_forest = RandomForestClassifier(n_estimators=100, random_state=random_state)
    random_forest.fit(features_train, labels_train)
    return random_forest

def predict_dino_forest(image, random_forest, dinov2_model='s', pad_mode='reflect', rgb=True, interpolate_features=True):
    padded_image = pad_to_patch(image, "bottom", "right", pad_mode=pad_mode, patch_size=(14,14))
    patch_features_flat = extract_features(padded_image, dinov2_model, pad_mode, rgb)
    num_features = patch_features_flat.shape[1]
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
    return pred_img_recrop

def selfpredict_dino_forest(image, labels, dinov2_model='s', pad_mode='reflect', random_state=None, rgb=True, interpolate_features=True):
    # TRAIN
    padded_image = pad_to_patch(image, "bottom", "right", pad_mode=pad_mode, patch_size=(14,14))
    padded_labels = pad_to_patch(labels, "bottom", "right", pad_mode="constant", patch_size=(14,14))
    patch_features_flat = extract_features(padded_image, dinov2_model, pad_mode, rgb)
    num_features = patch_features_flat.shape[1]
    features_annot, targets = get_annot_features_and_targets(patch_features_flat, padded_labels, interpolate_features=interpolate_features)
    # split_dataset = train_test_split(features, targets, test_size=0.2, random_state=42)
    # features_train, features_test, labels_train, labels_test = split_dataset
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
    return pred_img_recrop

### HELPER FUNCTIONS ###

def normalize_np_array(array, new_mean, new_sd, axis=(0,1)):
    '''
    Normalizes a numpy array to a new mean and standard deviation.
    '''
    current_mean, current_sd = np.mean(array, axis=axis), np.std(array, axis=axis)
    # Avoid division by zero; leads to setting channels with all the same value to the new mean
    current_sd[current_sd == 0] = 1
    new_mean, new_sd = np.array(new_mean), np.array(new_sd)
    array_norm = (array - current_mean) / current_sd
    array_norm = array_norm * new_sd + new_mean
    return array_norm

def pad_to_patch(image, vert_pos="center", hor_pos="center", pad_mode='constant', patch_size=(14,14)):
    '''
    Pads an image to the next multiple of patch size.
    The pad position can be chosen on both axis in the tuple (vert, hor), where vert can be "top", "center" or "bottom" and hor can be "left", "center" or "right".
    pad_mode can be chosen according to numpy pad method.
    '''
    # If image is an rgb image, run this function on each channel
    if len(image.shape) == 3:
        channel_list = np.array([pad_to_patch(image[:,:, channel], vert_pos, hor_pos, pad_mode, patch_size) for channel in range(image.shape[2])])
        rgb_padded = np.moveaxis(channel_list, 0, 2)
        return rgb_padded
    # For a greyscale image (or each separate RGB channel):
    h, w = image.shape
    ph, pw = patch_size
    # Calculate how much padding has to be done in total on each axis
    # The total pad on one axis is a patch size minus whatever remains when dividing the picture size including the extra pads by the patch size
    # The  * (h % ph != 0) term (and same with wdith) ensure that the pad is 0 if the shape is already a multiple of the patch size
    vertical_pad = (ph - h % ph) * (h % ph != 0)
    horizontal_pad = (pw - w % pw) * (w % pw != 0)
    # Define the paddings on each side depending on the chosen positions
    top_pad = {"top": vertical_pad,
               "center": np.ceil(vertical_pad/2),
               "bottom": 0
               }[vert_pos]
    bot_pad = vertical_pad - top_pad
    left_pad = {"left": horizontal_pad,
                "center": np.ceil(horizontal_pad/2),
                "right": 0
                }[hor_pos]
    right_pad = horizontal_pad - left_pad
    # Make sure paddings are ints
    top_pad, bot_pad, left_pad, right_pad = int(top_pad), int(bot_pad), int(left_pad), int(right_pad)
    # Pad the image using the pad sizes as calculated and the mode given as input
    image_padded = np.pad(image, ((top_pad, bot_pad), (left_pad, right_pad)), mode=pad_mode)
    return image_padded

def reshape_patches_to_img(patches, image_shape, patch_size=(14,14), interpolation_order=None):
    if not (image_shape[0]%patch_size[0] == 0 and image_shape[1]%patch_size[1] == 0):
        raise ValueError('Image shape must be multiple of patch size')
    if len(patches.shape) == 1:
        patch_as_pix_shape = int(image_shape[0] / patch_size[0]), int(image_shape[1] / patch_size[1])
    elif len(patches.shape) == 2:
        patch_as_pix_shape = int(image_shape[0] / patch_size[0]), int(image_shape[1] / patch_size[1]), patches.shape[1]
    else:
        raise ValueError('Patches must have one or two dimensions')
    patch_img = np.reshape(patches, patch_as_pix_shape)
    if interpolation_order is None or interpolation_order == 0:
        # Repeat each row and each column according to the patch size to recreate the patches
        patch_img = np.repeat(patch_img, patch_size[0], axis=0)
        patch_img = np.repeat(patch_img, patch_size[1], axis=1)
    else:
        patch_img = resize(patch_img, image_shape[0:2], mode='edge', order=interpolation_order, preserve_range=True)
    return patch_img