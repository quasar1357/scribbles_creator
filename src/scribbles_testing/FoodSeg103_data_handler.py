from datasets import load_dataset
import numpy as np
import pandas as pd
import re
from PIL import Image
import napari
from time import time

from scribbles_creator import create_even_scribbles
from napari_convpaint.conv_paint_utils import compute_image_stats, normalize_image
from scribbles_testing.image_analysis_helpers import single_img_stats
from seghub.ilastik_utils import get_ila_feature_space
from seghub.vgg16_utils import get_vgg16_feature_space
from seghub.dino_utils import get_dinov2_feature_space
from seghub.rf_utils import selfpredict_segforest_single_image

from scribbles_testing.convpaint_helpers import generate_convpaint_tag, time_convpaint
from scribbles_testing.ilastik_helpers import time_ilastik
from scribbles_testing.dino_helpers import time_dino



def load_food_data(img_num: int, load_image=True, load_gt=True):
    '''
    Load a single image and/or ground truth from the FoodSeg103 dataset.
    '''
    dataset = load_dataset("EduardoPacheco/FoodSeg103")
    if load_image:
        img = dataset['train'][img_num]['image']
        img = np.array(img)
    else:
        img = None
    if load_gt:
        ground_truth = dataset['train'][img_num]['label']
        ground_truth = np.array(ground_truth)
        ground_truth = ground_truth + 1
    else:
        ground_truth = None
    return img, ground_truth

def load_food_batch(img_num_list: list, load_images=True, load_gts=True):
    '''
    Load a batch of images and/or ground truths from the FoodSeg103 dataset.
    '''
    dataset = load_dataset("EduardoPacheco/FoodSeg103")
    img_dict = {}
    ground_truth_dict = {}
    for img_num in img_num_list:
        if load_images:
            img = dataset['train'][img_num]['image']
            img = np.array(img)
            img_dict[img_num] = img
        if load_gts:
            ground_truth = dataset['train'][img_num]['label']
            ground_truth = np.array(ground_truth)
            ground_truth = ground_truth + 1
            ground_truth_dict[img_num] = ground_truth
    return img_dict, ground_truth_dict

def get_food_img_data(folder_path, img_num, load_scribbles=False, mode="all", bin="NA", scribble_width=None, suff=False, load_pred=False, pred_tag="convpaint"):
    '''
    Create names/paths for the scribbles and/or prediction for the given image number and folder path and optionally load them.
    INPUT:
        folder_path (str): path to the folder containing the image and the ground truth
        img_num (int): number of the image to be processed
        load_scribbles (bool): if True, the scribbles will be loaded
        mode (str): scribble mode of the scribbles to consider
        bin (float): percentage of the scribbles to consider
        suff (str): suffix of the scribbles file name
        load_pred (bool): if True, the prediction will be loaded
        pred_tag (str): tag to be used for the prediction file name
    OUTPUT:
        img_data (dict): dictionary containing the paths and the loaded images, ground truth, scribbles and prediction
            keys: "scribbles_path", "pred_path", "scribbles", "pred"
            NOTE: If a key ("scribbles", "pred") is not loaded, the corresponding value will be None
    '''    
    folder_path = folder_path if folder_path[-1] == "/" else folder_path + "/"
    img_base = str(img_num).zfill(4)

    suff = "" if not suff else "_" + suff
    width_suff = "" if scribble_width is None else f"_w{scribble_width}"

    scribbles_path = folder_path + img_base + f"_scribbles_{mode}_{bin_for_file(bin)}{width_suff}{suff}.png"
    if load_scribbles:
        scribbles = np.array(Image.open(scribbles_path))
    else:
        scribbles = None

    pred_path = folder_path + img_base + f"_{pred_tag}_{mode}_{bin_for_file(bin)}{width_suff}{suff}.png"
    if load_pred:
        pred = np.array(Image.open(pred_path))
    else:
        pred = None

    img_data = {"scribbles_path": scribbles_path, "pred_path": pred_path,
                "scribbles": scribbles, "pred": pred}

    return img_data

def bin_for_file(bin):
    if bin == "NA":
        return "NA"
    return str(int(bin*1000)).zfill(5)



def create_food_scribble(ground_truth, folder_path, img_num, bin=0.1, margin=0.75, rel_scribble_len=False, scribble_width=1, mode="all", class_dist="relative", enforce_max_perc=False, save_res=False, suff=False, show_res=False, image=None, print_steps=False):
    '''
    Create scribbles on the given ground truth. Scribbles are created by sampling a certain percentage of the ground truth pixels and then expanding the scribbles to the given scribble width.
    The scribbles can be saved as an image and can be shown in a napari viewer if desired.
    INPUT:
        ground_truth (np.array): the ground truth to be used for creating the scribbles
        folder_path (str): path to the folder for saving the scribbles
        img_num (int): number of the image to be processed
        bin (float): percentage of the ground truth pixels to be sampled for the scribbles; the scribbles will hold close to and not more than this percentage of the image pixels
        rel_scribble_len (int/bool): length of the single scribbles relative to pixel dimensions, i.e. the number of scribbles that would fit the image (empirical default value: 20/(max_perc**0.25))
        mode (str): scribble mode; "prim_sk" for scribbles from the skeletonized ground truth, "sek_sk" from the secondary skeleton, "lines" for lines from the skeleton to the edge, "both_sk" and "all" for combinations
        save_res (bool): if True, the scribbles will be saved as an image
        suff (str): suffix to be added to the scribbles file name
        show_res (bool): if True, the scribbles will be shown in a napari viewer
        image (np.array): the image to be shown in the napari viewer; if None, the image will not be shown
        print_steps (bool): if True, the steps of the scribble creation will be printed
        scribble_width (int): width of the scribbles
    OUTPUT:
        scribbles (np.array): the created scribbles
        perc_labelled (float): percentage of the image pixels that are labelled in the scribbles
    NOTE: Set the random seed by calling np.random.seed(seed) before calling this function if you want to reproduce the scribbles
    '''
    # Create the scribbles
    scribbles = create_even_scribbles(ground_truth, max_perc=bin, margin=margin, rel_scribble_len=rel_scribble_len, scribble_width=scribble_width, mode=mode, class_dist=class_dist, print_steps=print_steps, enforce_max_perc=enforce_max_perc)
    perc_labelled = np.sum(scribbles>0) / (scribbles.shape[0] * scribbles.shape[1]) * 100

    if save_res:
        # Get the scribbles path for saving
        img_data = get_food_img_data(folder_path, img_num, mode=mode, bin=bin, scribble_width=scribble_width, suff=suff)
        # Save the scribble annotation as an image
        scribbles_path = img_data["scribbles_path"]
        scribble_img = Image.fromarray(scribbles)
        scribble_img.save(scribbles_path)

    if show_res:
        # Show the image, ground truth and the scribbles
        v = napari.Viewer()
        if image is not None:
            v.add_image(image)
        v.add_labels(ground_truth)
        v.add_labels(scribbles)

    return scribbles, perc_labelled



def pred_food(image, folder_path, img_num, pred_type="convpaint", mode="all", bin="NA", scribble_width=None, suff=False, save_res=False, show_res=False, ground_truth=None, random_state=0, **pred_kwargs):
    '''
    Load the scribbles and predict segmentation of the given image using the given prediction method. Optionally save the prediction and show the results in a napari viewer.
    INPUT:
        image (np.array): the image to be segmented
        folder_path (str): path to the folder containing the scribbles, and for saving the prediction
        img_num (int): number of the image to be processed
        pred_type (str): type of the prediction method; "convpaint" for ConvPaint, "ilastik" for Ilastik, "dino" for DINOv2
        mode (str): scribble mode of the scribbles to consider
        bin (float): percentage of the scribbles to consider
        suff (str): scribbles suffix of the file name
        save_res (bool): if True, the prediction will be saved as an image
        show_res (bool): if True, the results will be shown in a napari viewer
        ground_truth (np.array): the ground truth to be shown in the napari viewer (optional)
        pred_kwargs (dict): keyword arguments for the prediction function
    OUTPUT:
        prediction (np.array): the predicted image
    '''
    # Generate the convpaint model prefix given the model, the layer list and the scalings
    if pred_type == "convpaint":
        pred_tag = generate_convpaint_tag(pred_kwargs["layer_list"], pred_kwargs["scalings"], pred_kwargs["model_name"])
    else:
        pred_tag = pred_type

    # Load the labels
    img_data = get_food_img_data(folder_path, img_num, load_scribbles=True, mode=mode, bin=bin, scribble_width=scribble_width, suff=suff, load_pred=False, pred_tag=pred_tag)
    labels = img_data["scribbles"]

    # Pre-process the image to (0, 1) and (H, W, C) format
    image = image/255
    if image.ndim == 3 and image.shape[0] < 4:
        image = np.moveaxis(image, 0, -1)

    # Predict the image
    features_func = {"convpaint": get_vgg16_feature_space, "ilastik": get_ila_feature_space, "dino": get_dinov2_feature_space}[pred_type]
    prediction = selfpredict_segforest_single_image(image, labels, features_func, random_state=random_state, features_cfg=pred_kwargs)

    if pred_type == "ilastik":
        prediction = post_proc_ila_pred(prediction, labels)

    if save_res:
        # Save the scribble annotation as an image
        pred_path = img_data["pred_path"]
        pred_image = Image.fromarray(prediction)
        pred_image.save(pred_path)

    if show_res:
        # Show the results and the image (and the ground truth if available)
        v = napari.Viewer()
        v.add_image(image)
        if ground_truth is not None:
            v.add_labels(ground_truth)
        v.add_labels(prediction, name=pred_tag)
        v.add_labels(labels)

    return prediction

def post_proc_ila_pred(prediction, labels):
    '''Post-process the Ilastik prediction to match the labels of the labels and therefore the FoodSeg103 ground truth.'''
    # Sort the labels and use them to assign the correct labels to the Ilastik prediction
    pred_new = prediction.copy()
    labels = np.unique(labels[labels!=0])
    for i, l in enumerate(labels):
        pred_new[prediction == i+1] = l
    return pred_new

def pred_food_convpaint(image, folder_path, img_num, mode="all", bin="NA", scribble_width=None, suff=False, save_res=False, show_res=False, ground_truth=None,
                        layer_list=[0], scalings=[1,2], model="vgg16", random_state=None):
    '''Shortcut for pred_food() with pred_type="convpaint" (see pred_cellpose() for details).'''
    prediction = pred_food(image, folder_path, img_num, pred_type="convpaint", mode=mode, bin=bin, scribble_width=scribble_width, suff=suff, save_res=save_res, show_res=show_res, ground_truth=ground_truth, random_state=random_state,
                           layer_list=layer_list, scalings=scalings, model_name=model)
    return prediction

def pred_food_ilastik(image, folder_path, img_num, mode="all", bin="NA", scribble_width=None, suff=False, save_res=False, show_res=False, ground_truth=None,
                      random_state=None):
    '''Shortcut for pred_food() with pred_type="ilastik" (see pred_cellpose() for details).'''
    prediction = pred_food(image, folder_path, img_num, pred_type="ilastik", mode=mode, bin=bin, scribble_width=scribble_width, suff=suff, save_res=save_res, show_res=show_res, ground_truth=ground_truth, random_state=random_state)
    return prediction

def pred_food_dino(image, folder_path, img_num, mode="all", bin="NA", scribble_width=None, suff=False, save_res=False, show_res=False, ground_truth=None,
                   dinov2_model='s_r', upscale_order=False, random_state=None):
    '''Shortcut for pred_food() with pred_type="dino" (see pred_cellpose() for details).'''
    prediction = pred_food(image, folder_path, img_num, pred_type="dino", mode=mode, bin=bin, scribble_width=scribble_width, suff=suff, save_res=save_res, show_res=show_res, ground_truth=ground_truth, random_state=random_state,
                           dinov2_model=dinov2_model, interpolate_features=upscale_order)
    return prediction



def analyse_food_single_file(ground_truth, folder_path, img_num, mode="all", bin=0.1, scribble_width=None, suff=False, pred_tag="convpaint", show_res=False, image=None):
    ''' 
    Analyse the scribbles and the prediction for a single image. Optionally show the results in a napari viewer.
    INPUT:
        ground_truth (np.array): the ground truth to be used for the analysis
        folder_path (str): path to the folder containing the scribbles and prediciton
        img_num (int): number of the image to be processed
        mode (str): scribble mode of the scribbles to consider
        bin (float): percentage of the scribbles to consider
        suff (str): scribble suffix of the file names
        pred_tag (str): tag to be used for the prediction
        show_res (bool): if True, the results will be shown in a napari viewer
        image (np.array): the image to be shown in the napari viewer (not necessary for the analysis itself); if None, the image will not be shown
    OUTPUT:
        res (pd.DataFrame): dataframe containing the analysis results (one row)
            keys:   "img_num", "prediction type", "scribbles mode", "scribbles bin", "suffix", "max_class_pix_gt", "min_class_pix_gt",
                    "pix_labelled", "max_class_pix_labelled", "min_class_pix_labelled", "pix_in_img", "perc. labelled", "accuracy",
                    "scribbles", "prediction"
    '''
    img_data = get_food_img_data(folder_path, img_num, load_scribbles=True, mode=mode, bin=bin, scribble_width=scribble_width, suff=suff, load_pred=True, pred_tag=pred_tag)
    scribbles_path = img_data["scribbles_path"]
    pred_path = img_data["pred_path"]
    # Read the images
    labels = img_data["scribbles"]
    prediction = img_data["pred"]

    # Add columns for the layers and scalings (extracted from the prediction type tag)
    if "convpaint" in pred_tag and pred_tag != "convpaint":
        l_tag = re.split("_", pred_tag)[1]
        layers = l_tag[2:]
        # layers = re.split("-", l_tag)[1:]
        s_tag = re.split("_", pred_tag)[2]
        scalings = s_tag[2:]
        # scalings = re.split("-", s_tag)[1:]
    else:
        layers = None
        scalings = None
    # Add prediction type
    # if pred_tag == 'convpaint_l-0_s-1-2': pred_type = 'convpaint (default)'
    if pred_tag[:10] == 'convpaint_': pred_type = 'convpaint' # If there is still an underline (i.e. it wasn't renamed to default), just call it "convpaint" (layers and scalings are already in the columns)
    if pred_tag == 'ilastik': pred_type = 'classical filters'
    if pred_tag == 'dino': pred_type = 'DINOv2'

    # Calculate stats
    class_pix_gt = [np.sum(ground_truth == val) for val in np.unique(ground_truth) if val > 0] # There shouldn't be any 0s in the ground truth, but we exclude them just in case
    max_class_pix_gt = max(class_pix_gt)
    min_class_pix_gt = min(class_pix_gt)
    pix_labelled = np.sum(labels > 0)
    class_pix_labelled = [np.sum(labels == val) for val in np.unique(labels) if val > 0]
    max_class_pix_labelled = max(class_pix_labelled)
    min_class_pix_labelled = min(class_pix_labelled)
    pix_in_img = (labels.shape[0] * labels.shape[1])
    perc_labelled = pix_labelled / pix_in_img * 100
    acc, mPrec, mRecall, mIoU, mF1 = single_img_stats(prediction, ground_truth)

    if show_res:
        # Show the image, ground truth and the scribble annotation
        v = napari.Viewer()
        if image is not None:
            image = img_data["img"]
            v.add_image(image)
        v.add_labels(ground_truth)
        v.add_labels(labels)
        v.add_labels(prediction)

    res = pd.DataFrame({'img_num': img_num,
                        'prediction tag': pred_tag,
                        'prediction type': pred_type,
                        'layers': layers,
                        'scalings': scalings,
                        'scribbles mode': mode,
                        'scribbles bin': bin,
                        'suffix': suff,
                        'min_class_pix_gt': min_class_pix_gt,
                        'max_class_pix_gt': max_class_pix_gt,
                        'pix_labelled': pix_labelled,
                        'min_class_pix_labelled': min_class_pix_labelled,
                        'max_class_pix_labelled': max_class_pix_labelled,
                        'pix_in_img': pix_in_img,
                        'perc. labelled': perc_labelled,
                        'accuracy': acc,
                        'mPrecision': mPrec,
                        'mRecall': mRecall,
                        'mIoU': mIoU,
                        'mF1': mF1,
                        'scribbles': scribbles_path,
                        'prediction': pred_path}, index=[0])
    
    return res



def time_food(image, ground_truth, folder_path, img_num, mode="all", bin=0.1, scribble_width=None, suff=False, pred_tag="convpaint", **kwargs):
    '''
    Load and measure prediciton time of a FoodSeg103 image.    
    '''
    img_data = get_food_img_data(folder_path, img_num, load_scribbles=True, mode=mode, bin=bin, scribble_width=scribble_width, suff=suff, load_pred=False, pred_tag=pred_tag)
    scribbles_path = img_data["scribbles_path"]
    # Read the images
    labels = img_data["scribbles"]
    # Ensure the image has the right shape
    if image.ndim == 3 and image.shape[2] < 4:
        image = np.moveaxis(image, 2, 0)

    # Add columns for the layers and scalings (extracted from the prediction tag)
    if "convpaint" in pred_tag and pred_tag != "convpaint":
        l_tag = re.split("_", pred_tag)[1]
        layers = l_tag[2:]
        # layers = re.split("-", l_tag)[1:]
        s_tag = re.split("_", pred_tag)[2]
        scalings = s_tag[2:]
        # scalings = re.split("-", s_tag)[1:]
    else:
        layers = None
        scalings = None
    # Add prediction type
    # if pred_tag == 'convpaint_l-0_s-1-2': pred_type = 'convpaint (default)'
    if pred_tag[:9] == 'convpaint': pred_type = 'convpaint'
    if pred_tag == 'ilastik': pred_type = 'classical filters'
    if pred_tag == 'dino': pred_type = 'DINOv2'

    # Calculate stats
    class_pix_gt = [np.sum(ground_truth == val) for val in np.unique(ground_truth) if val > 0] # There shouldn't be any 0s in the ground truth, but we exclude them just in case
    max_class_pix_gt = max(class_pix_gt)
    min_class_pix_gt = min(class_pix_gt)
    pix_labelled = np.sum(labels > 0)
    class_pix_labelled = [np.sum(labels == val) for val in np.unique(labels) if val > 0]
    max_class_pix_labelled = max(class_pix_labelled)
    min_class_pix_labelled = min(class_pix_labelled)
    pix_in_img = (labels.shape[0] * labels.shape[1])
    perc_labelled = pix_labelled / pix_in_img * 100

    # Measure the prediction time
    time_func = {"convpaint": time_convpaint, "classical filters": time_ilastik, "DINOv2": time_dino}[pred_type]
    t = time_func(image, labels, **kwargs)
    t_load, t_features_full, t_features_train, t_train, t_pred, t_selfpred = t

    res = pd.DataFrame({'img_num': img_num,
                        'prediction tag': pred_tag,
                        'prediction type': pred_type,
                        'layers': layers,
                        'scalings': scalings,
                        'scribbles mode': mode,
                        'scribbles bin': bin,
                        'suffix': suff,
                        'min_class_pix_gt': min_class_pix_gt,
                        'max_class_pix_gt': max_class_pix_gt,
                        'pix_labelled': pix_labelled,
                        'min_class_pix_labelled': min_class_pix_labelled,
                        'max_class_pix_labelled': max_class_pix_labelled,
                        'pix_in_img': pix_in_img,
                        'perc. labelled': perc_labelled,
                        't_load': t_load,
                        't_features_full': t_features_full,
                        't_features_train': t_features_train,
                        't_train': t_train,
                        't_pred': t_pred,
                        't_selfpred': t_selfpred,
                        'scribbles': scribbles_path}, index=[0])

    return res
