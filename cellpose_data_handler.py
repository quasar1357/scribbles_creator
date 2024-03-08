
import numpy as np
import pandas as pd
from PIL import Image
import os
import napari

from scribbles_creator import create_even_scribble
from convpaint_helpers import *

def get_cellpose_img_data(folder_path, img_num, load_img=False, load_gt=False, load_scribbles=False, mode="NA", bin="NA", suff=False, load_pred=False, pred_tag="convpaint"):
    
    folder_path = folder_path if folder_path[-1] == "/" else folder_path + "/"
    img_base = str(img_num).zfill(3)

    masks_path = folder_path + img_base + "_masks.png"

    img_path = folder_path + img_base + f"_img.png"
    if load_img:
        img = np.array(Image.open(img_path))
    else:
        img = None

    gt_path = folder_path + img_base + "_ground_truth.png"
    if load_gt:
        ground_truth = np.array(Image.open(gt_path))
    else:
        ground_truth = None

    suff = "" if not suff else "_" + suff

    scribbles_path = folder_path + img_base + f"_scribbles_{mode}_{bin}{suff}.png"
    if load_scribbles:
        scribbles = np.array(Image.open(scribbles_path))
    else:
        scribbles = None

    pred_path = folder_path + img_base + f"_{pred_tag}_{mode}_{bin}{suff}.png"
    if load_pred:
        pred = np.array(Image.open(pred_path))  
    else:
        pred = None

    img_data = {"img_path": img_path, "gt_path": gt_path, "scribbles_path": scribbles_path, "pred_path": pred_path, "masks_path": masks_path,
                "img": img, "gt": ground_truth, "scribbles": scribbles, "pred": pred}

    return img_data



def create_cellpose_gt(folder_path, img_num, save_res=True, show_res=False):
    
    img_data = get_cellpose_img_data(folder_path, img_num, load_img=show_res)
    masks_path = img_data["masks_path"]
    ground_truth = np.array(Image.open(masks_path))
    # Summarize all non-background classes into one class (we are testing semantic segmentation not instance segmentation)
    ground_truth[ground_truth>0] = 2
    # Set the background class to 1 (since the scribble annotation assumes class 0 to represent non-annotated pixels)
    ground_truth[ground_truth==0] = 1

    if save_res:
        gt_path = img_data["gt_path"]
        if not os.path.exists(gt_path):
            gt_img = Image.fromarray(ground_truth)
            gt_img.save(gt_path)
    
    if show_res:
        image = img_data["img"]
        v = napari.Viewer()
        v.add_image(image)
        v.add_labels(ground_truth)
    
    return ground_truth



def create_cellpose_scribble(folder_path, img_num, bin=0.1, sq_scaling=False, mode="all", save_res=False, suff=False, show_res=False):

    # Load the ground truth and get the scribbles path for saving; note that if we want to show the results, we also load the image
    img_data = get_cellpose_img_data(folder_path, img_num, load_img=show_res, load_gt=True, mode=mode, bin=bin, suff=suff)
    ground_truth = img_data["gt"]
    # Create the scribbles
    scribbles = create_even_scribble(ground_truth, max_perc=bin, sq_scaling=sq_scaling, mode=mode)
    perc_labelled = np.sum(scribbles>0) / (scribbles.shape[0] * scribbles.shape[1]) * 100

    if save_res:
        # Save the scribble annotation as an image
        scribbles_path = img_data["scribbles_path"]
        scribble_img = Image.fromarray(scribbles)
        scribble_img.save(scribbles_path)

    if show_res:
        # Show the image, ground truth and the scribbles
        image = img_data["img"]
        v = napari.Viewer()
        v.add_image(image)
        v.add_labels(ground_truth)
        v.add_labels(scribbles)

    return scribbles, perc_labelled



def pred_cellpose_convpaint(folder_path, img_num, mode="NA", bin="NA", suff=False, layer_list=[0], scalings=[1,2], save_res=False, show_res=False):
    # Load the image, labels and the ground truth
    img_data = get_cellpose_img_data(folder_path, img_num, load_img=True, load_gt=True, load_scribbles=True, mode=mode, bin=bin, suff=suff, load_pred=False)
    image = img_data["img"]
    labels = img_data["scribbles"]
    ground_truth = img_data["gt"]
    
    # Predict the image
    prediction = selfpred_convpaint(image, labels, layer_list, scalings)

    if save_res:
        # Save the scribble annotation as an image
        pred_path = img_data["pred_path"]
        pred_image = Image.fromarray(prediction)
        pred_image.save(pred_path)

    if show_res:
        # Show the ground truth and the scribble annotation
        v = napari.Viewer()
        v.add_image(image)
        v.add_labels(ground_truth)
        v.add_labels(prediction)
        v.add_labels(labels)

    return prediction



def analyse_cellpose_single_file(folder_path, img_num, mode="all", bin=0.1, suff=False, pred_tag="convpaint", show_res=False):

    img_data = get_cellpose_img_data(folder_path, img_num, load_img=show_res, load_gt=True, load_scribbles=True, mode=mode, bin=bin, suff=suff, load_pred=True, pred_tag=pred_tag)
    image_path = img_data["img_path"]
    ground_truth_path = img_data["gt_path"]
    scribbles_path = img_data["scribbles_path"]
    pred_path = img_data["pred_path"]
    # Read the images
    ground_truth = img_data["gt"]
    labels = img_data["scribbles"]
    prediction = img_data["pred"]

    # Calculate stats
    acc = np.mean(ground_truth == prediction)
    perc_labelled = np.sum(labels>0) / (labels.shape[0] * labels.shape[1]) * 100

    if show_res:
        image = img_data["img"]
        # Show the image, ground truth and the scribble annotation
        v = napari.Viewer()
        v.add_image(image)
        v.add_labels(ground_truth)
        v.add_labels(labels)
        v.add_labels(prediction)

    res = pd.DataFrame({'group': f"{str(img_num)}_{mode}_{str(bin)}",
                        'image': image_path,
                        'ground truth': ground_truth_path,
                        'scribbles': scribbles_path,
                        'prediction': pred_path,
                        'mode': mode,
                        'bin': bin,
                        'perc. labelled': perc_labelled,
                        'accuracy': acc}, index=[0])
    
    return res