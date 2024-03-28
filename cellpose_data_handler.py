import numpy as np
import pandas as pd
from PIL import Image
import os
import napari

from scribbles_creator import create_even_scribble
from convpaint_helpers import *
from ilastik_helpers import pixel_classification_ilastik, pixel_classification_ilastik_multichannel
from napari_convpaint.conv_paint_utils import compute_image_stats, normalize_image


def get_cellpose_img_data(folder_path, img_num, load_img=False, load_gt=False, load_scribbles=False, mode="NA", bin="NA", suff=False, load_pred=False, pred_tag="convpaint"):
    
    folder_path = folder_path if folder_path[-1] == "/" else folder_path + "/"
    img_base = str(img_num).zfill(3)

    masks_path = folder_path + img_base + "_masks.png"

    img_path = folder_path + img_base + f"_img.png"
    if load_img:
        img = np.array(Image.open(img_path))#[:,:,1] # NOTE: If we only want to use 1 channel, we can filter here
        img = preprocess_img(img)
    else:
        img = None

    gt_path = folder_path + img_base + "_ground_truth.png"
    if load_gt:
        ground_truth = np.array(Image.open(gt_path))
    else:
        ground_truth = None

    suff = "" if not suff else "_" + suff

    scribbles_path = folder_path + img_base + f"_scribbles_{mode}_{bin_for_file(bin)}{suff}.png"
    if load_scribbles:
        scribbles = np.array(Image.open(scribbles_path))
    else:
        scribbles = None

    pred_path = folder_path + img_base + f"_{pred_tag}_{mode}_{bin_for_file(bin)}{suff}.png"
    if load_pred:
        pred = np.array(Image.open(pred_path))
    else:
        pred = None

    img_data = {"img_path": img_path, "gt_path": gt_path, "scribbles_path": scribbles_path, "pred_path": pred_path, "masks_path": masks_path,
                "img": img, "gt": ground_truth, "scribbles": scribbles, "pred": pred}

    return img_data



def bin_for_file(bin):
    return str(int(bin*1000)).zfill(5)



def preprocess_img(img):
    # Ensure right shape and dimension order    
    if len(img.shape) == 3 and img.shape[2] < 4:
        img = np.moveaxis(img, -1, 0) # ConvPaint expects (C, H, W)

    # If some channel(s) contain(s) no values, remove them
    if len(img.shape) == 3 and img.shape[0] == 3:
        # Check which channels contain values
        img_r_is_active = np.count_nonzero(img[0])>0
        img_g_is_active = np.count_nonzero(img[1])>0
        img_b_is_active = np.count_nonzero(img[2])>0
        num_active = sum((img_r_is_active, img_g_is_active, img_b_is_active))
        # Remove the inactive channel(s)
        if num_active < 3:
            # If there are 2 active channels, only take those
            img = img[[img_r_is_active, img_g_is_active, img_b_is_active]]
            # If there is just one, only pick the one and reduce the dimensions
            if num_active == 1:
                img = np.squeeze(img, axis=0)
            print(f"Active channels: R={img_r_is_active}, G={img_g_is_active}, B={img_b_is_active} --> Removed {3-num_active} channel(s) --> shape: {img.shape}")
        else:
            print("All channels contain values.")

    # Normalize the image
    img_mean, img_std = compute_image_stats(img, ignore_n_first_dims=img.ndim-2)
    img = normalize_image(img, img_mean, img_std)
    
    return img



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



def create_cellpose_scribble(folder_path, img_num, bin=0.1, sq_scaling=False, mode="all", save_res=False, suff=False, show_res=False, print_steps=False):

    # Load the ground truth and get the scribbles path for saving; note that if we want to show the results, we also load the image
    img_data = get_cellpose_img_data(folder_path, img_num, load_img=show_res, load_gt=True, mode=mode, bin=bin, suff=suff)
    ground_truth = img_data["gt"]
    # Create the scribbles
    scribbles = create_even_scribble(ground_truth, max_perc=bin, sq_scaling=sq_scaling, mode=mode, print_steps=print_steps)
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



def pred_cellpose_convpaint(folder_path, img_num, mode="NA", bin="NA", suff=False, layer_list=[0], scalings=[1,2], model="vgg16", random_state=None, save_res=False, show_res=False):
    # Load the image, labels and the ground truth
    model_pref = f'_{model}' if model != 'vgg16' else ''
    layer_pref = f'_l-{str(layer_list)[1:-1].replace(", ", "-")}'# if layer_list != [0] else ''
    scalings_pref = f'_s-{str(scalings)[1:-1].replace(", ", "-")}'# if scalings != [1,2] else ''
    pred_tag = f"convpaint{model_pref}{layer_pref}{scalings_pref}"
    img_data = get_cellpose_img_data(folder_path, img_num, load_img=True, load_gt=True, load_scribbles=True, mode=mode, bin=bin, suff=suff, load_pred=False, pred_tag=pred_tag)
    image = img_data["img"]
    labels = img_data["scribbles"]
    ground_truth = img_data["gt"]

    # Predict the image
    prediction = selfpred_convpaint(image, labels, layer_list, scalings, model, random_state)

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
        v.add_labels(prediction, name="convpaint")
        v.add_labels(labels)

    return prediction



def pred_cellpose_ilastik(folder_path, img_num, mode="NA", bin="NA", suff=False, save_res=False, show_res=False):
    # Load the image, labels and the ground truth
    img_data = get_cellpose_img_data(folder_path, img_num, load_img=True, load_gt=True, load_scribbles=True, mode=mode, bin=bin, suff=suff, load_pred=False, pred_tag="ilastik")
    image = img_data["img"]
    labels = img_data["scribbles"]
    ground_truth = img_data["gt"]
    
    # Predict the image
    if len(image.shape) > 1:
        prediction = pixel_classification_ilastik_multichannel(image, labels)
    else:
        prediction = pixel_classification_ilastik(image, labels)

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
        v.add_labels(prediction, name="ilastik")
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
    class_1_pix_gt = np.sum(ground_truth == 1)
    class_2_pix_gt = np.sum(ground_truth == 2)
    pix_labelled = np.sum(labels>0)
    class_1_pix_labelled = np.sum(labels == 1)
    class_2_pix_labelled = np.sum(labels == 2)
    pix_in_img = (labels.shape[0] * labels.shape[1])
    perc_labelled = pix_labelled / pix_in_img * 100
    acc = np.mean(ground_truth == prediction)

    if show_res:
        image = img_data["img"]
        # Show the image, ground truth and the scribble annotation
        v = napari.Viewer()
        v.add_image(image)
        v.add_labels(ground_truth)
        v.add_labels(labels)
        v.add_labels(prediction)

    res = pd.DataFrame({'img_num': img_num,
                        'prediction type': pred_tag,
                        'scribbles mode': mode,
                        'scribbles bin': bin,
                        'suffix': suff,
                        'class_1_pix_gt': class_1_pix_gt,
                        'class_2_pix_gt': class_2_pix_gt,
                        'pix_labelled': pix_labelled,
                        'class_1_pix_labelled': class_1_pix_labelled,
                        'class_2_pix_labelled': class_2_pix_labelled,
                        'pix_in_img': pix_in_img,
                        'perc. labelled': perc_labelled,
                        'accuracy': acc,
                        'image': image_path,
                        'ground truth': ground_truth_path,
                        'scribbles': scribbles_path,
                        'prediction': pred_path}, index=[0])
    
    return res