from datasets import load_dataset
import numpy as np
import pandas as pd
from PIL import Image
import napari

from scribbles_creator import create_even_scribbles
from convpaint_helpers import selfpred_convpaint, generate_convpaint_tag
from ilastik_helpers import selfpred_ilastik
from dino_helpers import selfpred_dino



def load_food_data(img_num: int, load_image=True, load_gt=True):
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



def get_food_img_data(folder_path, img_num, load_scribbles=False, mode="all", bin="NA", suff=False, load_pred=False, pred_tag="convpaint"):
    
    folder_path = folder_path if folder_path[-1] == "/" else folder_path + "/"
    img_base = str(img_num).zfill(4)

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

    img_data = {"scribbles_path": scribbles_path, "pred_path": pred_path,
                "scribbles": scribbles, "pred": pred}

    return img_data

def bin_for_file(bin):
    return str(int(bin*1000)).zfill(5)



def create_food_scribble(ground_truth, folder_path, img_num, bin=0.1, sq_scaling=False, mode="all", save_res=False, suff=False, show_res=False, image=None, print_steps=False, scribble_width=1):

    # Create the scribbles
    scribbles = create_even_scribbles(ground_truth, max_perc=bin, sq_scaling=sq_scaling, mode=mode, print_steps=print_steps, scribble_width=scribble_width)
    perc_labelled = np.sum(scribbles>0) / (scribbles.shape[0] * scribbles.shape[1]) * 100

    if save_res:
        # Get the scribbles path for saving
        img_data = get_food_img_data(folder_path, img_num, mode=mode, bin=bin, suff=suff)
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



def pred_food_convpaint(image, folder_path, img_num, mode="all", bin="NA", suff=False, layer_list=[0], scalings=[1,2], model="vgg16", random_state=None, save_res=False, show_res=False, ground_truth=None):
    pred_tag = generate_convpaint_tag(layer_list, scalings, model)
    # Load the image and labels
    img_data = get_food_img_data(folder_path, img_num, load_scribbles=True, mode=mode, bin=bin, suff=suff, load_pred=False, pred_tag=pred_tag)
    labels = img_data["scribbles"]

    # Predict the image
    prediction = selfpred_convpaint(image, labels, layer_list, scalings, model, random_state)

    if save_res:
        # Save the scribble annotation as an image
        pred_path = img_data["pred_path"]
        pred_image = Image.fromarray(prediction)
        pred_image.save(pred_path)

    if show_res:
        # Show the results
        v = napari.Viewer()
        v.add_image(image)
        if ground_truth is not None:
            v.add_labels(ground_truth)
        v.add_labels(prediction, name="convpaint")
        v.add_labels(labels)

    return prediction



def pred_food_ilastik(image, folder_path, img_num, mode="all", bin="NA", suff=False, random_state=None, save_res=False, show_res=False, ground_truth=None):
    # Load the image, labels and the ground truth
    img_data = get_food_img_data(folder_path, img_num, load_scribbles=True, mode=mode, bin=bin, suff=suff, load_pred=False, pred_tag="ilastik")
    labels = img_data["scribbles"]
    
    # Predict the image
    prediction = selfpred_ilastik(image, labels, random_state)
    pred = post_proc_ila_pred(prediction, labels)

    if save_res:
        # Save the scribble annotation as an image
        pred_path = img_data["pred_path"]
        pred_image = Image.fromarray(pred)
        pred_image.save(pred_path)

    if show_res:
        # Show the results
        v = napari.Viewer()
        v.add_image(image)
        if ground_truth is not None:
            v.add_labels(ground_truth)
        v.add_labels(pred, name="ilastik")
        v.add_labels(labels)

    return pred

def post_proc_ila_pred(prediction, labels):
    # Sort the labels and use them to assign the correct labels to the Ilastik prediction
    pred_new = prediction.copy()
    labels = np.unique(labels[labels!=0])
    for i, l in enumerate(labels):
        pred_new[prediction == i+1] = l
    return pred_new



def pred_food_dino(image, folder_path, img_num, mode="all", bin="NA", suff=False, dinov2_model='s', dinov2_layers=(), dinov2_scales=(), upscale_order=1, random_state=None, save_res=False, show_res=False, ground_truth=None):
    # Load the image, labels and the ground truth
    img_data = get_food_img_data(folder_path, img_num, load_scribbles=True, mode=mode, bin=bin, suff=suff, load_pred=False, pred_tag="dino")
    labels = img_data["scribbles"]
    
    # Predict the image
    prediction = selfpred_dino(image, labels, dinov2_model=dinov2_model, dinov2_layers=dinov2_layers, dinov2_scales=dinov2_scales, upscale_order=upscale_order, random_state = random_state)

    if save_res:
        # Save the scribble annotation as an image
        pred_path = img_data["pred_path"]
        pred_image = Image.fromarray(prediction)
        pred_image.save(pred_path)

    if show_res:
        # Show the results
        v = napari.Viewer()
        v.add_image(image)
        if ground_truth is not None:
            v.add_labels(ground_truth)
        v.add_labels(prediction, name="dino")
        v.add_labels(labels)

    return prediction



def analyse_food_single_file(ground_truth, folder_path, img_num, mode="all", bin=0.1, suff=False, pred_tag="convpaint", show_res=False, image=None):

    img_data = get_food_img_data(folder_path, img_num, load_scribbles=True, mode=mode, bin=bin, suff=suff, load_pred=True, pred_tag=pred_tag)
    scribbles_path = img_data["scribbles_path"]
    pred_path = img_data["pred_path"]
    # Read the images
    labels = img_data["scribbles"]
    prediction = img_data["pred"]

    # Calculate stats
    class_pix_gt = [np.sum(ground_truth == val) for val in np.unique(ground_truth)]
    max_class_pix_gt = np.max(class_pix_gt)
    min_class_pix_gt = np.min(class_pix_gt)
    pix_labelled = np.sum(labels>0)
    class_pix_labelled = [np.sum(labels == val) for val in np.unique(labels)]
    max_class_pix_labelled = np.max(class_pix_labelled)
    min_class_pix_labelled = np.min(class_pix_labelled)
    pix_in_img = (labels.shape[0] * labels.shape[1])
    perc_labelled = pix_labelled / pix_in_img * 100
    acc = np.mean(ground_truth == prediction)

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
                        'prediction type': pred_tag,
                        'scribbles mode': mode,
                        'scribbles bin': bin,
                        'suffix': suff,
                        'max_class_pix_gt': max_class_pix_gt,
                        'min_class_pix_gt': min_class_pix_gt,
                        'pix_labelled': pix_labelled,
                        'max_class_pix_labelled': max_class_pix_labelled,
                        'min_class_pix_labelled': min_class_pix_labelled,
                        'pix_in_img': pix_in_img,
                        'perc. labelled': perc_labelled,
                        'accuracy': acc,
                        'scribbles': scribbles_path,
                        'prediction': pred_path}, index=[0])
    
    return res