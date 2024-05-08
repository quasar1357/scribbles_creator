import os
import numpy as np
import pandas as pd
from PIL import Image
import napari

from scribbles_creator import create_even_scribbles
from convpaint_helpers import selfpred_convpaint, generate_convpaint_tag
from ilastik_helpers import selfpred_ilastik
from dino_helpers import selfpred_dino
from napari_convpaint.conv_paint_utils import compute_image_stats, normalize_image
from image_analysis_helpers import single_img_stats



def get_cellpose_img_data(folder_path, img_num, load_img=False, load_gt=False, load_scribbles=False, mode="all", bin="NA", scribble_width=None, suff=False, load_pred=False, pred_tag="convpaint"):
    '''
    Create names/paths for the image, ground truth, scribbles and/or prediction for the given image number and folder path and optionally load them.
    INPUT:
        folder_path (str): path to the folder containing the image and the ground truth
        img_num (int): number of the image to be processed
        load_img (bool): if True, the image will be loaded
        load_gt (bool): if True, the ground truth will be loaded
        load_scribbles (bool): if True, the scribbles will be loaded
        mode (str): scribble mode of the scribbles to consider
        bin (float): percentage of the scribbles to consider
        suff (str): suffix of the scribbles file name
        load_pred (bool): if True, the prediction will be loaded
        pred_tag (str): tag to be used for the prediction file name
    OUTPUT:
        img_data (dict): dictionary containing the paths and the loaded images, ground truth, scribbles and prediction
            keys: "img_path", "gt_path", "scribbles_path", "pred_path", "masks_path", "img", "gt", "scribbles", "pred"
            NOTE: If a key ("img", "gt", "scribbles", "pred") is not loaded, the corresponding value will be None
    '''
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

    img_data = {"img_path": img_path, "gt_path": gt_path, "scribbles_path": scribbles_path, "pred_path": pred_path, "masks_path": masks_path,
                "img": img, "gt": ground_truth, "scribbles": scribbles, "pred": pred}

    return img_data

def bin_for_file(bin):
    return str(int(bin*1000)).zfill(5)



def preprocess_img(img):
    '''
    Preprocess the image for the ConvPaint model.
    The shape is ensured to be (C, H, W). The channels are checked for values, and if a channel contains no values, it is removed.The image is normalized.
    INPUT:
        img (np.array): the image to be preprocessed
    OUTPUT:
        img (np.array): the preprocessed image
    '''
    # Ensure right shape and dimension order    
    if img.ndim == 3 and img.shape[2] < 4:
        img = np.moveaxis(img, -1, 0) # ConvPaint expects (C, H, W)

    # If some channel(s) contain(s) no values, remove them
    if img.ndim == 3 and img.shape[0] == 3:
        # Check which channels contain values
        img_r_is_active = np.count_nonzero(img[0]) > 0
        img_g_is_active = np.count_nonzero(img[1]) > 0
        img_b_is_active = np.count_nonzero(img[2]) > 0
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
    '''
    Load the masks from the cellpose dataset, summarize the classes and save the ground truth as an image.
    INPUT:
        folder_path (str): path to the folder containing the mask
        img_num (int): image number of the mask to be processed
        save_res (bool): if True, the ground truth will be saved as an image
        show_res (bool): if True, the ground truth will be shown in a napari viewer
    OUTPUT:
        ground_truth (np.array): the created ground truth (binary image with 1 for background and 2 for cells/foreground)
    '''
    img_data = get_cellpose_img_data(folder_path, img_num, load_img=show_res)
    masks_path = img_data["masks_path"]
    ground_truth = np.array(Image.open(masks_path))
    # Summarize all non-background classes into one class (we are testing semantic segmentation not instance segmentation)
    ground_truth[ground_truth > 0] = 2
    # Set the background class to 1 (since the scribble annotation assumes class 0 to represent non-annotated pixels)
    ground_truth[ground_truth == 0] = 1

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



def create_cellpose_scribble(folder_path, img_num, bin=0.1, margin=0.75, rel_scribble_len=False, scribble_width=1, mode="all", enforce_max_perc=False, save_res=False, suff=False, show_res=False, show_img=True, print_steps=False):
    '''
    Load the ground truth and create scribbles for the given image. Scribbles are created by sampling a certain percentage of the ground truth pixels and then expanding the scribbles to the given scribble width.
    The scribbles can be saved as an image and can be shown in a napari viewer if desired.
    INPUT:
        folder_path (str): path to the folder containing the image and the ground truth, and for saving the scribbles
        img_num (int): number of the image to be processed
        bin (float): percentage of the ground truth pixels to be sampled for the scribbles; the scribbles will hold close to and not more than this percentage of the image pixels
        rel_scribble_len (int/bool): length of the single scribbles relative to pixel dimensions, i.e. the number of scribbles that would fit the image (empirical default value: 20/(max_perc**0.25))        mode (str): scribble mode; "prim_sk" for scribbles from the skeletonized ground truth, "sek_sk" from the secondary skeleton, "lines" for lines from the skeleton to the edge, "both_sk" and "all" for combinations
        save_res (bool): if True, the scribbles will be saved as an image
        suff (str): suffix to be added to the scribbles file name
        show_res (bool): if True, the scribbles will be shown in a napari viewer
        show_img (bool): if True, the image will be shown in the napari viewer together with the scribbles and the ground truth (only applies if show_res is True)
        print_steps (bool): if True, the steps of the scribble creation will be printed
        scribble_width (int): width of the scribbles
    OUTPUT:
        scribbles (np.array): the created scribbles
        perc_labelled (float): percentage of the image pixels that are labelled in the scribbles
    NOTE: Set the random seed by calling np.random.seed(seed) before calling this function if you want to reproduce the scribbles
    '''
    # We only need to load the image if we want to show the results and it is specified that the image should be shown
    if not show_res: show_img = False
    # Load the ground truth and get the scribbles path for saving; note that if we want to show the results, we also load the image
    img_data = get_cellpose_img_data(folder_path, img_num, load_img=show_img, load_gt=True, mode=mode, bin=bin, scribble_width=scribble_width, suff=suff)
    ground_truth = img_data["gt"]
    # Create the scribbles
    scribbles = create_even_scribbles(ground_truth, max_perc=bin, margin=margin, rel_scribble_len=rel_scribble_len, scribble_width=scribble_width, mode=mode, print_steps=print_steps, enforce_max_perc=enforce_max_perc)
    perc_labelled = np.sum(scribbles>0) / (scribbles.shape[0] * scribbles.shape[1]) * 100

    if save_res:
        # Save the scribble annotation as an image
        scribbles_path = img_data["scribbles_path"]
        scribble_img = Image.fromarray(scribbles)
        scribble_img.save(scribbles_path)

    if show_res:
        # Show the image (if intended), ground truth and the scribbles
        v = napari.Viewer()
        if show_img:
            image = img_data["img"]
            v.add_image(image)
        v.add_labels(ground_truth)
        v.add_labels(scribbles)

    return scribbles, perc_labelled



def pred_cellpose(folder_path, img_num, pred_type="convpaint", mode="all", bin="NA", scribble_width=None, suff=False, save_res=False, show_res=False, show_gt=True, **pred_kwargs):
    '''
    Load the image and the scribbles and predict segmentation of the image using the given prediction method. Optionally save the prediction and show the results in a napari viewer.
    INPUT:
        folder_path (str): path to the folder containing the image and the scribbles (and the ground truth if intended), and for saving the prediction
        img_num (int): number of the image to be processed
        pred_type (str): type of the prediction method; "convpaint" for ConvPaint, "ilastik" for Ilastik, "dino" for DINOv2
        mode (str): scribble mode of the scribbles to consider
        bin (float): percentage of the scribbles to consider
        suff (str): scribbles suffix of the file name
        save_res (bool): if True, the prediction will be saved as an image
        show_res (bool): if True, the results will be shown in a napari viewer
        show_gt (bool): if True, the ground truth will be shown in the napari viewer (only applies if show_res is True)
        pred_kwargs (dict): keyword arguments for the prediction function
    OUTPUT:
        prediction (np.array): the predicted image
    '''
    # Generate the convpaint model prefix given the model, the layer list and the scalings
    if pred_type == "convpaint":
        pred_tag = generate_convpaint_tag(pred_kwargs["layer_list"], pred_kwargs["scalings"], pred_kwargs["model"])
    else:
        pred_tag = pred_type
    
    # We only need to load the image if we want to show the results and it is specified that the image should be shown
    if not show_res: show_gt = False
    # Load the image and labels
    img_data = get_cellpose_img_data(folder_path, img_num, load_img=True, load_gt=show_gt, load_scribbles=True, mode=mode, bin=bin, scribble_width=scribble_width, suff=suff, load_pred=False, pred_tag=pred_tag)
    image = img_data["img"]
    labels = img_data["scribbles"]
    print(image.shape, labels.shape)

    # Predict the image
    pred_func = {"convpaint": selfpred_convpaint, "ilastik": selfpred_ilastik, "dino": selfpred_dino}[pred_type]
    prediction = pred_func(image, labels, **pred_kwargs)

    if save_res:
        # Save the scribble annotation as an image
        pred_path = img_data["pred_path"]
        pred_image = Image.fromarray(prediction)
        pred_image.save(pred_path)

    if show_res:
        # Show the results and the image (and the ground truth if intended)
        v = napari.Viewer()
        v.add_image(image)
        if show_gt:
            ground_truth = img_data["gt"]
            v.add_labels(ground_truth)
        v.add_labels(prediction, name=pred_tag)
        v.add_labels(labels)

    return prediction


def pred_cellpose_convpaint(folder_path, img_num, mode="all", bin="NA", scribble_width=None, suff=False, save_res=False, show_res=False, show_gt=True,
                            layer_list=[0], scalings=[1,2], model="vgg16", random_state=None):
    '''Shortcut for pred_cellpose() with pred_type="convpaint" (see pred_cellpose() for details).'''
    prediction = pred_cellpose(folder_path, img_num, pred_type="convpaint", mode=mode, bin=bin, scribble_width=scribble_width, suff=suff, save_res=save_res, show_res=show_res, show_gt=show_gt,
                               layer_list=layer_list, scalings=scalings, model=model, random_state=random_state)
    return prediction

def pred_cellpose_ilastik(folder_path, img_num, mode="all", bin="NA", scribble_width=None, suff=False, save_res=False, show_res=False, show_gt=True,
                          random_state=None):
    '''Shortcut for pred_cellpose() with pred_type="ilastik" (see pred_cellpose() for details).'''
    prediction = pred_cellpose(folder_path, img_num, pred_type="ilastik", mode=mode, bin=bin, scribble_width=scribble_width, suff=suff, save_res=save_res, show_res=show_res, show_gt=show_gt,
                               random_state=random_state)
    return prediction

def pred_cellpose_dino(folder_path, img_num, mode="all", bin="NA", scribble_width=None, suff=False, save_res=False, show_res=False, show_gt=True,
                       dinov2_model='s', dinov2_layers=(), dinov2_scales=(), upscale_order=1, random_state=None):
    '''Shortcut for pred_cellpose() with pred_type="dino" (see pred_cellpose() for details).'''
    prediction = pred_cellpose(folder_path, img_num, pred_type="dino", mode=mode, bin=bin, scribble_width=scribble_width, suff=suff, save_res=save_res, show_res=show_res, show_gt=show_gt,
                               dinov2_model=dinov2_model, dinov2_layers=dinov2_layers, dinov2_scales=dinov2_scales, upscale_order=upscale_order, random_state=random_state)
    return prediction



def analyse_cellpose_single_file(folder_path, img_num, mode="all", bin=0.1, scribble_width=None, suff=False, pred_tag="convpaint", show_res=False):
    ''' 
    Load and nalyse the scribbles and the prediction for a single image. Optionally show the results in a napari viewer.
    INPUT:
        folder_path (str): path to the folder containing the ground truth, the scribbles and the prediction
        img_num (int): number of the image to be processed
        mode (str): scribble mode of the scribbles to consider
        bin (float): percentage of the scribbles to consider
        suff (str): scribble suffix of the file names
        pred_tag (str): tag to be used for the prediction
        show_res (bool): if True, the results will be shown in a napari viewer
    OUTPUT:
        res (pd.DataFrame): dataframe containing the analysis results (one row)
            keys:   "img_num", "prediction type", "scribbles mode", "scribbles bin", "suffix", "class_1_pix_gt", "class_2_pix_gt",
                    "pix_labelled", "class_1_pix_labelled", "class_2_pix_labelled", "pix_in_img", "perc. labelled", "accuracy",
                    "image", "ground truth", "scribbles", "prediction"
    '''
    img_data = get_cellpose_img_data(folder_path, img_num, load_img=show_res, load_gt=True, load_scribbles=True, mode=mode, bin=bin, scribble_width=scribble_width, suff=suff, load_pred=True, pred_tag=pred_tag)
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
    acc, mPrec, mRecall, mIoU, mF1 = single_img_stats(prediction, ground_truth)

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
                        'mPrecision': mPrec,
                        'mRecall': mRecall,
                        'mIoU': mIoU,
                        'mF1': mF1,
                        'image': image_path,
                        'ground truth': ground_truth_path,
                        'scribbles': scribbles_path,
                        'prediction': pred_path}, index=[0])
    
    return res