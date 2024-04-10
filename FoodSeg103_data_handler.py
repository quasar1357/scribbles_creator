from datasets import load_dataset
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('../cellpose'))
from scribbles_creator import create_even_scribble
from convpaint_helpers import selfpred_convpaint
from ilastik_helpers import pixel_classification_ilastik, pixel_classification_ilastik_multichannel


def load_food_data(img_num: int):
    dataset = load_dataset("EduardoPacheco/FoodSeg103")
    img = dataset['train'][img_num]['image']
    img = np.array(img)
    ground_truth = dataset['train'][img_num]['label']
    ground_truth = np.array(ground_truth)
    ground_truth = ground_truth + 1
    return img, ground_truth

def load_food_batch(img_num_list: list):
    dataset = load_dataset("EduardoPacheco/FoodSeg103")
    img_list = []
    ground_truth_list = []
    # data_list = []
    for img_num in img_num_list:
        img = dataset['train'][img_num]['image']
        img = np.array(img)
        ground_truth = dataset['train'][img_num]['label']
        ground_truth = np.array(ground_truth)
        ground_truth = ground_truth + 1
        img_list.append(img)
        ground_truth_list.append(ground_truth)
        # data_list.append((img, ground_truth))
    return img_list, ground_truth_list #data_list

def pred_food_ilastik(image, labels):
    # Predict the image
    if image.ndim > 2:
        prediction = pixel_classification_ilastik_multichannel(image, labels)
    else:
        prediction = pixel_classification_ilastik(image, labels)
    # Sort the labels and use them to assign the correct labels to the Ilastik prediction
    pred_new = prediction.copy()
    labels = np.unique(labels[labels!=0])
    for i, l in enumerate(labels):
        pred_new[prediction == i+1] = l
    return pred_new