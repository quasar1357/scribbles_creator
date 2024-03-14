import numpy as np
import pandas as pd
from datetime import datetime

from scribbles_creator import *
from cellpose_data_handler import *


# Define folder path
folder_path = "./imgs/cellpose_train_imgs/"

# Create ground truths
# for img_num in range(0, 540):
#     create_cellpose_gt(folder_path, img_num, save_res=True, show_res=False)

# Define parameters
mode = "all"
bins = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1]
all_suff = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
suff = all_suff[:5]

# Create scribbles
for img_num in range(0, 540):
    for bin in bins:
        for s in suff:
            print(f"IMG {img_num}: {bin}_{s}")
            create_cellpose_scribble(folder_path, img_num, bin=bin, mode=mode, save_res=True, suff=s, show_res=False, print_steps=True)
            print("\n")

# Create predictions
layer_list = [0]
scalings = [1, 2]
for img_num in range(0, 540):
    for bin in bins:
        for s in suff:
            pred_cellpose_convpaint(folder_path, img_num, mode=mode, bin=bin, suff=s, layer_list=layer_list, scalings=scalings, save_res=True, show_res=False)
            pred_cellpose_ilastik(folder_path, img_num, mode=mode, bin=bin, suff=s, save_res=True, show_res=False)

# Analyse results
prediction_types = ["convpaint", "ilastik"]
df = pd.DataFrame(columns=['img_num', 'prediction type', 'scribbles mode', 'scribbles bin', 'suffix', 'class_1_pix_gt', 'class_2_pix_gt', 'pix_labelled', 'class_1_pix_labelled', 'class_2_pix_labelled', 'pix_in_img', 'perc. labelled', 'accuracy', 'image', 'ground truth', 'scribbles', 'prediction'])
for img_num in range(0, 540):
    for bin in bins:
        for s in suff:
            for pred in prediction_types:
                res = analyse_cellpose_single_file(folder_path, img_num, mode=mode, bin=bin, suff=s, pred_tag=pred, show_res=False)
                df = pd.concat([df, res], ignore_index=True)
# Save numerical data to csv
time_stamp = datetime.now().strftime("%y%m%d%H%M%S")
file_name = f"test_labels_vs_acc_{time_stamp}.csv"
df.to_csv(file_name, index=False)