# scribbles creator

Welcome to scribbles creator. This is a little tool to automatically create scribble annotations similar to if you would draw them by hand, based on the ground truth of an image. It can be very useful for testing tools for semantic segmentation based on spares annotations. Here, also python scripts and notebooks are provided to test segmentation on two different datasets: cellpose and FoodSeg103.

If you decide to use some of my code in any sort of public work, please do contact and cite me.

## Installation
You can install scribbles_creator via pip using

    pip install git+https://github.com/quasar1357/scribbles_creator.git

## Main script
[scribbles_creator.py](scribbles_creator.py) is the core tool, providing the functions to create scribble annotations based on a ground truth.

## Helper functions for segmentations
The scripts [convpaint_helper.py](convpaint_helper.py), [ilastik_helper.py](ilastik_helper.py) and [dino_helper.py](dino_helper.py) provide functions for semantic segmentation (here limited to training and prediction of the same image) using the three tools Convpaint (with the CNN "VGG16" as feature extractor), Ilastik (using classical filters) and DINOv2 as a feature extractor (implemented in the Convpaint framework in my work [dino_paint](https://github.com/quasar1357/dino_paint)).

## Data handlers
The scripts [cellpose_data_handler.py](cellpose_data_handler.py) and [FoodSeg103_data_handler.py](FoodSeg103_data_handler.py) provide helpful functions to handle the two datasets. In particular, they provide wrapper functions to directly create scribbles, predict segmentations using the three tools mentioned above based on images from the datasets as well as to analyse the results.

## Notebooks
These are examples for using the scripts mentioned above. They are of course specific for the respective environment that is used. The workflow basically consists of the 4 steps:

1) Create scribbles based on the ground truth provided (save them as files for reproducibility)
2) Predict semantic segmentation based on the scribbles, using one of the 3 tools mentioned above
3) Analyse the results by calculating various metrics
4) Plot the results

Since steps 1)-3) are different for the two datasets, separate notebooks are provided. Meanwhile, the plotting for both datasets is unified in one notebook.

## Results
The results folders contain results from my own test-runs. They can be used as references.