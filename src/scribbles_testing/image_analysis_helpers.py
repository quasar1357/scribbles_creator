import numpy as np
import pandas as pd

def single_img_stats(pred, gt, print_results=False):
    '''Takes in a single image's prediction and ground truth and returns the overall accuracy, precision, recall, IoU, and F1 score as a mean over all classes in the image.'''
    df_list = []
    for class_val in np.unique(gt):
        tp = np.sum((pred == class_val) & (gt == class_val))
        fp = np.sum((pred == class_val) & (gt != class_val))
        fn = np.sum((pred != class_val) & (gt == class_val))
        tn = np.sum((pred != class_val) & (gt != class_val))
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        iou = tp / (tp + fp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        new_row = pd.DataFrame({"Class": class_val, "Precision": precision, "Recall": recall, "IoU": iou, "F1": f1}, index=[0])
        df_list.append(new_row)
        if print_results:
            print(f"Class {class_val} | TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
            print(f"Class {class_val} | Precision: {precision:.3f}, Recall: {recall:.3f}, IoU: {iou:.3f}, F1: {f1:.3f}")
    results = pd.concat(df_list, ignore_index=True)
    acc = np.sum(pred == gt) / gt.size
    if print_results:
        print(f"Overall Accuracy: {acc:.3f}")
        print(f"mPrecission: {results.Precision.mean():.3f}, mRecall: {results.Recall.mean():.3f}, mIoU: {results.IoU.mean():.3f}, mF1: {results.F1.mean():.3f}")
    return acc, results.Precision.mean(), results.Recall.mean(), results.IoU.mean(), results.F1.mean()
