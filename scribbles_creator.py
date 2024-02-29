from skimage.morphology import *
import numpy as np
import napari
from scipy.spatial import distance
from skimage.draw import line

def create_scribble(ground_truth, num_lines=5, keep_sec_sk=False):
    '''
    Generate the scribble annotation for the ground truth.
    Input:
        ground_truth (numpy array): the fully annotated image
        num_lines (int): the num_lines to be drawn
    Output:
        output (numpy array): the scribble annotation        
    '''
    scribble = np.zeros_like(ground_truth, dtype=np.uint8)
    # For each class (= value) in the ground truth, generate the scribble annotation
    for class_val in set(ground_truth.flatten()):
        # Skip the background class
        if class_val == 0:
            continue
        # Generate the scribble annotation for the class
        class_scribble = scribble_class(ground_truth, class_val, num_lines, keep_sec_sk=keep_sec_sk)
        # Add the scribble annotation of this class to the full scribble (which is valid, because there is no overlap between the classes)
        scribble += class_scribble.astype(np.uint8)
    return scribble

def scribble_class(gt, class_val, num_lines, scribble_width=3, keep_sec_sk=False):
    # Generate a boolean mask for the ground truth matching the class_id
    gt_class_mask = (gt == class_val)
    # Initialize the scribble for the class with zeros
    class_scribble = np.zeros_like(gt, dtype=np.int32)
    # For each slice of the ground truth, generate the skeleton and connecting lines and add them to the scribble
    for i in range(gt_class_mask.shape[0]):
        # If there is no annotation in the slice, then skip the slice
        if np.sum(gt_class_mask[i]) == 0:
            continue
        # Extract the mask of ground truth for the class in this slice
        gt_class_slice_mask = gt_class_mask[i]
        # Generate the primary and secondary skeleton for the class in this slice
        prim_sk, sec_sk = double_sk_class_2d(gt_class_slice_mask)
        # Check if a skeleton was created, raise an error if not
        if np.sum(sec_sk) == 0:
            raise ValueError(f"No skeleton was created for class {class_val} in slice {i}.")
        # Create lines leading from the secondary skeleton to the edge of the mask
        lines_mask = create_lines(sec_sk, gt_class_slice_mask, num_lines)
        if keep_sec_sk:
            class_slice_scribble_mask = np.logical_or(lines_mask, sec_sk)
        else:
            class_slice_scribble_mask = lines_mask
        # Dilate the scribble to make them wider
        class_slice_scribble_mask = dilation(class_slice_scribble_mask, square(scribble_width))
        # Add the scribble to this slice of the overall scribble
        class_scribble[i] = class_slice_scribble_mask * class_val
    # class_scribble = dilation(class_scribble, square(scribble_width))
    return class_scribble

def double_sk_class_2d(gt_mask_2d, closing_first=10):
    '''
    Create a skeleton as well as a secondary skeleton of the ground truth
    Input:
        gt_mask_2d (numpy array): the ground truth mask
        closing_first (int): the size of the structuring element for the closing operation
    Output:
        prim_sk (numpy array): the primary skeleton
        sec_sk (numpy array): the secondary skeleton
    '''
    # Create a skeleton of the ground truth
    gt_mask2d_with_prim_sk = gt_mask_2d.copy()
    prim_sk = skeletonize(gt_mask_2d, method='lee') != 0
    if closing_first: prim_sk = binary_closing(prim_sk, square(closing_first))

    # Add the skeleton to the ground truth mask
    prim_sk_mask = prim_sk == 1
    gt_mask2d_with_prim_sk[prim_sk_mask] = False

    sec_sk = skeletonize(prim_sk, method='lee') != 0
    # sec_sk = binary_closing(sec_sk, square(3))

    return prim_sk, sec_sk

def create_lines(sk_2d, gt_mask_2d, num_lines):
    all_lines = np.zeros_like(gt_mask_2d, dtype=np.bool8)
    for i in range(num_lines):
        line = draw_line(sk_2d, gt_mask_2d, num_lines=5, dist_to_edge=10)
        all_lines = np.logical_or(all_lines, line)
    return all_lines

def draw_line(sk_mask_2d, gt_mask_2d, num_lines=5, dist_to_edge=5):
    '''
    Take a random point on the skeleton (= True in the mask) and draw a line to the nearest edge point
    '''
    # Choose a random point from the skeleton
    sk_coordinates = np.argwhere(sk_mask_2d)
    total_points = sk_coordinates.shape[0]
    step_size = total_points // (num_lines * 2)
    possible_points = np.arange(total_points, step=step_size)
    random_point_index = np.random.choice(possible_points)
    random_point = sk_coordinates[random_point_index]
    # Erode the gt_mask, so that the line will have a distance to the edge
    eroded_gt_mask = erosion(gt_mask_2d, square(dist_to_edge*2))
    # Find the shortest path from the random point to the edge of the mask and return the mask of the path
    shortest_path = point_to_edge(random_point, eroded_gt_mask)
    shortest_path_mask = shortest_path == 1
    return shortest_path_mask

def point_to_edge(start_point, segmentation_mask):
    # Find the coordinates of the edges of the segmentation mask
    edge_coordinates = np.argwhere(segmentation_mask == False)
    # Compute distances from the start point to all edge points
    distances = distance.cdist([start_point], edge_coordinates)
    # Find the index of the closest edge point
    closest_edge_index = np.argmin(distances)
    # min_dist = distances[0, closest_edge_index]
    # Retrieve the coordinates of the closest edge point
    closest_edge_point = edge_coordinates[closest_edge_index]
    # Create an empty image to draw the line on
    path_mask = np.zeros_like(segmentation_mask)
    # Draw the line on the image
    rr, cc = line(start_point[0], start_point[1], closest_edge_point[0], closest_edge_point[1])
    path_mask[rr, cc] = 1
    return path_mask
