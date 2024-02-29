from skimage.morphology import *
import numpy as np
import napari
from scipy.spatial import distance
from skimage.draw import line

def create_scribble(ground_truth, num_lines=5, scribble_width=3, min_line_pix=10):
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
        class_scribble = scribble_class(ground_truth, class_val, num_lines, scribble_width, min_line_pix)
        # Add the scribble annotation of this class to the full scribble (which is valid, because there is no overlap between the classes)
        scribble += class_scribble.astype(np.uint8)
    return scribble

def scribble_class(gt, class_val, num_lines=5, scribble_width=3, min_line_pix=10):
    '''
    Generate the scribble annotation for a specific class in the ground truth.
    Input:
        gt (numpy array): the ground truth
        class_val (int): the value of the class
        num_lines (int): the num_lines to be drawn
    Output:
        class_scribble (numpy array): the scribble annotation for the class
    '''
    # Generate a boolean mask for the ground truth matching the class_id
    gt_class_mask = (gt == class_val)
    # Initialize the scribble for the class with zeros
    class_scribble = np.zeros_like(gt, dtype=np.int32)
    # Generate the primary and secondary skeleton for the class in this slice
    prim_sk, sec_sk = double_sk_class_2d(gt_class_mask)
    # Check if a skeleton was created, raise an error if not
    if np.sum(sec_sk) == 0:
        raise ValueError(f"No skeleton was created for class {class_val} in slice {i}.")
    # Create lines leading from the skeleton to the edge of the mask
    lines_mask = create_lines(prim_sk, gt_class_mask, num_lines, min_line_pix)
    # lines_mask = create_lines(sec_sk, gt_class_mask, num_lines)
    # class_slice_scribble_mask = np.logical_or(lines_mask, sec_sk)
    class_slice_scribble_mask = lines_mask
    # Dilate the scribble to make them wider
    class_slice_scribble_mask = dilation(class_slice_scribble_mask, square(scribble_width))
    # Add the scribble to this slice of the overall scribble
    # class_scribble[i] = class_slice_scribble_mask * class_val
    class_scribble = class_slice_scribble_mask * class_val
    
    # v = napari.Viewer()
    # v.add_image(gt_class_mask)
    # v.add_labels(prim_sk)
    # v.add_labels(sec_sk)

    # class_scribble = dilation(class_scribble, square(scribble_width))
    return class_scribble

def double_sk_class_2d(gt_mask_2d, closing_first=0):
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
    
    # Dilate the primary skeleton before generating the secondary skeleton (otherwise it interprets it as gaps in the primary skeleton)
    prim_sk_dilated = binary_dilation(prim_sk, square(3))

    # Add the dilated skeleton to the ground truth mask
    prim_sk_mask = prim_sk_dilated == 1
    gt_mask2d_with_prim_sk[prim_sk_mask] = False

    # Create the secondary skeleton
    sec_sk = skeletonize(gt_mask2d_with_prim_sk, method='lee') != 0
    # sec_sk = binary_closing(sec_sk, square(3))

    return prim_sk, sec_sk

def create_lines(sk_2d, gt_mask_2d, num_lines, min_line_pix=10):
    all_lines = np.zeros_like(gt_mask_2d, dtype=np.bool8)
    num_created_lines = 0
    attempts = 0
    while num_created_lines < num_lines:
        attempts += 1
        line = draw_line(sk_2d, gt_mask_2d, dist_to_edge=2)
        if np.sum(line) < min_line_pix:
            continue
        else:
            all_lines = np.logical_or(all_lines, line)
            num_created_lines += 1
        if attempts > np.sum(sk_2d) * num_lines:
            print("Warning: Could not create all lines from the skeleton to the edge.")
            break
    return all_lines

def draw_line(sk_mask_2d, gt_mask_2d, dist_to_edge=5):
    '''
    Take a random point on the skeleton (= True in the mask) and draw a line to the nearest edge point of the ground truth mask
    '''
    # Choose a random point from the skeleton
    sk_coordinates = np.argwhere(sk_mask_2d)
    total_points = sk_coordinates.shape[0]
    step_size = 1 #total_points // (num_lines * 2)
    possible_points = np.arange(total_points, step=step_size)
    random_point_index = np.random.choice(possible_points)
    random_point = sk_coordinates[random_point_index]
    # Erode the gt_mask, so that the line will have a distance to the edge
    if dist_to_edge:
        eroded_gt_mask = erosion(gt_mask_2d, square(dist_to_edge*2))
    else:
        eroded_gt_mask = gt_mask_2d
    # Find the shortest path from the random point to the edge of the mask and return the mask of the path
    shortest_path = point_to_edge(random_point, eroded_gt_mask)
    shortest_path_mask = shortest_path == 1
    return shortest_path_mask

def point_to_edge(start_point, segmentation_mask):
    '''
    Find the shortest path from a point to the edge of a segmentation mask
    Input:
        start_point (tuple): the coordinates of the starting point
        segmentation_mask (numpy array): the segmentation mask
    Output:
        path_mask (numpy array): the mask of the shortest path
        min_dist (float): the length of the shortest path
    '''
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
    path_mask[rr, cc] = True
    return path_mask
