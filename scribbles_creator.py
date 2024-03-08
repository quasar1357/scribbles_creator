import numpy as np
import napari
from skimage.morphology import *
from skimage.draw import line
from scipy.spatial import distance

def create_even_scribble(ground_truth, max_perc=0.2, sq_scaling=False, mode="all"):
    '''Generate the scribble annotation for the ground truth using an even distribution of pixels among the chosen scribble types (all, both skeletons or individual skeletons and lines).
    This function uses a scribble_width of 1, a formula to determine the square size and a range for pixels inside a square or line of half to double one square side length.
    These parameters should be suited for max_perc values between approximately 0.05 and 1.
    Input:
        ground_truth (numpy array): the fully annotated image
        max_perc (float): the maximum percentage of pixels that should be picked (from skeletons and lines)
        mode (str): the scribble types to use (lines, prim_sk, sec_sk, both_sk, all)
    Output:
        scribbles (numpy array): the scribble annotation
    '''    
    # Calculate parameters for the scribble annotation
    num_annots = {"lines": 1, "prim_sk": 1, "sec_sk": 1, "both_sk" : 2, "all": 3}
    max_perc_per_mode = max_perc / num_annots[mode]
    scribble_width = 1
    if not sq_scaling: sq_scaling = 400/(max_perc**0.5)
    sq_size = (ground_truth.shape[0] * ground_truth.shape[1] // sq_scaling) ** 0.5
    sq_size = int(sq_size)

    # Generate the scribble annotation for the ground truth
    scribbles = create_scribble(ground_truth, scribble_width=scribble_width, sk_max_perc=max_perc_per_mode, sq_size=sq_size, sq_pix_range=False, lines_max_perc=max_perc_per_mode, line_pix_range=False, mode=mode)
    return scribbles

def create_scribble(ground_truth, scribble_width=1, sk_max_perc=0.05, sq_size=20, sq_pix_range=False, lines_max_perc=0.05, line_pix_range=False, mode="all"):
    '''
    Generate the scribble annotation for the ground truth.
    Input:
        ground_truth (numpy array): the fully annotated image
        scribble_width (int): the width of the scribble lines
        sk_max_perc (float): the maximum/approximate percentage of pixels of the ground truth that should be picked (from skeletons)
        sq_size (int): the size of the squares (side length)
        sq_pix_range (int): the range that the number of pixels in a square shall be in
        lines_max_perc (float): the maximum/approximate percentage of pixels of the ground truth that should be created by drawing lines
        line_pix_range (int): the range that the number of pixels for a line shall be in
        mode (str): the scribble types to use (lines, prim_sk, sec_sk, both_sk, all)
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
        class_scribble = scribble_class(ground_truth, class_val, scribble_width, sk_max_perc, sq_size, sq_pix_range, lines_max_perc, line_pix_range, mode)
        # Add the scribble annotation of this class to the full scribble (which is valid, because there is no overlap between the classes)
        scribble += class_scribble.astype(np.uint8)
    return scribble

def scribble_class(gt, class_val, scribble_width=1, sk_max_perc=0.05, sq_size=20, sq_pix_range=False, lines_max_perc=0.05, line_pix_range=False, mode="all", print_steps=False):
    '''
    Generate the scribble annotation for a specific class in the ground truth.
    Input:
        gt (numpy array): the ground truth
        class_val (int): the value of the class
        scribble_width (int): the width of the scribble lines
        sk_max_perc (float): the maximum/approximate percentage of pixels of the ground truth that should be picked (from skeletons)
        sq_size (int): the size of the squares (side length)
        sq_pix_range (int): the range that the number of pixels in a square shall be in
        lines_max_perc (float): the maximum/approximate percentage of pixels of the ground truth that should be created by drawing lines
        line_pix_range (int): the range that the number of pixels for a line shall be in
        mode (str): the scribble types to use (lines, prim_sk, sec_sk, both_sk, all)
    Output:
        class_scribble (numpy array): the scribble annotation for the class
    '''
    # Generate a boolean mask for the ground truth matching the class_id
    gt_class_mask = (gt == class_val)
    # Initialize the scribble for the class with zeros
    class_scribble = np.zeros_like(gt, dtype=np.int32)

    # Generate the primary and secondary skeleton for the class in this slice
    prim_sk, sec_sk = double_sk_class(gt_class_mask)
    # v = napari.Viewer()
    # v.add_image(prim_sk)
    # v.add_image(sec_sk)
    # Check if a skeleton was created, raise an error if not
    if np.sum(sec_sk) == 0:
        raise ValueError(f"No skeleton was created for class {class_val}.")
    # Pick random squares from the skeletons
    sk_max_pix = max(2, int(np.sum(gt_class_mask) * sk_max_perc / 100)) # Ensure that at least two pixels are picked
    # Ensure that each square is allowed to as little pixels as the maximum total pixels in all squares
    sq_pix_range = (min(sq_size//2, sk_max_pix), sq_size*2) if not sq_pix_range else sq_pix_range
    if print_steps:
        print(f"class {class_val}:")
        print(f"sk_max_pix: {sk_max_pix}, sq_size: {sq_size}, sk_pix_range: {sq_pix_range}")
    prim_sk_squares = pick_sk_squares(prim_sk, sk_max_pix=sk_max_pix, sq_size=sq_size, sq_pix_range=sq_pix_range)
    sec_sk_squares = pick_sk_squares(sec_sk, sk_max_pix=sk_max_pix, sq_size=sq_size, sq_pix_range=sq_pix_range)
    both_sk_squares = np.logical_or(prim_sk_squares, sec_sk_squares)

    # Create lines leading from the primary skeleton to the edge of the mask
    lines_max_pix = max(2, int(np.sum(gt_class_mask) * lines_max_perc / 100)) # Ensure that at least two pixels are picked
    # Ensure that the line is allowed to be as short as the maximum total pixels in all lines
    line_pix_range = (min(sq_size//2, lines_max_pix), sq_size*2) if not line_pix_range else line_pix_range 
    if print_steps:
        print(f"lines_max_pix: {lines_max_pix}, line_pix_range: {line_pix_range}")
    lines = create_lines(prim_sk, gt_class_mask, lines_max_pix, line_pix_range)
    lines_and_squares = np.logical_or(lines, both_sk_squares)

    # Print some intermediate values (change False to True to print them)
    if print_steps:
        print(f"   pix: {np.sum(lines_and_squares)} = {np.sum(lines_and_squares)/np.sum(gt_class_mask)*100:.2f}%")
        print(f"   prim_sk_squares: {np.sum(prim_sk_squares)} = {np.sum(prim_sk_squares)/np.sum(gt_class_mask)*100:.2f}%")
        print(f"   sec_sk_squares: {np.sum(sec_sk_squares)} = {np.sum(sec_sk_squares)/np.sum(gt_class_mask)*100:.2f}%")
        print(f"   lines: {np.sum(lines)} = {np.sum(lines)/np.sum(gt_class_mask)*100:.2f}%")

    # Define the scribble type to use
    if mode == "lines": class_scribble_mask = lines
    elif mode == "prim_sk": class_scribble_mask = prim_sk_squares
    elif mode == "sec_sk": class_scribble_mask = sec_sk_squares
    elif mode == "both_sk": class_scribble_mask = both_sk_squares
    elif mode == "all": class_scribble_mask = lines_and_squares

    # Dilate the scribble to make them wider
    class_scribble_mask = dilation(class_scribble_mask, square(scribble_width))
    # Ensure that the scribble is within the ground truth mask
    class_scribble_mask = np.logical_and(class_scribble_mask, gt_class_mask)
    # Add the scribble as class value to the overall scribble
    class_scribble = class_scribble_mask * class_val
    return class_scribble

def double_sk_class(gt_mask, closing_prim=0, closing_sec=0):
    '''
    Create a skeleton as well as a secondary skeleton of the ground truth.
    Input:
        gt_mask (numpy array): the ground truth mask
        closing_prim (int): the size of the structuring element for the closing operation on the primary skeleton
        closing_sec (int): the size of the structuring element for the closing operation on the secondary skeleton
    Output:
        prim_sk (numpy array): the primary skeleton
        sec_sk (numpy array): the secondary skeleton
    '''
    # Prepare a copy of the ground truth mask for the secondary skeleton
    gt_mask2d_with_prim_sk = gt_mask.copy()
    # Create a (primary) skeleton of the ground truth
    prim_sk = skeletonize(gt_mask, method='lee') != 0
    if closing_prim: prim_sk = binary_closing(prim_sk, square(closing_prim))
    
    # Create a dilated version of the primary skeleton for generating the secondary skeleton (otherwise it interprets it as gaps in the primary skeleton)
    prim_sk_dilated = binary_dilation(prim_sk, square(3))
    # Add the dilated skeleton to the ground truth mask
    gt_mask2d_with_prim_sk[prim_sk_dilated] = False
    # Create the secondary skeleton
    sec_sk = skeletonize(gt_mask2d_with_prim_sk, method='lee') != 0
    if closing_sec: sec_sk = binary_closing(sec_sk, square(closing_prim))

    return prim_sk, sec_sk

def pick_sk_squares(sk, sk_max_pix=20, sq_size=20, sq_pix_range=(10, 40)):
    '''
    Pick random squares from the skeleton.
    Input:
        sk (numpy array): the skeleton
        sk_max_pix (int): the approximate number of pixels that should be picked
        sq_size (int): the size of the squares (side length)
        sq_pix_range (int): the range that the number of pixels in a square shall be in
    Output:
        all_squares (numpy array): the mask of all squares
    '''
    pix_in_sk = np.sum(sk)
    all_squares = np.zeros_like(sk, dtype=np.bool8)
    added_pix = 0
    attempts = 0
    overshoots = 0
    # Loop until the total number of pixels in all squares approaches the threshold or the maximum number of attempts is reached
    while overshoots < 10:
        attempts += 1
        # If the number of attempts is too high or 90% of the skeleton are annotated, print a warning and break the loop
        if attempts > pix_in_sk * sk_max_pix or added_pix > pix_in_sk * 0.99:
            print("Warning: Could not create enough squares from the skeleton ({sk}).")
            return all_squares
        # Pick a random square from the skeleton
        square = pick_square(sk, sq_size)
        pix_in_sq = np.sum(square)
        # If there are too few or too many pixels in the square, skip it
        if pix_in_sq < sq_pix_range[0] or pix_in_sq > sq_pix_range[1]:
            continue
        # If the square would push the total number of pixels on lines above the maximum number of pixels, skip it and count the overshoot
        elif added_pix + pix_in_sq > sk_max_pix:
            overshoots += 1
            continue
        # If the square is valid, add it to the mask of all lines
        else:
            all_squares = np.logical_or(all_squares, square)
            added_pix = np.sum(all_squares)
    # If no squares were added, try again with smaller squares and a range starting at a lower value (allowing fewer pixels in a square)
    if added_pix == 0:
        all_squares = pick_sk_squares(sk, sk_max_pix, sq_size//2, (sq_pix_range[0]//2, sq_pix_range[1]))
    return all_squares

def pick_square(mask, sq_size=20):
    '''
    Take a random point on a mask (= True) and return the part inside a square around it.
    Input:
        mask (numpy array): the mask
        sq_size (int): the size of the squares (side length)
    Output:
        square_mask (numpy array): the mask inside the square
    '''
    # Choose a random point from the mask
    mask_coordinates = np.argwhere(mask)
    total_points = mask_coordinates.shape[0]
    random_point_index = np.random.choice(total_points)
    random_point = mask_coordinates[random_point_index]
    # Create an empty image to draw the square on
    square_mask = np.zeros_like(mask)
    # Draw the square on the image
    square_mask[random_point[0]-sq_size//2:random_point[0]+sq_size//2, random_point[1]-sq_size//2:random_point[1]+sq_size//2] = mask[random_point[0]-sq_size//2:random_point[0]+sq_size//2, random_point[1]-sq_size//2:random_point[1]+sq_size//2]
    return square_mask

def create_lines(sk, gt_mask, lines_max_pix=20, line_pix_range=(10, 40)):
    '''
    Create lines leading from a skeleton to the edge of the mask.
    Input:
        sk (numpy array): the skeleton mask
        gt_mask (numpy array): the ground truth mask
        lines_max_pix (int): the maximum/approximate number of pixels that should be picked with all lines
        line_pix_range (int): the range that the number of pixels for a single line shall be in
    Output:
        all_lines (numpy array): the mask of all lines
    '''
    # Initialize the mask of all lines
    all_lines = np.zeros_like(gt_mask, dtype=np.bool8)
    added_pix = 0
    attempts = 0
    overshoots = 0
    # Loop until the pixels in all lines approach the threshold or the maximum number of attempts is reached
    while overshoots < 10:
        attempts += 1
        # If the number of attempts is too high, print a warning and break the loop
        if attempts > np.sum(sk) * lines_max_pix:
            print("Warning: Could not create enough lines from the skeleton to the edge.")
            return all_lines   
        # Draw a line from the skeleton to the edge of the mask
        line = draw_line(sk, gt_mask, dist_to_edge=2)
        pix_in_line = np.sum(line)
        # If the line is too short or too long, skip it
        if pix_in_line < line_pix_range[0] or pix_in_line > line_pix_range[1]:
            continue
        # If the line would push the total number of pixels on lines above the maximum number of pixels, skip it and count the overshoot
        elif added_pix + pix_in_line > lines_max_pix:
            overshoots += 1
            continue
        # If the line is valid, add it to the mask of all lines
        else:
            # Add the line to the mask of all lines
            all_lines = np.logical_or(all_lines, line)
            added_pix = np.sum(all_lines)
    # If no lines were added, try again with a range starting at a lower value (allowing fewer pixels per line)
    if added_pix == 0:
        all_lines = create_lines(sk, gt_mask, lines_max_pix, (line_pix_range[0]//2, line_pix_range[1]))
    return all_lines

def draw_line(sk_mask, gt_mask, dist_to_edge=5):
    '''
    Take a random point on the skeleton (= True in the mask) and draw a line to the nearest edge point of the ground truth mask.
    Input:
        sk_mask (numpy array): the skeleton mask
        gt_mask (numpy array): the ground truth mask
        dist_to_edge (int): the distance of the line to the edge
    Output:
        shortest_path_mask (numpy array): the mask of the shortest path
    '''
    # Choose a random point from the skeleton
    sk_coordinates = np.argwhere(sk_mask)
    total_points = sk_coordinates.shape[0]
    step_size = 1 #total_points // (num_lines * 2)
    possible_points = np.arange(total_points, step=step_size)
    random_point_index = np.random.choice(possible_points)
    random_point = sk_coordinates[random_point_index]
    # Erode the gt_mask, so that the line will have a distance to the edge
    if dist_to_edge:
        eroded_gt_mask = erosion(gt_mask, square(dist_to_edge*2))
    else:
        eroded_gt_mask = gt_mask
    # Find the shortest path from the random point to the edge of the mask and return the mask of the path
    shortest_path = point_to_edge(random_point, eroded_gt_mask)
    shortest_path_mask = shortest_path == 1
    return shortest_path_mask

def point_to_edge(start_point, segmentation_mask):
    '''
    Find the shortest path from a point to the edge of a segmentation mask.
    Input:
        start_point (tuple): the coordinates of the starting point
        segmentation_mask (numpy array): the segmentation mask
    Output:
        path_mask (numpy array): the mask of the shortest path
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