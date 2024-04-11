import numpy as np
import napari
from skimage.morphology import *
from skimage.draw import line
from scipy.spatial import distance

def create_even_scribbles(ground_truth, max_perc=0.2, sq_scaling=False, mode="all", print_steps=False, scribble_width=1):
    '''Generate the scribble annotation for the ground truth using an even distribution of pixels among the chosen scribble types (all, both skeletons or individual skeletons and lines).
    This function uses a scribble_width of 1, a formula to determine the square size and a range for pixels inside a square or line of half to double one square side length.
    These parameters should be suited for max_perc values between approximately 0.05 and 1.
    Input:
        ground_truth (numpy array): the fully annotated image
        max_perc (float): the maximum percentage of pixels that should be picked (from skeletons and lines)
        sq_scaling (int): the scaling factor for the square size (side length), i.e. the number of squares that would fit the image (default: 400/(max_perc**0.5))
        mode (str): the scribble types to use (lines, prim_sk, sec_sk, both_sk, all)
        print_steps (bool): whether to print the steps of the scribble creation
        scribble_width (int): the width of the scribble lines; IMPORTANT: the width is created through downstream dilation which alters the percentage...
    Output:
        scribble (numpy array): the scribble annotation
    '''    
    # Calculate parameters for the scribble annotation
    num_annots = {"lines": 1, "prim_sk": 1, "sec_sk": 1, "both_sk" : 2, "all": 3}
    max_perc_per_type = max_perc / num_annots[mode]
    if not sq_scaling: sq_scaling = 400/(max_perc**0.5)
    sq_size = (ground_truth.shape[0] * ground_truth.shape[1] // sq_scaling) ** 0.5
    sq_size = int(sq_size)

    # Generate the scribble annotation for the ground truth
    scribble = create_scribble(ground_truth, scribble_width=scribble_width, sk_max_perc=max_perc_per_type, sq_size=sq_size, sq_pix_range=False, lines_max_perc=max_perc_per_type, line_pix_range=False, mode=mode, print_steps=print_steps)

    # Handle edge cases where too many pixels were picked
    # (Should only happen if the total maximum is <3 and the minimum of one pixel was picked per scribble type, even though this pushed the total percentage above the maximum)
    # Do this for each class (= value) in the ground truth
    for class_val in set(ground_truth.flatten()):
        # Skip the background class
        if class_val == 0:
            continue
        else:
            # Find the maximum number of pixels that should be picked for the class
            gt_class_mask = (ground_truth == class_val)
            tot_class_pix = int(np.sum(gt_class_mask))
            max_pix = int(tot_class_pix * max_perc / 100)
            # If the maximum number of pixels is below 1, raise a warning and pick 1 pixel instead (avoiding empty scribble annotations)
            if max_pix < 1:
                print(f"WARNING: The theoretical maximum number of pixels for the entire class {class_val} ({max_pix:.2f}) is below 1. Instead, 1 pixel is picked.")
                max_pix = 1
            # If too many pixels are present in this class in the scribble, raise a warning and pick the requested number of pixels
            scribble_class_mask = scribble == class_val
            num_pix_in_scribble = np.sum(scribble_class_mask)
            # Note that when inflating the scribble width, the number of pixels will increase, so the maximum percentage no longer has to be guaranteed
            if num_pix_in_scribble > max_pix and scribble_width == 1:
                print(f"WARNING: The total number of pixels for class {class_val} ({num_pix_in_scribble}) exceeds the maximum ({max_pix:.2f}). Removing pixels...")
                scribble_class_coord = np.where(scribble_class_mask)
                scribble[scribble_class_coord[0][max_pix:], scribble_class_coord[1][max_pix:]]
                new_num_pix_in_scribble = np.sum(scribble == class_val)
                print(f"   New total number of pixels for the class: {new_num_pix_in_scribble} ({new_num_pix_in_scribble/tot_class_pix*100:.4f}%)")

    return scribble

def create_scribble(ground_truth, scribble_width=1, sk_max_perc=0.05, sq_size=20, sq_pix_range=False, lines_max_perc=0.05, line_pix_range=False, mode="all", print_steps=False):
    '''
    Generate the scribble annotation for the ground truth.
    Input:
        ground_truth (numpy array): the fully annotated image
        scribble_width (int): the width of the scribble lines; IMPORTANT: the width is created through dilation which alters the percentage...
        sk_max_perc (float): the maximum percentage of pixels of the ground truth that should be picked (from each of the skeletons)
        sq_size (int): the size of the squares (side length)
        sq_pix_range (int): the range that the number of pixels in a square shall be in
        lines_max_perc (float): the maximum percentage of pixels of the ground truth that should be created by drawing lines
        line_pix_range (int): the range that the number of pixels for a line shall be in
        mode (str): the scribble types to use (lines, prim_sk, sec_sk, both_sk, all)
        print_steps (bool): whether to print the steps of the scribble creation
    Output:
        output (numpy array): the scribble annotation
    '''
    scribble = np.zeros_like(ground_truth, dtype=np.uint8)
    # For each class (= value) in the ground truth, generate the scribble annotation
    for class_val in set(ground_truth.flatten()):
        # Skip the background class
        if class_val == 0:
            continue
        if print_steps:
            print(f"CLASS {class_val}:")
        # Generate the scribble annotation for the class
        class_scribble = scribble_class(ground_truth, class_val, scribble_width, sk_max_perc, sq_size, sq_pix_range, lines_max_perc, line_pix_range, mode, print_steps=print_steps)
        # Add the scribble annotation of this class to the full scribble (which is valid, because there is no overlap between the classes)
        scribble += class_scribble.astype(np.uint8)
    return scribble

def scribble_class(gt, class_val, scribble_width=1, sk_max_perc=0.05, sq_size=20, sq_pix_range=False, lines_max_perc=0.05, line_pix_range=False, mode="all", print_steps=False):
    '''
    Generate the scribble annotation for a specific class in the ground truth.
    Input:
        gt (numpy array): the ground truth
        class_val (int): the value of the class
        scribble_width (int): the width of the scribble lines (NOTE: the width is created through dilation which alters the percentage...)
        sk_max_perc (float): the maximum percentage of pixels of the ground truth that should be picked (from skeletons)
        sq_size (int): the size of the squares (side length)
        sq_pix_range (int): the range that the number of pixels in a square shall be in
        lines_max_perc (float): the maximum percentage of pixels of the ground truth that should be created by drawing lines
        line_pix_range (int): the range that the number of pixels for a line shall be in
        mode (str): the scribble types to use (lines, prim_sk, sec_sk, both_sk, all)
        print_steps (bool): whether to print the steps of the scribble creation
    Output:
        class_scribble (numpy array): the scribble annotation for the class
    '''
    # Generate a boolean mask for the ground truth matching the class_id
    gt_class_mask = (gt == class_val)
    tot_class_pix = int(np.sum(gt_class_mask))
    # Initialize the scribble for the class with zeros
    class_scribble = np.zeros_like(gt, dtype=np.int32)

    # Generate the primary and secondary skeleton for the class in this slice
    prim_sk, sec_sk = double_sk_class(gt_class_mask)
    # v = napari.Viewer()
    # v.add_image(prim_sk)
    # v.add_image(sec_sk)
    # Check if a skeleton was created, raise an error if not
    if np.sum(prim_sk) == 0:
        raise ValueError(f"No skeleton was created for class {class_val}.")

    # PICK SKELETON SQUARES
    if mode in ("prim_sk", "sec_sk", "both_sk", "all"):
        # Calculate how many pixels of each skeleton are allowed in this class given the percentage
        sk_max_pix = tot_class_pix * sk_max_perc / 100
        # Ensure that the maximum number of pixels is at least 1 (i.e. at least one pixel is picked)
        if sk_max_pix < 1:
            print(f"WARNING: The theoretical maximum number of pixels for the skeletons ({sk_max_pix:.2f}) is below 1 for class {class_val}. Instead, 1 pixel will be picked.")
            sk_max_pix = 1
        # Ensure that each square is allowed to contain as little pixels as the maximum total pixels in all squares
        sq_pix_range = (min(sq_size//2, int(sk_max_pix)), sq_size*2) if not sq_pix_range else sq_pix_range
        if print_steps:
            print(f"sk_max_pix: {sk_max_pix:.2f}, sq_size: {sq_size}, sk_pix_range: {sq_pix_range}")
        # If the primary skeleton is needed, pick squares of it
        if mode in ("prim_sk", "both_sk", "all"):
            prim_sk_squares = pick_sk_squares(prim_sk, sk_max_pix=sk_max_pix, sq_size=sq_size, sq_pix_range=sq_pix_range)
            if print_steps:
                print(f"   prim_sk_squares pix: {np.sum(prim_sk_squares)} = {np.sum(prim_sk_squares)/np.sum(gt_class_mask)*100:.2f}%")    
        # If the secondary skeleton is needed, pick squares of it
        if mode in ("sec_sk", "both_sk", "all"):
            sec_sk_squares = pick_sk_squares(sec_sk, sk_max_pix=sk_max_pix, sq_size=sq_size, sq_pix_range=sq_pix_range)
            if print_steps:
                print(f"   sec_sk_squares pix: {np.sum(sec_sk_squares)} = {np.sum(sec_sk_squares)/np.sum(gt_class_mask)*100:.2f}%")
        # If both skeletons are needed, combine the squares of both skeletons
        if mode in ("both_sk", "all"):
            both_sk_squares = np.logical_or(prim_sk_squares, sec_sk_squares)

    # PICK LINES
    # If lines are needed, create and pick them (lines leading from the primary skeleton to the edge of the mask)
    if mode in ("lines", "all"):
        # Calculate how many pixels of lines are allowed in this class given the percentage
        lines_max_pix = tot_class_pix * lines_max_perc / 100
        # Ensure that the maximum number of pixels is at least 1 (i.e. at least one pixel is picked)
        if lines_max_pix < 1:
            print(f"WARNING: The theoretical maximum number of pixels for the lines ({lines_max_pix:.2f}) is below 1 for class {class_val}. Instead, 1 pixel will be picked.")
            lines_max_pix = 1
        # Ensure that the line is allowed to be as short as the maximum total pixels in all lines
        line_pix_range = (min(sq_size//2, int(lines_max_pix)), sq_size*2) if not line_pix_range else line_pix_range
        if print_steps:
            print(f"lines_max_pix: {lines_max_pix:.2f}, line_pix_range: {line_pix_range}")
        lines = create_lines(prim_sk, gt_class_mask, lines_max_pix, line_pix_range)
        if print_steps:
            print(f"   lines pix: {np.sum(lines)} = {np.sum(lines)/np.sum(gt_class_mask)*100:.2f}%")
    if mode =="all":
        lines_and_squares = np.logical_or(lines, both_sk_squares)

    # Define the scribble type to use
    if mode == "lines": class_scribble_mask = lines
    elif mode == "prim_sk": class_scribble_mask = prim_sk_squares
    elif mode == "sec_sk": class_scribble_mask = sec_sk_squares
    elif mode == "both_sk": class_scribble_mask = both_sk_squares
    elif mode == "all": class_scribble_mask = lines_and_squares

    # Print the total picked pixels
    if print_steps:
        print(f"TOTAL pix: {np.sum(class_scribble_mask)} = {np.sum(class_scribble_mask)/np.sum(gt_class_mask)*100:.2f}%")    

    # Dilate the scribble to make them wider; NOTE: this dilation will alter the percentage!!!
    class_scribble_mask = dilation(class_scribble_mask, square(scribble_width))
    
    # Ensure that the scribble is within the ground truth mask
    class_scribble_mask = np.logical_and(class_scribble_mask, gt_class_mask)
    # Use the mask to add the class values to the scribble
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
    
    # Create a dilated version of the primary skeleton for generating the secondary skeleton (otherwise it interprets sees gaps in the primary skeleton)
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
    # Shuffle the coordinates of the skeleton to loop over them in a random order
    sk_coordinates = np.argwhere(sk)
    np.random.shuffle(sk_coordinates)
    # Initialize the mask of all squares
    all_squares = np.zeros_like(sk, dtype=np.bool8)
    added_pix = 0
    idx = 0
    overshoots = 0
    # Loop until the total number of pixels in all squares approaches the threshold or the end of all pixels in the skeleton is reached
    while overshoots < 100 and idx < pix_in_sk:
        # Pick a random square from the skeleton
        current_coordinate = sk_coordinates[idx]
        idx += 1        
        square = get_square(sk, current_coordinate, sq_size)
        pix_in_sq = np.sum(square)
        # If the square would push the total number of pixels in all squares above the maximum, skip it and count the overshoot
        if added_pix + pix_in_sq > sk_max_pix:
            overshoots += 1
            continue
        # If there are too few or too many pixels in the square, skip it
        elif pix_in_sq < sq_pix_range[0] or pix_in_sq > sq_pix_range[1]:
            continue
        # If the square is valid, add it to the mask of all lines
        else:
            all_squares = np.logical_or(all_squares, square)
            added_pix = np.sum(all_squares)
    # If no squares were added, try again with smaller squares and a range starting at a lower value (allowing fewer pixels in a square)
    if added_pix == 0:
        # Do not reduce the square size below 1
        if sq_size >= 2:
            # Reduce the square size
            sq_size = sq_size//2
            # Adjust the range accordingly
            sq_pix_range = (min(sq_size//2, int(sk_max_pix)), sq_pix_range[1])
            print("Adjusting square size and range to", sq_size, sq_pix_range)
            all_squares = pick_sk_squares(sk, sk_max_pix, sq_size, sq_pix_range)
        else:
            print("ERROR: No squares were added!")
    return all_squares

def get_square(mask, coord, sq_size=20):
    '''
    Take a point on a mask and return the part inside a square around it.
    Input:
        mask (numpy array): the mask
        coord (numpy array): the coordinates of the center of the square
        sq_size (int): the size of the square (side length)
    Output:
        square_mask (numpy array): the mask inside the square
    '''
    # Create an empty image to draw the square on
    square_mask = np.zeros_like(mask)
    # Draw the square on the image
    red = int(np.floor(sq_size/2))
    red = min(red, coord[0]) # Ensure that the square does not exceed the mask
    inc = int(np.ceil(sq_size/2)) # Here, the index can exceed the mask because slicing will stop at the end of the mask
    square_mask[coord[0]-red:coord[0]+inc, coord[1]-red:coord[1]+inc] = mask[coord[0]-red:coord[0]+inc, coord[1]-red:coord[1]+inc]
    return square_mask

def create_lines(sk, gt_mask, lines_max_pix=20, line_pix_range=(10, 40), dist_to_edge=2):
    '''
    Create lines leading from a skeleton to the edge of the mask.
    Input:
        sk (numpy array): the skeleton mask
        gt_mask (numpy array): the ground truth mask
        lines_max_pix (int): the maximum number of pixels that should be picked with all lines
        line_pix_range (int): the range that the number of pixels for a single line shall be in
        dist_to_edge (int): the distance of the line to the edge
    Output:
        all_lines (numpy array): the mask of all lines
    '''
    # Initialize the mask of all lines
    pix_in_sk = np.sum(sk)
    # Shuffle the coordinates of the skeleton to loop over them in a random order
    sk_coordinates = np.argwhere(sk)
    np.random.shuffle(sk_coordinates)
    # Initialize the mask of all lines
    all_lines = np.zeros_like(gt_mask, dtype=np.bool8)
    added_pix = 0
    idx = 0
    overshoots = 0
    # Loop until the pixels in all lines approach the threshold or the end of all pixels in the skeleton is reached
    while overshoots < 100 and idx < pix_in_sk:
        # Draw a line from the skeleton to the edge of the mask
        current_coordinate = sk_coordinates[idx]
        idx += 1
        line = get_line(current_coordinate, gt_mask, dist_to_edge=dist_to_edge)
        pix_in_line = np.sum(line)
        # If the line would push the total number of pixels on lines above the maximum, skip it and count the overshoot
        if added_pix + pix_in_line > lines_max_pix:
            overshoots += 1
            continue
        # If the line is too short or too long, skip it
        elif pix_in_line < line_pix_range[0] or pix_in_line > line_pix_range[1]:
            continue
        # If the line is valid, add it to the mask of all lines
        else:
            # Add the line to the mask of all lines
            all_lines = np.logical_or(all_lines, line)
            added_pix = np.sum(all_lines)
    # If no lines were added, try again with adjusted parameters
    if added_pix == 0:
        # If the line range is too small, make it larger (especially decreasing the minimum) and try again
        # NOTE: if the upper bound is still too low, this is not a big deal, because we will instead shorten the lines
        if line_pix_range[0] > 1: # or line_pix_range[1] > max(gt_mask.shape) // 2:
            line_pix_range = (line_pix_range[0]//2, line_pix_range[1] * 2)
            print("Adjusting line range to" , line_pix_range)
            all_lines = create_lines(sk, gt_mask, lines_max_pix, line_pix_range)
        # If this did not work (i.e. the lines are longer than the lines_max), shorten the lines by increasing the distance to the edge
        elif dist_to_edge < max(gt_mask.shape) // 2:
            # Take a minimum distance of 2 pixels to the edge, to not get stuck at 1 pixel...
            new_dist_to_edge = max(2, int(dist_to_edge * 1.5))
            print("Adjusting distance to edge to", new_dist_to_edge)
            all_lines = create_lines(sk, gt_mask, lines_max_pix, line_pix_range, new_dist_to_edge)
        else:
            print("ERROR: No lines were added!")
    return all_lines

def get_line(coord, gt_mask, dist_to_edge=2):
    '''
    Take a point on the skeleton (= True in the mask) and draw a line to the nearest edge point of the ground truth mask.
    Input:
        coord (tuple): the coordinates of the starting point
        gt_mask (numpy array): the ground truth mask
        dist_to_edge (int): the distance of the line to the edge
    Output:
        shortest_path_mask (numpy array): the mask of the shortest path
    '''
    # Erode the gt_mask, so that the line will have a distance to the edge
    if dist_to_edge:
        eroded_gt_mask = erosion(gt_mask, square(dist_to_edge*2))
    else:
        eroded_gt_mask = gt_mask
    # Find the shortest path from the random point to the edge of the mask and return the mask of the path
    shortest_path_mask = point_to_edge(coord, eroded_gt_mask)
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