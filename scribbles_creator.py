import numpy as np
from skimage.morphology import *
from skimage.draw import line
from scipy.spatial import distance

def create_even_scribbles(ground_truth, max_perc=0.2, margin=0.75, rel_scribble_len=False, mode="all", print_steps=False, scribble_width=1):
    '''Generate the scribble annotation for the ground truth using an even distribution of pixels among the chosen scribble types (all, both skeletons or individual skeletons and lines).
    This function uses a default scribble_width of 1, a formula to determine the square size and a range for pixels inside a square or line of half to double one square side length.
    These parameters should be suited for max_perc values between approximately 0.05 and 1.
    Input:
        ground_truth (numpy array): the fully annotated image
        max_perc (float): the maximum percentage of pixels that should be picked (from skeletons and lines)
        margin (float): the margin for minimum nr. pixels that should be picked, as a proportion of the pixels given by max_perc (default: 0.75)
        rel_scribble_len (int/bool): length of the single scribbles relative to pixel dimensions, i.e. the number of scribbles that would fit the image (empirical default value: 20/(max_perc**0.25))        mode (str): the scribble types to use (lines, prim_sk, sec_sk, both_sk, all)
        print_steps (bool): whether to print the steps of the scribble creation
        scribble_width (int): the width of the individual scribbles
    Output:
        scribbles (numpy array): the scribble annotation
    '''
    # Calculate parameters for the scribble annotation
    num_types = {"lines": 1, "prim_sk": 1, "sec_sk": 1, "both_sk" : 2, "all": 3}[mode]
    # Calculate the maximum percentage of pixels per type
    max_perc_per_type = max_perc / num_types
    # If the relative scribble length is not given, calculate it by an empirical formula
    if not rel_scribble_len: rel_scribble_len = 20/(max_perc**0.25)
    # The square size is calculated by the number of pixels that would fit into the dimensions of the image
    sq_size = (ground_truth.shape[0] * ground_truth.shape[1] // rel_scribble_len**2) ** 0.5
    sq_size = int(sq_size)

    # Generate the scribble annotation for the ground truth
    scribbles = create_scribbles(ground_truth, scribble_width=scribble_width, sk_max_perc=max_perc_per_type, sk_margin=margin, sq_size=sq_size, sq_pix_range=False, lines_max_perc=max_perc_per_type, lines_margin=margin, line_pix_range=False, mode=mode, print_steps=print_steps)

    # Handle edge cases where too many pixels were picked
    # (Should only happen if the total maximum is < 3 and the minimum of one pixel was picked per scribble type, even though this pushed the total percentage above the maximum)
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
                print(f"\nWARNING: The theoretical maximum number of pixels for the ENTIRE CLASS {class_val} ({max_pix}) is below 1. Instead, 1 pixel is picked.")
                max_pix = 1
            # If too many pixels are present in this class in the scribble, raise a warning and pick the requested number of pixels
            scribble_class_mask = scribbles == class_val
            num_pix_in_scribble = np.sum(scribble_class_mask)
            # Remove pixels if the total number of pixels exceeds the maximum
            if num_pix_in_scribble > max_pix:
                print(f"\nWARNING: The total number of picked pixels for class {class_val} ({num_pix_in_scribble}) exceeds the maximum ({max_pix}). Removing pixels...")
                scribble_class_coord = np.where(scribble_class_mask)
                scribbles[scribble_class_coord[0][max_pix:], scribble_class_coord[1][max_pix:]] = 0
                new_num_pix_in_scribble = np.sum(scribbles == class_val)
                print(f"New total number of pixels for this class: {new_num_pix_in_scribble} ({new_num_pix_in_scribble/tot_class_pix*100:.4f}%)")

    return scribbles

def create_scribbles(ground_truth, scribble_width=1, sk_max_perc=0.05, sk_margin=0.75, sq_size=20, sq_pix_range=False, lines_max_perc=0.05, lines_margin=0.75, line_pix_range=False, mode="all", print_steps=False):
    '''
    Generate the scribble annotation for the ground truth.
    Input:
        ground_truth (numpy array): the fully annotated image
        scribble_width (int): the width of the individual scribbles
        sk_max_perc (float): the maximum percentage of pixels of the ground truth that should be picked (from each of the skeletons)
        sk_margin (float): for the skeletons - the margin for minimum nr. pixels that should be picked, as a proportion of the pixels given by max_perc (default: 0.75)
        sq_size (int): the size of the squares (side length)
        sq_pix_range (int): the range that the number of pixels in a square shall be in
        lines_max_perc (float): the maximum percentage of pixels of the ground truth that should be created by drawing lines
        lines_margin (float): for the lines - the margin for minimum nr. pixels that should be picked, as a proportion of the pixels given by max_perc (default: 0.75)
        line_pix_range (int): the range that the number of pixels for a line shall be in
        mode (str): the scribble types to use (lines, prim_sk, sec_sk, both_sk, all)
        print_steps (bool): whether to print the steps of the scribble creation
    Output:
        scribble_annotation (numpy array): the scribble annotation
    '''
    scribble_annotation = np.zeros_like(ground_truth, dtype=np.uint8)
    # For each class (= value) in the ground truth, generate the scribble annotation
    for class_val in set(ground_truth.flatten()):
        # Skip the background class
        if class_val == 0:
            continue
        if print_steps:
            print(f"CLASS {class_val}:")
        # Generate the scribble annotation for the class
        class_scribble_annotation = scribble_class(ground_truth, class_val, scribble_width, sk_max_perc, sk_margin, sq_size, sq_pix_range, lines_max_perc, lines_margin, line_pix_range, mode, print_steps=print_steps)
        # Add the scribble annotation of this class to the full scribble (which is valid, because there is no overlap between the classes)
        scribble_annotation += class_scribble_annotation.astype(np.uint8)
    return scribble_annotation

def scribble_class(gt, class_val, scribble_width=1, sk_max_perc=0.05, sk_margin=0.75, sq_size=20, sq_pix_range=False, lines_max_perc=0.05, lines_margin=0.75, line_pix_range=False, mode="all", print_steps=False):
    '''
    Generate the scribble annotation for a specific class in the ground truth.
    Input:
        gt (numpy array): the ground truth
        class_val (int): the value of the class
        scribble_width (int): the width of the individual scribbles
        sk_max_perc (float): the maximum percentage of pixels of the ground truth that should be picked (from skeletons)
        sk_margin (float): for the skeleton - the margin for minimum nr. pixels that should be picked, as a proportion of the pixels given by max_perc (default: 0.75)
        sq_size (int): the size of the squares (side length)
        sq_pix_range (int): the range that the number of pixels in a square shall be in
        lines_max_perc (float): the maximum percentage of pixels of the ground truth that should be created by drawing lines
        lines_margin (float): for the lines - the margin for minimum nr. pixels that should be picked, as a proportion of the pixels given by max_perc (default: 0.75)
        line_pix_range (int): the range that the number of pixels for a line shall be in
        mode (str): the scribble types to use (lines, prim_sk, sec_sk, both_sk, all)
        print_steps (bool): whether to print the steps of the scribble creation
    Output:
        class_scribble (numpy array): the scribble annotation for the class
    '''
    # Generate a boolean mask for the ground truth matching the class_id
    gt_class_mask = (gt == class_val)
    tot_class_pix = int(np.sum(gt_class_mask))

    # Generate the primary and secondary skeleton for the class in this slice
    prim_sk, sec_sk = double_sk_class(gt_class_mask)

    # Check if a skeleton was created, raise an error if not
    if np.sum(prim_sk) == 0:
        raise ValueError(f"No skeleton was created for class {class_val}.")

    # PICK SKELETON SQUARES
    if mode in ("prim_sk", "sec_sk", "both_sk", "all"):
        # Calculate how many TOTAL pixels of each skeleton are allowed in this class given the percentage
        sk_max_pix = tot_class_pix * sk_max_perc / 100
        # Ensure that the TOTAL maximum number of pixels is at least scribble_width**2 (avoiding empty scribble annotations, i.e. allowing for at least a "point scribble")
        if sk_max_pix < scribble_width**2:
            print(f"   WARNING: The theoretical maximum number of pixels for the SQUARES ({sk_max_pix:.2f}) is below scribble_width**2. Instead, scribble_width**2 pixel(s) is/are picked.")
            sk_max_pix = scribble_width**2
        # If the maximum number of pixels is below the square of the scribble width, the square might be decreased to 1 unecessarily; thus adjust the scribble width accordingly
        # if sk_max_pix < scribble_width**2:
        #     print(f"   WARNING: The theoretical maximum number of pixels for the SQUARES ({sk_max_pix:.2f}) is below the square of the scribble_width ({scribble_width**2}). The scribble width is adjusted to {int(sk_max_pix**0.5)}.")
        #     scribble_width_for_squares = int(sk_max_pix**0.5)
        # else:
        #     scribble_width_for_squares = scribble_width
        # Define the range of pixels in a SINGLE square
        sq_pix_max = sq_size*2
        sq_pix_min = sq_size//2
        # Make sure the minumim cannot be 0
        sq_pix_min = max(1, sq_pix_min)
        # Adjust the range to scribble_width (if the scribble is wider, the range needs to be larger to have similar lengths of the scribbles)
        sq_pix_min, sq_pix_max = int(sq_pix_min * scribble_width), int(sq_pix_max * scribble_width)
        # Make sure that the minimum to pick is not above the total maximum allowed
        sq_pix_min = min(sq_pix_min, int(sk_max_pix))
        # Use these values if no range was given
        sq_pix_range = (sq_pix_min, sq_pix_max) if not sq_pix_range else sq_pix_range
        if print_steps:
            print(f"   sk_max_pix: {sk_max_pix:.2f}, sq_size: {sq_size}, sq_pix_range: {sq_pix_range}")
        # If the primary skeleton is needed, pick squares of it
        if mode in ("prim_sk", "both_sk", "all"):
            prim_sk_squares = pick_sk_squares_optim(prim_sk, gt_class_mask, sk_max_pix=sk_max_pix, sk_margin=sk_margin, sq_size=sq_size, sq_pix_range=sq_pix_range, scribble_width=scribble_width, print_steps=print_steps)
            if print_steps:
                print(f"      prim_sk_squares pix: {np.sum(prim_sk_squares)} = {np.sum(prim_sk_squares)/np.sum(gt_class_mask)*100:.2f}%")
        # If the secondary skeleton is needed, pick squares of it
        if mode in ("sec_sk", "both_sk", "all"):
            sec_sk_squares = pick_sk_squares_optim(sec_sk, gt_class_mask, sk_max_pix=sk_max_pix, sk_margin=sk_margin, sq_size=sq_size, sq_pix_range=sq_pix_range, scribble_width=scribble_width, print_steps=print_steps)
            if print_steps:
                print(f"      sec_sk_squares pix: {np.sum(sec_sk_squares)} = {np.sum(sec_sk_squares)/np.sum(gt_class_mask)*100:.2f}%")
        # If both skeletons are needed, combine the squares of both skeletons
        if mode in ("both_sk", "all"):
            both_sk_squares = np.logical_or(prim_sk_squares, sec_sk_squares)

    # PICK LINES
    # If lines are needed, create and pick them (lines leading from the primary skeleton to the closest edge of the mask)
    if mode in ("lines", "all"):
        # Calculate how many TOTAL pixels of lines are allowed in this class given the percentage
        lines_max_pix = tot_class_pix * lines_max_perc / 100
        # Ensure that the TOTAL maximum number of pixels is at least scribble_width**2 (avoiding empty scribble annotations, i.e. allowing for at least a "point scribble")
        if lines_max_pix < scribble_width**2:
            print(f"   WARNING: The theoretical maximum number of pixels for the LINES ({lines_max_pix:.2f}) is below scribble_width**2. Instead, scribble_width**2 pixel(s) is/are picked.")
            lines_max_pix = scribble_width**2
        # # If the maximum number of pixels is below the square of the scribble width, even a line of only length 1 will overshoot; thus adjust the scribble width accordingly
        # if lines_max_pix < scribble_width**2:
        #     print(f"   WARNING: The theoretical maximum number of pixels for the LINES ({lines_max_pix:.2f}) is below the square of the scribble_width ({scribble_width**2}). The scribble width is adjusted to {int(lines_max_pix**0.5)}.")
        #     scribble_width_for_lines = int(lines_max_pix**0.5)
        # else:
        #     scribble_width_for_lines = scribble_width
        # Define the range of pixels in a SINGLE line
        line_pix_max = sq_size*2
        line_pix_min = sq_size//2
        # Adjust the range to the scribble_width (if the scribble is wider, the range need to be higher to have similar lengths of the scribbles)
        line_pix_min, line_pix_max = int(line_pix_min * scribble_width), int(line_pix_max * scribble_width)
        # Ensure that the line is allowed to be as short as the maximum total pixels in all lines
        line_pix_min = min(line_pix_min, int(lines_max_pix))
        # Use these values if no range was given
        line_pix_range = (line_pix_min, line_pix_max) if not line_pix_range else line_pix_range
        if print_steps:
            print(f"   lines_max_pix: {lines_max_pix:.2f}, line_pix_range: {line_pix_range}")
        lines = create_lines_optim(prim_sk, gt_class_mask, lines_max_pix, lines_margin, line_pix_range, scribble_width, print_steps=print_steps)
        if print_steps:
            print(f"      lines pix: {np.sum(lines)} = {np.sum(lines)/np.sum(gt_class_mask)*100:.2f}%")
    if mode == "all":
        lines_and_squares = np.logical_or(lines, both_sk_squares)

    # Define the scribble type to use
    class_scribble_mask = {"lines": lines,
                           "prim_sk": prim_sk_squares,
                           "sec_sk": sec_sk_squares,
                           "both_sk": both_sk_squares,
                           "all": lines_and_squares}[mode]

    # Ensure that the scribble is within the ground truth mask
    before = np.sum(class_scribble_mask)
    class_scribble_mask = np.logical_and(class_scribble_mask, gt_class_mask)
    after = np.sum(class_scribble_mask)
    # Print the total picked pixels
    if print_steps:
        print(f"   Checking for pixels OUTSIDE the class ground truth (all scribbles types together) - before: {before} = {before/np.sum(gt_class_mask)*100:.2f}%")
        print(f"      {before - after} pixel(s) removed from the scribble because they were outside the ground truth mask (probably due to dilation to scribble width).")
        removed_pix = np.argwhere((class_scribble_mask == True) & (gt_class_mask == False))
        print(f"      Coordinates of removed pixels: {removed_pix}")
        print(f"   TOTAL pix - after: {after} = {after/np.sum(gt_class_mask)*100:.2f}%")
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

def pick_sk_squares(sk, gt_mask, sk_max_pix=20, sq_size=20, sq_pix_range=(10,40), scribble_width=1, print_details=False):
    '''
    Pick random squares from the skeleton.
    Input:
        sk (numpy array): the skeleton
        sk_max_pix (int): the approximate number of pixels that should be picked
        sq_size (int): the size of the squares (side length)
        sq_pix_range (int): the range that the number of pixels in a square shall be in
        scribble_width (int): the width of the individual scribbles
    Output:
        all_squares (numpy array): the mask of all squares
    '''
    pix_in_sk = np.sum(sk)
    # Shuffle the coordinates of the skeleton to loop over them in a random order
    sk_coordinates = np.argwhere(sk)
    np.random.shuffle(sk_coordinates)
    # Create a dilated version of the skeleton to picke the squares from
    sk_dilated = binary_dilation(sk, square(scribble_width))
    # Ensure all pixels of the dilated skeleton are within the mask
    sk_dilated = np.logical_and(sk_dilated, gt_mask)
    # Initialize the mask of all squares and variables for the loop
    all_squares = np.zeros_like(sk_dilated, dtype=np.bool8)
    added_pix = 0
    idx = 0
    idx_step = 1
    overshoots = 0
    if print_details:
        print("--- Sampling squares: pix_in_sk", pix_in_sk, "indx_step", idx_step)
    # Loop until the total number of pixels in all squares approaches the threshold or the end of all pixels in the skeleton is reached
    while overshoots < 100 and idx < pix_in_sk:
        # Pick a random square from the skeleton
        current_coordinate = sk_coordinates[idx]
        idx += idx_step
        sk_square = get_square(sk_dilated, current_coordinate, sq_size)
        pix_in_sq = np.sum(sk_square)
        future_total = added_pix + pix_in_sq
        if print_details:
            print(f"---    current_coordinate: {current_coordinate} | added_pix (before): {added_pix}, pix_in_square: {pix_in_sq}, future_total: {future_total}")
        # If the square would push the total number of pixels in all squares above the maximum, skip it and count the overshoot
        if future_total > sk_max_pix:
            if print_details:
                print("---    overshoot nr.", overshoots)
            overshoots += 1
            continue
        # If there are too few or too many pixels in the square, skip it
        elif pix_in_sq < sq_pix_range[0] or pix_in_sq > sq_pix_range[1]:
            if print_details:
                print("---    outside sq_pix_range", sq_pix_range)
            continue
        # If the square is valid, add it to the mask of all lines
        else:
            all_squares = np.logical_or(all_squares, sk_square)
            added_pix = np.sum(all_squares)
        if added_pix == int(sk_max_pix):
            if print_details:
                print("---    sk_max_pix reached (no improvement possible)")
            break
        if sk_max_pix - added_pix < sq_pix_range[0]:
            if print_details:
                print("---    the remaining total pixels are too few for a square")
            break
    if print_details:
        print("--- Done sampling squares")
    return all_squares

def pick_sk_squares_optim(sk, gt_mask, sk_max_pix=20, sk_margin=0.75, sq_size=20, sq_pix_range=(10,0), scribble_width=1, print_steps=False):
    '''
    Pick random squares from the skeleton. Use the base function and adjust the parameters while too little squares were added.
    Input:
        sk (numpy array): the skeleton
        sk_max_pix (int): the approximate number of pixels that should be picked
        sk_margin (float): the margin for minimum nr. pixels that should be picked, as a proportion of the pixels given by max_perc (default: 0.75)
        sq_size (int): the size of the squares (side length)
        sq_pix_range (int): the range that the number of pixels in a square shall be in
        scribble_width (int): the width of the individual scribbles
    Output:
        squares (numpy array): the mask of all squares
    '''
    squares = pick_sk_squares(sk, gt_mask, sk_max_pix, sq_size, sq_pix_range, scribble_width)
    added_pix = np.sum(squares)
    # If not enough squares were added, try again with smaller squares and a range starting at a lower value (allowing fewer pixels in a square)
    # NOTE: We check at least for 1 pixel, since otherwise we would allow for empty scribbles; on the other hand we take floor, to ensure it cannot be expected to be above the allowed maximum
    while added_pix < max(1, np.floor(sk_max_pix * sk_margin)):
        # Do not reduce the square size below scribble_width
        if sq_size > scribble_width:
            # Reduce the square size
            diff_to_widht = sq_size - scribble_width
            sq_size = scribble_width + diff_to_widht//2
            # Adjust the range accordingly
            # Make sure that the minimum to pick is not above the total maximum allowed
            sq_pix_min = min(sq_pix_range[0]//2, int(sk_max_pix))
            # Make sure the minumim cannot be 0
            sq_pix_min = max(1, sq_pix_min)
            sq_pix_range = (sq_pix_min, sq_pix_range[1])
            if print_steps:
                print("         Adjusting square size and range to", sq_size, sq_pix_range)
        # If we are reaching the limit of what we can sample, break the loop and raise a warning (if at least some squares were added) or an error (if no squares were added)
        else:
            if added_pix == 0:
                print("   ERROR: No squares were added!")
            else:
                print(f"   WARNING: It was not possible to sample {sk_margin * 100}% of the requested pixels. Only {added_pix} pixels in squares were added!")
            break
        # Create new squares with the adjusted parameters and try again; add them to the squares
        sk_max_pix_left = sk_max_pix - added_pix
        if print_steps:
            print("         sk_max_pix_left:", sk_max_pix_left)
        # Sample again and add the new squares to the existing ones
        new_squares = pick_sk_squares(sk, gt_mask, sk_max_pix_left, sq_size, sq_pix_range, scribble_width)
        squares = np.logical_or(squares, new_squares)
        added_pix = np.sum(squares)
    return squares

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

def create_lines(sk, gt_mask, lines_max_pix=20, line_pix_range=(10, 40), scribble_width=1, line_crop=2, print_details=False):
    '''
    Create lines leading from a skeleton to the edge of the mask.
    Input:
        sk (numpy array): the skeleton mask
        gt_mask (numpy array): the ground truth mask
        lines_max_pix (int): the maximum number of pixels that should be picked with all lines
        line_pix_range (int): the range that the number of pixels for a single line shall be in
        scribble_width (int): the width of the individual scribbles
        line_crop (int): the crop of the line (on both sides; i.e. distance to edge and and skeleton will each be half of the crop)
    Output:
        all_lines (numpy array): the mask of all lines
    '''
    pix_in_sk = np.sum(sk)
    # Shuffle the coordinates of the skeleton to loop over them in a random order
    sk_coordinates = np.argwhere(sk)
    np.random.shuffle(sk_coordinates)
    # Initialize the mask of all picked lines and variables for the loop
    all_lines = np.zeros_like(gt_mask, dtype=np.bool8)
    added_pix = 0
    idx = 0
    idx_step = min(5, int(np.ceil(pix_in_sk/250))) # 1
    overshoots = 0
    tried_lines = []
    if print_details:
        print("--- Sampling lines: pix_in_sk", pix_in_sk, "indx_step", idx_step, "line_crop", line_crop)
    # Loop until the pixels in all lines approach the threshold (keeps overshooting) or the end of all pixels in the skeleton is reached
    while overshoots < 100 and idx < pix_in_sk:
        # Draw a line from the skeleton to the edge of the mask
        current_coordinate = sk_coordinates[idx]
        idx += idx_step
        single_pix_line = get_line(current_coordinate, gt_mask, line_crop=line_crop)
        tried_lines.append(single_pix_line)
        # Dilate the line to the scribble width
        line = binary_dilation(single_pix_line, square(scribble_width))
        # Ensure all pixels of the dilated line are within the mask
        line = np.logical_and(line, gt_mask)
        pix_in_line = np.sum(line)
        future_total = added_pix + pix_in_line
        if print_details:
            print(f"---    current_coordinate: {current_coordinate}, line {(np.argwhere(single_pix_line)[0], np.argwhere(single_pix_line)[-1]) if np.sum(single_pix_line) > 0 else 'empty'} | added_pix (before): {added_pix}, pix_in_line: {pix_in_line}, future_total: {future_total}")
        # If the line would push the total number of pixels on lines above the maximum, skip it and count the overshoot
        if future_total > lines_max_pix:
            if print_details:
                print("---    overshoot nr.", overshoots)
            overshoots += 1
            continue
        # If the line is too short or too long, skip it
        elif pix_in_line < line_pix_range[0] or pix_in_line > line_pix_range[1]:
            if print_details:
                print("---    outside line_pix_range", line_pix_range)
            continue
        # If the line is valid, add it to the mask of all lines
        else:
            # Add the line to the mask of all lines
            all_lines = np.logical_or(all_lines, line)
            added_pix = np.sum(all_lines)
        if added_pix == int(lines_max_pix):
            if print_details:
                print("---    lines_max_pix reached (no improvement possible)")
            break
        if lines_max_pix - added_pix < line_pix_range[0]:
            if print_details:
                print("---    the remaining total pixels are too few for a line")
            break
    if print_details:
        avg_length_tried = get_lines_stats(tried_lines)
        print("--- Done sampling lines: avg_length_tried", avg_length_tried)
    return all_lines, tried_lines

def create_lines_optim(sk, gt_mask, lines_max_pix=20, lines_margin=0.75, line_pix_range=(10, 40), scribble_width=1, init_line_crop=2, print_steps=False):
    '''
    Create lines leading from a skeleton to the edge of the mask. Use the base function and adjust the parameters if no lines were added.
    Input:
        sk (numpy array): the skeleton mask
        gt_mask (numpy array): the ground truth mask
        lines_max_pix (int): the maximum number of pixels that should be picked with all lines
        lines_margin (float): the margin for minimum nr. pixels that should be picked, as a proportion of the pixels given by max_perc (default: 0.75)
        line_pix_range (int): the range that the number of pixels for a single line shall be in
        scribble_width (int): the width of the individual scribbles
        init_line_crop (int): the crop of the line to start with (on both sides; i.e. distance to edge and and skeleton will each be half of the crop), may be changed during optimization
    Output:
        lines (numpy array): the mask of all lines
    '''
    # If the lines are dilated (also on each side), the crop needs to be adjusted accordingly
    line_crop = init_line_crop + scribble_width - 1
    lines, tried_lines = create_lines(sk, gt_mask, lines_max_pix, line_pix_range, scribble_width, line_crop)
    avg_length_tried = get_lines_stats(tried_lines)
    added_pix = np.sum(lines)
    # If not enough lines were added, try again with adjusted parameters
    # NOTE: We check at least for 1 pixel, since otherwise we would allow for empty scribbles; on the other hand we take floor, to ensure it cannot be expected to be above the allowed maximum
    while added_pix < max(1, np.floor(lines_max_pix * lines_margin)):
        # If the line range is too small, make it larger (especially decreasing the minimum) and try again
        # NOTE: if the upper bound is still too low, this is not a big deal, because we can instead shorten the lines
        # NOTE: since the minimum (line_pix_range[0]) is an integer and is checked to be > 1, it will not be reduced to 0
        if line_pix_range[0] > 1: # or line_pix_range[1] > max(gt_mask.shape) // 2:
            line_pix_range = (line_pix_range[0]//2, line_pix_range[1] * 2)
            if print_steps:
                print("         Adjusting line range to", line_pix_range)
        # If this did not work (i.e. the lines are longer than the lines_max_pix), shorten the lines by increasing the lines crop
        elif line_crop < max(gt_mask.shape) / 2:
            # If the average pixels of the lines tried ~ (len * width) in the last run is still far from the lines_max_pix, increase the lines crop by a larger amount
            dil_pix_tried = avg_length_tried * scribble_width + (scribble_width-1) * scribble_width
            if dil_pix_tried > lines_max_pix * 5:
                pix_to_remove = int(np.ceil((dil_pix_tried - lines_max_pix) * 0.75))
                # The crop increase is dependent of the scribble width, because the lines are wider than the scribbles (plus there are overhangs on each side)
                crop_increase = int( ( pix_to_remove - (scribble_width-1) * scribble_width ) / scribble_width )
                # print(f"avg_length_tried ({avg_length_tried:.2f}) * scribble_width ({scribble_width}) > 5x lines_max_pix ({lines_max_pix:.2f}) --> crop_increase = ({avg_length_tried:.2f} - {lines_max_pix:.2f}) * 0.75 = {crop_increase}")
            # If the average pixels of the lines tried ~ (len * width) in the last run is close to the lines_max_pix, increase the lines crop by a smaller amount
            else:
                # Ensure that the steps are not becoming too large to fit inside the lines_max_pix (also considering the scribble_width)
                needed_line_len = int( ( lines_max_pix - (scribble_width-1) * scribble_width ) / scribble_width )
                # Rather make the crop steps a bit smaller than absolutely necessary
                crop_increase = int(np.ceil(0.75 * needed_line_len))
                # print(f"avg_length_tried ({avg_length_tried:.2f}) < 5x lines_max_pix ({lines_max_pix:.2f}) --> crop_increase = {crop_increase}")
            # Increase by at least 1, since otherwise nothing will be changed
            crop_increase = max(1, crop_increase)
            line_crop = line_crop + crop_increase
            if print_steps:
                print("         Adjusting line_crop to", line_crop)
        # If we are reaching the limit of what we can sample, break the loop and raise a warning (if at least some lines were added) or an error (if no lines were added)
        else:
            if added_pix == 0:
                print("   ERROR: No lines were added!")
            else:
                print(f"   WARNING: It was not possible to sample {lines_margin * 100}% of the requested pixels. Only {added_pix} pixels in lines were added!")
            break
        # Create new lines with the adjusted parameters and try again; add them to the lines
        lines_max_pix_left = lines_max_pix - added_pix
        if print_steps:
            print("         lines_max_pix_left:", lines_max_pix_left)
        new_lines, tried_lines = create_lines(sk, gt_mask, lines_max_pix_left, line_pix_range, scribble_width, line_crop=line_crop)
        avg_length_tried = get_lines_stats(tried_lines)
        lines = np.logical_or(lines, new_lines)
        added_pix = np.sum(lines)
    return lines

def get_lines_stats(line_list):
    tried_line_lengths = [np.sum(line) for line in line_list]
    tried_lines_tot_pix = np.sum(tried_line_lengths)
    tries = len(line_list)
    min_length_tried = np.min(tried_line_lengths)
    max_length_tried = np.max(tried_line_lengths)
    avg_length_tried = np.mean(tried_line_lengths)
    return avg_length_tried

def get_line(coord, gt_mask, line_crop=2):
    '''
    Take a point on the skeleton (= True in the mask) and draw a line to the nearest edge point of the ground truth mask.
    Input:
        coord (tuple): the coordinates of the starting point
        gt_mask (numpy array): the ground truth mask
        line_crop (int): the crop of the line (on both sides; i.e. distance to edge and and skeleton will each be half of the crop)
    Output:
        final_path_mask (numpy array): the mask of the shortest path, optionally cropped on both sides
    '''
    shortest_path_mask = point_to_edge(coord, gt_mask)
    # Crop the line as specified
    if line_crop:
        start, end = int(np.floor(line_crop/2)), int(- np.ceil(line_crop/2))
        coords = np.argwhere(shortest_path_mask)
        final_path_mask = np.zeros_like(shortest_path_mask)
        for coord in coords[start:end]:
            final_path_mask[coord[0], coord[1]] = True
    else:
        final_path_mask = shortest_path_mask
    return final_path_mask

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
