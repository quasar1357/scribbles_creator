import numpy as np
from skimage.morphology import *
from skimage.draw import line
from scipy.spatial import distance

def create_even_scribbles(ground_truth, max_perc=0.2, margin=0.75, rel_scribble_len=False, scribble_width=1, mode="all", class_dist="balanced", enforce_max_perc=False, print_steps=False):
    '''Generate the scribble annotation for the ground truth using an even distribution of pixels among the chosen scribble types (all, both skeletons or individual skeletons and lines).
    This function uses a default scribble_width of 1, a formula to determine the square size and a range for pixels inside a square or line of half to double one square side length.
    These parameters should be suited for max_perc values between approximately 0.05 and 1.
    Input:
        ground_truth (numpy array): the fully annotated image
        max_perc (float): the maximum percentage of pixels that should be picked (from skeletons and lines)
        margin (float): the margin for minimum nr. pixels that should be picked, as a proportion of the pixels given by max_perc (default: 0.75)
        rel_scribble_len (int/bool): length of the single scribbles relative to pixel dimensions, i.e. the number of scribbles that would fit the image (empirical default value: 20/(max_perc**0.25))
        scribble_width (int): the width of the individual scribbles
        mode (str): the scribble types to use (lines, prim_sk, sec_sk, both_sk, all)
        class_dist (str or float): the distribution of the classes in the ground truth: relative (keep % annot for each class), even (same num pix per class), or balanced (mean of relative and even); can also be given as a float between 0 and 1 for a linear interpolation between even and relative (1=even)
        enforce_max_perc (bool): whether to enforce the maximum percentage of pixels in the scribble annotation (in case it has to be surpassed, e.g. due to the scribble width)
        print_steps (bool): whether to print the steps of the scribble creation
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
    # Make sure the square is not bigger than the image
    sq_size = min(sq_size, ground_truth.shape[0], ground_truth.shape[1])
    # Make sure the square is an integer
    sq_size = int(sq_size)

    if print_steps:
        print(f"\nmax. perc.: {max_perc}, margin: {margin}, rel_scribble_len: {rel_scribble_len:.2f}, width: {scribble_width}, mode: {mode}, class_dist: {class_dist}, enforce_max_perc: {enforce_max_perc}, print_steps: {print_steps}\n")

    # Generate the scribble annotation for the ground truth
    scribbles = create_scribbles(ground_truth, scribble_width=scribble_width, sk_max_perc=max_perc_per_type, sk_margin=margin, sq_size=sq_size, sq_pix_range=False, lines_max_perc=max_perc_per_type, lines_margin=margin, line_pix_range=False, mode=mode, class_dist=class_dist, enforce_max_perc=enforce_max_perc, print_steps=print_steps)            
    # Calculate the percentage of annotated pixels and optionally print it
    pix_annot = np.sum(scribbles != 0)
    pix_tot = np.sum(ground_truth != 0)
    percent_annot = pix_annot / pix_tot * 100
    if print_steps:
        print(f"TOTAL annotation: {pix_annot} = {percent_annot:.3f}% \n")

    return scribbles

def create_scribbles(ground_truth, scribble_width=1, sk_max_perc=0.05, sk_margin=0.75, sq_size=20, sq_pix_range=False, lines_max_perc=0.05, lines_margin=0.75, line_pix_range=False, mode="all", class_dist="balanced", enforce_max_perc=False, print_steps=False):
    '''
    Generate the scribble annotation for the ground truth.
    Input:
        ground_truth (numpy array): the fully annotated image
        scribble_width (int): the width of the individual scribbles
        sk_max_perc (float): the maximum percentage of pixels of the ground truth that should be picked from each of the skeletons
        sk_margin (float): for the skeletons - the margin for minimum nr. pixels that should be picked, as a proportion of the pixels given by max_perc (default: 0.75)
        sq_size (int): the size of the squares (side length)
        sq_pix_range (int): the range that the number of pixels in a square shall be in
        lines_max_perc (float): the maximum percentage of pixels of the ground truth that should be created by drawing lines
        lines_margin (float): for the lines - the margin for minimum nr. pixels that should be picked, as a proportion of the pixels given by max_perc (default: 0.75)
        line_pix_range (int): the range that the number of pixels for a line shall be in
        mode (str): the scribble types to use (lines, prim_sk, sec_sk, both_sk, all)
        class_dist (str or float): the distribution of the classes in the ground truth: relative (keep % annot for each class), even (same num pix per class), or balanced (mean of relative and even); can also be given as a float between 0 and 1 for a linear interpolation between even and relative (1=even)
        enforce_max_perc (bool): whether to enforce the maximum percentage of pixels in the scribble annotation (in case it has to be surpassed, e.g. due to the scribble width)
        print_steps (bool): whether to print the steps of the scribble creation
    Output:
        scribble_annotation (numpy array): the scribble annotation
    '''
    scribble_annotation = np.zeros_like(ground_truth, dtype=np.uint8)
    class_values = [class_val for class_val in set(ground_truth.flatten()) if class_val != 0]
    num_classes = len(class_values)
    # For each class (= value) in the ground truth, generate the scribble annotation
    for class_val in class_values:
        img_pix = np.sum(ground_truth != 0)
        class_pix = np.sum(ground_truth == class_val)
        # Calculate the maximum percentage of pixels for the class according to the distribution chosen
        even_factor = img_pix / (num_classes * class_pix) # this is the factor to use to turn relative (same percentage) into even (same absolute number) distribution
        if class_dist == "even":
            class_sk_max_perc = sk_max_perc * even_factor
            class_lines_max_perc = lines_max_perc * even_factor
        elif class_dist == "balanced":
            class_sk_max_perc = (sk_max_perc + sk_max_perc * even_factor) / 2
            class_lines_max_perc = (lines_max_perc + lines_max_perc * even_factor) / 2
        elif class_dist == "relative":
            class_sk_max_perc = sk_max_perc
            class_lines_max_perc = lines_max_perc
        # We also allow for a linear interpolation between the even and relative distribution; this is done if class_dist is given as a float between 0 and 1
        elif 0 <= class_dist <= 1:
            class_sk_max_perc = (1 - class_dist) * sk_max_perc + class_dist * sk_max_perc * even_factor
            class_lines_max_perc = (1 - class_dist) * lines_max_perc + class_dist * lines_max_perc * even_factor
        else:
            raise ValueError(f"Invalid class distribution: {class_dist}. Choose from 'relative', 'even', 'balanced'; or pass the 'evenness' as a float between 0 and 1.")

        # Calculate the total maximum percentage of pixels for the class according to the mode chosen
        if mode == "all": tot_max_perc = 2*class_sk_max_perc + class_lines_max_perc
        elif mode == "both_sk": tot_max_perc = 2*class_sk_max_perc
        elif mode in ("prim_sk", "sec_sk"): tot_max_perc = class_sk_max_perc
        elif mode == "lines": tot_max_perc = class_lines_max_perc
        else: raise ValueError(f"Invalid mode: {mode}. Choose from 'prim_sk', 'sec_sk', 'both_sk', 'lines', 'all'.")
        if print_steps:
            print(f"CLASS {class_val}, max. pixel: {tot_max_perc:.3f}% = {int(class_pix*tot_max_perc/100)} pixels")

        # Generate the scribble annotation for the class
        # NOTE: We are not enforcing the max. pixels for all types individually, since we do it in the end for the whole class (avoiding multiple very small scribbles)
        class_scribble_annotation = scribble_class(ground_truth=ground_truth, class_val=class_val, scribble_width=scribble_width, sk_max_perc=class_sk_max_perc, sk_margin=sk_margin, sq_size=sq_size, sq_pix_range=sq_pix_range, lines_max_perc=class_lines_max_perc, lines_margin=lines_margin, line_pix_range=line_pix_range, mode=mode, enforce_max_perc=False, print_steps=print_steps)
        class_pix_tot = np.sum(ground_truth==class_val)
        class_pix_annot = np.sum(class_scribble_annotation==class_val)

        if enforce_max_perc:
            # Calculate the number of pixels that should be picked for this class
            class_max_pix = int(class_pix_tot * tot_max_perc / 100)
            # If the scribble is too large, reduce it to the maximum number of pixels
            class_pix_annot_before = class_pix_annot
            class_scribble_annotation = reduce_scribble(class_scribble_annotation, class_max_pix)
            class_pix_annot_after = np.sum(class_scribble_annotation==class_val)
            if print_steps and class_pix_annot_after != class_pix_annot_before:
                print(f"   Enforcing the max. percentage ({tot_max_perc:.3f}%) of pixels in the scribble annotation of the entire CLASS {class_val}")
                print(f"      Pixels before: {class_pix_annot_before} = {class_pix_annot_before/class_pix*100:.3f}% | new pixels: {class_pix_annot_after} = {class_pix_annot_after/class_pix*100:.3f}%")
                if class_max_pix < 1:
                    print(f"      WARNING: The theoretical maximum number of pixels for the CLASS {class_val} ({class_max_pix:.2f}) is below 1. Instead, 1 pixel was kept.")
            class_pix_annot = class_pix_annot_after

        if print_steps:
            print(f"CLASS {class_val} pixels: {class_pix_annot} = {class_pix_annot/class_pix*100:.3f}% \n")

        # Add the scribble annotation of this class to the full scribble (which is valid, because there is no overlap between the classes)
        scribble_annotation += class_scribble_annotation.astype(np.uint8)
    return scribble_annotation

def scribble_class(ground_truth, class_val, scribble_width=1, sk_max_perc=0.05, sk_margin=0.75, sq_size=20, sq_pix_range=False, lines_max_perc=0.05, lines_margin=0.75, line_pix_range=False, mode="all", enforce_max_perc=False, print_steps=False):
    '''
    Generate the scribble annotation for a specific class in the ground truth.
    Input:
        ground_truth (numpy array): the ground truth
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
    gt_class_mask = (ground_truth == class_val)
    tot_class_pix = int(np.sum(gt_class_mask))

    # Generate the primary and secondary skeleton for the class in this slice
    prim_sk, sec_sk = double_sk_class(gt_class_mask)
    # Check if a skeleton was created, raise an error if not
    if np.sum(prim_sk) == 0:
        raise ValueError(f"No skeleton was created for class {class_val}.")

    # PICK SKELETON SQUARES
    if mode in ("prim_sk", "sec_sk", "both_sk", "all"):
        # Calculate how many TOTAL pixels of each skeleton are allowed in this class given the percentage
        sk_max_pix = int(tot_class_pix * sk_max_perc / 100)
        # Store the original value for when we need it while we change the used one
        sk_max_pix_orig = sk_max_pix
        # Ensure that the TOTAL maximum number of pixels is at least scribble_width**2 (avoiding empty scribble annotations, i.e. allowing for at least a "point scribble")
        if sk_max_pix < scribble_width**2:
            print(f"   WARNING: The theoretical maximum number of pixels for the SQUARES ({sk_max_pix:.2f}) is below scribble_width**2 ({scribble_width**2}). Instead, {scribble_width**2} pixel(s) is/are sampled.")
            sk_max_pix = scribble_width**2
            sq_size = scribble_width
        # Define the range of pixels in a SINGLE square
        sq_pix_max = sq_size*2
        sq_pix_min = sq_size//2
        # Make sure the minumim cannot be 0
        sq_pix_min = max(1, sq_pix_min)
        # Adjust the range to scribble_width (if the scribble is wider, the range needs to be larger to have similar lengths of the scribbles)
        sq_pix_min, sq_pix_max = int(sq_pix_min * scribble_width), int(sq_pix_max * scribble_width)
        # Make sure that the minimum to pick is not above the total maximum allowed
        sq_pix_min = min(sq_pix_min, int(sk_max_pix))
        # Use these values if no range was specified
        sq_pix_range = (sq_pix_min, sq_pix_max) if not sq_pix_range else sq_pix_range
        if print_steps:
            print(f"   sk_max_pix: {sk_max_pix:.2f}, sq_size: {sq_size}, sq_pix_range: {sq_pix_range}")
        # If the primary skeleton is needed, pick squares of it
        if mode in ("prim_sk", "both_sk", "all"):
            prim_sk_squares = pick_sk_squares_optim(prim_sk, gt_class_mask, sk_max_pix=sk_max_pix, sk_margin=sk_margin, sq_size=sq_size, sq_pix_range=sq_pix_range, scribble_width=scribble_width, print_steps=print_steps)
            # Ensure that the scribble is within the ground truth mask
            prim_sk_squares = np.logical_and(prim_sk_squares, gt_class_mask)
            if print_steps:
                print(f"      prim_sk_squares pix: {np.sum(prim_sk_squares)} = {np.sum(prim_sk_squares)/np.sum(gt_class_mask)*100:.3f}%")
            # If enforce_max_perc option is turned on, remove pixels such that the max_perc is ensured
            if enforce_max_perc:
                prim_sk_pix_before = np.sum(prim_sk_squares)
                prim_sk_squares = reduce_scribble(prim_sk_squares, sk_max_pix_orig)
                prim_sk_pix_after = np.sum(prim_sk_squares)
                if print_steps and prim_sk_pix_after != prim_sk_pix_before:
                    print(f"      -> Enforcing the max. percentage ({sk_max_perc:.3f}%) of pixels in the PRIMARY skeleton scribbles - new nr. pix: {prim_sk_pix_after} = {prim_sk_pix_after/np.sum(gt_class_mask)*100:.3f}%")
                    if sk_max_pix_orig < 1:
                        print(f"      WARNING: The theoretical maximum number of pixels for the SKELETON SQUARES ({sk_max_pix_orig:.2f}) is below 1. Instead, 1 pixel was kept.")
        # If the secondary skeleton is needed, pick squares of it
        if mode in ("sec_sk", "both_sk", "all"):
            sec_sk_squares = pick_sk_squares_optim(sec_sk, gt_class_mask, sk_max_pix=sk_max_pix, sk_margin=sk_margin, sq_size=sq_size, sq_pix_range=sq_pix_range, scribble_width=scribble_width, print_steps=print_steps)
            # Ensure that the scribble is within the ground truth mask
            sec_sk_squares = np.logical_and(sec_sk_squares, gt_class_mask)
            if print_steps:
                print(f"      sec_sk_squares pix: {np.sum(sec_sk_squares)} = {np.sum(sec_sk_squares)/np.sum(gt_class_mask)*100:.3f}%")
            # If enforce_max_perc option is turned on, remove pixels such that the max_perc is ensured
            if enforce_max_perc:
                sec_sk_pix_before = np.sum(sec_sk_squares)
                sec_sk_squares = reduce_scribble(sec_sk_squares, sk_max_pix_orig)
                sec_sk_pix_after = np.sum(sec_sk_squares)
                if print_steps and sec_sk_pix_after != sec_sk_pix_before:
                    print(f"      -> Enforcing the max. percentage ({sk_max_perc:.3f}%) of pixels in the SECONDARY skeleton scribbles - new nr. pix: {sec_sk_pix_after} = {sec_sk_pix_after/np.sum(gt_class_mask)*100:.3f}%")
                    if sk_max_pix_orig < 1:
                        print(f"      WARNING: The theoretical maximum number of pixels for the SKELETON SQUARES ({sk_max_pix_orig:.2f}) is below 1. Instead, 1 pixel was kept.")
        # If both skeletons are needed, combine the squares of both skeletons
        if mode in ("both_sk", "all"):
            both_sk_squares = np.logical_or(prim_sk_squares, sec_sk_squares)

    # PICK LINES
    # If lines are needed, create and pick them (lines leading from the primary skeleton to the closest edge of the mask)
    if mode in ("lines", "all"):
        # Calculate how many TOTAL pixels of lines are allowed in this class given the percentage
        lines_max_pix = int(tot_class_pix * lines_max_perc / 100)
        # Store the original value in case we need it but change the actually used one
        lines_max_pix_orig = lines_max_pix
        # Ensure that the TOTAL maximum number of pixels is at least scribble_width**2 (avoiding empty scribble annotations, i.e. allowing for at least a "point scribble")
        if lines_max_pix < scribble_width**2:
            print(f"   WARNING: The theoretical maximum number of pixels for the LINES ({lines_max_pix:.2f}) is below scribble_width**2 ({scribble_width**2}). Instead, {scribble_width**2} pixel(s) is/are sampled.")
            lines_max_pix = scribble_width**2
            avg_line_len = scribble_width
        else:
            avg_line_len = sq_size
        # Define the range of pixels in a SINGLE line
        line_pix_max = avg_line_len*2
        line_pix_min = avg_line_len//2
        # Make sure the minumim cannot be 0
        line_pix_min = max(1, line_pix_min)
        # Adjust the range to the scribble_width (if the scribble is wider, the range need to be higher to have similar lengths of the scribbles)
        line_pix_min, line_pix_max = int(line_pix_min * scribble_width), int(line_pix_max * scribble_width)
        # Ensure that the line is allowed to be as short as the maximum total pixels in all lines
        line_pix_min = min(line_pix_min, int(lines_max_pix))
        # Use these values if no range was given
        line_pix_range = (line_pix_min, line_pix_max) if not line_pix_range else line_pix_range
        if print_steps:
            print(f"   lines_max_pix: {lines_max_pix:.2f}, line_pix_range: {line_pix_range}")
        lines = create_lines_optim(prim_sk, gt_class_mask, lines_max_pix, lines_margin, line_pix_range, scribble_width, print_steps=print_steps)
        # Ensure that the scribble is within the ground truth mask
        lines = np.logical_and(lines, gt_class_mask)
        if print_steps:
            print(f"      lines pix: {np.sum(lines)} = {np.sum(lines)/np.sum(gt_class_mask)*100:.3f}%")
        # If option is turned on, remove pixels such that the max_perc is ensured
        if enforce_max_perc:
            lines_pix_before = np.sum(lines)
            lines = reduce_scribble(lines, lines_max_pix_orig)
            lines_pix_after = np.sum(lines)
            if print_steps and lines_pix_after != lines_pix_before:
                print(f"      -> Enforcing the max. percentage ({lines_max_perc:.3f}%) of pixels in the LINES scribbles - new nr. pix: {lines_pix_after} = {lines_pix_after/np.sum(gt_class_mask)*100:.3f}%")
                if lines_max_pix_orig < 1:
                    print(f"      WARNING: The theoretical maximum number of pixels for the LINES ({lines_max_pix_orig:.2f}) is below 1. Instead, 1 pixel was kept.")
    if mode == "all":
        lines_and_squares = np.logical_or(lines, both_sk_squares)

    # Define the scribble type to use
    if mode == "lines": class_scribble_mask = lines
    elif mode == "prim_sk": class_scribble_mask = prim_sk_squares
    elif mode == "sec_sk": class_scribble_mask = sec_sk_squares
    elif mode == "both_sk": class_scribble_mask = both_sk_squares
    elif mode == "all": class_scribble_mask = lines_and_squares

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

def reduce_scribble(scribbles, max_pix):
    '''Reduce the number of pixels in a scribble to a maximum number of pixels.'''
    # If the maximum number of pixels is below 1, raise a warning and pick 1 pixel instead (avoiding empty scribble annotations)
    if max_pix < 1:
        max_pix = 1
    # If too many pixels are present in the scribble, raise a warning and pick the requested number of pixels
    num_pix_in_scribble = np.sum(scribbles)
    # Remove pixels if the total number of pixels exceeds the maximum
    if num_pix_in_scribble > max_pix:
        scribble_coord = np.where(scribbles)
        scribbles[scribble_coord[0][max_pix:], scribble_coord[1][max_pix:]] = 0
    return scribbles

def pick_sk_squares(sk, gt_mask, sk_max_pix=20, sq_size=20, sq_pix_range=(10,40), scribble_width=1, print_details=False):
    '''
    Pick random squares from the skeleton.
    Input:
        sk (numpy array): the skeleton
        gt_mask (numpy array): the ground truth mask
        sk_max_pix (int): the approximate number of pixels that should be picked
        sq_size (int): the size of the squares (side length)
        sq_pix_range (int): the range that the number of pixels in a square shall be in
        scribble_width (int): the width of the individual scribbles
        print_details (bool): whether to print the details of the function
    Output:
        all_squares (numpy array): the mask of all squares
    '''
    pix_in_sk = np.sum(sk)
    idx_step = 1
    # Get the coordinates of the skeleton, using steps of idx_step
    sk_coordinates = np.argwhere(sk)[::idx_step]
    num_coords = len(sk_coordinates)
    # Shuffle the coordinates of the skeleton to loop over them in a random order
    np.random.shuffle(sk_coordinates)
    # Create a dilated version of the skeleton to pick the squares from
    sk_dilated = binary_dilation(sk, square(scribble_width))
    # Ensure all pixels of the dilated skeleton are within the mask
    sk_dilated = np.logical_and(sk_dilated, gt_mask)
    # Initialize the mask of all squares and variables for the loop
    all_squares = np.zeros_like(sk_dilated, dtype=np.bool8)
    added_pix = 0
    idx = 0
    overshoots = 0
    if print_details:
        print("--- Sampling squares: pix_in_sk", pix_in_sk, "indx_step", idx_step, "num_coords", num_coords)
    # Loop until the total number of pixels in all squares approaches the threshold or the end of all pixels in the skeleton is reached
    while overshoots < 100 and idx < num_coords:
        # Pick a random square from the skeleton (note that the coordinates were shuffled above)
        current_coordinate = sk_coordinates[idx]
        idx += 1
        sk_square = get_square(sk_dilated, current_coordinate, sq_size)
        pix_in_sq = np.sum(sk_square)
        future_all_squares = np.logical_or(all_squares, sk_square)
        future_total = np.sum(future_all_squares)
        if print_details:
            print(f"---    current_coordinate: {current_coordinate} | added_pix (before): {added_pix}, pix_in_square: {pix_in_sq}, new pix: {future_total-added_pix}, future_total: {future_total}")
        # If the square would push the total number of pixels in all squares above the maximum, skip it and count the overshoot
        if future_total > sk_max_pix:
            if print_details:
                print("---    overshoot nr.", overshoots)
            overshoots += 1
            continue
        # If there are too few or too many pixels in the square, skip it (without overshoot)
        elif pix_in_sq < sq_pix_range[0] or pix_in_sq > sq_pix_range[1]:
            if print_details:
                print("---    outside sq_pix_range", sq_pix_range)
            continue
        # If the square is valid, add it to the mask of all squares and update the total number of pixels
        else:
            all_squares = future_all_squares
            added_pix = future_total
        # If we have reached the goal maximum number of pixels, we can stop the loop, since no improvement is possible
        if added_pix == int(sk_max_pix):
            if print_details:
                print("---    sk_max_pix reached (no improvement possible)")
            break
        # If the remaining total pixels are too few for a square, we can also stop the loop
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
        sk (numpy array): the skeleton to sample squares from
        sk_max_pix (int): the approximate number of pixels that should be picked
        sk_margin (float): the margin for minimum nr. pixels that should be picked, as a proportion of the pixels given by max_perc (default: 0.75)
        sq_size (int): the size of the squares (side length)
        sq_pix_range (int): the range that the number of pixels in a square shall be in
        scribble_width (int): the width of the individual scribbles
        print_steps (bool): whether to print the steps of the function
    Output:
        squares (numpy array): the mask of all squares
    '''
    # For checking how many pixels can be sampled max.: Create a dilated version of the skeleton (same as the one that the squares are picked from)
    sk_dilated = binary_dilation(sk, square(scribble_width))
    # Ensure all pixels of the dilated skeleton are within the mask
    sk_dilated = np.logical_and(sk_dilated, gt_mask)
    num_pix_dil = np.sum(sk_dilated)
    # If there are too little pixels in the dilated skeleton, raise a warning and return the entire skeleton
    if num_pix_dil <= sk_max_pix:
        print(f"      NOTE: All pixels in the skeleton were added ({num_pix_dil}).")
        squares = sk_dilated
        added_pix = np.sum(squares)
    else:
        # Sample squares from the skeleton
        squares = pick_sk_squares(sk, gt_mask, sk_max_pix, sq_size, sq_pix_range, scribble_width)
        added_pix = np.sum(squares)
    # If not enough squares were added, try again with smaller squares and a range starting at a lower value (allowing fewer pixels in a square)
    # NOTE: We check at least for 1 pixel, since otherwise we would allow for empty scribbles; on the other hand we take floor, to ensure it cannot be expected to be above the allowed maximum
    while np.all((added_pix < max(1, np.floor(sk_max_pix * sk_margin)), # Stop if the required minimum number of pixels was added
                  sq_size > scribble_width, # Do not reduce the square size below scribble_width
                  added_pix < np.sum(sk_dilated))): # Skip if all pixels in the dilated skeleton were added
        # Reduce the square size
        diff_to_width = sq_size - scribble_width
        sq_size = scribble_width + diff_to_width//2
        # Adjust the range accordingly
        # Make sure that the minimum to pick is not above the total maximum allowed
        sq_pix_min = min(sq_pix_range[0]//2, int(sk_max_pix))
        # Make sure the minumim cannot be 0
        sq_pix_min = max(1, sq_pix_min)
        sq_pix_range = (sq_pix_min, sq_pix_range[1])
        if print_steps:
            print("         Adjusting square size and range to", sq_size, sq_pix_range)
        # Sample new squares with the adjusted parameters and try again; add them to the squares
        sk_max_pix_left = sk_max_pix - added_pix
        # We only need to draw squares around the pixels that are still missing
        sk_left = np.logical_and(sk, np.logical_not(squares))
        if print_steps:
            print(f"         Sampling skeleton squares - sk_max_pix_left: {sk_max_pix_left}")
        new_squares = pick_sk_squares(sk_left, gt_mask, sk_max_pix_left, sq_size, sq_pix_range, scribble_width)
        squares = np.logical_or(squares, new_squares)
        added_pix = np.sum(squares)
    # If we have finished looping because we reached the limit of what we can sample, raise a warning if too little squares were added or an error if no squares were added
    if added_pix == 0:
        print("      ERROR: No squares were added!")
    elif added_pix < max(1, np.floor(sk_max_pix * sk_margin)):
        print(f"      WARNING: It was not possible to sample {sk_margin * 100}% of the requested pixels. Only {added_pix} pixels in squares were added!")
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
    red = int(np.ceil(sq_size/2))
    # Ensure that the square does not exceed the mask
    red = [min(red, coord[0]), min(red, coord[1])]
    inc = int(np.floor(sq_size/2)) # Here, the index can exceed the mask because slicing will stop at the end of the mask
    square_mask[coord[0]-red[0]:coord[0]+inc, coord[1]-red[1]:coord[1]+inc] = mask[coord[0]-red[0]:coord[0]+inc, coord[1]-red[1]:coord[1]+inc]
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
        print_details (bool): whether to print the details of the function
    Output:
        all_lines (numpy array): the mask of all lines
    '''
    pix_in_sk = np.sum(sk)
    # Make sure the lines are not sampled too densely
    idx_step = scribble_width + 2
    # Make sure to take at least a certain amount of positions on the skeleton (to avoid too few lines)
    idx_step = min(idx_step, scribble_width + int(np.ceil(pix_in_sk/250))) # 1
    # Get the coordinates of the skeleton, using steps of idx_step
    sk_coordinates = np.argwhere(sk)[::idx_step]
    num_coords = len(sk_coordinates)
    # Shuffle the coordinates of the skeleton to loop over them in a random order
    np.random.shuffle(sk_coordinates)
    # Initialize the mask of all picked lines and variables for the loop
    all_lines = np.zeros_like(gt_mask, dtype=np.bool8)
    added_pix = 0
    idx = 0
    overshoots = 0
    tried_lines = []
    if print_details:
        print("--- Sampling lines: pix_in_sk", pix_in_sk, "indx_step", idx_step, "num_coords", num_coords, "line_crop", line_crop)
    # Loop until the pixels in all lines approach the threshold (keeps overshooting) or the end of all pixels in the skeleton is reached
    while overshoots < 100 and idx < num_coords:
        # Draw a line from the skeleton to the edge of the mask
        current_coordinate = sk_coordinates[idx]
        idx += 1
        single_pix_line = get_line(current_coordinate, gt_mask, line_crop=line_crop)
        tried_lines.append(single_pix_line)
        # Dilate the line to the scribble width
        line = binary_dilation(single_pix_line, square(scribble_width))
        # Ensure all pixels of the (dilated )line are within the mask
        line = np.logical_and(line, gt_mask)
        pix_in_line = np.sum(line)
        future_all_lines = np.logical_or(all_lines, line)
        future_total = np.sum(future_all_lines)
        if print_details:
            print(f"---    current_coordinate: {current_coordinate}, line {(np.argwhere(single_pix_line)[0], np.argwhere(single_pix_line)[-1]) if np.sum(single_pix_line) > 0 else 'empty'} | added_pix (before): {added_pix}, pix_in_line: {pix_in_line}, new pix: {future_total-added_pix}, future_total: {future_total}")
        # If the line would push the total number of pixels on lines above the maximum, skip it and count the overshoot
        if future_total > lines_max_pix:
            if print_details:
                print("---    overshoot nr.", overshoots)
            overshoots += 1
            continue
        # If the line is too short or too long, skip it (without overshoot)
        elif pix_in_line < line_pix_range[0] or pix_in_line > line_pix_range[1]:
            if print_details:
                print("---    outside line_pix_range", line_pix_range)
            continue
        # If the line is valid, add it to the mask of all lines and update the total number of pixels
        else:
            all_lines = future_all_lines
            added_pix = future_total
        # If we have reached the goal maximum number of pixels, we can stop the loop, since no improvement is possible
        if added_pix == int(lines_max_pix):
            if print_details:
                print("---    lines_max_pix reached (no improvement possible)")
            break
        # If the remaining total pixels are too few for a line, we can also stop the loop
        if lines_max_pix - added_pix < line_pix_range[0]:
            if print_details:
                print("---    the remaining total pixels are too few for a line")
            break
    if print_details:
        avg_length_tried = get_lines_stats(tried_lines)[0]
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
        print_steps (bool): whether to print the steps of the function
    Output:
        lines (numpy array): the mask of all lines
    '''
    line_crop = init_line_crop
    lines, tried_lines = create_lines(sk, gt_mask, lines_max_pix, line_pix_range, scribble_width, line_crop)
    avg_length_tried = get_lines_stats(tried_lines)[0]
    added_pix = np.sum(lines)
    lines_max_pix_left = lines_max_pix - added_pix

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
        # If this did not work (which generally means the lines are longer than the lines_max_pix), shorten the lines by increasing the lines crop
        elif (line_crop < max(gt_mask.shape) / 2) and (avg_length_tried > 0):
            # If the average pixels of the lines tried ~ (len * width) in the last run is still far from the lines_max_pix_left, increase the lines crop by a larger amount
            dil_pix_tried = avg_length_tried * scribble_width #avg_length_tried * scribble_width + (scribble_width-1) * scribble_width
            if dil_pix_tried > lines_max_pix_left * 5:
                pix_to_remove = int(np.ceil((dil_pix_tried - lines_max_pix_left) * 0.75))
                # The crop increase has to be adjusted dependent of the scribble width, since the crop refers to the single pix line (plus there are overhangs on each side)
                crop_increase = int(pix_to_remove / scribble_width) #int( ( pix_to_remove - (scribble_width-1) * scribble_width ) / scribble_width )
            # If the average pixels of the lines tried ~ (len * width) in the last run is close to the lines_max_pix_left, increase the lines crop by a smaller amount
            else:
                # Ensure that the steps are not becoming too large to fit inside the lines_max_pix_left (also considering the scribble_width)
                needed_line_len = int(lines_max_pix_left / scribble_width) #int( ( lines_max_pix_left - (scribble_width-1) * scribble_width ) / scribble_width )
                crop_increase = avg_length_tried - needed_line_len
                # Rather make the crop steps a bit smaller than absolutely necessary
                crop_increase = int(np.ceil(0.75 * crop_increase))
            # Increase by at least 1, since otherwise nothing will be changed
            crop_increase = max(1, crop_increase)
            line_crop = line_crop + crop_increase
            if print_steps:
                print("         Adjusting line_crop to", line_crop)
        else:
            print("      NOTE: No more options to adjust the parameters.")
            break
        # Create new lines with the adjusted parameters and try again; add them to the lines
        lines_max_pix_left = lines_max_pix - added_pix
        if print_steps:
            print(f"         Sampling lines - lines_max_pix_left: {lines_max_pix_left}")
        new_lines, tried_lines = create_lines(sk, gt_mask, lines_max_pix_left, line_pix_range, scribble_width, line_crop=line_crop)
        avg_length_tried = get_lines_stats(tried_lines)[0]
        lines = np.logical_or(lines, new_lines)
        added_pix = np.sum(lines)

    # If we have finished looping because we reached the limit of what we can sample, raise a warning if too little lines were added or an error if no lines were added
    if added_pix == 0:
        print("      ERROR: No lines were added!")
    elif added_pix < max(1, np.floor(lines_max_pix * lines_margin)):
        print(f"      WARNING: It was not possible to sample {lines_margin * 100}% of the requested pixels. Only {added_pix} pixels in lines were added!")

    return lines

def get_lines_stats(line_list):
    '''Take a list of lines (masks) and return statistics of them.'''
    line_lengths = [np.sum(line) for line in line_list]
    lines_tot_pix = np.sum(line_lengths)
    num_lines = len(line_list)
    min_length = np.min(line_lengths)
    max_length = np.max(line_lengths)
    avg_length = np.mean(line_lengths)
    return avg_length, min_length, max_length, lines_tot_pix, num_lines

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
