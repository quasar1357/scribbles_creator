# -*- coding: utf-8 -*-
# Author: Shuojue Yang (main contribution) and Xiangde Luo (minor modification for WORD and other datasets).
# Date:   16 Dec. 2021
# Implementation for simulation of the sparse scribble annotation based on the dense annotation for the WORD dataset and other datasets.
# # Reference:
# @article{luo2022scribbleseg,
# title={Scribble-Supervised Medical Image Segmentation via Dual-Branch Network and Dynamically Mixed Pseudo Labels Supervision},
# author={Xiangde Luo, Minhao Hu, Wenjun Liao, Shuwei Zhai, Tao Song, Guotai Wang, Shaoting Zhang},
# journal={Medical Image Computing and Computer Assisted Intervention -- MICCAI 2022},
# year={2022},
# pages={528--538}}

# @article{luo2022word,
# title={{WORD}: A large scale dataset, benchmark and clinical applicable study for abdominal organ segmentation from CT image},
# author={Xiangde Luo, Wenjun Liao, Jianghong Xiao, Jieneng Chen, Tao Song, Xiaofan Zhang, Kang Li, Dimitris N. Metaxas, Guotai Wang, and Shaoting Zhang},
# journal={Medical Image Analysis},
# volume={82},
# pages={102642},
# year={2022},
# publisher={Elsevier}}

# @misc{wsl4mis2020,
# title={{WSL4MIS}},
# author={Luo, Xiangde},
# howpublished={\url{https://github.com/Luoxd1996/WSL4MIS}},
# year={2021}}
# If you have any questions, please contact Xiangde Luo (https://luoxd1996.github.io).


import glob
import math
import random
import sys

import cv2
import numpy as np
import SimpleITK as sitk
from PIL import Image
from scipy import ndimage
from skimage.morphology import *

sys.setrecursionlimit(1000000)
seed = 2022
np.random.seed(seed)
random.seed(seed)


def random_rotation(image, max_angle=15):
    '''Rotate the image with a random angle between -max_angle and max_angle.'''
    angle = np.random.uniform(-max_angle, max_angle)
    img = Image.fromarray(image)
    img_rotate = img.rotate(angle)
    return img_rotate


def translate_img(img, x_shift, y_shift):
    '''Translate the image with x_shift and y_shift.'''
    (height, width) = img.shape[:2]
    matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    trans_img = cv2.warpAffine(img, matrix, (width, height))
    return trans_img


def get_largest_two_component_2D(img, print_info=False, threshold=None):
    """
    Get the largest two components of a binary volume
    inputs:
        img: the input 2D volume
        threshold: a size threshold
    outputs:
        out_img: the output volume 
    """
    s = ndimage.generate_binary_structure(2, 2)  # iterate structure
    labeled_array, numpatches = ndimage.label(img, s)  # labeling
    sizes = ndimage.sum(img, labeled_array, range(1, numpatches+1))
    sizes_list = [sizes[i] for i in range(len(sizes))]
    sizes_list.sort()
    if(print_info):
        print('component size', sizes_list)
    if(len(sizes) == 1):
        out_img = [img]
    else:
        if(threshold):
            max_size1 = sizes_list[-1]
            max_label1 = np.where(sizes == max_size1)[0] + 1
            if max_label1.shape[0] > 1:
                max_label1 = max_label1[0]
            component1 = labeled_array == max_label1
            out_img = [component1]
            for temp_size in sizes_list:
                if(temp_size > threshold):
                    temp_lab = np.where(sizes == temp_size)[0] + 1
                    temp_cmp = labeled_array == temp_lab[0]
                    out_img.append(temp_cmp)
            return out_img
        else:
            max_size1 = sizes_list[-1]
            max_size2 = sizes_list[-2]
            max_label1 = np.where(sizes == max_size1)[0] + 1
            max_label2 = np.where(sizes == max_size2)[0] + 1
            if max_label1.shape[0] > 1:
                max_label1 = max_label1[0]
            if max_label2.shape[0] > 1:
                max_label2 = max_label2[0]
            component1 = labeled_array == max_label1
            component2 = labeled_array == max_label2
            if(max_size2*10 > max_size1):
                out_img = [component1, component2]
            else:
                out_img = [component1]
    return out_img


class Cutting_branch(object):
    '''Cutting the branches of the skeleton map to simulate the scribble annotation.'''
    def __init__(self):
        '''Initialization of the class
        lst_bifur_pt: the last bifurcation point
        branch_state: the state of the branch
        lst_branch_state: the last state of the branch
        direction2delta: the direction to the delta'''
        self.lst_bifur_pt = 0
        self.branch_state = 0
        self.lst_branch_state = 0
        self.direction2delta = {0: [-1, -1], 1: [-1, 0], 2: [-1, 1], 3: [
            0, -1], 4: [0, 0], 5: [0, 1], 6: [1, -1], 7: [1, 0], 8: [1, 1]}

    def __find_start(self, lab):
        '''Find the start point of the skeleton map.
        lab: the skeleton map
        start: the start point of the skeleton map
        directions: the directions of the skeleton map
        d: the direction of the skeleton map
        idxes: the indexes of the skeleton map
        pt: the point of the skeleton map        '''
        y, x = lab.shape
        idxes = np.asarray(np.nonzero(lab))
        for i in range(idxes.shape[1]):
            pt = tuple([idxes[0, i], idxes[1, i]])
            assert lab[pt] == 1
            directions = []
            for d in range(9):
                if d == 4:
                    continue
                if self.__detect_pt_bifur_state(lab, pt, d):
                    directions.append(d)
            if len(directions) == 1:
                start = pt
                self.start = start
                self.output[start] = 1
                return start
        start = tuple([idxes[0, 0], idxes[1, 0]])
        self.output[start] = 1
        self.start = start
        return start

    def __detect_pt_bifur_state(self, lab, pt, direction):
        '''Detect the bifurcation state of the point.'''
        d = direction
        y = pt[0] + self.direction2delta[d][0]
        x = pt[1] + self.direction2delta[d][1]
        if lab[y, x] > 0:
            return True
        else:
            return False

    def __detect_neighbor_bifur_state(self, lab, pt):
        '''Detect the bifurcation state of the neighbor.
        direction: the direction of the skeleton map
        next_pt: the next point of the skeleton map
        lst_output: the last output of the skeleton map
        previous_bifurPts: the previous bifurcation points
        end: the end point of the skeleton map'''
        directions = []
        for i in range(9):
            if i == 4:
                continue
            if self.output[tuple([pt[0] + self.direction2delta[i][0], pt[1] + self.direction2delta[i][1]])] > 0:
                continue
            if self.__detect_pt_bifur_state(lab, pt, i):
                directions.append(i)

        if len(directions) == 0:
            self.end = pt
            return False
        else:
            direction = random.sample(directions, 1)[0]
            next_pt = tuple([pt[0] + self.direction2delta[direction]
                            [0], pt[1] + self.direction2delta[direction][1]])
            if len(directions) > 1 and pt != self.start:
                self.lst_output = self.output*1
                self.previous_bifurPts.append(pt)
            self.output[next_pt] = 1
            pt = next_pt
            self.__detect_neighbor_bifur_state(lab, pt)

    def __detect_loop_branch(self, end):
        '''Detect the loop branch of the skeleton map.
        d: the direction of the skeleton map
        y: the y coordinate of the skeleton map
        x: the x coordinate of the skeleton map
        end: the end point of the skeleton map
        previous_bifurPts: the previous bifurcation points
        output: the output of the skeleton map
        lst_output: the last output of the skeleton map'''
        for d in range(9):
            if d == 4:
                continue
            y = end[0] + self.direction2delta[d][0]
            x = end[1] + self.direction2delta[d][1]
            if (y, x) in self.previous_bifurPts:
                self.output = self.lst_output * 1
                return True

    def __call__(self, lab, seg_lab, iterations=1):
        '''Cut the branches of the skeleton map.
        lab: the skeleton map
        seg_lab: the segmentation label
        iterations: the iterations of the cutting
        output: the output of the skeleton map
        shift_y: the shift of the y coordinate
        shift_x: the shift of the x coordinate
        output: the output of the skeleton map'''
        self.previous_bifurPts = []
        self.output = np.zeros_like(lab)
        self.lst_output = np.zeros_like(lab)
        components = get_largest_two_component_2D(lab, threshold=15)
        if len(components) > 1:
            for c in components:
                start = self.__find_start(c)
                self.__detect_neighbor_bifur_state(c, start)
        else:
            c = components[0]
            start = self.__find_start(c)
            self.__detect_neighbor_bifur_state(c, start)
        self.__detect_loop_branch(self.end)
        struct = ndimage.generate_binary_structure(2, 2)
        output = ndimage.morphology.binary_dilation(
            self.output, structure=struct, iterations=iterations)
        shift_y = random.randint(-6, 6)
        shift_x = random.randint(-6, 6)
        if np.sum(seg_lab) > 1000:
            output = translate_img(output.astype(np.uint8), shift_x, shift_y)
            output = random_rotation(output)
        output = output * seg_lab
        return output


def scrible_2d(label, iteration=[4, 10]):
    '''Generate the scribble annotation for the 2D segmentation.
    label: the dense annotation
    iteration: the iterations of the cutting
    lab: the label of the dense annotation
    skeleton_map: the skeleton map of the dense annotation
    sk_slice: the skeleton map of the slice
    slic: the slice of the dense annotation
    struct: the structure of the dense annotation'''
    lab = label
    # Initialize the skeleton map with zeros
    skeleton_map = np.zeros_like(lab, dtype=np.int32)
    # For each slice of the dense annotation, generate the skeleton map
    for i in range(lab.shape[0]):
        # If there is no annotation in the slice, then skip the slice
        if np.sum(lab[i]) == 0:
            continue
        # Create a binary structuring element for the erosion
        struct = ndimage.generate_binary_structure(2, 2)



        if not (isinstance(iteration, int) or len(iteration) == 2):
            raise ValueError("iteration should be an integer or a list of two integers.")
        iteration = iteration if isinstance(iteration, list) else [0, iteration]
        # If the sum of the annotation in the slice is greater than 900 and the iteration is not 0, then erode the annotation
        if np.sum(lab[i]) > 900 and iteration != 0 and iteration != [0] and iteration != None:
            iter_num = math.ceil(
                iteration[0]+random.random() * (iteration[1]-iteration[0]))
            slic = ndimage.morphology.binary_erosion(
                lab[i], structure=struct, iterations=iter_num)
        else:
            slic = lab[i]
        sk_slice = skeletonize(slic, method='lee')
        sk_slice = np.asarray((sk_slice == 255), dtype=np.int32)
        skeleton_map[i] = sk_slice



    return skeleton_map


def scribble4class(label, class_id, iteration=[4, 10], cut_branch=True):
    '''Generate the scribble annotation for the segmentation class.
    label: the dense annotation
    class_id: the class id
    iteration: the iterations of the cutting
    cut_branch: the flag of the cutting
    sk_map: the skeleton map of the dense annotation
    lab: the label of the dense annotation'''
    # Generate a boolean map for the dense annotation matching the class_id
    label = (label == class_id)
    # Generate the scribble annotation for the class
    sk_map = scrible_2d(label, iteration=iteration)
    # If the flag of the cutting is True and the class_id is not 0, then cut the branches of the skeleton map
    if cut_branch and class_id != 0:
        cut = Cutting_branch()
        for i in range(sk_map.shape[0]):
            lab = sk_map[i]
            if lab.sum() < 1:
                continue
            sk_map[i] = cut(lab, seg_lab=label[i])
    # If the class_id is 0, then set the class_id to one above the highest class_id, i.e. a new class --> output contains no classes of value 0
    # if class_id == 0:
    #     class_id = class_num
    # Return the scribble annotation for the class, i.e. the skeleton map of the dense annotation multiplied by the class_id (--> initial class value)
    return sk_map * class_id


def generate_scribble(label, iterations, cut_branch=True):
    '''Generate the scribble annotation for the dense annotation.
    label: the dense annotation
    iterations: the iterations of the cutting
    cut_branch: the flag of the cutting
    class_num: the total number of classes'''
    class_num = np.max(label) + 1
    print(f"Class number: {class_num}", f"Label shape: {label.shape}", f"Iterations: {iterations}", sep='\n')
    output = np.zeros_like(label, dtype=np.uint8)
    # For each class (= value) in the dense annotation, generate the scribble annotation
    for i in range(class_num):
        # If iterations is a list, then the iterations for each class is different, otherwise the iterations for all classes is the same
        it = iterations[i] if (isinstance(iterations, list) and len(iterations) == class_num) else iterations
        # Generate the scribble annotation for the class
        scribble = scribble4class(label, i, class_num, it, cut_branch=cut_branch)
        # Add the scribble annotation to the output (which is valid, because there is no overlap between the classes)
        output += scribble.astype(np.uint8)
    return output


##################### ROMAN's CODE #####################

def generate_scribble_2(ground_truth, num_lines=5):
    '''
    Generate the scribble annotation for the dense annotation.
    Input:
        ground_truth (numpy array): the fully annotated image
        num_lines (int): the num_lines to be drawn
    Output:
        output (numpy array): the scribble annotation        
    '''
    scribble = np.zeros_like(ground_truth, dtype=np.uint8)
    # For each class (= value) in the dense annotation, generate the scribble annotation
    for class_val in set(ground_truth.flatten()):
        if class_val == 0:
            continue
        # Generate the scribble annotation for the class
        class_scribble = scribble_class(ground_truth, class_val, num_lines)
        # Add the scribble annotation of this class to the full scribble (which is valid, because there is no overlap between the classes)
        scribble += class_scribble.astype(np.uint8)
    return scribble

def scribble_class(gt, class_val, num_lines):
    # Generate a boolean map for the dense annotation matching the class_id
    gt_class_map = (gt == class_val)
    # Initialize the skeleton map with zeros
    class_scribble = np.zeros_like(gt, dtype=np.int32)
    # For each slice of the dense annotation, generate the scribble annotation
    for i in range(gt_class_map.shape[0]):
        # If there is no annotation in the slice, then skip the slice
        if np.sum(gt_class_map[i]) == 0:
            continue
        gt_slice = gt_class_map[i]
        sk_2d = double_sk_class_slice(gt_slice)
        # Check if a skeleton was created, raise an error if not
        if np.sum(sk_2d) == 0:
            raise ValueError(f"No skeleton was created for class {class_val} in slice {i}.")
        lines = create_lines(sk_2d, gt_slice, num_lines)
        class_scribble[i] = lines * class_val
    return class_scribble


def sk_class_slice(gt_map_2d, num_lines):
    sk_slice = skeletonize(gt_map_2d, method='lee')
    sk_slice = dilation(sk_slice, square(3))
    # sk_slice = medial_axis(gt_map_2d)
    sk_slice = np.asarray((sk_slice == 255), dtype=np.int32)
    return sk_slice

import napari

def double_sk_class_slice(gt_map_2d):

    first_sk = skeletonize(gt_map_2d, method='lee')
    first_sk = np.asarray((first_sk == 255), dtype=np.int32)
    first_sk = dilation(first_sk, square(3))
    first_sk = binary_closing(first_sk, square(30))

    mask = first_sk == 1
    gt_map_2d[mask] = False    

    second_sk = skeletonize(gt_map_2d, method='lee')
    second_sk = np.asarray((second_sk == 255), dtype=np.int32)
    # second_sk = binary_closing(second_sk, square(3))
    second_sk = dilation(second_sk, square(3))

    if False:
        v = napari.Viewer()
        v.add_labels(first_sk)
        v.add_labels(gt_map_2d)
        v.add_labels(second_sk)

    return second_sk

def create_lines(sk_2d, gt_map_2d, num_lines):
    all_lines = np.zeros_like(gt_map_2d, dtype=np.int32)
    for i in range(num_lines):
        line = draw_line(sk_2d, gt_map_2d)
        line_map = np.asarray((line), dtype=np.int32)
        all_lines += line_map
    return all_lines

def draw_line(sk_2d, gt_map_2d):
    # Take a random point on the skeleton and draw a line to the nearest edge point
    line_coordinates = np.array(np.where(sk_2d == 1)).T
    random_point_index = np.random.randint(0, len(line_coordinates))
    random_point = line_coordinates[random_point_index]
    shortest_path = find_shortest_path(gt_map_2d, random_point)
    shortest_path_map = shortest_path == 1
    return shortest_path_map

from scipy.spatial import distance
from skimage.draw import line

def find_shortest_path(segmentation_map, start_point):
    # Find the coordinates of the edges of the segmentation map
    edge_coordinates = np.array(np.where(segmentation_map == False)).T
    # Compute distances from the start point to all edge points
    distances = distance.cdist([start_point], edge_coordinates)
    # Find the index of the closest edge point
    closest_edge_index = np.argmin(distances)
    # Retrieve the coordinates of the closest edge point
    closest_edge_point = edge_coordinates[closest_edge_index]
    # Use OpenCV's line function to draw a line from start_point to closest_edge_point
    # path_map = cv2.line(np.zeros_like(segmentation_map), tuple(start_point[::-1]), tuple(closest_edge_point[::-1]), color=255, thickness=3)

    # Create an empty image to draw the line on
    path_map = np.zeros_like(segmentation_map)
    # Draw the line on the image
    rr, cc = line(start_point[0], start_point[1], closest_edge_point[0], closest_edge_point[1])
    path_map[rr, cc] = 1
    return path_map