import pickle
import os, sys, math
import uuid, pdb
import cv2, openslide
import xml.etree.ElementTree as ET

import h5py, time, copy
from numba import jit

import numpy as np
import scipy.sparse
import scipy.ndimage as ndi

from scipy.io import loadmat
from ..proj_utils.local_utils import RGB2GREY, getfilelist, getfolderlist

# from .papsmear import *

from multiprocessing import Pool
#from .voc_eval import voc_eval
# from utils.yolo import preprocess_train
debug_mode = False
fill_val = np.pi * 1e-8



def im_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

@jit
def overlay_bbox(img, bbox,linewidth=1, labels=None, cls_color=None):
    labels = np.zeros((len(bbox))) if labels is None else labels
    cls_color = {0:[255,0,0]}  if cls_color is None else cls_color

    for bb, this_label in zip(bbox, labels):
        this_color = cls_color[this_label]
        x_min_, y_min_, x_max_, y_max_ = bb
        x_min_, y_min_, x_max_, y_max_ = int(x_min_),int( y_min_), int(x_max_), int(y_max_)
        img[:,:,0] = change_val(img[:,:,0], this_color[0], linewidth, x_min_, y_min_, x_max_, y_max_)
        img[:,:,1] = change_val(img[:,:,1], this_color[1], linewidth,  x_min_, y_min_, x_max_, y_max_)
        img[:,:,2] = change_val(img[:,:,2], this_color[2], linewidth,  x_min_, y_min_, x_max_, y_max_)
    return img


def get_anchor(img_shape, board_ratio = 0.5, use_random=True):
    half_ratio = board_ratio/2
    row_size, col_size = img_shape
    row_shift_range, col_shift_range = int(row_size*half_ratio), int(col_size*half_ratio)

    if use_random is True :
        if random.random() > 0.2:
            r_min = random.randint(row_shift_range//4, 2*row_shift_range//3 )
            c_min = random.randint(col_shift_range//4, 2*col_shift_range//3 )

            r_max   = random.randint(row_size - 2*row_shift_range//3 - 1,
                                    row_size - 1 - row_shift_range//4)
            c_max   = random.randint(col_size - 2*col_shift_range//3 - 1,
                                    col_size - 1- col_shift_range //4)
        else:
            r_min = random.randint(row_shift_range//2, 2*row_shift_range//3 )
            c_min = random.randint(col_shift_range//2, 2*col_shift_range//3 )

            r_max   = random.randint(row_size - 2*row_shift_range//3 - 1,
                                    row_size - 1 - row_shift_range//2)
            c_max   = random.randint(col_size - 2*col_shift_range//3 - 1,
                                    col_size - 1- col_shift_range //2)

    else:
        r_min = row_shift_range//2
        c_min = col_shift_range//2

        r_max = row_size - row_shift_range//2 - 1
        c_max = col_size - col_shift_range//2 - 1

    r_min, c_min = max(0, r_min), max(0, c_min)
    r_max, c_max = min(row_size, r_max), min(col_size, c_max)

    return r_min, c_min, r_max, c_max

def _get_next(inputs):
    img_data, img_cls, img_shape = inputs
    angle = np.random.randint(0, 359)
    # rotate image here
    if random.random() > 0.8:
        img_data = scipy.ndimage.rotate(img_data, angle= angle, mode='reflect')
    elif random.random() > 0.3:
        angle = random.choice( [90, 180, 270])
        img_data = scipy.ndimage.rotate(img_data, angle= angle, mode='reflect')

    org_shape = img_data.shape[0:2]
    r_min, c_min, r_max, c_max = get_anchor(org_shape, board_ratio=0.5)

    this_patch = img_data[r_min:r_max, c_min:c_max, :]
    this_patch = imresize_shape(this_patch, img_shape)

    this_patch = this_patch.transpose(2, 0, 1)

    return [this_patch], [img_cls], img_data

def aggregate_label(label_list):
    '''
       label_list:
    '''
    num_cell = len(label_list)
    label_dict = {}
    unique_lab = np.unique(label_list)
    for this_key in unique_lab:
        label_dict[this_key] = []
    for idx, this_label in  enumerate(label_list):
        label_dict[this_label].append(idx)
    return label_dict


def get_all_imgs(rootFolder, inputext = ['.png', '.tif'],
                 class_map_dict=None, pre_load=True, use_grey=False):
    '''
    Given a root folder, this function needs to return 2 lists. imglist and clslist:
        (img_data, label)
    '''
    sub_folder_list, sub_folder_name = getfolderlist(rootFolder)
    img_list, label_list = [], []
    #print(sub_folder_name, '\n')

    for idx, (this_folder, this_folder_name) in enumerate( zip(sub_folder_list, sub_folder_name) ):
        filelist, filenames = getfilelist(this_folder, inputext, with_ext=False)
        this_cls = class_map_dict[this_folder_name]
        #print(filenames)
        for this_file_path, this_file_name in zip(filelist, filenames):
            if pre_load is True:
                this_img = scipy.misc.imread(this_file_path)[:,:,0:3]
                #this_tuple = (this_img, this_cls, idx)
                if use_grey:
                    this_img = RGB2GREY(this_img)[:,:,None].astype(np.uint8)
                img_list.append(this_img)
            else:
                img_list.append(this_file_path)
            label_list.append(this_cls)
    return img_list, label_list


def get_indices_balance(label_dict, num_sampling, batch_size, class_ratio_array):
    '''
    Parameters:
    -----------
        label_dict:   a dictionary of cls:idx
        num_sampling: the number of chosen class in each run
        batch_size:   totoal number of samples in each run
        class_ratio_array: class_ratio_array[8] store prob of class_reverse_label[8]
    Return:
    -----------
        indices of the sample id
    '''
    indices = []
    key_list = list(label_dict.keys())
    #random.shuffle(key_list)
    prob = class_ratio_array.copy()[0:len(key_list)]
    #import pdb; pdb.set_trace()
    for idx, this_k in enumerate(key_list):
        #label_key   = folder_reverse_map[this_k]
        prob[ idx ] = class_ratio_array[ this_k ]
    prob = prob/np.sum(prob)
    each_chosen = math.ceil(batch_size/num_sampling)

    if num_sampling <= len(key_list):
        chosen_key = np.random.choice(key_list, num_sampling, replace=False, p=prob )
    else:
        chosen_key = np.random.choice(key_list, num_sampling, replace=True,  p=prob )

    for idx, this_key in enumerate(chosen_key):
        #this_key = chosen_key[idx]
        this_ind = label_dict[this_key] #idx set of all the img in this classes.
        this_num = min(each_chosen,  batch_size - each_chosen*idx)

        if this_num <= len(this_ind):
            this_choice = np.random.choice(this_ind, this_num, replace=False )
        else:
            this_choice = np.random.choice(this_ind, this_num, replace=True )

        indices.extend( this_choice )

    # need to double check the correctness
    return indices
