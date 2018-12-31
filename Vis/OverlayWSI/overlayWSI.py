# -*- coding: utf-8 -*-

import os, sys
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import deepdish as dd

sys.path.append(osp.join(osp.dirname(__file__), 'kfb'))
import kfb_deepzoom, kfbslide


def load_embolus_kfb(kfb_filepath, level=3, wsi_dim=None):
    kfb_slide = kfb_deepzoom.KfbDeepZoomGenerator(kfbslide.KfbSlide(kfb_filepath))
    tile_index = kfb_slide._dz_levels - 1 - level
    x_count, y_count = kfb_slide._t_dimensions[tile_index]
    # substract 1 to crop boundary
    x_count, y_count = x_count - 1, y_count - 1
    x_dim, y_dim = kfb_slide._z_dimensions[tile_index]
    assert x_count*256 <= x_dim and y_count*256 <= y_dim

    wsi_img = np.zeros((y_count*256, x_count*256, 3)) # Crop boundary
    for index_x in range(x_count):
        for index_y in range(y_count):
            start_x, start_y = index_x*256, index_y*256
            wsi_img[start_y:start_y+256, start_x:start_x+256, :] = kfb_slide.get_tile(tile_index, (index_x, index_y))
    # Select regions
    if wsi_dim != None:
        wsi_img = wsi_img[:wsi_dim[0], :wsi_dim[1], :]
    # wsi_img = wsi_img / 255.0

    return wsi_img


def overlayWSI(wsi_path, att_dict=None, num=100):
    wsi_img = load_embolus_kfb(wsi_path).astype(np.uint8)
    alpha = np.ones((wsi_img.shape[0], wsi_img.shape[1], 1), np.uint8) * 100
    att_vals, bboxes = att_dict['probs'], att_dict['bboxes']
    att_num = min(len(att_vals), num)
    for ib in np.arange(att_num):
        val = int(att_vals[ib] * 255)
        h_start = bboxes[ib][1]
        w_start = bboxes[ib][0]
        h_end = h_start + bboxes[ib][3]
        w_end = w_start + bboxes[ib][2]
        # import pdb; pdb.set_trace()
        alpha[h_start:h_end, w_start:w_end, 0] = val

    wsi_rgba = np.concatenate((wsi_img, alpha), axis=2)
    plt.imshow(wsi_rgba, interpolation='none')
    plt.axis('off')
    plt.show()

def load_att(patch_feas):
    feas = dd.io.load(patch_feas)
    prob_list, bboxes = feas['prob'], feas['bbox']

    non_benign_probs = []
    for patch_prob in prob_list:
        non_benign_probs.append(patch_prob[-1])

    full_boxes = []
    for cur_box in bboxes:
        x_start = cur_box[0] - int(cur_box[2]/2)
        y_start = cur_box[1] - int(cur_box[3]/2)
        x_len = cur_box[2] * 2 + 1
        y_len = cur_box[3] * 2 + 1
        full_boxes.append([x_start, y_start, x_len, y_len])

    att_dict = {
        'probs': non_benign_probs,
        'bboxes': full_boxes,
    }

    return att_dict


if __name__ == "__main__":
    wsi_path = "./data/Slides/1238349.kfb"
    patch_feas = "./data/Feas/1238349.h5"

    att_dict = load_att(patch_feas)
    overlayWSI(wsi_path, att_dict, num=20)
