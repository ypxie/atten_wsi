# -*- coding: utf-8 -*-

import os, sys
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import deepdish as dd
from skimage import filters, io
import skimage.io as sio
import skimage

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


def overlayWSI(wsi_path, patch_att_path, overlay_path, num=100, alp=0.65):
    wsi_img = load_embolus_kfb(wsi_path)
    wsi_img = (wsi_img * alp).astype(np.uint8)

    alpha = np.zeros((wsi_img.shape[0], wsi_img.shape[1]), np.uint8)
    att_dict = dd.io.load(patch_att_path)
    att_vals, bboxes = att_dict['probs'], att_dict['bboxes']
    att_vals = np.array(att_vals) / max(att_vals)

    att_num = min(len(att_vals), num)
    for ib in np.arange(att_num):
        val = int(att_vals[ib] * 255)
        h_start = int(bboxes[ib][1])
        w_start = int(bboxes[ib][0])
        h_end = h_start + int(bboxes[ib][3])
        w_end = w_start + int(bboxes[ib][2])
        alpha[h_start:h_end, w_start:w_end] = val


    alpha = filters.gaussian(alpha, sigma=60)
    cmap = plt.get_cmap('jet')
    heat_map = cmap(alpha)[:, :, :-1]
    alpha = (heat_map * (1 - alp) * 255.0).astype(np.uint8)
    alpha_wsi = wsi_img + alpha

    save_name = os.path.splitext(os.path.basename(wsi_path))[0]
    save_pathname = os.path.join(overlay_path, save_name + ".png")
    io.imsave(save_pathname, alpha_wsi)

    # plt.imshow(alpha_wsi)
    # plt.axis('off')
    # plt.show()


if __name__ == "__main__":
    filename = "1239928"
    wsi_path = "./data/Slides/{}.kfb".format(filename)
    patch_att_path = "./data/Feas/{}.h5".format(filename)
    overlay_path = "./data/Overlays"

    # att_dict = load_att(patch_feas)
    overlayWSI(wsi_path, patch_att_path, overlay_path, num=1000)
