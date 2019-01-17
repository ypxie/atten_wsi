# -*- coding: utf-8 -*-

import os, sys
import warnings
warnings.filterwarnings("ignore")

import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import deepdish as dd
from skimage import filters, io
import skimage.io as sio
import skimage
import openslide


def load_wsi(wsi_filepath, level=3, wsi_dim=None):
	wsi_header = openslide.OpenSlide(wsi_filepath)
	wsi_size = wsi_header.level_dimensions[level]

	wsi_img = wsi_header.read_region((0,0), level, wsi_size)

	wsi_img = np.array(wsi_img)[:, :, :-1]

	return wsi_img


def overlayWSI(wsi_path, patch_att_path, ori_path, overlay_path, num=100, alp=0.65):
    save_name = os.path.splitext(os.path.basename(wsi_path))[0]

    wsi_img = load_wsi(wsi_path, level=3)
    io.imsave(os.path.join(ori_path, save_name+".jpg"), wsi_img)

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


    save_pathname = os.path.join(overlay_path, save_name + ".png")
    io.imsave(save_pathname, alpha_wsi)

    plt.imshow(alpha_wsi)
    plt.axis('off')
    plt.show()



if __name__ == "__main__":
    database = "Mucosa"
    filename = ""
    wsi_path = "./data/{}/TIFFs/{}.tiff".format(database, filename)
    patch_att_path = "./data/{}/Feas/{}.h5".format(database, filename)
    ori_path = "./data/{}/OriImgs".format(database)
    overlay_path = "./data/{}/OverlayTIFF".format(database)

    overlayWSI(wsi_path, patch_att_path, ori_path, overlay_path, num=1000)
