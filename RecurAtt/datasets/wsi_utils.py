# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import deepdish as dd

from ..proj_utils.local_utils import getfolderlist, getfilelist


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


def get_all_files(rootFolder, inputext = ['.h5'], class_map_dict=None, pre_load=True):
    '''
    Given a root folder, this function needs to return 2 lists. imglist and clslist:
        (img_data, label)
    '''

    sub_folder_list, sub_folder_name = getfolderlist(rootFolder)
    file_list, label_list = [], []

    for idx, (this_folder, this_folder_name) in enumerate(zip(sub_folder_list, sub_folder_name)):
        filelist, filenames = getfilelist(this_folder, inputext, with_ext=False)
        this_cls = class_map_dict[this_folder_name]

        for this_file_path, this_file_name in zip(filelist, filenames):
            if pre_load is True:
                try:
                    data = dd.io.load(this_file_path)
                except Exception as err:
                    import traceback
                    traceback.print_tb(err.__traceback__)
                    print('cannot process data[feat] in {}'.format(this_file_path))
                    data = None
                file_list.append(data)

            else:
                file_list.append(this_file_path)
            label_list.append(this_cls)
    return file_list, label_list
