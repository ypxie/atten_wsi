# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import deepdish as dd
import math, random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from .thyroid_config import multi_class_map_dict, class_reverse_map
from .thyroid_config import folder_map_dict, folder_reverse_map, folder_ratio_map

from .data_utils import aggregate_label
from ..proj_utils.local_utils import getfolderlist, getfilelist


def get_all_files(rootFolder, inputext = ['.json'], class_map_dict=None, pre_load=True):
    '''
    Given a root folder, this function needs to return 2 lists. imglist and clslist:
        (img_data, label)
    '''

    sub_folder_list, sub_folder_name = getfolderlist(rootFolder)
    file_list, label_list = [], []
    #print(sub_folder_name, '\n')

    for idx, (this_folder, this_folder_name) in enumerate( zip(sub_folder_list, sub_folder_name) ):
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
                # print('Finish {}'.format(this_file_path))
                file_list.append(data)

            else:
                file_list.append(this_file_path)
            label_list.append(this_cls)
    return file_list, label_list



def sample_gumbel(logits, eps=1e-20):
    #U = torch.rand(logits.size()).cuda()
    U = logits.new_zeros(logits.size()).uniform_()
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits)
    return F.softmax(y / temperature, dim=-1)



class ThyroidDataSet(Dataset):
    def __init__(self, data_dir, testing=False, testing_num=128):
        self.data_dir = data_dir
        self.testing = testing
        self.pre_load = False
        self.testing_num = testing_num

        self.class_map_dict = multi_class_map_dict
        self.class_reverse_map = class_reverse_map
        self.folder_map_dict = folder_map_dict
        self.folder_reverse_map = folder_reverse_map
        self.folder_ratio_map = folder_ratio_map

        # calculate class_ratio_array
        class_ratio_array = [None]*len(self.folder_map_dict.keys())
        for this_k in self.folder_map_dict.keys():
            class_ratio_array[self.folder_map_dict[this_k]] = self.folder_ratio_map[this_k]

        class_ratio_array = np.asarray(class_ratio_array).astype(np.float)
        class_ratio_array = class_ratio_array/np.sum(class_ratio_array)
        self.class_ratio_array = class_ratio_array

        file_list, label_list = get_all_files(data_dir, inputext = ['.h5'],
                                              class_map_dict=self.folder_map_dict,
                                              pre_load=self.pre_load)
        self.file_list   = file_list
        self.label_list  = label_list

        summery_label_dict = aggregate_label(label_list)
        # for k,v in summery_label_dict.items():
        #     print('The number of ', k, 'is: ', len(v))

        # the following line get {1:[1,2,3,4], 2:[23,25], ...}
        self.label_dict   =  summery_label_dict
        self.img_num      =  len(self.file_list)

        self.count = 0
        self.batch_count = 0
        self.start = 0

        self._shuffle = True
        self.indices = list(range(self.img_num))
        self.temperature = 0.5

        ## doubt about the following two
        self.chosen_num_list = list(range(100, 140))
        self.fixed_num = 20

        self.max_num = len(self.file_list)
        self.max_num = 140 if not self.testing else self.testing_num
        self.data_cache = [None]*len(self.file_list)


    def get_true_label(self, label):
        new_label =  self.class_map_dict[self.folder_reverse_map[label]]
        return new_label

    def __len__(self):
        return self.img_num


    def __getitem__(self, index):
        while True:
            try:
                if self.pre_load == True:
                    data = self.file_list[index]
                else:
                    this_data_path = self.file_list[index]
                    data = dd.io.load(this_data_path)
                chosen_num = random.choice(self.chosen_num_list)

                label, logits, feat = data['cls_labels'], data['cls_pred'], data['feat']
                feat = np.squeeze(feat)
                #mix_feat = np.concatenate( [feat, logits], axis=1)
                mix_feat = feat

                total_ind  = np.array(range(0, len(label)))
                feat_placeholder = np.zeros((self.max_num, mix_feat.shape[1]), dtype=np.float32)

                if self.testing:
                    chosen_total_ind_ = total_ind[0:self.testing_num]
                else:
                    additoinal_num = 10
                    logits          = torch.from_numpy(logits).float() #.cuda()
                    neg_logits      = logits[self.fixed_num+additoinal_num::, 0] + 1e-5
                    gumbel_probs    = gumbel_softmax_sample(neg_logits, self.temperature)
                    this_probs_norm = gumbel_probs.cpu().numpy()

                    # Use positive samples for selection
                    this_probs_norm = [1.0 - ele for ele in this_probs_norm]
                    this_probs_norm = this_probs_norm / np.sum(this_probs_norm)

                    # combine fixed number + random chosen number
                    fixed_chosen_ind = total_ind[0:self.fixed_num+additoinal_num]
                    fixed_chosen_ind = np.random.choice(total_ind[0:self.fixed_num+additoinal_num], self.fixed_num)
                    random_chosen_ind = np.random.choice(total_ind[self.fixed_num+additoinal_num::],
                                                         chosen_num-self.fixed_num,
                                                         replace=False, p=this_probs_norm)
                    chosen_total_ind_ = np.concatenate([fixed_chosen_ind, random_chosen_ind], 0 )

                chosen_total_ind_ = chosen_total_ind_.reshape( (chosen_total_ind_.shape[0],) )
                chosen_feat = mix_feat[chosen_total_ind_]
                true_num = chosen_feat.shape[0]
                feat_placeholder[0:true_num] = chosen_feat
                this_true_label = self.get_true_label(self.label_list[index])

                return feat_placeholder, this_true_label, true_num

            except Exception as err:
                #print(err)
                import traceback
                traceback.print_tb(err.__traceback__)
                print("Having problem processing index {}".format(index) )
                index = random.choice(self.indices)

    def __iter__(self):
        return self



class BatchSampler(object):
    def __init__(self, label_dict=None, batch_size=32,
                 class_ratio_array=None, num_sampling=8, data_len=None):
        self.label_dict = label_dict
        self.batch_size = batch_size
        self.num_sampling = num_sampling
        self.class_ratio_array = class_ratio_array
        self.data_len = data_len
        self.num_batches = self.data_len // self.batch_size


    def __iter__(self):
        for idx in range( self.num_batches):
            batch = get_indices_balance(self.label_dict, self.num_sampling,
                                        self.batch_size, self.class_ratio_array)
            yield batch

    def __len__(self):
        return self.num_batches


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
    prob = class_ratio_array.copy()[0:len(key_list)]

    for idx, this_k in enumerate(key_list):
        prob[idx] = class_ratio_array[this_k]
    prob = prob/np.sum(prob)

    true_sampling = min(num_sampling, len(key_list))
    each_chosen = math.ceil(batch_size/true_sampling)

    chosen_key = np.random.choice(key_list, true_sampling, replace=False, p=prob)
    for idx, this_key in enumerate(chosen_key):
        #this_key = chosen_key[idx]
        this_ind = label_dict[this_key] #idx set of all the img in this classes.
        this_num = min(each_chosen,  batch_size - each_chosen*idx)

        if this_num <= len(this_ind):
            this_choice = np.random.choice(this_ind, this_num, replace=False)
        else:
            this_choice = np.random.choice(this_ind, this_num, replace=True)

        indices.extend(this_choice)

    return indices
