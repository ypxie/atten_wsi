# -*- coding: utf-8 -*-

import os, sys


folder_ratio_map = {
    "Benign": 0.4,
    "Uncertain": 0.3,
    "Yes": 0.3
}


multi_class_map_dict = {
    "Benign": 0,
    "Uncertain": 1,
    "Yes": 2,
}


"""
class_reverse_map = {
    0: "Benign",
    1: "Uncertain",
    2: "Yes",
}
"""
class_reverse_map = {}
for k, v in multi_class_map_dict.items():
    class_reverse_map[v] = k


"""
folder_map_dict = {
    "Benign": 0,
    "Uncertain": 1,
    "Yes": 2,
}
"""
folder_map_dict = {}
for idx, (k, v ) in enumerate(folder_ratio_map.items()):
    folder_map_dict[k] = idx


"""
folder_reverse_map = {
    0: "Benign",
    1: "Uncertain",
    2: "Yes",
}
"""
folder_reverse_map = {}
for k, v in folder_map_dict.items():
    folder_reverse_map[v] = k
