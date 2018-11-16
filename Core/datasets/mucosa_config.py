# -*- coding: utf-8 -*-

import os, sys


folder_ratio_map = {
    "0": 0.25,
    "1": 0.25,
    "2": 0.25,
    "3": 0.25,
}


multi_class_map_dict = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
}


"""
class_reverse_map = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
}
"""
class_reverse_map = {}
for k, v in multi_class_map_dict.items():
    class_reverse_map[v] = k


"""
folder_map_dict = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
}
"""
folder_map_dict = {}
for idx, (k, v ) in enumerate(folder_ratio_map.items()):
    folder_map_dict[k] = idx


"""
folder_reverse_map = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
}
"""
folder_reverse_map = {}
for k, v in folder_map_dict.items():
    folder_reverse_map[v] = k
