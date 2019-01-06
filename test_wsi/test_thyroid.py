# -*- coding: utf-8 -*-

import os, sys
import numpy as np
PRJ_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PRJ_PATH)

import argparse
import torch
from torch.utils.data import DataLoader

from RecurAtt.datasets.thyroid_dataset import ThyroidDataSet
from RecurAtt.models.wsinet  import WsiNet
from RecurAtt.test_eng import test_cls


def set_args():
    parser = argparse.ArgumentParser(description = 'WSI diagnois')
    parser.add_argument("--model_path",      type=str,   default="")
    # model setting
    parser.add_argument("--patch_mix",       type=str,   default="pool")
    parser.add_argument("--fea_mix",         type=str,   default="global")
    parser.add_argument("--data_dir",        type=str,   default="../data")
    parser.add_argument("--dataset",         type=str,   default="Thyroid")
    parser.add_argument("--num_mlp_layer",   type=int,   default=1)
    parser.add_argument("--use_w_loss",      action='store_true')
    parser.add_argument("--pre_load",        action='store_true')
    parser.add_argument("--class_num",       type=int,   default=3)
    parser.add_argument("--input_fea_num",   type=int,   default=2048)
    parser.add_argument("--recur_steps",     type=int,   default=3)

    parser.add_argument("--seed",            type=int,   default=1234)
    args = parser.parse_args()
    return args


if  __name__ == '__main__':
    args = set_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    import torch.backends.cudnn as cudnn
    torch.backends.cudnn.deterministic=True
    cudnn.benchmark = True

    # Network and GPU setting
    net = WsiNet(class_num=args.class_num, in_channels=args.input_fea_num, patch_mix=args.patch_mix,
                 fea_mix=args.fea_mix, recur_steps=args.recur_steps, num_mlp_layer = args.num_mlp_layer,
                 use_w_loss=args.use_w_loss, dataset=args.dataset)
    net.cuda()

    # prepare data locations
    thyroid_data_root = os.path.join(args.data_dir, args.dataset+"Data")
    test_data_root = os.path.join(thyroid_data_root, "Test")

    # prepare data locations
    thyroid_data_root = os.path.join(args.data_dir, args.dataset+"Data")
    test_dataset = ThyroidDataSet(test_data_root, testing=True, testing_num=128, pre_load=args.pre_load)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, num_workers=0, pin_memory=True)
    print(">> START testing")
    model_root = os.path.join(args.data_dir, "Models", args.dataset)
    test_cls(test_dataloader, model_root, net, args)
