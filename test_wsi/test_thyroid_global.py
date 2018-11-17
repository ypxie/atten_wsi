# -*- coding: utf-8 -*-

import os, sys
PRJ_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PRJ_PATH)

import argparse
import torch
from torch.utils.data import DataLoader


from Core.datasets.thyroid_dataset import ThyroidDataSet
from Core.models.wsinet  import logistWsiNet
from Core.test_eng import test_cls


def set_args():
    parser = argparse.ArgumentParser(description = 'WSI diagnois by feature fusion using global attention')
    parser.add_argument("--model_path",      type=str,   default="900/global-epoch-289.pth")
    # model setting
    parser.add_argument("--model_name",      type=str,   default="global")
    parser.add_argument("--data_dir",        type=str,   default="../data")
    parser.add_argument("--dataset",         type=str,   default="Thyroid")
    parser.add_argument("--pre_load",        type=bool,  default=True)
    parser.add_argument("--class_num",       type=int,   default=3)
    parser.add_argument("--input_fea_num",   type=int,   default=2048)

    args = parser.parse_args()
    return args

if  __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    args = set_args()

    # Network and GPU setting
    net = logistWsiNet(class_num=args.class_num, in_channels=args.input_fea_num, use_self=args.model_name)
    cuda_avail = torch.cuda.is_available()
    if cuda_avail :
        net.cuda()
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

    # prepare data locations
    thyroid_data_root = os.path.join(args.data_dir, args.dataset+"Data")
    test_data_root = os.path.join(thyroid_data_root, "Test")

    # prepare data locations
    thyroid_data_root = os.path.join(args.data_dir, args.dataset+"Data")
    test_dataset = ThyroidDataSet(test_data_root, testing=True, testing_num=128, pre_load=args.pre_load)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1,
        num_workers=0, pin_memory=True)
    print(">> START testing")
    model_root = os.path.join(args.data_dir, "Models", args.dataset)
    test_cls(test_dataloader, model_root, args.model_name, net, args)
