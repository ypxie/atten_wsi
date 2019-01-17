# -*- coding: utf-8 -*-

import os, sys
PRJ_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PRJ_PATH)
import argparse

import torch
import torch.multiprocessing
from torch.utils.data import DataLoader
from RecurAtt.datasets.mucosa_dataset import MucosaDataSet
from RecurAtt.datasets.mucosa_dataset import BatchSampler
from RecurAtt.models.wsinet  import WsiNet
from RecurAtt.train_eng import train_cls


def set_args():
    parser = argparse.ArgumentParser(description = 'WSI diagnois')
    parser.add_argument("--batch_size",      type=int,   default=32,      help="batch size")
    parser.add_argument("--lr",              type=float, default=1.0e-2,  help="learning rate (default: 0.01)")
    parser.add_argument("--momentum",        type=float, default=0.9,     help="SGD momentum (default: 0.5)")
    parser.add_argument("--weight_decay",    type=float, default=5.0e-4,  help="weight decay for training")
    parser.add_argument("--maxepoch",        type=int,   default=300,     help="number of epochs to train")
    parser.add_argument("--decay_epoch",     type=int,   default=1,       help="lr start to decay linearly from decay_epoch")
    parser.add_argument("--display_freq",    type=int,   default=10,      help="plot the results every {} batches")
    parser.add_argument("--save_freq",       type=int,   default=1,       help="how frequent to save the model")
    parser.add_argument("--session",         type=int,   default=0,       help="training session")
    # model setting
    parser.add_argument("--class_num",       type=int,   default=4)
    parser.add_argument("--input_fea_num",   type=int,   default=2048)
    parser.add_argument("--data_dir",        type=str,   default="../data")
    parser.add_argument("--dataset",         type=str,   default="Mucosa")
    parser.add_argument("--pre_load",        action='store_true')
    parser.add_argument("--use_w_loss",      action='store_true')
    parser.add_argument("--patch_mix",       type=str,   default="pool")
    parser.add_argument("--fea_mix",         type=str,   default="self")
    parser.add_argument("--recur_steps",     type=int,   default=1)
    parser.add_argument("--num_mlp_layer",   type=int,   default=1)

    args = parser.parse_args()
    return args


if  __name__ == '__main__':
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    import torch.backends.cudnn as cudnn
    torch.backends.cudnn.deterministic=True
    cudnn.benchmark = True

    args = set_args()
    # Network and GPU setting
    net = WsiNet(class_num=args.class_num, in_channels=args.input_fea_num, patch_mix=args.patch_mix,
                 fea_mix=args.fea_mix, num_mlp_layer = args.num_mlp_layer,
                 use_w_loss=args.use_w_loss, dataset=args.dataset)
    net.cuda()

    # prepare data locations
    mucosa_data_root = os.path.join(args.data_dir, args.dataset+"Data")
    train_data_root = os.path.join(mucosa_data_root, "Train")
    # val_data_root = os.path.join(mucosa_data_root, "Val")
    val_data_root = os.path.join(mucosa_data_root, "Test")
    # create dataset
    train_dataset = MucosaDataSet(train_data_root, testing=False, pre_load=args.pre_load)
    val_dataset = MucosaDataSet(val_data_root, testing=True, testing_num=50, pre_load=args.pre_load)
    # create dataloader
    batch_sampler  = BatchSampler(label_dict=train_dataset.label_dict, batch_size=args.batch_size,
        data_len=len(train_dataset), class_ratio_array=train_dataset.class_ratio_array, num_sampling=8)
    train_dataloader = DataLoader(dataset=train_dataset, batch_sampler=batch_sampler,
        num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=1,
        num_workers=0, pin_memory=True)

    print(">> START training")
    model_root = os.path.join(args.data_dir, "Models", args.dataset)
    train_cls(train_dataloader, val_dataloader, model_root, args.fea_mix, net, args)
