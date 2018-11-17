# -*- coding: utf-8 -*-

import os, sys
PRJ_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PRJ_PATH)

import argparse
import torch
from torch.utils.data import DataLoader

from Core.datasets.thyroid_dataset import ThyroidDataSet
from Core.datasets.thyroid_dataset import BatchSampler

from Core.models.wsinet  import logistWsiNet
from Core.train_eng import train_cls


def set_args():
    parser = argparse.ArgumentParser(description = 'WSI diagnois by feature fusion using global attention')
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
    parser.add_argument("--model_name",      type=str,   default="global")
    parser.add_argument("--data_dir",        type=str,   default="../data")
    parser.add_argument("--dataset",         type=str,   default="Thyroid")
    parser.add_argument("--pre_load",        type=bool,  default=True)
    parser.add_argument("--class_num",       type=int,   default=3)
    parser.add_argument("--input_fea_num",   type=int,   default=2048)

    args = parser.parse_args()
    return args


if  __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    torch.manual_seed(1234)
    args = set_args()

    # Network and GPU setting
    net = logistWsiNet(class_num=args.class_num, in_channels=args.input_fea_num, use_self=args.model_name)
    cuda_avail = torch.cuda.is_available()
    if cuda_avail :
        torch.cuda.manual_seed(1234)
        net.cuda()
        import torch.backends.cudnn as cudnn
        torch.backends.cudnn.deterministic=True
        cudnn.benchmark = True

    # prepare data locations
    thyroid_data_root = os.path.join(args.data_dir, args.dataset+"Data")
    train_data_root = os.path.join(thyroid_data_root, "Train")
    # val_data_root = os.path.join(thyroid_data_root, "Val")
    val_data_root = os.path.join(thyroid_data_root, "Test")
    # create dataset
    train_dataset = ThyroidDataSet(train_data_root, testing=False, pre_load=args.pre_load)
    val_dataset = ThyroidDataSet(val_data_root, testing=True, testing_num=128, pre_load=args.pre_load)
    # create dataloader
    batch_sampler  = BatchSampler(label_dict=train_dataset.label_dict, batch_size=args.batch_size,
        data_len=len(train_dataset), class_ratio_array=train_dataset.class_ratio_array, num_sampling=8)
    train_dataloader = DataLoader(dataset=train_dataset, batch_sampler=batch_sampler,
        num_workers=0, pin_memory=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size= args.batch_size,
        num_workers=0, pin_memory=True)

    print(">> START training")
    model_root = os.path.join(args.data_dir, "Models", args.dataset)
    train_cls(train_dataloader, val_dataloader, model_root, args.model_name, net, args)
