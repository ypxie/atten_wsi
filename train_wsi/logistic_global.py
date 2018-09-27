import os, sys, pdb
sys.path.insert(0, '..')
import torch.multiprocessing


import argparse
import torch
from torch.utils.data import DataLoader

from Core.proj_utils.local_utils import mkdirs
from Core.datasets.wsi_dataset import wsiDataSet as DataSet
from Core.datasets.wsi_dataset import BatchSampler 

from Core.cfgs.config_pap import cfg
from Core.models.wsinet  import logistWsiNet as wsiNet
#from Core.models.clsnet  import inceptionCellNet as cellNet
from Core.train_wsi_parallel import train_cls

proj_root = os.path.join('..')
home = os.path.expanduser('~')

model_root = os.path.join(proj_root, 'Model')
#model_root = os.path.join(home, 'ganData', 'TCT', 'Models')

mkdirs([model_root])

## ----------------training parameters setting--------------------------
proj_root = os.path.join('..')

#train_data_root   = os.path.join(home, 'DataSet', 'TCT_DATA', 'split_h5', 'train')
#test_data_root    = os.path.join(home, 'DataSet', 'TCT_DATA', 'split_h5', 'test')
    
#train_data_root   = os.path.join(home, 'DataSet', 'TCT_DATA', 'split_h5_mil_160_feat', 'total')
#test_data_root    = os.path.join(home, 'DataSet', 'TCT_DATA', 'split_h5_mil_160_feat', 'test')

#train_data_root   = os.path.join(home, 'DataSet', 'TCT_DATA', 'debug_h5', 'train')
#test_data_root    = os.path.join(home, 'DataSet', 'TCT_DATA', 'debug_h5', 'test')

#train_data_root = os.path.join(proj_root, 'Data', 'split_h5', 'train')
#test_data_root  = os.path.join(proj_root, 'Data', 'split_h5', 'test')

train_data_root   = os.path.join(proj_root, 'Data', 'split_bin_clinical_160_209_feat', 'train')
test_data_root    = os.path.join(proj_root, 'Data', 'split_bin_clinical_160_209_feat', 'test')

#/home/yuanpu/DataSet/TCT_DATA/split_json

#train_data_root   = os.path.join(home, 'ganData', 'TCT', 'Data',  'crop_patch')
#test_data_root    = os.path.join(home, 'ganData', 'TCT', 'Data', 'crop_patch')

#save_root = os.path.join(home, 'DataSet', 'TCT_DATA', 'dataloader_sanity')
check_sanity = False
#mkdirs(save_root)
model_name = 'logist_global_model'
use_self = 'global'

def set_args():
    # Arguments setting
    parser = argparse.ArgumentParser(description = 'Pap Smear Bounding Box Detection')

    parser.add_argument('--batch_size', type=int, default = 32, help='batch size.')
    parser.add_argument("--buffer_size",     type=int, default = 128, help="steps frozen")
    
    #parser.add_argument('--img_size',   default = [256], help='output image size')
    
    #parser.add_argument('--start_seg',       type=int,   default = 100,    help='number of epochs before we train the seg part')
    parser.add_argument('--lr',              type=float, default = 1e-2, help='learning rate (default: 0.01)')
    parser.add_argument('--momentum',        type=float, default = 0.9,  help='SGD momentum (default: 0.5)')
    parser.add_argument('--weight_decay',    type=float, default = 5e-5, help='weight decay for training')
    
    parser.add_argument('--reuse_weights',   action='store_true', default=True, help='continue from last checkout point')
    parser.add_argument('--load_from_epoch', type=int, default= 0, help='load from epoch')
    # 450-last epoch before adding tongs of negtive samples
    
    parser.add_argument('--maxepoch',        type=int, default = 1001,    help='number of epochs to train (default: 10)')
    parser.add_argument("--decay_epoch",     type=int, default = 1, help="lr start to decay linearly from decay_epoch")
    parser.add_argument("--frozen_step",     type=int, default = 0, help="steps frozen")
    
    parser.add_argument('--display_freq',    type=int, default= 300, help='plot the results every {} batches')
    parser.add_argument('--save_freq',       type=int, default= 10,   help='how frequent to save the model')
    parser.add_argument('--model_name',      type=str, default=model_name)

    args = parser.parse_args()
    return args

if  __name__ == '__main__':
    args = set_args()
    torch.multiprocessing.set_start_method('forkserver', force=True)

    # DatasetDir = "/data/.data1/pingjun/Datasets/PapSmear"

    # net = wsiNet(class_num=2, in_channels= 66, num_clusters=3)
    net = wsiNet(class_num=2, in_channels= 2048, use_self = use_self)

    print(net)
    
    train_dataset = DataSet( train_data_root, multi_class=False, 
                             check_sanity=check_sanity, use_grey=True, testing=False,
                            )

    test_dataset = DataSet(  test_data_root,  multi_class=False, testing_num = 384,
                             check_sanity=check_sanity, use_grey=True, testing=True,
                            )
    
    batch_sampler  = BatchSampler(  label_dict=train_dataset.label_dict, batch_size=args.batch_size,
                                    data_len= len(train_dataset), 
                                    class_ratio_array=train_dataset.class_ratio_array, 
                                    num_sampling = 8)

    train_dataloader = DataLoader(  dataset=train_dataset, batch_sampler = batch_sampler, 
                                    num_workers = 16, pin_memory=True)
    
    test_dataloader = DataLoader(   dataset=test_dataset,  batch_size= args.batch_size,
                                    num_workers = 16, pin_memory=True)


    # CUDA Settings
    cuda_avail = torch.cuda.is_available()
    if cuda_avail :
        net.cuda()
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

    # print ('>> START training ')
    
    train_cls(train_dataloader, test_dataloader, model_root, args.model_name, net, args)

