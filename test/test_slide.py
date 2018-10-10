import os, sys
import csv
import argparse, time
import torch, glob
import deepdish as dd
import numpy as np

proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(proj_root)
sys.path.insert(0, proj_root)

from Core.proj_utils.torch_utils import to_device

cfg = None
use_mixed = False
in_channels = 2048
use_refer = True
model_root = os.path.join(proj_root, 'Model')

if 1:
    det_model_name       =  'tct_unet_tiny'
    det_load_from_epoch  =  8100

    cls_model_name       = 'random_grey_bin_clinical_160'
    cls_load_from_epoch  =  209 #222

    from Core.models.wsinet  import liteWsiNet as wsiNet
    wsi_model_name       = 'lite_model'
    wsi_load_from_epoch  =  970 #222
    use_self = None
    use_aux = False

    wsi_weightspath  = os.path.join(model_root, wsi_model_name, 'weights_epoch_{}.pth'.format(wsi_load_from_epoch))

    wsi_net          = wsiNet(class_num=2, in_channels= in_channels,  use_self = use_self, use_aux=use_aux)

    #use_grey         = classifier.in_channels == 1 and net.in_channels == 1 #classifier.in_channels == 1
    marker = det_model_name +'_'+ cls_model_name+'_{}_'.format(cls_load_from_epoch) + wsi_model_name+'_{}_'.format(wsi_load_from_epoch)

wsi_weights_dict = torch.load(wsi_weightspath, map_location=lambda storage, loc: storage)
wsi_net.load_state_dict(wsi_weights_dict)
wsi_net.eval()
print('reload clssifier weights from {}'.format(wsi_weightspath))


def get_testing_feat(pos_ratio, logits, feat, testing_num=128 ):
    '''
    Parameters:
    ---------------------
    pos_ratio:  a scalar
    logits:    Bxc layer activation before softmax.
    feat:      B x D, the patch features.
    testing_num: how many patches to use.
    Returns:
    ---------------------
    returned_composed feature:  B X D_F
    pos_ratio:    (B, )
    '''
    feat = np.squeeze(feat)
    if use_mixed:
        mix_feat = np.concatenate( [feat, logits], axis=1)
    else:
        mix_feat = feat
    total_ind  = np.array( range(0, logits.shape[0]) )
    chosen_total_ind_ = total_ind[0:testing_num]
    chosen_total_ind_ = chosen_total_ind_.reshape( (chosen_total_ind_.shape[0],) )

    chosen_feat = mix_feat[chosen_total_ind_]

    this_true_label =  None
    pos_ratio = np.asarray([pos_ratio] )
    return chosen_feat[None], pos_ratio

def pos_or_neg(pos_ratio, slide_pos_prob):
    #if (pos_ratio > 0.0008 and slide_pos_prob > 0.3) or (pos_ratio > 0.02) or (slide_pos_prob>0.95):
    if (pos_ratio > 0.0008 and slide_pos_prob > 0.4) or (pos_ratio > 0.02) or (slide_pos_prob>0.95):

    #if (pos_ratio > 0.001 and slide_pos_prob > 0.5) or (pos_ratio > 0.02) or (slide_pos_prob>0.95):
        diagnosis_label = 1
    else:
        diagnosis_label = 0
    return diagnosis_label

def cal_tct_feat(pos_ratio, cls_pred, feat):
    '''
    take in feature and compute the slide label
    Parameters:
    ----------------

    Returns:
    ----------------

    '''
    testing_feat, ratio_feat = get_testing_feat(pos_ratio, cls_pred, feat, testing_num=128)
    testing_feat = to_device(testing_feat, wsi_net.device_id)
    ratio_feat   = to_device(ratio_feat, wsi_net.device_id)
    slide_pos_prob = wsi_net(testing_feat, ratio_feat)
    slide_pos_prob = slide_pos_prob.cpu().data.numpy()[0, 1].tolist() # the positive probability
    slide_label = pos_or_neg(pos_ratio, slide_pos_prob)

    return {'pos_ratio':pos_ratio, 'cls_pred':cls_pred, 'feat':feat,
            'slide_pos_prob':slide_pos_prob, 'slide_label':slide_label}


if  __name__ == '__main__':

    from Core.proj_utils.local_utils import imread, mkdirs, getfilelist, walk_dir
    from torch.utils.data.dataset import Dataset
    from torch.utils.data import DataLoader
    import json, argparse

    parser = argparse.ArgumentParser(description = 'Pap Smear Bounding Box Detection')

    parser.add_argument('--test_data', default = '', help='which dataset to test')
    parser.add_argument('--start_idx', default = 0, type=int, help='start index for testing')

    args = parser.parse_args()

    home = os.path.expanduser('~')
    test_data  = args.test_data

    folder_list = [\
                    #'/home/yuanpu/Data/YuanpuDisk/TCT_Slides',
                    os.path.join(proj_root, 'Data', 'hsj_feat', 'tct_unet_tiny_random_grey_bin_clinical_160_209_feat_save_results' )
                    #os.path.join(proj_root, 'Data', 'jinyu_feat', 'tct_unet_tiny_random_grey_bin_clinical_160_209_feat_save_results' )

                    #'/media/yuanpu/Seagate Backup Plus Drive/HSJ_local'
                  ]
    save_root = os.path.join(proj_root, 'Data', 'hsj_feat')

    file_types  = ['.h5'] # the tuple of file types

    for this_data_root in folder_list:
        slide_name_list, h5_path_list  = [], []
        this_list = walk_dir(this_data_root, file_types)

        for h5_path in this_list:
            this_root, this_h5_name = os.path.dirname(h5_path), os.path.basename(h5_path)
            this_naked_name, slide_ext = os.path.splitext(this_h5_name)
            #save_folder = os.path.join(this_root, marker + 'save_results')
            img_name = this_naked_name[0:-4] + '.kfb'
            slide_name_list.append(img_name)
            h5_path_list.append(h5_path)

        test_name_list, test_label_list, name_label_dict = [], [], {}
        test_pos_ratio_list, test_pos_prob_list = [], []

        for this_h5, this_img_name in zip(h5_path_list, slide_name_list):

            data = dd.io.load(this_h5)
            pos_ratio, label, logits, feat = data['pos_ratio'], data['cls_labels'], data['cls_pred'], data['feat']
            pos_ratio, label, logits, feat = np.asarray(pos_ratio),  np.asarray(label), np.asarray(logits), np.asarray(feat)
            #import pdb; pdb.set_trace()
            results = cal_tct_feat(pos_ratio, logits, feat)
            slide_label = results['slide_label']
            pos_ratio   = results['pos_ratio']
            cls_pred   = results['cls_pred']
            slide_pos_prob = results['slide_pos_prob']

            print(this_img_name, slide_label)
            test_name_list.append(this_img_name)
            test_label_list.append(slide_label)
            test_pos_ratio_list.append(pos_ratio)
            test_pos_prob_list.append(slide_pos_prob)

            name_label_dict[this_img_name]  =  [slide_label, pos_ratio, slide_pos_prob ]


            final_name_list = test_name_list
            final_label_list = test_label_list
            final_pos_ratio_list = test_pos_ratio_list
            final_slide_pos_prob_list = test_pos_prob_list

            slide_label, pos_ratio, slide_pos_prob

            csv_file = os.path.join(save_root, marker+'_summary.csv')

            with open(csv_file, 'w') as outfile:
                writer = csv.writer(outfile, delimiter=',')
                writer.writerow( ['slide_name', 'slide_label', 'pos_ratio',  'slide_pos_prob'] )
                for name, label, pos_ratio, slide_pos_prob in zip(final_name_list, final_label_list,
                                                            final_pos_ratio_list, final_slide_pos_prob_list):
                    writer.writerow([name, label, pos_ratio, slide_pos_prob])
                #writer.writerow(final_label_list)
