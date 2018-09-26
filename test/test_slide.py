import os, sys
import csv
import argparse, time
import torch, glob
import deepdish as dd
import numpy as np

_this_abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
proj_root  = os.path.join(_this_abs_path)
print(proj_root)

sys.path.insert(0, proj_root)

from papSmear.wsi_mixed_unet import  test_mixed_io, test_mixed_np, test_mixed_slide
from papSmear.models.unet import UNetPap as detModel

#from papSmear.models.wsinet  import lateFeatWsiNet as wsiNet
from papSmear.models.wsinet  import denseWsiNet as wsiNet

from papSmear.models.clsnet_customized import inceptionCellNet as clsNet
from papSmear.io.io_image import patch_read_slide
from papSmear.proj_utils.torch_utils import to_device

cfg = None

model_root = os.path.join(proj_root, 'Model')

if 1:
    det_model_name       =  'tct_unet_tiny'
    det_load_from_epoch  =  8100

    #cls_model_name       = 'random_grey_bin_clinical_160'
    #cls_load_from_epoch  =  200 #222
    
    #wsi_model_name       = 'lateFeat_wsi_model'
    #wsi_load_from_epoch  =  1690 #222
    
    # cls_model_name       = 'mil_bin_160'
    # cls_load_from_epoch  = 160 #222
    # atten_type  =  'mil'

    # wsi_model_name       = 'dense_wsi_model'
    # wsi_load_from_epoch  =  1460 #222
    
    cls_model_name       = 'random_grey_bin_clinical_160'
    cls_load_from_epoch  =  209 #222

    from papSmear.models.wsinet  import denseWsiNet as wsiNet
    wsi_model_name       = 'dense_wsi_model_nologit'
    wsi_load_from_epoch  =  340 
    use_self = None
    use_aux = False
    use_refer = False
    in_channels = 2048

    net_weightspath  = os.path.join(model_root, det_model_name, 'weights_epoch_{}.pth'.format(det_load_from_epoch))
    cls_weightspath  = os.path.join(model_root, cls_model_name,  'weights_epoch_{}.pth'.format(cls_load_from_epoch))
    wsi_weightspath  = os.path.join(model_root, wsi_model_name, 'weights_epoch_{}.pth'.format(wsi_load_from_epoch))


    net              = detModel(in_channels=1, n_classes=1)
    classifier       = clsNet(class_num=2, in_channels=1, large=False, atten_type=atten_type)
    #wsi_net          = wsiNet(class_num=2, in_channels= 2050)
    wsi_net          = wsiNet(class_num=2, in_channels= in_channels,  use_self = use_self, use_aux=use_aux)

    use_grey         = classifier.in_channels == 1 and net.in_channels == 1 #classifier.in_channels == 1
    
    
    marker = det_model_name +'_'+ cls_model_name+'_{}_'.format(cls_load_from_epoch) + wsi_model_name+'_{}_'.format(wsi_load_from_epoch) 
    
net_weights_dict = torch.load(net_weightspath, map_location=lambda storage, loc: storage)
net.load_state_dict(net_weights_dict)
net.eval()
print('reload detection net weights from {}'.format(net_weightspath))

cls_weights_dict = torch.load(cls_weightspath, map_location=lambda storage, loc: storage)
classifier.load_state_dict(cls_weights_dict)
classifier.eval()
print('reload clssifier weights from {}'.format(cls_weightspath))

wsi_weights_dict = torch.load(wsi_weightspath, map_location=lambda storage, loc: storage)
wsi_net.load_state_dict(wsi_weights_dict)
wsi_net.eval()
print('reload clssifier weights from {}'.format(wsi_weightspath))



def get_testing_feat(pos_ratio, logits, feat, testing_num=128 ):
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
    #if (pos_ratio > 0.0007 and slide_pos_prob > 0.5) or (pos_ratio > 0.01) or (slide_pos_prob>0.85):
    if (pos_ratio > 0.0008 and slide_pos_prob > 0.4) or (pos_ratio > 0.02) or (slide_pos_prob>0.95):
        diagnosis_label = 1
    else:
        diagnosis_label = 0
    return diagnosis_label

def bbox2contour(bboxes):
    '''
    Parameters:
    -----------
    bboxes: N x 4, each one is [xmin, ymin, xmax, ymax]
    
    Returns:
    ----------
    contours:  list of [p1, p2, p3, p4, p5], in [x, y] order
    '''
    contours = []
    for bb in bboxes:
        x_min_, y_min_, x_max_, y_max_ = bb.astype(np.float32)
        p1 = [x_min_, y_min_]
        p2 = [x_max_, y_min_]
        p3 = [x_max_, y_max_]
        p4 = [x_min_, y_max_]
        p5 = [x_min_, y_min_]
        this_cont = np.asarray([p1, p2, p3, p4, p5])
        #contours.append( this_cont[:,::-1]  )
        contours.append( this_cont[:]  )
    return contours

def cal_tct_np(im_np, use_cuda=True, resize_ratio=[0.5], save_folder=None, 
               use_grey=use_grey, pick_num = None, **args):
    if use_cuda and torch.cuda.is_available():
        net.cuda()
        classifier.cuda()
        wsi_net.cuda()
        print('You are using CUDA')
    else:
        print('You are using CPU')
    im_np = im_np[:,:,0:3]
    results = test_mixed_slide( im_np, net, classifier, 
                                save_folder = save_folder,
                                resize_ratio=resize_ratio,  
                                cfg=cfg, use_grey=use_grey, 
                                **args)
    pos_ratio, bboxes,  cls_labels, cls_pred, feat = results['pos_ratio'], results['bboxes'], results['cls_labels'],\
                                                    results['cls_pred'] , results['feat']

    # now we only pick the first pick_num patches having the hignest non-neg probability
    sorted_ind = np.argsort(1- cls_pred[:, 1]) 
    if pick_num is not None:
        real_pick  = min(pick_num, cls_pred.shape[0] )
        sorted_ind = sorted_ind[0:pick_num]
    bboxes     = bboxes[sorted_ind]
    cls_labels = cls_labels[sorted_ind]
    cls_pred   = cls_pred[sorted_ind]
    feat       = feat[sorted_ind]
    
    # now jsonnize it. bbox->contour->list
    cls_labels = cls_labels.tolist()
    contours   = bbox2contour(bboxes)
    #print(pos_ratio, contours, cls_labels)
    testing_feat, ratio_feat = get_testing_feat(pos_ratio, cls_pred, feat, testing_num=384)
    testing_feat = to_device(testing_feat, wsi_net.device_id)
    ratio_feat   = to_device(ratio_feat, wsi_net.device_id)
    slide_pos_prob = wsi_net(testing_feat, ratio_feat)
    slide_pos_prob = slide_pos_prob.cpu().data.numpy()[0, 1].tolist() # the positive probability
    slide_label = pos_or_neg(pos_ratio, slide_pos_prob)

    return {'pos_ratio':pos_ratio, 'contours': contours, 'cls_labels':cls_labels,
            'cls_pred':cls_pred, 'feat':feat, 'slide_pos_prob':slide_pos_prob, 'slide_label':slide_label}

def cal_tct_io(io_cls, use_cuda=True, resize_ratio=[0.5], save_folder=None, 
               use_grey=use_grey, pick_num = None, **args):
    if use_cuda and torch.cuda.is_available():
        net.cuda()
        classifier.cuda()
        wsi_net.cuda()
        print('You are using CUDA')
    else:
        print('You are using CPU')
    
    results = test_mixed_io(io_cls, net, classifier, 
                            save_folder = save_folder, 
                            use_grey=use_grey, cfg=cfg, 
                            resize_ratio=resize_ratio,  **args)

    pos_ratio, bboxes, cls_labels, cls_pred, feat = results['pos_ratio'], results['bboxes'], results['cls_labels'],\
                                                    results['cls_pred'] , results['feat']   
    # now we only pick the first pick_num patches having the hignest non-neg probability
    sorted_ind = np.argsort(1- cls_pred[:, 1]) 
    if pick_num is not None:
        real_pick  = min(pick_num, cls_pred.shape[0] )
        sorted_ind = sorted_ind[0:pick_num]
    bboxes     = bboxes[sorted_ind]
    cls_labels = cls_labels[sorted_ind]
    cls_pred   = cls_pred[sorted_ind]
    feat       = feat[sorted_ind]
    
    # now jsonnize it. bbox->contour->list
    cls_labels = cls_labels.tolist()
    contours   = bbox2contour(bboxes)
    #print(pos_ratio, contours, cls_labels)

    testing_feat, ratio_feat = get_testing_feat(pos_ratio, cls_pred, feat, testing_num=384)
    testing_feat = to_device(testing_feat, wsi_net.device_id)
    ratio_feat   = to_device(ratio_feat, wsi_net.device_id)
    slide_pos_prob = wsi_net(testing_feat, ratio_feat)
    slide_pos_prob = slide_pos_prob.cpu().data.numpy()[0, 1].tolist() # the positive probability
    
    slide_label = pos_or_neg(pos_ratio, slide_pos_prob)
    return {'pos_ratio':pos_ratio, 'contours': contours, 'cls_labels':cls_labels,
            'cls_pred':cls_pred, 'feat':feat, 'slide_pos_prob':slide_pos_prob, 'slide_label':slide_label}
          

if  __name__ == '__main__':
    
    from papSmear.proj_utils.local_utils import imread, mkdirs, getfilelist, walk_dir
    from torch.utils.data.dataset import Dataset
    from torch.utils.data import DataLoader
    import json, argparse

    class wsiDataSet(Dataset):
        def __init__(self, file_list, level=None, use_grey=None):
            self.file_list = file_list
            self.level = level
            self.use_grey = use_grey
        def __len__(self):
            return len(self.file_list)
        def __getitem__(self, index):
            while True:
                try:
                    img_path = self.file_list[index]
                    this_img_name =  os.path.basename(img_path)
                    this_naked_name, slide_ext = os.path.splitext(this_img_name)
                    if slide_ext == ".svs": # level 1 is 0.25
                        img_np   = patch_read_slide(img_path, level = 1, use_grey=self.use_grey)
                    else:
                        img_np   = patch_read_slide(img_path, level = 2, use_grey=self.use_grey)
                    return img_np, index
                except:
                    print('failed to process {}'.format(img_path))
                    index = index + 1

    parser = argparse.ArgumentParser(description = 'Pap Smear Bounding Box Detection')
    
    parser.add_argument('--test_data', default = '', help='which dataset to test')
    parser.add_argument('--start_idx', default = 0, type=int, help='start index for testing')

    args = parser.parse_args()
    
    home = os.path.expanduser('~')
    test_data  = args.test_data

    gt_path = 'summary.csv'
    refer_name = False
    if os.path.exists(gt_path):
        ref_name_list, _shao_list, _hsj_list = [], [], []
        _name_res_dict = {}
        #import pdb; pdb.set_trace()
        with open(gt_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                slide_name, shao, hsj = row['slide_name'], int(row['shao']), int(row['hsj'])
                ref_name_list.append( slide_name )
                _shao_list.append( shao )
                _hsj_list.append( hsj )
                _name_res_dict[slide_name] = (shao, hsj)
        refer_name = True
        
    folder_list = [\
                    #'/home/yuanpu/Data/YuanpuDisk/TCT_Slides',
                    '/home/yuanpu/Desktop/HSJ_local',
                    #'/media/yuanpu/Seagate Backup Plus Drive/HSJ_local'
                  ]
                  
    file_types  = ['.svs', '.kfb'] # the tuple of file types
    
    for this_data_root in folder_list:
        slide_list  = []
        this_list = walk_dir(this_data_root, file_types)
        #slide_list.extend(this_list)    
        for img_path in this_list:
            this_root, this_img_name = os.path.dirname(img_path), os.path.basename(img_path)
            this_naked_name, slide_ext = os.path.splitext(this_img_name)
            save_folder = os.path.join(this_root, marker + 'save_results')
            h5_path   = os.path.join(save_folder, this_naked_name + '_res.h5')
            #if not os.path.exists(h5_path):#  \
                #and ("2017-08-15" in img_path or "Dr_Li" in img_path or "TCT_test" in img_path):
            slide_list.append(img_path)

        save_root  = os.path.join(this_data_root)
        data_set    = wsiDataSet(slide_list, level = 2, use_grey = use_grey)
        data_loader = DataLoader(dataset=data_set, batch_size = 1, num_workers = 4, pin_memory=False)
        label_list, name_list = [], []
        name_label_dict = {}
        for im_in, index in data_loader: # batch_size can only be 1

            ratio_list = [] 
            im_in, index = im_in[0].numpy(), index[0]
            img_path = slide_list[index]
            print('now processing {}'.format(img_path))
            this_root, this_img_name   = os.path.dirname(img_path), os.path.basename(img_path)
            this_naked_name, slide_ext = os.path.splitext(this_img_name)
            
            save_folder = os.path.join(this_root, marker + 'save_results')
            mkdirs([save_folder])

            json_path = os.path.join(save_folder, this_naked_name + '_res.json')
            h5_path   = os.path.join(save_folder, this_naked_name + '_res.h5')        
            
            
            if 1:
                results = cal_tct_np(im_in,save_folder=save_folder, pick_num=100,
                                    resize_ratio=[1], naked_name=this_naked_name)
                pos_ratio, contours, cls_labels, cls_pred, feat, slide_pos_prob, slide_label = \
                                                    results['pos_ratio'], results['contours'], \
                                                    results['cls_labels'], results['cls_pred'], \
                                                    results['feat'], results['slide_pos_prob'], \
                                                    results['slide_label']
                
                result_dict = { 'pos_ratio':pos_ratio, 'contours':contours, 'cls_labels':np.asarray(cls_labels), \
                                'cls_pred': np.asarray(cls_pred).astype(np.float16), 'feat':feat.astype(np.float16), 
                                'slide_pos_prob':slide_pos_prob, 'slide_label':slide_label
                                }

                # 'contours': np.asarray(contours).tolist() \
                
                with open(json_path,'w') as jfile:
                    json.dump({'pos_ratio':pos_ratio, 'slide_pos_prob':slide_pos_prob, 
                            'slide_label':slide_label}, jfile)
                ratio_list.append(pos_ratio)

                dd.io.save(h5_path, result_dict)
                print(pos_ratio, slide_pos_prob)

                name_list.append(this_img_name)
                label_list.append(slide_label)
                name_label_dict[this_img_name]  =  slide_label

        if refer_name == True:
            final_name_list, final_label_list = [], []
            for _this_name_ in ref_name_list:

                final_name_list.append(_this_name_)
                final_label_list.append( name_label_dict[_this_name_]   )
            

            from sklearn.metrics import precision_recall_fscore_support as score
            from sklearn.metrics import confusion_matrix

            #precision, recall, fscore, support = score(label_gt_list, final_label_list)
            _shao_con_mat = confusion_matrix(_shao_list, final_label_list)
            _hsj_con_mat  = confusion_matrix(_hsj_list, final_label_list)

            #print('  p:  {}\n  r:  {}\n  f1: {} \n'.format(precision, recall, fscore))
            print('  _shao_con_mat matrix: \n')
            print(_shao_con_mat)

            print('  _hsj_con_mat matrix: \n')
            print(_hsj_con_mat)
        else:
            final_name_list = name_list
            final_label_list = label_list
        
        csv_file = os.path.join(save_root, marker+'_summary.csv')
        with open(csv_file, 'w') as outfile:
            writer = csv.writer(outfile, delimiter=',')
            writer.writerow(final_name_list)
            writer.writerow(final_label_list)
            