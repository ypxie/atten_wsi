
import math
import json
import os
import scipy.io as sio
import numpy as np
from numba import jit

try:
    from .utils import yolo as yolo_utils
    from .utils.cython_yolo import yolo_to_bbox
except:
    pass
    
from .datasets.cls_config import class_reverse_map

from .proj_utils.torch_utils import to_device
from .proj_utils.local_utils import Indexflow, imshow, mkdirs
from .proj_utils.model_utils import resize_layer


import torch
from   torch.autograd import Variable
import torch.nn.functional as F


@jit(nopython=True)
def change_val(img, val, len, x_min, y_min, x_max, y_max):
    left_len  = (len-1)//2
    right_len = (len-1) - left_len
    row_size, col_size = img.shape[0:2]
    for le in range(-left_len, right_len + 1):
        y_min_ = max(0, y_min + le )
        x_min_ = max(0, x_min + le )
        
        y_max_ = min(row_size, y_max - le )
        x_max_ = min(col_size, x_max - le )

        img[y_min_:y_max_, x_min_:x_min_+1] = val
        img[y_min_:y_min_+1, x_min_:x_max_] = val
        img[y_min_:y_max_, x_max_:x_max_+1] = val
        img[y_max_:y_max_+1, x_min_:x_max_] = val
    return img

@jit
def overlay_bbox(img, bbox,linewidth=1, labels=None, cls_color=None):
    labels = np.zeros((len(bbox))) if labels is None else labels
    cls_color = {0:[255,0,0], 1:[0,255, 0]}  if cls_color is None else cls_color
    for bb, this_label in zip(bbox, labels):
        this_color = cls_color[this_label]
        x_min_, y_min_, x_max_, y_max_ = bb
        x_min_, y_min_, x_max_, y_max_ = int(x_min_),int( y_min_), int(x_max_), int(y_max_)
        for idx in range(img.shape[2]):
            change_val(img[:,:,idx], this_color[idx], linewidth, x_min_, y_min_, x_max_, y_max_)
    return img

def batch_forward(cls, BatchData, batch_size, **kwards):
    total_num = BatchData.shape[0]
    results = {'bbox':[],'iou':[], 'prob':[]}

    for ind in Indexflow(total_num, batch_size, False):
        data = BatchData[ind]
        
        data = to_device(data, cls.device_id, volatile=True)
        bbox_pred, iou_pred, prob_pred = cls.forward(data, **kwards)
        #print('data shape: ',bbox_pred.size(), iou_pred.size(), prob_pred.size())
        results['bbox'].append(bbox_pred.cpu().data.numpy())
        results['iou'].append(iou_pred.cpu().data.numpy())
        results['prob'].append(prob_pred.cpu().data.numpy())

    for k, v in results.items():
        results[k] = np.concatenate(v, 0)
    return results


def split_testing(cls, org_img,  batch_size = 4, windowsize=None,
                  thresh= None, cfg=None, args=None, do_seg=False, 
                  ext_len = 0):
    
    # since inputs is (B, T, C, Row, Col), we need to make (B*T*C, Row, Col)
    #windowsize = self.row_size

    do_seg = args.do_seg if args is not None else do_seg
    board = 0
    adptive_batch_size=False # cause we dont need it for fixed windowsize.
    chn, org_row, org_col = org_img.shape
    pad_row, pad_col = max(0, windowsize-org_row), max(0, windowsize-org_col)
    if pad_row > 0 or pad_col > 0:
        org_img = np.pad(org_img, ((0,0), (0, pad_row), (0, pad_col)), mode='constant')
    #print('org_img type: ', org_img.dtype)
    #img = to_device(org_img[None], cls.device_id, volatile=True)

    results = {'bbox':[],'iou':[], 'prob':[]}
    chan, row_size, col_size = org_img.shape
    org_img = org_img[None]

    if windowsize is None:
        row_window = row_size
        col_window = col_size
    else:
        row_window = min(windowsize, row_size)
        col_window = min(windowsize, col_size)
    
    # print(row_window, col_window)
    num_row, num_col = math.ceil(row_size/row_window),  math.ceil(col_size/col_window) # lower int
    feat_map = None
    for row_idx in range(num_row):
        row_start = row_idx * row_window
        if row_start + row_window > row_size:
            row_start = row_size - row_window
        row_end   = row_start + row_window

        for col_idx in range(num_col):
            col_start = col_idx * col_window
            if col_start + col_window > col_size:
                col_start = col_size - col_window
            col_end   = col_start + col_window

            batch_data = org_img[:,:, row_start:row_end+ext_len, col_start:col_end+ext_len]
            #print('this_patch shape: ', this_patch.shape)
            # feedforward normaization to this to save memory
            batch_data = batch_data.astype(np.float32)
            #batch_data = batch_data * (2. / 255) - 1.
            with torch.no_grad():
                batch_data = to_device(batch_data, cls.device_id, volatile=True)
                bbox_pred, iou_pred, prob_pred, large_map  = cls.forward(batch_data)

            bbox_pred = bbox_pred.cpu().data.numpy()
            iou_pred  = iou_pred.cpu().data.numpy()
            if prob_pred is None:
                prob_pred = np.ones_like(iou_pred).astype(np.float32)
            else:
                prob_pred = prob_pred.cpu().data.numpy()
            large_map = large_map

            if feat_map is None and do_seg is True:
                chnn     = large_map.size()[1]
                feat_map = Variable(torch.zeros(1, chnn, row_size, col_size), volatile=True)

            #print(large_map[:,:, 0:row_end-row_start, 0:col_end-col_start].size(), feat_map[:,:, row_start:row_end, col_start:col_end].size())

            if do_seg:
                feat_map[:,:, row_start:row_end, col_start:col_end] = large_map[:,:, 0:row_end-row_start, 0:col_end-col_start]
            
            H, W = cls.out_size
            x_ratio, y_ratio = cls.x_ratio, cls.y_ratio

            bbox_pred = yolo_to_bbox(
                        np.ascontiguousarray(bbox_pred, dtype=np.float),
                        np.ascontiguousarray(cfg.anchors, dtype=np.float),
                        H, W,
                        x_ratio, y_ratio)

            np_start  = np.array([[col_start , row_start, col_start, row_start]]  )

            results['bbox'].append(bbox_pred +  np_start )
            results['iou'].append(iou_pred)
            results['prob'].append(prob_pred)
            
    #results['feat_map'] = feat_map[:,:,0:org_row, 0:org_col] if do_seg else None
    results['bbox'] = np.concatenate(results['bbox'], 1)
    results['iou']  = np.concatenate(results['iou'], 1)
    results['prob'] = np.concatenate(results['prob'], 1)

    return results
    
def get_feat_bbox(pred_boxes, featMaps, dest_size=[32,32], org_img=None, 
                  board_ratio=0.1, device_id=None):
    croped_feat_list=[]
    with torch.no_grad():
        featMaps = Variable(featMaps)
    
    org_size, org_coord, patch_list = [], [], []
    img_row, img_col = featMaps.size()[2::]
    for img_idx, this_bbox_list in enumerate(pred_boxes):
        for bidx, bb in enumerate(this_bbox_list):
            x_min_, y_min_, x_max_, y_max_ = bb
            x_min_, y_min_, x_max_, y_max_ = int(x_min_),int( y_min_), int(x_max_), int(y_max_)
            col_size  = x_max_ - x_min_ + 1
            row_size  = y_max_ - y_min_ + 1

            boarder_row = int(board_ratio * row_size)
            boarder_col = int(board_ratio * col_size)

            xmin, ymin = x_min_ - boarder_col, y_min_ - boarder_row
            xmax, ymax = x_max_ + boarder_col, y_max_ + boarder_col

            xmin, ymin = max(xmin, 0), max(ymin,0)
            xmax, ymax = min(xmax, img_col), min(ymax,img_row)

            final_col_size  = xmax - xmin + 1
            final_row_size  = ymax - ymin + 1
            
            this_featmap   = featMaps[img_idx:img_idx+1,:, ymin:ymax+1, xmin:xmax+1]
            
            #this_featmap   = to_device(this_featmap, device_id, volatile=True)
            #this_patch     = org_img[:, ymin:ymax+1, xmin:xmax+1]
            #import pdb; pdb.set_trace();
            resize_featmap = resize_layer(this_featmap, dest_size)
            croped_feat_list.append(resize_featmap.cpu())
            org_size.append([final_row_size, final_col_size])
            org_coord.append([ymin, xmin])
            patch_list.append(None)
    
    if len(croped_feat_list) > 0: 
        croped_feat_nd = torch.cat(croped_feat_list, 0)
        #import pdb; pdb.set_trace()
        #croped_feat_nd = croped_feat_nd * (2. / 255) - 1.
        return croped_feat_nd, org_size, org_coord, patch_list
    else:
        return None, None, None, None

def process_wsi_bbox(net, pred_boxes, featMaps, dest_size=[128, 128], 
                     org_img=None, board_ratio=0.1, batch_size=128):
    
    croped_feat_list=[]
    batch_ind = 0
    pool_size = 1024

    #featMaps = featMaps[None]
    with torch.no_grad():
        #featMaps = Variable(featMaps)
        chn = featMaps.size()[1]
        #batch_array = np.zeros((batch_size, chn, dest_size[0],  
        #                         dest_size[1]), dtype=np.float32)

        #print('batch_tensor size: ',batch_size, chn, dest_size[0],dest_size[1] )
        #batch_tensor =  Variable(torch.zeros((pool_size, chn, dest_size[0],  
        #                         dest_size[1]), dtype=torch.float32) )
        batch_tensor =  Variable(featMaps.new(pool_size, chn, 
                                 dest_size[0], dest_size[1]) )

    org_size, org_coord, patch_list = [], [], []
    
    img_row, img_col = featMaps.shape[2::]
    mask_pred_list = []

    
    for img_idx, this_bbox_list in enumerate(pred_boxes):
        for bidx, bb in enumerate(this_bbox_list):
            x_min_, y_min_, x_max_, y_max_ = bb
            x_min_, y_min_, x_max_, y_max_ = int(x_min_),int( y_min_), int(x_max_), int(y_max_)
            col_size  = x_max_ - x_min_ + 1
            row_size  = y_max_ - y_min_ + 1

            boarder_row = int(board_ratio * row_size)
            boarder_col = int(board_ratio * col_size)

            xmin, ymin = x_min_ - boarder_col, y_min_ - boarder_row
            xmax, ymax = x_max_ + boarder_col, y_max_ + boarder_col

            xmin, ymin = max(xmin, 0), max(ymin,0)
            xmax, ymax = min(xmax, img_col), min(ymax,img_row)

            final_col_size  = xmax - xmin + 1
            final_row_size  = ymax - ymin + 1
            
            this_featmap   = featMaps[img_idx:img_idx+1,:, ymin:ymax+1, xmin:xmax+1]
            
            #this_featmap   = this_featmap.astype(np.float32)
            #this_featmap   = this_featmap * (2. / 255) - 1.
            
            #with torch.no_grad():
                #this_featmap   = Variable(torch.from_numpy(this_featmap) )
                #this_featmap = this_featmap.float32()
                #this_featmap   = to_device(this_featmap, net.device_id, volatile=True)
            #this_patch     = org_img[:, ymin:ymax+1, xmin:xmax+1]
            #import pdb; pdb.set_trace();
            resize_featmap = resize_layer(this_featmap, dest_size)
            #croped_feat_list.append(resize_featmap.cpu())
            #print('resize_featmap size, this_featmap size ', 
            #resize_featmap.size(), this_featmap.size(), batch_tensor.size())
            batch_tensor[batch_ind] = resize_featmap
            batch_ind += 1
            

            if batch_ind == pool_size: # can be further optimized 
                #this_feat = to_device(batch_tensor, net.device_id, volatile=True)
                mask_pred = batch_mask_forward(net, batch_tensor, batch_size= batch_size )
                #mask_pred = net(this_feat)
                mask_pred_list.append(mask_pred.cpu())
                batch_ind = 0

            org_size.append([final_row_size, final_col_size])
            org_coord.append([ymin, xmin])
            patch_list.append(None)
    
    if batch_ind != 0 :
        #this_feat = to_device(batch_tensor[0:batch_ind], net.device_id, volatile=True)
        #this_feat = batch_tensor[0:batch_ind]
        #mask_pred = net(this_feat)

        mask_pred = batch_mask_forward(net, batch_tensor[0:batch_ind], batch_size=batch_size )
        mask_pred_list.append(mask_pred.cpu())
        batch_ind = 0

    if len(mask_pred_list) > 0: 
        total_pred = torch.cat(mask_pred_list, 0)
        return total_pred
    else:
        return None


def batch_mask_forward(net, feat_nd, batch_size=128 ):
    mask_pred_list = []
    if type(feat_nd) == np.ndarray:
        total_num = feat_nd.shape[0]
    else:
        total_num = feat_nd.size()[0]

    with torch.no_grad():
        for ind in Indexflow(total_num, batch_size, False):
            st, end   = np.min(ind), np.max(ind)+1
            this_feat = feat_nd[st:end]
            
            this_feat = to_device(this_feat, net.device_id, volatile=True)
            #print(this_feat.get_device(), net.device_id.get_device() )
            #print('insider shape: ', this_feat.size(), feat_nd.size())
            mask_pred = net(this_feat).cpu().detach()
            mask_pred_list.append(mask_pred)
            this_feat = None
        total_pred = torch.cat(mask_pred_list, 0)
        return total_pred


def save_boundingbox(im_np, bboxes, cls_labels, pos_neg_labels, 
                     save_folder, naked_name, marker = 'patches'):
    extra_ratio=0.8
    import uuid
    from scipy import misc
    img_row, img_col = im_np.shape[0:2]

    for bidx, (bb, this_label, this_pos_neg) in \
            enumerate(zip(bboxes, cls_labels, pos_neg_labels)):
        
        this_cls_folder = class_reverse_map[this_label]
        if this_pos_neg is 1:
            print('this bb: and im shape: ', bb, im_np.shape)
            x_min_, y_min_, x_max_, y_max_ = bb
            x_min_, y_min_, x_max_, y_max_ = int(x_min_), int( y_min_), int(x_max_), int(y_max_)
            col_size  = x_max_ - x_min_ + 1
            row_size  = y_max_ - y_min_ + 1
            
            boarder_row = int( (extra_ratio/2) * row_size)
            boarder_col = int( (extra_ratio/2) * col_size)

            xmin, ymin = x_min_ - boarder_col, y_min_ - boarder_row
            xmax, ymax = x_max_ + boarder_col, y_max_ + boarder_col

            xmin, ymin = max(xmin, 0), max(ymin,0)
            xmax, ymax = min(xmax, img_col), min(ymax, img_row)

            this_patch_name = str(bidx) + '_' + uuid.uuid4().hex
            this_patch = im_np[ymin:ymax, xmin:xmax,:]

            this_save_folder = os.path.join(save_folder, marker , naked_name, this_cls_folder)
            if not os.path.exists(this_save_folder):
                os.makedirs(this_save_folder)
            
            this_save_path = os.path.join(this_save_folder, this_patch_name+'.png')

            misc.imsave(this_save_path, this_patch)
            this_patch = None


def split_patch_testing(net, org_img,  batch_size = 64, windowsize=None, 
                        step_size = 200, pool_size = 512):
    
    # since inputs is (B, T, C, Row, Col), we need to make (B*T*C, Row, Col)
    #windowsize = self.row_size

    bbox_list, cls_pred_list = [], []

    chn, org_row, org_col = org_img.shape
    pad_row, pad_col = max(0, windowsize-org_row), max(0, windowsize-org_col)
    if pad_row > 0 or pad_col > 0:
        org_img = np.pad(org_img, ((0,0), (0, pad_row), (0, pad_col)), mode='constant')

    chan, row_size, col_size = org_img.shape
    org_img = org_img[None]

    if windowsize is None:
        row_window = row_size
        col_window = col_size
    else:
        row_window = min(windowsize, row_size)
        col_window = min(windowsize, col_size)
    
    with torch.no_grad():
        #featMaps = Variable(featMaps)
        #chn = featMaps.size()[1]
        batch_tensor = np.zeros((pool_size, chan, row_window, col_window), dtype=np.float32)

        #print('batch_tensor size: ',batch_size, chn, dest_size[0],dest_size[1] )
        #batch_tensor =  Variable(torch.zeros((pool_size, chn, dest_size[0],  
        #                         dest_size[1]), dtype=torch.float32) )
        #batch_tensor =  Variable(featMaps.new(pool_size, chan, row_window, col_window) )
    
    row_step, col_step = step_size, step_size
    
    # print(row_window, col_window)
    num_row, num_col = math.ceil(row_size/row_step),  math.ceil(col_size/col_step) # lower int
    batch_ind = 0
    for row_idx in range(num_row):
        row_start = row_idx * row_step
        if row_start + row_window > row_size:
            row_start = row_size - row_window
        row_end   = row_start + row_window

        for col_idx in range(num_col):
            col_start = col_idx * col_step
            if col_start + col_window > col_size:
                col_start = col_size - col_window
            col_end   = col_start + col_window

            batch_data = org_img[:,:, row_start:row_end, col_start:col_end]
            # print('this_patch shape: ', this_patch.shape)
            # feedforward normaization to this to save memory
            batch_data = batch_data.astype(np.float32)
            #batch_data = batch_data * (2. / 255) - 1.
            batch_tensor[batch_ind] = batch_data
            batch_ind += 1

            bbox_pred = np.array( [[col_start, row_start, col_end, row_end ]]   )
            bbox_list.append( bbox_pred )

            if batch_ind == pool_size: # can be further optimized 
                #this_feat = to_device(batch_tensor, net.device_id, volatile=True)
                
                cls_pred = batch_mask_forward(net, batch_tensor, batch_size= batch_size )
                cls_pred_list.append(cls_pred.cpu().data)
                batch_ind = 0
    
    if batch_ind != 0 :
        cls_pred = batch_mask_forward(net, batch_tensor[0:batch_ind], batch_size=batch_size )
        cls_pred_list.append(cls_pred.cpu().data)
        batch_ind = 0

    #results['feat_map'] = feat_map[:,:,0:org_row, 0:org_col] if do_seg else None
    results = {}
    results['bbox'] = np.concatenate(bbox_list, 0)
    
    results['prob'] = torch.cat(cls_pred_list, 0)

    return results


@jit(nopython=True)
def get_bin(bboxes, num_row, num_col, row_step, col_step):
    count_bin  = np.zeros((num_row, num_col, 1), dtype = np.uint32)
    center_bin = np.zeros((num_row, num_col, 2), dtype = np.float32) # [row, col] order
    n_box = bboxes.shape[0]
    for idx in range(n_box):
        this_bbox = bboxes[idx]
        col_min, row_min, col_max, row_max = this_bbox
        org_row_cent, org_col_cent = (0.5*(row_min+row_max)), (0.5*(col_min+col_max))
        bin_row_cent = int( org_row_cent/row_step  )
        bin_col_cent = int( org_col_cent/col_step  )

        cur_count = count_bin[bin_row_cent, bin_col_cent, 0]
        
        cur_row_mean = (center_bin[bin_row_cent, bin_col_cent, 0] * cur_count + org_row_cent)/(cur_count+1)
        cur_col_mean = (center_bin[bin_row_cent, bin_col_cent, 1] * cur_count + org_col_cent)/(cur_count+1)

        #cur_row_mean = (bin_row_cent*row_step + row_step//2)
        #cur_col_mean = (bin_col_cent*col_step + col_step//2)

        center_bin[bin_row_cent, bin_col_cent, 0]   = cur_row_mean
        center_bin[bin_row_cent, bin_col_cent, 1]   = cur_col_mean

        count_bin[bin_row_cent, bin_col_cent, 0]       += 1

    center_bin  = (np.floor(center_bin) ).astype(np.int64)
    return center_bin, count_bin

def split_patch_testing_atten(net, org_img, bboxes, batch_size = 64, 
                              windowsize=None, step_size = 200, 
                              pool_size = 512, get_feat=False):
    '''
    Parameters:
    ---------------
        net:            
        org_img: B*C*H*W
        bboxes:  B*4 [xmin, ymin, xmax, ymax] order
    Return:
    ---------------

    '''
    pool_size = batch_size
    bbox_list, cls_pred_list, feat_list = [], [], []

    chn, org_row, org_col = org_img.shape
    pad_row, pad_col = max(0, windowsize-org_row), max(0, windowsize-org_col)
    if pad_row > 0 or pad_col > 0:
        org_img = np.pad(org_img, ((0,0), (0, pad_row), (0, pad_col)), mode='constant')

    chan, row_size, col_size = org_img.shape
    org_img = org_img[None]

    if windowsize is None:
        row_window = row_size
        col_window = col_size
    else:
        row_window = min(windowsize, row_size)
        col_window = min(windowsize, col_size)
    
    #with torch.no_grad():
        #featMaps = Variable(featMaps)
        #chn = featMaps.size()[1]
    batch_tensor = np.zeros((pool_size, chan, row_window, col_window), dtype=np.float32)

        #print('batch_tensor size: ',batch_size, chn, dest_size[0],dest_size[1] )
        #batch_tensor =  Variable(torch.zeros((pool_size, chn, dest_size[0],  
        #                         dest_size[1]), dtype=torch.float32) )
        #batch_tensor =  Variable(featMaps.new(pool_size, chan, row_window, col_window) )
    
    row_step, col_step = step_size, step_size
    
    # print(row_window, col_window)
    num_row, num_col = math.ceil(row_size/row_step),  math.ceil(col_size/col_step) # lower int
    #----------------------------------------------
    #         use bin as attention
    #----------------------------------------------
    # I don't actually care about the boarder :)
    
    import time
    tic = time.time()
    center_bin, count_bin = get_bin(bboxes, num_row, num_col, row_step, col_step)
    
    #print('2d bin taking time: {}'.format(time.time()-tic))    
    # imshow(count_bin)
    #----------------------------------------------
    batch_ind = 0
    ccount = 0
    for row_idx in range(num_row):
        for col_idx in range(num_col):
            row_cent, col_cent   =  center_bin[row_idx, col_idx]

            if count_bin[row_idx, col_idx] > 0:
                ccount += 1
                row_start, col_start =  max(row_cent - row_window//2, 0), max(col_cent-col_window//2, 0)
                if row_start + row_window > row_size:
                    row_start = row_size - row_window
                row_end   = row_start + row_window

                if col_start + col_window > col_size:
                    col_start = col_size - col_window
                col_end   = col_start + col_window

                batch_data = org_img[:,:, row_start:row_end, col_start:col_end]
                # print('this_patch shape: ', this_patch.shape)
                # feedforward normaization to this to save memory
                batch_data = batch_data.astype(np.float32)
                #batch_data = batch_data * (2. / 255) - 1.
                batch_tensor[batch_ind] = batch_data
                batch_ind += 1

                bbox_pred = np.array( [[col_start, row_start, col_end, row_end ]]   )
                bbox_list.append( bbox_pred )

                if batch_ind == pool_size: # can be further optimized 
                    #this_feat = to_device(batch_tensor, net.device_id, volatile=True)
                    #cls_pred = batch_mask_forward(net, batch_tensor, batch_size= batch_size )
                    with torch.no_grad():
                        this_feat = to_device(batch_tensor, net.device_id, volatile=True)
                        cls_pred  = net(this_feat)
                    cls_pred_list.append(cls_pred.cpu().data)
                    if get_feat is True:
                        feat_list.append( net.feat.cpu().data   )
                    batch_ind = 0
        
    if batch_ind != 0 :
        #cls_pred = batch_mask_forward(net, batch_tensor[0:batch_ind], batch_size=batch_size )
        with torch.no_grad():
            this_feat = to_device(batch_tensor[0:batch_ind], net.device_id, volatile=True)
            cls_pred  = net(this_feat)

        cls_pred_list.append( cls_pred.cpu().data)
        if get_feat is True:
            feat_list.append( net.feat.cpu().data   )
        batch_ind = 0

    #print('do you save time?: # boxes {}, # effect squares {}'.format(bboxes.shape[0], ccount) )

    #results['feat_map'] = feat_map[:,:,0:org_row, 0:org_col] if do_seg else None
    results = {}
    if len(bbox_list) == 0:
        results['bbox'] = None
        results['prob'] = None
        results['feat'] = None
    else:    
        results['bbox'] = np.concatenate(bbox_list, 0)
        results['prob'] = torch.cat(cls_pred_list, 0)
        
        if get_feat is True:
            results['feat'] = torch.cat(feat_list, 0)
        else:
            results['feat'] = None
    return results


def split_fix_testing(net, org_img, bboxes, batch_size = 128, windowsize= 32, pool_size = 1024):
    '''
    Parameters:
    ---------------
        net:            
        org_img: B*C*H*W
        bboxes:  B*4 [xmin, ymin, xmax, ymax] order
    Return:
    ---------------

    '''
    bbox_list, cls_pred_list = [], []

    chn, org_row, org_col = org_img.shape
    pad_row, pad_col = max(0, windowsize-org_row), max(0, windowsize-org_col)
    if pad_row > 0 or pad_col > 0:
        org_img = np.pad(org_img, ((0,0), (0, pad_row), (0, pad_col)), mode='constant')

    chan, row_size, col_size = org_img.shape
    org_img = org_img[None]

    if windowsize is None:
        row_window = row_size
        col_window = col_size
    else:
        row_window = min(windowsize, row_size)
        col_window = min(windowsize, col_size)
    
    batch_tensor = np.zeros((pool_size, chan, row_window, col_window), dtype=np.float32)
    batch_ind = 0

    for this_bbox in bboxes:
        col_min, row_min, col_max, row_max = this_bbox
        row_cent, col_cent = int(0.5*(row_min+row_max)), int(0.5*(col_min+col_max))

        row_start, col_start =  max(row_cent - row_window//2, 0), max(col_cent-col_window//2, 0)
        if row_start + row_window > row_size:
            row_start = row_size - row_window
        row_end   = row_start + row_window
        
        if col_start + col_window > col_size:
            col_start = col_size - col_window
        col_end   = col_start + col_window

        batch_data = org_img[:,:, row_start:row_end, col_start:col_end]
        batch_data = batch_data.astype(np.float32)
        
        batch_tensor[batch_ind] = batch_data
        batch_ind += 1

        bbox_pred  = np.array( [[col_start, row_start, col_end, row_end ]]   )
        #bbox_pred  = np.array( [[col_min, row_min, col_max, row_max ]]   )

        bbox_list.append( bbox_pred )
        
        if batch_ind == pool_size: # can be further optimized 
            #this_feat = to_device(batch_tensor, net.device_id, volatile=True)
            cls_pred  = batch_mask_forward(net, batch_tensor, batch_size= batch_size )
            cls_pred_list.append(cls_pred.cpu().data)
            batch_ind = 0
        
    if batch_ind != 0 :
        cls_pred  = batch_mask_forward(net, batch_tensor[0:batch_ind], batch_size=batch_size )
        cls_pred_list.append(cls_pred.cpu().data)
        batch_ind = 0

    #results['feat_map'] = feat_map[:,:,0:org_row, 0:org_col] if do_seg else None
    results = {}
    if len(bbox_list) == 0:
        results['bbox'] = None
        results['prob'] = None
    else:    
        results['bbox'] = np.concatenate(bbox_list, 0)
        results['prob'] = torch.cat(cls_pred_list, 0)

    return results


from  skimage.feature import peak_local_max

import torch, os, sys
from torch.autograd import Variable
import numpy as np
import time, math
#import scipy.io as sio
#import matplotlib.pyplot as plt
#from .proj_utils.local_utils import *

def split_index(rowsize, colsize, windowsize=1000, board = 0, fixed_window = False, step_size = None):
    '''
        Parameters:
        -----------

        Returns:
        -----------

    '''
    IndexDict = {}
    identifier = -1
    PackList = []
    if windowsize is not None and  type(windowsize) is int:
        windowsize = (windowsize, windowsize)

    if windowsize is None or (rowsize <= windowsize[0] and colsize<=windowsize[1] ):

        place_slice  = (slice(0, rowsize), slice(0, colsize))
        output_slice = place_slice
        crop_patch_slice = (slice(0, rowsize), slice(0, colsize))
        thisSize =  (rowsize, colsize )
        identifier = identifier + 1

        if thisSize in IndexDict:
           IndexDict[thisSize].append(identifier)
        else:
           IndexDict[thisSize] = []
           IndexDict[thisSize].append(identifier)
        #PackList.append((crop_patch_slice, place_slice, output_slice, thisSize, identifier))
        PackList.append((crop_patch_slice, place_slice, output_slice, thisSize))
    else:

        hidden_windowsize = (windowsize[0]-2*board, windowsize[1]-2*board)
        
        if type(step_size) is int:
            step_size = (step_size, step_size)
        if step_size is None:
            step_size = hidden_windowsize

        numRowblocks = int(math.ceil(float(rowsize)/hidden_windowsize[0]))  # how many windows we need
        numColblocks = int(math.ceil(float(colsize)/hidden_windowsize[1]))
        
        # sanity check, make sure the image is at least of size window_size to the left-hand side if fixed_windows is true
        # which means,    -----*******|-----, left to the vertical board of original image is at least window_size.

        thisrowstart, thiscolstart =0, 0
        thisrowend,   thiscolend = 0,0
        
        for row_idx in range(numRowblocks):
            thisrowlen   = min(hidden_windowsize[0], rowsize - row_idx * step_size[0])
            # special case for the first row and column. 

            thisrowstart = 0 if row_idx == 0 else thisrowstart + step_size[0]

            thisrowend = thisrowstart + thisrowlen

            row_shift = 0
            if fixed_window:
                #if thisrowlen < hidden_windowsize[0]:
                #    row_shift = hidden_windowsize[0] - thisrowlen
                
                if thisrowend + board >= rowsize:  # if we can not even pad it. 
                    row_shift = (hidden_windowsize[0] - thisrowlen) + (thisrowend+board - rowsize)
                    
            for col_idx in range(numColblocks):
                #import pdb; pdb.set_trace()
                thiscollen = min(hidden_windowsize[1], colsize -  col_idx * step_size[1])
 
                thiscolstart = 0 if col_idx == 0 else thiscolstart + step_size[1]

                thiscolend = thiscolstart + thiscollen

                col_shift = 0
                if fixed_window:
                    # we need to shift the patch to left to make it at least windowsize.
                    #if thiscollen < hidden_windowsize[1]:
                    #    col_shift = hidden_windowsize[1] - thiscollen

                    if thiscolend + board >= colsize:  # if we can not even pad it. 
                        col_shift = (hidden_windowsize[1] - thiscollen) + (thiscolend+board - colsize)

                #
                #----board----******************----board----
                #

                if thisrowstart == 0:
                    #------ slice obj for crop from original image-----
                    crop_r_start = thisrowstart
                    crop_r_end  =  thisrowend + 2*board 

                    #------ slice obj for crop results from local patch-----
                    output_slice_row_start = 0
                    output_slice_row_end   = output_slice_row_start + thisrowlen 
                else:
                    #------ slice obj for crop from original image-----
                    crop_r_start = thisrowstart - board - row_shift 
                    crop_r_end  =  min(rowsize, thisrowend   + board)
                    
                    #------ slice obj for crop results from local patch-----
                    output_slice_row_start = board + row_shift
                    output_slice_row_end   = output_slice_row_start + thisrowlen
                
                if thiscolstart == 0:
                    #------ slice obj for crop from original image-----
                    crop_c_start = thiscolstart
                    crop_c_end  =  thiscolend + 2* board 
                    #------ slice obj for crop results from local patch-----
                    output_slice_col_start = 0
                    output_slice_col_end   = output_slice_col_start + thiscollen
                else:
                    #------ slice obj for crop from original image-----
                    crop_c_start =  thiscolstart - board - col_shift 
                    crop_c_end   =  min(colsize, thiscolend  + board) 
                    #------ slice obj for crop results from local patch-----
                    output_slice_col_start = board + col_shift
                    output_slice_col_end   = output_slice_col_start + thiscollen

                # slice on a cooridinate of the original image for the central part
                place_slice  = (slice(thisrowstart, thisrowend), slice(thiscolstart, thiscolend))
                
                # extract on local coordinate of a patch to fill place_slice
                output_slice = ( slice(output_slice_row_start, output_slice_row_end),
                                 slice(output_slice_col_start, output_slice_col_end))

                # here take care of the original image size
                crop_patch_slice = (slice(crop_r_start, crop_r_end), slice(crop_c_start, crop_c_end))
                thisSize = (thisrowlen + 2*board + row_shift, thiscollen + 2*board + col_shift)
                
                
                identifier =  identifier +1
                #PackList.append((crop_patch_slice, place_slice, output_slice, thisSize, identifier))
                PackList.append((crop_patch_slice, place_slice, output_slice, thisSize))
                if thisSize in IndexDict:
                   IndexDict[thisSize].append(identifier)
                else:
                   IndexDict[thisSize] = []
                   IndexDict[thisSize].append(identifier)
                #print('this crop and place: ', (crop_patch_slice, place_slice, output_slice, thisSize))

    return  PackList

def split_testing_unet(cls, imgs,  batch_size = 4, windowsize=512, 
                       board = 40, grid_probs_bag = None, step=1):
    '''
    Testing each region of an input image, and decide if this is a ROI
    Parameters:
    -----------
        cls: classification model
        imgs: chn x row x col image
        grid_probs_bag: dictionary of (row_, col_) indicating wheter or not to evaluate
    Returns:
    -----------
        output: 1xrowxcol a probability map
    '''

    
    chn, org_row, org_col = imgs.shape
    pad_row, pad_col = max(0, windowsize-org_row), max(0, windowsize-org_col)
    if pad_row > 0 or pad_col > 0:
        imgs = np.pad(imgs, ((0,0), (0, pad_row), (0, pad_col)), mode='constant')

    chn, row_size, col_size = imgs.shape

    assert len(imgs.shape) == 3, 'wrong input dimensio of input image'
    outputs    =   np.zeros((1, row_size, col_size), dtype=np.float32) # we don't consider multiple outputs
    PackList   =   split_index(row_size, col_size, windowsize = windowsize, 
                               board = board, fixed_window= True, 
                               step_size=None)
    total_testing = len(PackList)
    _,_,_, thisSize = PackList[0]

    #BatchData = np.zeros((batch_size, chn, thisSize[0], thisSize[1]) , dtype=np.float32)
    BatchData = np.zeros((batch_size, chn, int(thisSize[0]/step), int(thisSize[1]/step)) , dtype=np.float32)

    batch_count = 0
    results = []
    chosenPackList = []

    for idx in range(total_testing):
        crop_patch_slice, place_slice, output_slice, _ = PackList[idx]
        if grid_probs_bag is None or grid_probs_bag[(crop_patch_slice[0].__reduce__(), 
                                                     crop_patch_slice[1].__reduce__())] == 1 :
            chosenPackList.append(PackList[idx])
            this_patch = imgs[:,crop_patch_slice[0], crop_patch_slice[1]][:, 0::step, 0::step]
            BatchData[batch_count] = this_patch
            batch_count += 1
            if batch_count == batch_size or idx == total_testing - 1:
                data = BatchData[0:batch_count]
                with torch.no_grad():
                    data_variable = Variable(torch.from_numpy(data).float())
                    if cls.parameters().__next__().is_cuda:
                        data_variable = data_variable.cuda(cls.parameters().__next__().get_device())
                    
                results.append(cls.forward(data_variable).cpu().data.numpy() )
                batch_count = 0

    idx = 0
    for each_result in results:
        for b_idx in range(each_result.shape[0]):
            _, place_slice, output_slice, _ = chosenPackList[idx]
            outputs[:, place_slice[0], place_slice[1]] = each_result[b_idx, :,output_slice[0], output_slice[1]]
            idx += 1
    
    return outputs[:,0:org_row, 0:org_col]



# def split_testing_unet( cls, imgs,  batch_size = 4, windowsize=512, 
#                         board = 40, grid_probs_bag = None, step=1):
    
#     adptive_batch_size=False # cause we dont need it for fixed windowsize.

#     chn, org_row, org_col = imgs.shape
#     pad_row, pad_col = max(0, windowsize-org_row), max(0, windowsize-org_col)
#     if pad_row > 0 or pad_col > 0:
#         imgs = np.pad(imgs, ((0,0), (0, pad_row), (0, pad_col)), mode='constant')


#     chn, row_size, col_size = imgs.shape
#     assert len(imgs.shape) == 3, 'wrong input dimensio of input image'
#     outputs    = np.zeros((1, row_size, col_size), dtype=np.float32) # we don't consider multiple outputs
#     print('Finish allocate outputs')
    
#     PackList   =   split_index(row_size, col_size, windowsize = windowsize, 
#                                board = board, fixed_window= True, 
#                                step_size=None)
#     print('Finished splitting')
#     total_testing = len(PackList)
#     _,_,_, thisSize = PackList[0]

#     BatchData = np.zeros((batch_size, chn, thisSize[0], thisSize[1]) , dtype=np.float32)

#     batch_count = 0
#     results = []
#     for idx in range(total_testing):
#         crop_patch_slice, place_slice, output_slice, _ = PackList[idx]
#         BatchData[batch_count] = imgs[:,crop_patch_slice[0],crop_patch_slice[1]]
#         batch_count += 1
#         if batch_count == batch_size or idx == total_testing - 1:
#             data = BatchData[0:batch_count]
#             with torch.no_grad():
#                 data_variable = Variable(torch.from_numpy(data).float())
#                 if cls.parameters().__next__().is_cuda:
#                     data_variable = data_variable.cuda(cls.parameters().__next__().get_device())
                
#             results.append(cls.forward(data_variable).cpu().data.numpy() )
#             batch_count = 0

#     idx = 0
#     for each_result in results:
#         for b_idx in range(each_result.shape[0]):
#             _, place_slice, output_slice, _ = PackList[idx]
#             outputs[:, place_slice[0], place_slice[1]] = each_result[b_idx, :,output_slice[0], output_slice[1]]
#             idx += 1
    
#     return outputs[:,0:org_row, 0:org_col]

#------------

def get_coordinate( voting_map, threshhold = 0.1, min_len=5):
    voting_map[voting_map < threshhold*np.max(voting_map[:])] = 0
    coordinates = peak_local_max(voting_map, min_distance= min_len, indices = True) # N by 2
    if coordinates.size == 0:
        coordinates = None # np.asarray([])
        return coordinates  
    
    boxes_list = [coordinates[:, 1:2], coordinates[:, 0:1], coordinates[:, 1:2], coordinates[:, 0:1]]
    coordinates = np.concatenate(boxes_list, axis=1)    
    return coordinates


