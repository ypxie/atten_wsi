# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import torch
import time, json
import deepdish as dd
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical

from .proj_utils.torch_utils import load_partial_state_dict


def test_cls(dataloader, model_root, net, args):
    net.eval()
    model_path = os.path.join(model_root, "BestModel", args.model_path)
    assert os.path.exists(model_path), "Given model doesnot exist"
    print("Loaded model is {}".format(args.model_path))

    weights_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    load_partial_state_dict(net, weights_dict)

    start_timer = time.time()
    test_gt_pred = {}
    total_pred, total_pred_probs, total_gt = [], [], []
    total_names, total_feas = [], []

    for ind, data in enumerate(dataloader):
        if args.pre_load == True:
            test_data, test_label, test_num = data
        else:
            test_data, test_label, test_num, gt_bboxes = data

        test_data = test_data.cuda().float()
        test_num = test_num.cuda().long()

        test_pred_pro, assignments, fusionFea = net(test_data, true_num = test_num)

        # Generate ROI bbox in whole slide image
        if args.pre_load == False:
            topk_num = test_num.item()
            top_probs, top_inds = torch.topk(assignments, topk_num, dim=1)
            topk_assign = top_inds.cpu().numpy()
            heat_bboxes = np.zeros((gt_bboxes.shape[0], topk_num, 4))
            for ind in np.arange(gt_bboxes.shape[0]):
                heat_bboxes[ind] = np.take(gt_bboxes[ind], topk_assign[ind], axis=0)
            # BBoxes
            probs, att_vals = [], assignments[0][topk_assign]
            for ind in np.arange(topk_num):
                probs.append(att_vals[ind].item())
            att_dict = {
                "probs": probs,
                "bboxes": heat_bboxes[0]
            }
            cur_filename = os.path.splitext(dataloader.dataset.cur_filename)[0]
            save_path = os.path.join(os.path.dirname(model_path), cur_filename + ".h5")
            # dd.io.save(save_path, att_dict)
            total_names.append(cur_filename)

        test_pred_pro = test_pred_pro.cpu()
        _, cls_labels  = torch.topk(test_pred_pro, 1, dim=1)
        cls_labels    =  cls_labels.data.cpu().numpy()[:,0]
        cur_pred_probs = test_pred_pro.detach().numpy()[0].tolist()
        total_feas.append(fusionFea.cpu().detach().numpy()[0].tolist())

        batch_cls_labels = cls_labels.tolist()
        batch_gt_labels = test_label.tolist()
        total_pred.extend(batch_cls_labels)
        total_pred_probs.append(cur_pred_probs)
        total_gt.extend(batch_gt_labels)


    if args.pre_load == False:
        json_path = os.path.join(os.path.dirname(model_path), "pred_gt.json")
        test_gt_pred['gts'] = to_categorical(total_gt).tolist()
        test_gt_pred['preds'] = total_pred_probs
        test_gt_pred['feas'] = total_feas
        test_gt_pred['names'] = total_names
        with open(json_path, 'w') as fp:
            json.dump(test_gt_pred, fp)

    precision, recall, fscore, support = score(total_gt, total_pred)
    con_mat = confusion_matrix(total_gt, total_pred)
    print(' p:  {}\n r:  {}\n f1: {} \n'.format(precision, recall, fscore))
    print('confusion matrix: \n')
    print(con_mat)
    cls_acc = np.trace(con_mat) * 1.0 / np.sum(con_mat)
    print("Final classification accuracy is: {:.3f}".format(cls_acc))

    # total_time = time.time()-start_timer
    # print("It takes {} to finish testing on {} slides.".format(total_time, len(dataloader)))
