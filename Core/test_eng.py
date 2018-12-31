# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import torch
import time
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix

from .proj_utils.torch_utils import load_partial_state_dict


def test_cls(dataloader, model_root, net, args):
    net.eval()
    model_path = os.path.join(model_root, "BestModel", args.model_path)
    assert os.path.exists(model_path), "Given model doesnot exist"
    print("Loaded model is {}".format(args.model_path))

    weights_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    load_partial_state_dict(net, weights_dict)

    start_timer = time.time()
    total_pred, total_gt = [], []
    for ind, data in enumerate(dataloader):
        if args.pre_load == True:
            test_data, test_label, test_num = data
        else:
            test_data, test_label, test_num, gt_bboxes = data

        test_data = test_data.cuda().float()
        test_num = test_num.cuda().long()
        # test_data  =  to_device(test_data, net.device_id, volatile=True).float()
        # test_num   =  to_device(test_num, net.device_id, volatile=True).long()

        test_pred_pro, assignments = net(test_data, true_num = test_num)
        # Generate ROI bbox in whole slide image
        if args.pre_load == False:
            topk_num = 10
            top_probs, top_inds = torch.topk(assignments, 10, dim=1)
            topk_assign = top_inds.cpu().numpy()
            heat_bboxes = np.zeros((gt_bboxes.shape[0], topk_num, 4))
            for ind in range(gt_bboxes.shape[0]):
                heat_bboxes[ind] = np.take(gt_bboxes[ind], topk_assign[ind], axis=0)
        test_pred_pro = test_pred_pro.cpu()
        _, cls_labels  = torch.topk(test_pred_pro, 1, dim=1)
        cls_labels    =  cls_labels.data.cpu().numpy()[:,0]

        total_pred.extend(cls_labels.tolist())
        total_gt.extend(test_label.tolist())

    precision, recall, fscore, support = score(total_gt, total_pred)
    con_mat = confusion_matrix(total_gt, total_pred)

    print(' p:  {}\n r:  {}\n f1: {} \n'.format(precision, recall, fscore))
    print('confusion matrix: \n')
    print(con_mat)
    cls_acc = np.trace(con_mat) * 1.0 / np.sum(con_mat)
    print("Final classification accuracy is: {:.3f}".format(cls_acc))

    total_time = time.time()-start_timer
    print("It takes {} to finish testing on {} slides.".format(total_time, len(dataloader)))
