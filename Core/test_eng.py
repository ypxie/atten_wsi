# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import torch
import time
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix

from .proj_utils.torch_utils import to_device, load_partial_state_dict


def test_cls(dataloader, model_root, mode_name, net, args):
    net.eval()
    model_path = os.path.join(model_root, "BestModel", args.model_path)
    assert os.path.exists(model_path), "Given model doesnot exist"
    print("Loaded model is {}".format(args.model_path))

    weights_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    load_partial_state_dict(net, weights_dict)

    start_timer = time.time()
    total_pred, total_gt = [], []
    for test_data, test_aux, test_label, test_num in dataloader:
        test_data  =  to_device(test_data, net.device_id, volatile=True).float()
        test_aux   =  to_device(test_aux, net.device_id, volatile=True).float()
        test_num   =  to_device(test_num, net.device_id, volatile=True).long()

        test_pred_pro    = net(test_data, test_aux, true_num = test_num).cpu()
        _, cls_labels  = torch.topk(test_pred_pro, 1, dim=1)
        cls_labels    =  cls_labels.data.cpu().numpy()[:,0]

        total_pred.extend(cls_labels.tolist()  )
        total_gt.extend(test_label.tolist() )

    precision, recall, fscore, support = score(total_gt, total_pred)
    con_mat = confusion_matrix(total_gt, total_pred)

    print(' p:  {}\n r:  {}\n f1: {} \n'.format(precision, recall, fscore))
    print('confusion matrix: \n')
    print(con_mat)
    cls_acc = np.trace(con_mat) * 1.0 / np.sum(con_mat)
    print("Final classification accuracy is: {}".format(cls_acc))

    total_time = time.time()-start_timer
    print("It takes {} to finish testing on {} slides.".format(total_time, len(dataloader)))
