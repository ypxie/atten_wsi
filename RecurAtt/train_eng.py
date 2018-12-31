import os, sys
import numpy as np
import shutil
from random import shuffle
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix

import torch
from torch.multiprocessing import Pool
from torch.autograd import Variable
from torch.optim import lr_scheduler

# from .proj_utils.plot_utils import plot_scalar, plot_img, save_images
from .proj_utils.torch_utils import LambdaLR

def train_cls(dataloader, val_dataloader, model_root, mode_name, net, args):
    net.train()
    model_folder = os.path.join(model_root, mode_name + str(args.session))
    if os.path.exists(model_folder) == True:
        shutil.rmtree(model_folder)
    os.makedirs(model_folder)

    start_epoch = 1
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                                nesterov=True, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lr_lambda=LambdaLR(args.maxepoch, start_epoch, args.decay_epoch).step)
    # loss_train_plot   = plot_scalar(name="loss_cls",  env=mode_name, rate=args.display_freq)

    train_loss, step_cnt, batch_count  = 0.0, 0, 0
    cls_acc, best_acc = 0.0, 0.0
    for epoc_num in np.arange(start_epoch, args.maxepoch):
        for batch_idx, (batch_data, gt_classes, true_num) in enumerate(dataloader):
            batch_data, gt_classes, true_num = batch_data, gt_classes, true_num
            im_data   = batch_data.cuda().float()
            im_label  = gt_classes.cuda().long()
            num_data  = true_num.cuda().long()

            im_label = im_label.view(-1, 1)
            train_pred, assignments = net(im_data, im_label, true_num=num_data)

            vecloss = net.loss
            loss = torch.mean(vecloss)
            n_data = im_data.size()[0]
            num_sample  = im_data.size()[0]
            train_loss_val = loss.data.cpu().item()
            train_loss  += train_loss_val
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_cnt += 1
            batch_count += 1
            if batch_count % args.display_freq == 0:
                train_loss /= step_cnt
                print((' epoch {}/[{}/{}], loss: {}, sample_{}, learning rate: {:.5f}'. \
                        format(epoc_num, batch_idx, len(dataloader)  ,train_loss, num_sample, \
                        optimizer.param_groups[0]['lr'])))

                net.eval()
                total_pred, total_gt = [], []
                for val_data, val_label, val_num in val_dataloader:

                    val_data  =  val_data.cuda().float()
                    val_num   =  val_num.cuda().long()

                    val_pred_pro, assignments = net(val_data, true_num = val_num)
                    val_pred_pro = val_pred_pro.cpu()
                    _, cls_labels  = torch.topk(val_pred_pro, 1, dim=1)
                    cls_labels    =  cls_labels.data.cpu().numpy()[:,0]

                    total_pred.extend(cls_labels.tolist())
                    total_gt.extend(val_label.tolist())
                precision, recall, fscore, support = score(total_gt, total_pred)
                con_mat = confusion_matrix(total_gt, total_pred)
                print(' p:  {}\n r:  {}\n f1: {} \n'.format(precision, recall, fscore))
                print('confusion matrix:')
                print(con_mat)
                cls_acc = np.trace(con_mat) * 1.0 / np.sum(con_mat)
                print("\n Current classification accuracy is: {:.4f}".format(cls_acc))
                # plot_img(X=im_data[0][0].data.cpu().numpy(), win='cropped_img', env=mode_name)
                train_loss, step_cnt = 0, 0
                net.train()
                # loss_train_plot.plot(train_loss_val)
        lr_scheduler.step()
        if epoc_num > 100 and epoc_num % args.save_freq == 0 and cls_acc >= best_acc:
            save_model_name = '{}-epoch-{}-acc-{:.3f}.pth'.format(args.fea_mix, str(epoc_num).zfill(3), cls_acc)
            torch.save(net.state_dict(), os.path.join(model_folder, save_model_name))
            print('Model saved as {}'.format(save_model_name))
            best_acc = cls_acc

        if (epoc_num + 1) == args.maxepoch:
            save_model_name = '{}-epoch-{}-acc-{:.3f}.pth'.format(args.fea_mix, str(epoc_num).zfill(3), cls_acc)
            torch.save(net.state_dict(), os.path.join(model_folder, save_model_name))            