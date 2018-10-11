import os
import cv2
import torch
import numpy as np
import datetime, time
from random import shuffle
from torch.multiprocessing import Pool
from torch.autograd import Variable
from torch.optim import lr_scheduler

from .proj_utils.plot_utils import plot_scalar, plot_img, save_images
from .proj_utils.torch_utils import to_device, LambdaLR
from .proj_utils.torch_utils import adding_grad_noise, load_partial_state_dict


def train_cls(dataloader, val_dataloader, model_root, mode_name, net, args):
    net.train()
    frozen_step = getattr(args, 'frozen_step', 0)
    lr = args.lr
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=args.momentum,
                                nesterov=True, weight_decay=args.weight_decay)

    loss_train_plot   = plot_scalar(name="loss_cls",  env=mode_name, rate=args.display_freq)
    model_folder      = os.path.join(model_root, mode_name + str(args.session))
    if os.path.exists(model_folder) == False:
        os.makedirs(model_folder)


    if args.reuse_weights:
        weightspath = os.path.join(model_folder, args.reuse_model_path)
        if os.path.exists(weightspath):
            weights_dict = torch.load(weightspath, map_location=lambda storage, loc: storage)
            print('reload weights from {}'.format(weightspath))
            load_partial_state_dict(net, weights_dict)
            start_epoch = args.load_from_epoch + 1
        else:
            print('WRANING!!! {} do not exist!!'.format(weightspath))
            start_epoch = 1
    else:
        start_epoch = 1

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lr_lambda=LambdaLR(args.maxepoch, start_epoch, args.decay_epoch).step)
    train_loss = 0

    step_cnt = 0 # will be zeroed after a while
    batch_count = 0
    init_flag = True
    cls_acc = 0.0

    for epoc_num in range(start_epoch, args.maxepoch):
        #dataloader.dataset.img_shape = [args.img_size[ridx], args.img_size[ridx]]
        start_timer = time.time()

        #import pdb; pdb.set_trace()
        for batch_idx, (batch_data, batch_aux, gt_classes, true_num) in enumerate(dataloader):
            if batch_count < 3:
                pass
            else:
                _batch_data, _batch_aux, _gt_classes, _true_num = batch_data, batch_aux, gt_classes, true_num

                if batch_count > frozen_step and init_flag:
                    for p in net.parameters():
                        p.requires_grad = True  # to resume computation
                    init_flag = False
                    print('Set the training back to normal')
                im_data   = to_device(_batch_data, net.device_id).float()
                aux_data  = to_device(_batch_aux, net.device_id).float()
                im_label  = to_device(_gt_classes, net.device_id, requires_grad=False).long()
                num_data  = to_device(_true_num, net.device_id, requires_grad=False).long()

                im_label = im_label.view(-1, 1)
                # add noise to input data

                train_pred, assignments = net(im_data, aux_data, im_label, true_num=num_data)

                vecloss = net.loss
                loss = torch.mean(vecloss)
                n_data = im_data.size()[0]

                num_sample  = im_data.size()[0]
                #--------------------------------------------------------
                # backward
                train_loss_val = loss.data.cpu().item()
                train_loss  += train_loss_val

                optimizer.zero_grad()
                loss.backward()

                adding_grad_noise(net, 1, time_step=max(batch_count, epoc_num))
                optimizer.step()
                step_cnt += 1

                if batch_count % args.display_freq == 0:
                    train_loss /= step_cnt
                    print((' epoch {}/[{}/{}], loss: {}, sample_{}, learning rate: {}'. \
                            format(epoc_num, batch_idx, len(dataloader)  ,train_loss, num_sample, optimizer.param_groups[0]['lr'])))
                    print('scale is: ', net.atten.scale)
                    net.eval()
                    from sklearn.metrics import precision_recall_fscore_support as score
                    from sklearn.metrics import confusion_matrix
                    total_pred, total_gt = [], []

                    for val_data, val_aux, val_label, val_num in val_dataloader:
                        val_data  =  to_device(val_data, net.device_id, volatile=True).float()
                        val_aux   =  to_device(val_aux, net.device_id, volatile=True).float()
                        val_num   =  to_device(val_num, net.device_id, volatile=True).long()

                        val_pred_pro, assignments = net(val_data, val_aux, true_num = val_num)
                        val_pred_pro = val_pred_pro.cpu()
                        _, cls_labels  = torch.topk(val_pred_pro, 1, dim=1)
                        cls_labels    =  cls_labels.data.cpu().numpy()[:,0]

                        total_pred.extend(cls_labels.tolist()  )
                        total_gt.extend(val_label.tolist() )

                    precision, recall, fscore, support = score(total_gt, total_pred)
                    con_mat = confusion_matrix(total_gt, total_pred)

                    print(' p:  {}\n r:  {}\n f1: {} \n'.format(precision, recall, fscore))
                    print('confusion matrix: \n')
                    print(con_mat)

                    cls_acc = np.trace(con_mat) * 1.0 / np.sum(con_mat)
                    print("Current classification accuracy is: {}".format(cls_acc))

                    net.train()
                    plot_img(X=im_data[0][0].data.cpu().numpy(), win='cropped_img', env=mode_name)
                    train_loss = 0
                    step_cnt = 0

                    loss_train_plot.plot(train_loss_val)

            batch_count += 1
        lr_scheduler.step()

        if epoc_num > 0 and epoc_num % args.save_freq == 0:
            save_model_name = '{}-epoch-{}-acc-{:.3f}.pth'.format(args.model_name, str(epoc_num).zfill(3), cls_acc)
            torch.save(net.state_dict(), os.path.join(model_folder, save_model_name))
            print('Model saved as {}'.format(save_model_name))

        total_time = time.time()-start_timer
        print('it takes {} to finish one epoch.'.format(total_time))
