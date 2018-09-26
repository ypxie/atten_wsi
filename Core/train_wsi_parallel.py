import os
import cv2
import torch
import numpy as np
import datetime, time
from random import shuffle
from torch.multiprocessing import Pool
from torch.autograd import Variable

from .reply_buffer.reply_buffer import PrioritizedReplayBuffer as replyBuffer
from .proj_utils.plot_utils import plot_scalar, plot_img, save_images
from .proj_utils.torch_utils import *
from .proj_utils.local_utils import *
from .proj_utils.model_utils import resize_layer
from .train_utils import adding_grad_noise, load_partial_state_dict

def train_cls(dataloader, test_dataloader, model_root, mode_name, net, args):
    net.train()
    forzen_step = getattr(args, 'frozen_step', 2000)
    lr = args.lr
    reply_buffer = replyBuffer(args.buffer_size, alpha=1)

    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=args.momentum,
                                nesterov=True, weight_decay=args.weight_decay)
    #optimizer  = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.999))

    loss_train_plot   = plot_scalar(name = "loss_cls",  env= mode_name, rate = args.display_freq)
    model_folder    = os.path.join(model_root, mode_name)
    mkdirs([model_folder])

    if args.reuse_weights :
        weightspath = os.path.join(model_folder, 'weights_epoch_{}.pth'.format(args.load_from_epoch))
        if os.path.exists(weightspath):
            weights_dict = torch.load(weightspath, map_location=lambda storage, loc: storage)
            print('reload weights from {}'.format(weightspath))
            load_partial_state_dict(net, weights_dict)
            #net.load_state_dict(weights_dict)
            start_epoch = args.load_from_epoch + 1
            #if os.path.exists(plot_save_path):
            #    plot_dict = torch.load(plot_save_path)
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
    using_replay = False
    for epoc_num in range(start_epoch, args.maxepoch):
        #ridx = random.randint(0, len(args.img_size)-1 )
        #dataloader.dataset.img_shape = [args.img_size[ridx], args.img_size[ridx]]
        start_timer = time.time()
        
        #import pdb; pdb.set_trace()
        for batch_idx, (batch_data, batch_aux, gt_classes, true_num) in enumerate(dataloader) :
            #print('data and label: ', batch_data.size(), gt_classes)
            #if np.random.random() > 0.5:
            #    batch_data += 0.2/min(epoc_num**0.5, 10) *torch.randn_like(batch_data) 
            if using_replay:
                for iidx in range(0, batch_data.size()[0]):
                    _data = (batch_data[iidx], batch_aux[iidx], 
                                gt_classes[iidx], true_num[iidx], 0)
                    reply_buffer.add(_data, priority=None)

            if batch_count < 3:
                pass
                #import pdb; pdb.set_trace()
            else:
                if using_replay:
                    _buffer = reply_buffer.sample(args.batch_size*2, beta=1)
                    (_batch_data, _batch_aux, _gt_classes, 
                    _true_num, _, weights, idxes)   = _buffer
                else:
                    _batch_data, _batch_aux, _gt_classes, _true_num = batch_data, batch_aux, gt_classes, true_num
                #batch_data  = np.concatenate( [batch_data, _batch_data], axis=0 ) 
                #batch_aux   = np.concatenate( [batch_aux, _batch_aux], axis=0 )
                #gt_classes  = np.concatenate( [gt_classes, _gt_classes], axis=0 )
                #true_num    = np.concatenate( [true_num, _true_num], axis=0 )

                if batch_count > forzen_step and init_flag:
                    for p in net.parameters(): p.requires_grad = True  # to resume computation
                    init_flag = False
                    print('Set the training back to normal')
                    #import pdb; pdb.set_trace()
                im_data   = to_device(_batch_data, net.device_id).float()
                aux_data  = to_device(_batch_aux, net.device_id).float()
                im_label  = to_device(_gt_classes, net.device_id, requires_grad=False).long()
                num_data  = to_device(_true_num, net.device_id, requires_grad=False).long()
                #print('loading data time {}'.format(time.time() - start_timer) )
                
                #im_label = torch.unsqueeze(im_label, dim=1)
                im_label = im_label.view(-1, 1)
                # add noise to input data


                train_pred = net(im_data, aux_data, im_label, true_num=num_data)
                
                vecloss = net.loss
                loss = torch.mean(vecloss)

                n_data = im_data.size()[0]

                if using_replay:
                    vecloss_val = vecloss.cpu().data.numpy()+1e-10
                    vecloss_val = vecloss_val.tolist()
                    reply_buffer.update_priorities(idxes, vecloss_val)
                #for iidx in range(0, n_data):
                #    _data = (batch_data[iidx], batch_aux[iidx], gt_classes[iidx], true_num[iidx], 0)
                #    this_priority = vecloss_val[iidx]
                #    reply_buffer.add(_data, priority=this_priority)
                    # (obs_t, action, reward, obs_tp1, done), priority

                #--------------------------------------------------------
                #         Update the priority queue here
                #--------------------------------------------------------



                #probs = net(im_data)
                #loss  = F.nll_loss(F.log_softmax(probs), im_label) #, weight=cls_wgt
                num_sample  = im_data.size()[0]
                #--------------------------------------------------------
                # backward
                train_loss_val = loss.data.cpu().item()
                
                train_loss  += train_loss_val

                optimizer.zero_grad()
                loss.backward()
                #import pdb; pdb.set_trace()
                adding_grad_noise(net, 1, time_step = max(batch_count, epoc_num)  )

                optimizer.step()
                
                step_cnt += 1

                if batch_count % args.display_freq == 0:
                    train_loss /= step_cnt

                    end_timer = time.time()
                    
                    print((' epoch {}/[{}/{}], loss: {}, sample_{}, learning rate: {}'. \
                            format(epoc_num, batch_idx, len(dataloader)  ,train_loss, num_sample, optimizer.param_groups[0]['lr'])))
                    print('scale is: ', net.atten.scale)
                    net.eval()
                    from sklearn.metrics import precision_recall_fscore_support as score
                    from sklearn.metrics import confusion_matrix

                    total_pred, total_gt = [], []
                    
                    #for batch_idx, (batch_data, batch_aux, gt_classes) in enumerate(dataloader) :

                    for test_data, test_aux, test_label, test_num in test_dataloader:
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
                    
                    
                    _, cls_train  = torch.topk(train_pred, 1, dim=1)
                    print('last train_pred: ', cls_train.view(-1).cpu().data.numpy().tolist() )
                    print('last train_gt:   ', im_label.view(-1).cpu().data.numpy().tolist())
                    
                    print('predciton: ', total_pred )
                    print('groundtrt: ', total_gt)
                    #print('pos num {}, neg num {}\n'.format(sum(total_gt), len(total_gt)-sum(total_gt)  ))
                    print(' p:  {}\n r:  {}\n f1: {} \n'.format(precision, recall, fscore))
                    print('confusion matrix: \n')
                    print(con_mat)
                        
                    net.train()
                    
                    plot_img(X=im_data[0][0].data.cpu().numpy(), win='cropped_img', env=mode_name)
                    #org_img_contours = mark_contours(orgin_im[0][:,:], res_mat[0])
                    #plot_img(X=orgin_im[0], win='original image', env=mode_name)
                    train_loss = 0
                    #bbox_loss, iou_loss, cls_loss = 0., 0., 0.
                    step_cnt = 0 

                    loss_train_plot.plot(train_loss_val)   

                    torch.save(net.state_dict(), os.path.join(model_folder, 'weights_tmp.pth') )
                    print('finish saving tmp weights at {}'.format(model_folder))

            batch_count += 1
        lr_scheduler.step()
        
        if epoc_num>0 and epoc_num % args.save_freq == 0:
            torch.save(net.state_dict(), os.path.join(model_folder, 'weights_epoch_{}.pth'.format(epoc_num)))
            print('finish saving weights at {}'.format(model_folder))

        total_time = time.time()-start_timer
        print('it takes {} to finish one epoch.'.format(total_time))            
        