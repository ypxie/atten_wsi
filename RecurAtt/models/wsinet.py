# -*- coding: utf-8 -*-

import os, sys

import numpy as np
from numba import jit
import torch
import torch.nn as nn
import torch.nn.functional as F


@jit(nopython=True)
def get_mask(B, N, true_num=None):
    '''
    Parameters:
    ------------
        B@int: batch size
        N@int: length of N
        true_num: np array of int, of shape (B,)
    Returns:
    ------------
        mask: of type np.bool, of shape (B, N). 1 indicates valid, 0 invalid.
    '''
    dis_ = np.ones((B, N), dtype=np.int32)

    if true_num is not None:
        for idx in range(B):
            this_num = true_num[idx]
            if this_num < N:
                dis_[idx, this_num::] = 0
    return dis_


@jit(nopython=True)
def get_weight(preds, label, weight_mat):
    # out B X 1
    # LABEL: B x 1
    # weight_mat C * C
    weight_vec = np.zeros((preds.shape[0], ))
    for i in range(weight_vec.shape[0]):
        gt, pred = label[i], preds[i]
        weight_vec[i] = weight_mat[gt][pred]

    return weight_vec


class MILAtten(nn.Module):
    """MILAtten layer implementation"""

    def __init__(self, dim=128, dl=128, fea_mix="global"):
        """
        Args:
            dim : int
                Dimension of descriptors
        """
        super(MILAtten, self).__init__()

        self.fea_mix = fea_mix
        self.dim = dim
        self.out_dim = dim
        self.dl = dl

        if self.fea_mix == "global":
            self.atten_dim = self.dim * 2
        elif self.fea_mix == "self":
            self.atten_dim = self.dim
        else:
            print("Unknow fea fusion mode")
            raise

        self.V = nn.Parameter(torch.Tensor(self.atten_dim, self.dl), requires_grad=True)
        self.W = nn.Parameter(torch.Tensor(self.dl, 1), requires_grad=True)
        self.scale=1
        self.reset_params()

    def reset_params(self):
        std1 = 1./((self.dl*self.dim)**(1/2))
        self.V.data.uniform_(-std1, std1)

        std2 = 1./((self.dl)**(1/2))
        self.W.data.uniform_(-std2, std2)


    def forward(self, x, true_num=None, recur_steps=1):
        '''
        Parameters:
        -----------
            x: B x N x D
            true_num: B
        Return
            feat_fusion:
                Bxfeat_dim
            soft_assign
                BxN
        '''
        B, num_dis, D = x.size()
        if true_num is not None:
            _t_num = true_num.cpu().numpy().tolist()
            _mask  = get_mask(B, num_dis, _t_num)
        else:
            _mask  = np.ones((B, num_dis), dtype=np.int32)
        device_mask = x.new_tensor(_mask)

        if self.fea_mix == 'global':
            if true_num is not None:
                x_sum       =  torch.sum(x, dim=1, keepdim=True) # B x 1 x D
                _num_array  =  true_num.unsqueeze(-1).unsqueeze(-1).expand_as(x_sum)
                x_mean      =  x_sum/_num_array.float()
            else:
                x_mean      =  torch.mean(x, dim=1, keepdim=True) # B x 1 x D

            feat_ = x

            x_atten = x_mean # initialized as x_mean
            for step in range(recur_steps):
                _atten_feat =  torch.cat( [x_atten.expand_as(x), x] , dim=-1)
                x_   = torch.tanh(torch.matmul(_atten_feat,  self.V )) # BxNxL used to be torch.tanh
                dis_ = torch.matmul(x_, self.W).squeeze(-1) # BxN
                dis_ = dis_/np.sqrt(self.dl)

                # set unwanted value to 0, so it won't affect.
                dis_.masked_fill_(device_mask==0, -1e20)
                soft_assign_ = F.softmax(dis_, dim=1) # BxN

                soft_assign = soft_assign_.unsqueeze(-1).expand_as(feat_)  # BxNxD
                x_atten = torch.sum(soft_assign*feat_, 1, keepdim=True) # Bx1XD

            feat_fusion = torch.squeeze(x_atten, dim=1)
            return feat_fusion, soft_assign_
        elif self.fea_mix == "self":
            feat_ = x
            x_   = torch.tanh(torch.matmul(x,  self.V)) # BxNxL used to be torch.tanh
            dis_ = torch.matmul(x_, self.W).squeeze(-1) # BxN
            dis_ = dis_/np.sqrt(self.dl)

            # set unwanted value to 0, so it won't affect.
            dis_.masked_fill_(device_mask==0, -1e20)
            soft_assign_ = F.softmax(dis_, dim=1) # BxN

            soft_assign = soft_assign_.unsqueeze(-1).expand_as(feat_)  # BxNxD
            feat_fusion = torch.sum(soft_assign*feat_, 1, keepdim=False) # BxD

            return feat_fusion, soft_assign_
        else:
            raise NotImplementedError()


def batch_fea_pooling(feas, fea_num):
    batch_size = len(fea_num)
    assignments = torch.cuda.FloatTensor(batch_size, feas.shape[1]).fill_(0)
    vlad = torch.cuda.FloatTensor(batch_size, feas.shape[2]).fill_(0)
    for ip in range(batch_size):
        patch_num = fea_num[ip]
        assignments[ip][:patch_num].fill_(1.0/patch_num.item())
        vlad[ip] = torch.mean(feas[ip][:patch_num], dim=0)

    return vlad, assignments


class WsiNet(nn.Module):
    def __init__(self, class_num, in_channels, patch_mix="att", fea_mix="global", recur_steps=4,
                 num_mlp_layer=1, use_w_loss=False, dataset="Thyroid"):
        super(WsiNet, self).__init__()

        self.patch_mix = patch_mix
        self.in_channels = in_channels
        self.fea_mix = fea_mix
        self.recur_steps = recur_steps
        self.num_mlp_layer = num_mlp_layer
        self.use_w_loss = use_w_loss
        self.dataset = dataset

        # self.register_buffer('device_id', torch.IntTensor(1))
        self.atten = MILAtten(dim=in_channels, dl=64, fea_mix=self.fea_mix)
        self.fc = nn.Linear(in_features=self.atten.out_dim, out_features=class_num)
        # self.fc1 = nn.Linear(in_features=self.atten.out_dim, out_features=128)
        # self.fc2 = nn.Linear(in_features=128, out_features=class_num)

        self._loss = 0

        # Predefined weighted matrix for loss calculation
        self.weight_thyroid_mat = np.array([[0.1, 0.5, 1.0],
                                            [1.0, 0.1, 2.0],
                                            [2.0, 0.5, 0.1]])

        self.weight_mucosa_mat = np.array([[0.1, 0.6, 1.0, 0.3],
                                           [1.0, 0.1, 1.0, 0.3],
                                           [1.0, 0.6, 0.1, 0.3],
                                           [1.0, 1.0, 1.0, 0.1]])


    def forward(self, x, label=None, true_num=None):
        B, N, C = x.size()

        if self.patch_mix == "att":
            vlad, assignments = self.atten(x, true_num, recur_steps=self.recur_steps)
        elif self.patch_mix == "pool":
            vlad, assignments = batch_fea_pooling(x, true_num)
        else:
            raise NotImplementedError()

        if self.num_mlp_layer == 1:
            out = F.dropout(vlad, training=self.training)
            out = self.fc(out)
        # elif self.num_mlp_layer == 2:
        #     out = self.fc1(vlad)
        #     out = F.dropout(out, training=self.training)
        #     out = self.fc2(out)
        else:
            raise NotImplementedError()

        if self.training:
            assert label is not None, "invalid label for training mode"
            self._loss = F.nll_loss(F.log_softmax(out, dim=1), label.view(-1),
                                     reduction='none') # B x 1
            if self.use_w_loss == True:
                if self.dataset == "Thyroid":
                    weight_mat = self.weight_thyroid_mat
                elif self.dataset == "Mucosa":
                    weight_mat = self.weight_mucosa_mat
                else:
                    raise NotImplementedError()

                out_numpy = np.argmax(out.data.cpu().numpy(), axis=1)
                label_numpy = label.data.cpu().numpy().squeeze()
                this_weight = get_weight(out_numpy, label_numpy, weight_mat)  # B x 1
                this_weight_var = self._loss.new_tensor(torch.from_numpy(this_weight))
                self._loss = self._loss * this_weight_var
            return out, assignments
        else:
            cls_pred = F.softmax(out, dim=1)
            return cls_pred, assignments

    @property
    def loss(self):
        return self._loss
