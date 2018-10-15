# -*- coding: utf-8 -*-

import os, sys

import torch, math
from numba import jit
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from .attention import MultiHeadedAttention
import numpy as np


class MILAtten(nn.Module):
    """MILAtten layer implementation"""

    def __init__(self, dim=128, dl=128, use_self=None):
        """
        Args:
            dim : int
                Dimension of descriptors
        """
        super(MILAtten, self).__init__()

        self.use_self = use_self
        self.dim = dim
        self.out_dim = dim
        self.dl = dl

        if use_self is 'self_atten':
            self.atten_dim = 256
            self.f_linear = nn.Linear(self.dim, self.atten_dim)
            self.mh_atten = MultiHeadedAttention(h=4, d_model=self.atten_dim)
        elif use_self is 'global':
            self.atten_dim = self.dim * 2

        self.V = nn.Parameter(torch.Tensor(self.atten_dim, self.dl), requires_grad=True)
        self.W = nn.Parameter(torch.Tensor(self.dl, 1), requires_grad=True)
        self.scale=1
        self.reset_params()

    def reset_params(self):
        std1 = 1./((self.dl*self.dim)**(1/2))
        self.V.data.uniform_(-std1, std1)

        std2 = 1./((self.dl)**(1/2))
        self.W.data.uniform_(-std2, std2)

    def forward(self, x, true_num=None):
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

        if self.use_self is 'self_atten':
            self_atten_mask = torch.bmm(device_mask.unsqueeze(2), device_mask.unsqueeze(1))
            atten_x = self.f_linear(x)
            _atten_feat = self.mh_atten(atten_x, atten_x, atten_x, mask=self_atten_mask) # B x N x D
        elif self.use_self is 'global':
            if true_num is not None:
                x_sum       =  torch.sum(x, dim=1, keepdim=True) # B x 1 x D
                _num_array  =  true_num.unsqueeze(-1).unsqueeze(-1).expand_as(x_sum)
                x_mean      =  x_sum/_num_array.float()
            else:
                x_mean      =  torch.mean(x, dim=1, keepdim=True) # B x 1 x D

            _atten_feat =  torch.cat( [x_mean.expand_as(x), x] , dim=-1)
        else:
            _atten_feat = x
        feat_ = x

        x_   = torch.tanh(torch.matmul(_atten_feat,  self.V )) # BxNxL used to be torch.tanh
        dis_ = torch.matmul(x_, self.W).squeeze(-1) # BxN
        dis_ = dis_/math.sqrt(self.dl)

        # set unwanted value to 0, so it won't affect.
        dis_.masked_fill_(device_mask==0, -1e20)
        soft_assign_ = F.softmax(dis_, dim=1) # BxN

        soft_assign = soft_assign_.unsqueeze(-1).expand_as(feat_)  # BxNxD
        feat_fusion = torch.sum(soft_assign*feat_, 1, keepdim=False) # BxD

        return feat_fusion, soft_assign_


class logistWsiNet(nn.Module):
    def __init__(self, class_num, in_channels, use_self=None, use_aux=False):
        super(logistWsiNet, self).__init__()

        self.in_channels = in_channels
        self.use_self = use_self
        self.use_aux = use_aux

        self.register_buffer('device_id', torch.IntTensor(1))
        self.atten = MILAtten(dim=in_channels,  dl=64, use_self=self.use_self)

        layers = [
            nn.Dropout2d(0.5),
         ]
        self.conv1    = nn.Sequential(*layers)

        if self.use_aux:
            self.fc = nn.Linear(self.atten.out_dim+1, class_num, bias=True)
        else:
            self.fc = nn.Linear(self.atten.out_dim, class_num, bias=True)

        self._loss = 0

        # self.weight_mat = np.array([[0.1, 0.3, 2.0],
        #                             [0.7, 0.1, 1.0],
        #                             [3.0, 0.3, 0.1]])

        self.weight_mat = np.array([[0.1, 1.0, 1.0, 1.0],
                                    [1.0, 0.1, 1.0, 1.0],
                                    [1.0, 1.0, 0.1, 1.0],
                                    [1.0, 1.0, 1.0, 0.1]])

        # self.weight_mat = np.array([[0.1, 0.6, 1.0, 0.3],
        #                             [1.0, 0.1, 1.0, 0.3],
        #                             [1.0, 1.0, 0.1, 0.3],
        #                             [1.0, 1.0, 1.0, 0.1]])

    def forward(self, x, aux ,label=None, true_num= None):
        B, N, C = x.size()

        vlad, assignments = self.atten(x, true_num)
        out = self.conv1(vlad)

        if self.use_aux:
            aux = aux.view(B, 1)
            out = torch.cat([out, aux], dim=1)

        out = self.fc(out)

        if self.training:
            assert label is not None, "invalid label for training mode"
            self._loss = F.nll_loss(F.log_softmax(out, dim=1), label.view(-1),
                                     reduction='none') # B x 1
            out_numpy = np.argmax(out.data.cpu().numpy(), axis=1)
            label_numpy = label.data.cpu().numpy().squeeze()
            this_weight = get_weight(out_numpy, label_numpy, self.weight_mat)  # B x 1
            this_weight_var = self._loss.new_tensor(torch.from_numpy(this_weight))

            self._loss = self._loss * this_weight_var
            return out, assignments
        else:
            cls_pred = F.softmax(out, dim=1)
            return cls_pred, assignments

    @property
    def loss(self):
        return self._loss


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




class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, use_relu=True, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn   = nn.BatchNorm2d(out_channels, eps=0.001)
        self.use_relu = use_relu

    def forward(self, x):
        # change bn after relu
        x = self.conv(x)
        if self.use_relu:
            x = F.relu(x, inplace=True)
        else:
            x = F.leaky_relu(x, inplace=True)

        return self.bn(x)


class denseWsiNet(nn.Module):
    # input must be 299x299

    def __init__(self, class_num=2, in_channels=2, num_clusters=5,
                 multi_head=False, use_self=False , use_aux=False):
        super(denseWsiNet, self).__init__()

        self.in_channels = in_channels
        self.use_self = use_self
        self.use_aux = use_aux
        self.register_buffer('device_id', torch.IntTensor(1))
        #if multi_head:
        #    self.atten   = multHeadMILAtten(dim=in_channels, nhead = 4, dm = 32, dl = 32)
        #else:
        self.atten   = MILAtten(dim=in_channels,  dl = 64, use_self=use_self)

        self.conv1   =  BasicConv2d(self.atten.out_dim, 128, kernel_size=1, use_relu=False)
        self.conv2   =  BasicConv2d(128, 64, kernel_size=1, use_relu=False)
        self.conv3   =  BasicConv2d(192, 64, kernel_size=1, use_relu=False)
        self.conv4   =  BasicConv2d(256, 64, kernel_size=1, use_relu=False)
        #self.conv5   =  BasicConv2d(224, 32, kernel_size=1, use_relu=False)
        #self.conv6   =  BasicConv2d(256+5, 32, kernel_size=1, use_relu=False)
        if self.use_aux is True:
            self.out_conv = nn.Conv2d(320+1, class_num, kernel_size=1, bias=True)
        else:
            self.out_conv = nn.Conv2d(320, class_num, kernel_size=1, bias=True)
        self._loss = 0

    def forward(self, x, aux ,label=None, true_num= None):
        #import pdb; pdb.set_trace()
        B, N, C = x.size()[0:3]
        aux = aux.view(B, 1,1, 1)
        vlad, alpha = self.atten(x, true_num)
        vlad = vlad.unsqueeze(-1).unsqueeze(-1)

        conv1 = self.conv1(vlad)
        cout1_dense = torch.cat([conv1], 1)

        conv2 = self.conv2(cout1_dense)
        cout2_dense = torch.cat([conv1, conv2], 1)
        cout2_dense = F.dropout2d(cout2_dense, p=0.1, training=self.training)

        conv3 = self.conv3(cout2_dense)
        cout3_dense = torch.cat([conv1, conv2, conv3], 1)

        conv4 = self.conv4(cout3_dense)

        #import pdb; pdb.set_trace()
        if self.use_aux:
            cout4_dense = torch.cat([aux, conv1, conv2, conv3, conv4], 1)
        else:
            cout4_dense = torch.cat([conv1, conv2, conv3, conv4], 1)
        cout4_dense = F.dropout2d(cout4_dense, p=0.1, training=self.training)

        #conv5 = self.conv5(cout4_dense)
        #cout5_dense = torch.cat([aux, conv1, conv2, conv3, conv4, conv5], 1)

        out = self.out_conv(cout4_dense)
        out = out.squeeze(-1).squeeze(-1)

        if self.training:
            tensor_ = torch.tensor((0.2, 0.8), dtype=torch.float32)
            weights = out.new_tensor(tensor_)

            assert label is not None, "invalid label for training mode"
            self._loss = F.nll_loss(F.log_softmax(out, dim=1), label.view(-1),
                                    weight=weights, reduction='none')
            self._loss += nn.MultiMarginLoss(weight=weights,
                          reduction='none')(out, label.view(-1))

            return out
        else:
            cls_pred       = F.softmax(out, dim=1)
            return cls_pred

    @property
    def loss(self):
        return self._loss

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
