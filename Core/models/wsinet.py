import torch, math
from numba import jit
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from .attention import MultiHeadedAttention
import numpy as np

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

class wsiNet(nn.Module):
    # input must be 299x299
    
    def __init__(self, class_num=2, in_channels=2, num_clusters=5, use_self=None, use_aux=False):
        super(wsiNet, self).__init__()
        
        self.in_channels = in_channels
        self.use_self = use_self
        self.use_aux = use_aux
        self.register_buffer('device_id', torch.IntTensor(1))
        #self.netvlad = NetVLAD(num_clusters = num_clusters, dim = in_channels, alpha= 2.0)
        #if multi_head:
        #    self.atten   = multHeadMILAtten(dim=in_channels, nhead = 4, dm = 32, dl = 32)
        #else:
        self.atten   = MILAtten(dim=in_channels,  dl = 64, use_self=self.use_self)

        if self.use_aux:
            layers = [
                BasicConv2d(self.atten.out_dim+1, 64, kernel_size=1, use_relu=True),
                BasicConv2d(64, 64, kernel_size=1, use_relu=True),
                BasicConv2d(64, 64, kernel_size=1, use_relu=True),
                nn.Dropout2d(),
                BasicConv2d(64, 64, kernel_size=1, use_relu=True),
                BasicConv2d(64, 64, kernel_size=1, use_relu=True),
                nn.Dropout2d(),
                nn.Conv2d(64, class_num, kernel_size=1, bias=True)
            ]
        else:
            layers = [
                BasicConv2d(self.atten.out_dim, 64, kernel_size=1, use_relu=True),
                BasicConv2d(64, 64, kernel_size=1, use_relu=True),
                BasicConv2d(64, 64, kernel_size=1, use_relu=True),
                nn.Dropout2d(),
                BasicConv2d(64, 64, kernel_size=1, use_relu=True),
                BasicConv2d(64, 64, kernel_size=1, use_relu=True),
                nn.Dropout2d(),
                nn.Conv2d(64, class_num, kernel_size=1, bias=True)
            ]

        self.conv1    = nn.Sequential(*layers)
        #self.out_conv = nn.Conv2d(64+1, class_num, kernel_size=1, bias=True)

        self._loss = 0
        
    
    def forward(self, x, aux ,label=None, true_num= None):
        #import pdb; pdb.set_trace()
        B, N, C = x.size()[0:3]
        #x = x.view(B, N, C, 1, 1)
        # here deal with 
        vlad, alpha = self.atten(x, true_num)

        vlad = vlad.unsqueeze(-1).unsqueeze(-1)
        if self.use_aux:
            aux = aux.view(B, 1,1, 1)
            xx   = torch.cat([vlad, aux], dim = 1 )
        else:
            xx = vlad
        out = self.conv1(xx)
        #out = self.out_conv(out)
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


class liteWsiNet(nn.Module):
    # input must be 299x299
    
    def __init__(self, class_num=2, in_channels=2, num_clusters=5, 
                 use_self=None, use_aux=False):
        super(liteWsiNet, self).__init__()
        
        self.in_channels = in_channels
        self.use_self = use_self
        self.use_aux = use_aux

        self.register_buffer('device_id', torch.IntTensor(1))
        #self.netvlad = NetVLAD(num_clusters = num_clusters, dim = in_channels, alpha= 2.0)
        self.atten   = MILAtten(dim=in_channels,  dl = 64, use_self=self.use_self)
        
        layers = [
            BasicConv2d(self.atten.out_dim, 256, kernel_size=1, use_relu=False),
            nn.Dropout2d(0.5),
            #BasicConv2d(256, 128, kernel_size=1, use_relu=False),
         ]
        self.conv1    = nn.Sequential(*layers)

        if self.use_aux:
            self.out_conv = nn.Conv2d(256+1, class_num, kernel_size=1, bias=True)
        else:
            self.out_conv = nn.Conv2d(256, class_num, kernel_size=1, bias=True)

        self._loss = 0
        
    
    def forward(self, x, aux ,label=None, true_num= None):
        #import pdb; pdb.set_trace()
        B, N, C = x.size()[0:3]
        #x = x.view(B, N, C, 1, 1)
        aux = aux.view(B, 1,1, 1)
        # here deal with 

        vlad, alpha = self.atten(x, true_num)

        vlad = vlad.unsqueeze(-1).unsqueeze(-1)
        out = self.conv1(vlad)
        if self.use_aux:
            out   = torch.cat([out, aux], dim = 1 )

        out = self.out_conv(out)
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

class logistWsiNet(nn.Module):
    # input must be 299x299
    
    def __init__(self, class_num=2, in_channels=2, num_clusters=5, 
                 use_self=None, use_aux=False):
        super(logistWsiNet, self).__init__()
        
        self.in_channels = in_channels
        self.use_self = use_self
        self.use_aux = use_aux

        self.register_buffer('device_id', torch.IntTensor(1))
        #self.netvlad = NetVLAD(num_clusters = num_clusters, dim = in_channels, alpha= 2.0)
        self.atten   = MILAtten(dim=in_channels,  dl = 64, use_self=self.use_self)
        
        layers = [
            #BasicConv2d(self.atten.out_dim, 256, kernel_size=1, use_relu=False),
            nn.Dropout2d(0.5),
            #BasicConv2d(256, 128, kernel_size=1, use_relu=False),
         ]
        self.conv1    = nn.Sequential(*layers)

        if self.use_aux:
            self.out_conv = nn.Conv2d(self.atten.out_dim+1, class_num, kernel_size=1, bias=True)
        else:
            self.out_conv = nn.Conv2d(self.atten.out_dim, class_num, kernel_size=1, bias=True)

        self._loss = 0
        
    
    def forward(self, x, aux ,label=None, true_num= None):
        #import pdb; pdb.set_trace()
        B, N, C = x.size()[0:3]
        #x = x.view(B, N, C, 1, 1)
        aux = aux.view(B, 1,1, 1)
        # here deal with 

        vlad, alpha = self.atten(x, true_num)

        vlad = vlad.unsqueeze(-1).unsqueeze(-1)
        out = self.conv1(vlad)
        if self.use_aux:
            out   = torch.cat([out, aux], dim = 1 )

        out = self.out_conv(out)
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


class lateFeatWsiNet(nn.Module):
    # input must be 299x299
    
    def __init__(self, class_num=2, in_channels=2, num_clusters=5, use_self=None,
                use_aux = False):
        super(lateFeatWsiNet, self).__init__()
        
        self.in_channels = in_channels
        self.use_self = use_self
        self.use_aux  = use_aux

        self.register_buffer('device_id', torch.IntTensor(1))
        #self.netvlad = NetVLAD(num_clusters = num_clusters, dim = in_channels, alpha= 2.0)
        self.atten   = MILAtten(dim=in_channels,  dl = 64, use_self=self.use_self)
        
        layers = [
            BasicConv2d(self.atten.out_dim, 256, kernel_size=1, use_relu=False),
            BasicConv2d(256, 128, kernel_size=1, use_relu=False),
            nn.Dropout2d(),
            BasicConv2d(128, 64, kernel_size=1, use_relu=False),
            BasicConv2d(64, 64, kernel_size=1, use_relu=False),
        ]
        self.conv1    = nn.Sequential(*layers)
        if self.use_aux:
            self.out_conv = nn.Conv2d(64+1, class_num, kernel_size=1, bias=True)
        else:
            self.out_conv = nn.Conv2d(64, class_num, kernel_size=1, bias=True)

        self._loss = 0
        
    
    def forward(self, x, aux ,label=None, true_num= None):
        #import pdb; pdb.set_trace()
        B, N, C = x.size()[0:3]
        #x = x.view(B, N, C, 1, 1)
        aux = aux.view(B, 1,1, 1)
        # here deal with 

        vlad, alpha = self.atten(x, true_num)

        vlad = vlad.unsqueeze(-1).unsqueeze(-1)
        out = self.conv1(vlad)
        if self.use_aux:
            out   = torch.cat([out, aux], dim = 1 )

        out = self.out_conv(out)
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

class MILAtten(nn.Module):
    """MILAtten layer implementation"""

    def __init__(self, dim=128, dl= 128, use_self=None):
        """
        Args:
            dim : int
                Dimension of descriptors
        """
        super(MILAtten, self).__init__()

        self.use_self = use_self
        self.dim = dim
        #self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        #self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self.out_dim = dim
        self.dl = dl
        self.atten_dim = self.dim
        
        if use_self is 'self_atten':
            self.atten_dim = 256
            self.f_linear = nn.Linear(self.dim, self.atten_dim)
            self.mh_atten = MultiHeadedAttention(h=4, d_model=self.atten_dim)
        elif use_self is 'global':
            self.atten_dim = self.dim * 2

        self.V = nn.Parameter(torch.Tensor( self.atten_dim, self.dl), requires_grad=True)
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
        #import pdb; pdb.set_trace()
        B, num_dis, D = x.size()
        #import pdb; pdb.set_trace()
        #_t_num = true_num.cpu().numpy().tolist() if true_num is not None else None
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
            feat_ = x
        elif self.use_self is 'global':
            #import pdb; pdb.set_trace()
            x_sum       = torch.sum(x, dim=1, keepdim=True) # B x 1 x D
            _num_array  =  true_num.unsqueeze(-1).unsqueeze(-1).expand_as(x_sum) 
            x_mean      =  x_sum/_num_array.float()
            _atten_feat = torch.cat( [x_mean.expand_as(x), x ] , dim=-1  )
            feat_ = x
        else:
            _atten_feat = x
            feat_ = x

        x_   = torch.tanh( torch.matmul(_atten_feat,  self.V ) ) # BxNxL used to be torch.tanh
        dis_ = torch.matmul(x_, self.W).squeeze(-1) # BxN
        dis_ = dis_/math.sqrt(self.dl)
        # set unwanted value to 0, so it won't affect.
        dis_.masked_fill(device_mask==0, -1e20)

        # if true_num is not None:
        #     for idx in range(B):
        #         this_num = true_num[idx]
        #         if this_num < num_dis:
        #             dis_[idx, this_num::] = -1e20
        
        soft_assign_ = F.softmax(dis_, dim = 1) # BxN
        
        soft_assign = soft_assign_.unsqueeze(-1).expand_as(feat_)  # BxNxD
        feat_fusion = torch.sum(soft_assign*feat_, 1, keepdim=False) # BxD

        return feat_fusion, soft_assign_

class multHeadMILAtten(nn.Module):
    """MILAtten layer implementation"""

    def __init__(self, dim=128, nhead=4,  dm=32, dl=32):
        """
        Args:
            dim : int
                Dimension of descriptors
            nhead: int
                Number of heads of multi head attention
            dm: int
                dimensio of each head
            dl: int
                hidden dimension for distance
        """
        super(multHeadMILAtten, self).__init__()
        
        self.dim = dim
        #self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        #self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self.out_dim = nhead*dm
        self.quadratic = False

        self.dl = dl # quadratic form, it has to be
        self.nhead = nhead
        self.dm = dm
        
        self.WX = nn.Parameter(torch.Tensor(self.nhead, self.dim, self.dm), requires_grad=True)  
        self.V  = nn.Parameter(torch.Tensor(self.dm, self.dl), requires_grad=True)
        self.W  = nn.Parameter(torch.Tensor(self.nhead, self.dl), requires_grad=True)
        self.scale  = nn.Parameter(torch.Tensor(self.nhead,1), requires_grad=True)
        self.gate   = nn.Parameter(torch.Tensor(self.dim, self.dl), requires_grad=True)

        self.reset_params()

    def reset_params(self):
        std1 = 1./((self.dl*self.dim)**(1/2))
        self.V.data.uniform_(-std1, std1)
        
        std2 = 1./((self.dl)**(1/2))
        self.W.data.uniform_(-std2, std2)

        std3 = 1./((self.nhead*self.dim*self.dm)**(1/2))
        self.WX.data.uniform_(-std3, std3)

        std4 = 1./((self.dm*self.dl)**(1/2))
        self.gate.data.uniform_(-std4, std4)

        self.scale.data.uniform_(0.8, 1.2)

    def forward(self, x, true_num):
        '''
        Parameters:
        -----------
            x: B x N x D
        Return
            feat_fusion: 
                Bxfeat_dim
            soft_assign
                BxN
        '''
        # 
        B, num_dis, D = x.size()
        #import pdb; pdb.set_trace()
        mh_x = x.unsqueeze(0).expand(self.nhead, -1, -1, -1) #nheadxBxNxD
        wx_expand = self.WX.unsqueeze(1).expand(-1, B, -1, -1) # nheadxBxDxm
        mh_x_ = torch.matmul(mh_x, wx_expand) # nheadxBxNxm
        

        if 0: #linear distance
            gate  = torch.sigmoid( torch.matmul(mh_x, self.gate) ).squeeze(-1) # nheadxBxN
            psd_V =  torch.matmul(torch.transpose(self.V, 0, 1), self.V ) 
            w_expand = self.W.unsqueeze(1).unsqueeze(1) # nheadx1x1xL
            w_expand = w_expand.expand(-1,  B, num_dis, -1) # nheadxBxNxL
            res_  = mh_x_ - w_expand # nheadxBxNxL
            #import pdb; pdb.set_trace()
            dis_ = ( torch.matmul(res_,  psd_V ) )  # nheadxBxNxL
            dis_  = torch.sum( dis_*res_,  -1, keepdim=False   ) #nheadxBxN
            dis_ = dis_/math.sqrt(self.dl)  # nheadxBxN
            
            if true_num is not None:
                for idx in range(B):
                    this_num = true_num[idx]
                    if this_num < num_dis:
                        dis_[:, idx, this_num::] = -1e20

            #scale = torch.clamp(self.scale.unsqueeze(-1).expand(-1, B, num_dis), -5, 5) 
            scale = 1
            soft_assign = F.softmax(dis_*scale*gate, dim = 2) # nheadxBxN
            
        if 1:
            gate = torch.sigmoid( torch.matmul(mh_x, self.gate) )#.expand(-1,-1,-1,self.dl)# nheadxBxNxL
            dis_ = torch.tanh( torch.matmul(mh_x_,  self.V ) )*gate  # nheadxBxNxL
            
            w_expand = self.W.unsqueeze(1).unsqueeze(1) # nheadx1x1xL
            #w_norm   = torch.norm(w_expand, p=2, dim=-1, keepdim=True) # nheadx1x1xL
            #w_expand = w_expand/(w_norm+1e-8)
            w_expand = w_expand.expand(-1,  B, num_dis, -1) # nheadxBxNxL
            
            #dis_norm  = torch.norm(dis_, p=2, dim=-1, keepdim=True) # nheadx1x1xL
            #dis_  = dis_/(dis_norm+1e-8)
            dis_  = torch.sum( dis_*w_expand,  -1, keepdim=False   ) #nheadxBxN
            
            if true_num is not None:
                for idx in range(B):
                    this_num = true_num[idx]
                    if this_num < num_dis:
                        dis_[:, idx, this_num::] = -1e20
                    
            #dis_ = dis_/math.sqrt(self.dl)  # nheadxBxN
            #scale = torch.clamp(self.scale.unsqueeze(-1).expand(-1, B, num_dis), -5, 5) 
            scale = 1
            soft_assign_ = F.softmax(dis_*scale, dim = 2) # nheadxBxN

        soft_assign = soft_assign_.unsqueeze(-1).expand(-1, -1, -1, self.dm)  # nheadxBxNxm
        feat_fusion = torch.sum(soft_assign*mh_x_, 2, keepdim=False) # nheadxBxm
        feat_fusion = feat_fusion.permute(1,0,2) #Bxnheadxm
        feat_fusion = feat_fusion.contiguous().view(B, -1)

        return feat_fusion, soft_assign_[0]



class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128, alpha=2.0,
                 normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        #self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        #self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self.out_dim = num_clusters*dim
        
        self.centroids = nn.Parameter(torch.Tensor(num_clusters, dim), requires_grad=True)
        self.scale     = nn.Parameter(torch.Tensor(1, num_clusters), requires_grad=True)
        self.metric    = nn.Parameter(torch.Tensor(dim, dim), requires_grad=True)

        self.reset_params()

    def reset_params(self):
        std1 = 1./((self.num_clusters*self.dim)**(1/2))
        self.centroids.data.uniform_(-std1, std1)
        self.scale.data.uniform_(-1, 0)
        std2 = 1./((self.dim*self.dim)**(1/2))
        self.metric.data.uniform_(-std2, std2)


    def forward(self, x):
        '''
        Parameters:
        -----------
            x: B x N x D
        '''
        # import pdb; pdb.set_trace()
        B, num_dis, D = x.size()
        x = x.view(B*num_dis, D)
        N, D = x.size()[0:2]
        
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim
        
        #centroids_ = torch.matmul(self.centroids, self.metric) # KxD
        #x_ = torch.matmul(x, self.metric) # NxD
        centroids_ = self.centroids
        x_ = x

        x_ = x_.unsqueeze(1).expand(-1, self.num_clusters ,-1)   # NxKxD
        center_ = centroids_.unsqueeze(0).expand(N, -1, -1) # NxKxD
        
        res_vec = x_ - center_  # NxKxD
        #metric_dis = torch.matmul(res_vec, self.metric) # nxkxd
        metric_dis = torch.sum(res_vec*res_vec, dim=2, keepdim=False) #nxk
        #metric_dis = metric_dis * self.scale.expand(N,-1)
        
        soft_assign = F.softmax(metric_dis, dim = 1).unsqueeze(2) # NxKx1

        # # calculate residuals to each clusters
        x_expand = x.unsqueeze(0) # 1xNxD 
        x_expand = x_expand.expand(self.num_clusters, -1, -1) # KxNxD
        x_expand = x_expand.permute(1, 0, 2) # NxKxD
        center_expand = self.centroids.unsqueeze(0) # 1xKxD
        center_expand = center_expand.expand(x_expand.size(0), -1, -1) #NxKxD

        residual  = x_expand - center_expand
        residual *= soft_assign.expand(-1,-1,D) # NxKxD
        residual  = residual.view(B, num_dis , self.num_clusters , D)

        vlad = residual.sum(dim=1)
        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(B, -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad

