import torch, math
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from .inception_customized import BasicConv2d, InceptionA, InceptionB, InceptionC,\
                                  InceptionD, InceptionE, InceptionAux, InceptionFeat

class inceptionCellNet(nn.Module):
    # input must be 299x299
    
    def __init__(self, class_num=2, in_channels=3, 
                 large=False, atten_type=None):
        aux_logits=True
        transform_input=False
        super(inceptionCellNet, self).__init__()
        
        self.in_channels = in_channels
        self.register_buffer('device_id', torch.IntTensor(1))

        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.atten_type = atten_type

        #if large is True: Since small is the default settings.
        
        #    self.Conv2d_1a_3x3 = BasicConv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        #else:
        self.Conv2d_1a_3x3 = BasicConv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)

        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3, padding=1)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3, padding=1)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        if aux_logits:
            self.AuxLogits = InceptionAux(768, class_num)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)
        if self.atten_type == 'mil':
            self.atten_layer = MILAtten(dim=2048, dl= 2)
        elif self.atten_type == 'global':
            self.atten_layer = MILAtten(dim=4096, dl= 2)
        
        self.fc = nn.Linear(2048, class_num)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.numel()))
                values = values.view(m.weight.size())
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self._loss = 0
 
    def forward(self, x, label=None):
        #import pdb; pdb.set_trace()

        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768
        if self.training and self.aux_logits:
            #import pdb; pdb.set_trace()
            aux = self.AuxLogits(x)
        # 17 x 17 x 768
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        
        
        if self.atten_type == 'mil':
            feat, _ = self.atten_layer(x, x)
        elif self.atten_type == 'global':
            #import pdb; pdb.set_trace()
            glob_feat = F.adaptive_avg_pool2d(x, (1, 1))
            repeat_global = glob_feat.expand_as(x)
            cat_feat = torch.cat( [repeat_global, x], dim=1 )
            feat, _  = self.atten_layer(cat_feat, x)
        
        elif self.atten_type is None:
            feat = F.adaptive_avg_pool2d(x, (1, 1))
        
        # 1 x 1 x 2048
        x = F.dropout(feat, p=0.5, training=self.training)
        #x = feat
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048
        x = self.fc(x)
        # 1000 (num_classes)
 
        if self.training:
            assert label is not None, "invalid label for training mode"
            self._loss = F.nll_loss(F.log_softmax(x, dim=1), 
                        label.view(-1), reduction='none')
            #self._loss = nn.MultiMarginLoss()(x, label.view(-1))

            if self.aux_logits:
                #self._loss += F.nll_loss(F.log_softmax(aux, dim=1), label.view(-1))
                self._loss += nn.MultiMarginLoss(reduction='none')(aux, label.view(-1))

            return x
        else:
            self.feat   = feat.data.squeeze(-1).squeeze(-1)
            #cls_pred    = F.softmax(x, dim=1)
            return x
        
    @property
    def loss(self):
        return self._loss


class MILAtten(nn.Module):
    """MILAtten layer implementation"""

    def __init__(self, dim=128, dl= 128):
        """
        Args:
            dim : int
                Dimension of descriptors
            dl: hidden dim
        """
        super(MILAtten, self).__init__()
        
        self.dim = dim
        #self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        #self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self.out_dim = dim
        self.dl = dl

        self.V = nn.Parameter(torch.Tensor(self.dim, self.dl), requires_grad=True)
        self.W = nn.Parameter(torch.Tensor(self.dl, 1), requires_grad=True)
        self.scale=1
        self.reset_params()

    def reset_params(self):
        std1 = 1./((self.dl*self.dim)**(1/2))
        self.V.data.uniform_(-std1, std1)
        
        std2 = 1./((self.dl)**(1/2))
        self.W.data.uniform_(-std2, std2)

    def forward(self, x, f):
        '''
        Parameters:
        -----------
            x: B x C x W x H used to generate attention
            f: B x D x W x H used to return feature
        Return
            feat_fusion: 
                Bxfeat_dim
            soft_assign
                BxN
        '''
        # import pdb; pdb.set_trace()
        B, C, W, H   = x.size()
        _, Cf, Wf, Hf = f.size()

        x = x.view(B, C, W*H)
        x = x.permute(0, 2, 1) 

        f = f.view(B, Cf, Wf*Hf)
        f = f.permute(0, 2, 1) 

        B, num_dis, D = x.size()
        
        x_   = torch.tanh( torch.matmul(x,  self.V ) ) # BxNxL
        dis_ = torch.matmul(x_, self.W).squeeze(-1) # BxN
        dis_ = dis_/math.sqrt(self.dl)
        # set unwanted value to 0, so it won't affect.
        soft_assign_ = F.softmax(dis_, dim = 1) # BxN
        
        soft_assign = soft_assign_.unsqueeze(-1).expand(-1, -1, Cf)  # BxNxD
        feat_fusion = torch.sum(soft_assign*f, 1, keepdim=False) # BxD

        return feat_fusion, soft_assign_

