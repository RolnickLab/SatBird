import torch
import torchvision.models as models
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import math


class ResLayer(nn.Module):
    def __init__(self, linear_size):
        super(ResLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y

        return out


class FCNet(nn.Module):
    def __init__(self, num_inputs, num_classes, num_filts, ebd = False):
        super(FCNet, self).__init__()
        self.ebd = ebd
        self.inc_bias = False
        if self.ebd:
            self.ebd_emb = nn.Sequential(nn.Linear(num_filts, 824, bias=self.inc_bias),
                                         nn.ReLU(inplace=True))
            self.class_emb = nn.Linear(824, num_classes, bias=self.inc_bias)
        else:
            self.class_emb = nn.Linear(num_filts, num_classes, bias=self.inc_bias)
       # self.user_emb = nn.Linear(num_filts, num_users, bias=self.inc_bias)
        

        self.feats = nn.Sequential(nn.Linear(num_inputs, num_filts),
                                    nn.ReLU(inplace=True),
                                    ResLayer(num_filts),
                                    ResLayer(num_filts),
                                    ResLayer(num_filts),
                                    ResLayer(num_filts))
        

        
    def forward(self, x, class_of_interest=None, return_feats=False):
        
        loc_emb = self.feats(x)
        if return_feats:
            return loc_emb
        if class_of_interest is None:
            if self.ebd:
                loc_emb = self.ebd_emb(loc_emb)
        class_pred = self.class_emb(loc_emb)
          
        return torch.sigmoid(class_pred)

    def eval_single_class(self, x, class_of_interest):
        if self.inc_bias:
            return torch.matmul(x, self.class_emb.weight[class_of_interest, :]) + self.class_emb.bias[class_of_interest]
        else:
            return torch.matmul(x, self.class_emb.weight[class_of_interest, :])

        
class MLPDecoder(nn.Module):
    def __init__(self, input_size, target_size, flatten = True):
        super().__init__()
        self.mlp_dim = input_size
        if flatten:
            modules = [nn.AdaptiveAvgPool2d((1,1)), 
                       nn.Flatten(), 
                       nn.Linear(self.mlp_dim, self.mlp_dim), 
                       nn.ReLU(), 
                       nn.Linear(self.mlp_dim, target_size)]
        else: 
            modules = [nn.Linear(self.mlp_dim, self.mlp_dim), 
                       nn.ReLU(), 
                       nn.Linear(self.mlp_dim, target_size)]
        self.model = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.model(x)