import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pdb import set_trace as stop
import copy
import math
from typing import Optional, List
from torch import nn, Tensor

def weights_init(module):
    """ Initialize the weights """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        stdv = 1. / math.sqrt(module.weight.size(1))
        module.weight.data.uniform_(-stdv, stdv)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.uniform_(-stdv, stdv)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

def custom_replace(tensor,on_neg_1,on_zero,on_one):
    res = tensor.clone()
    res[tensor==-1] = on_neg_1
    res[tensor==0] = on_zero
    res[tensor==1] = on_one
    return res


def get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))