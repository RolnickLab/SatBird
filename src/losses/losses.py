import torch
import torch.nn as nn
import torchmetrics


class CustomCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def __call__(self, pred, target):
        return (-target * torch.log(pred) - (1-target) *torch.log(1 - pred)).sum()

#TODO
#def get_eval_metrics(pred, target):
#    dic = {"MSE":}