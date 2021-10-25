import torch
import torch.nn as nn


class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, lambd_pres = 1, lambd_abs = 1):
        super().__init__()
        self.lambd_abs = lambd_abs
        self.lambd_pres =lambd_pres
    def __call__(self, pred, target):
        return (-self.lambd_pres *target * torch.log(pred) - self.lambd_abs * (1-target) *torch.log(1 - pred)).sum()


class TopKAccuracy(nn.Module):
    def __init__(self,k=30):
        super().__init__()
        self.k=k
    def __call__(self, pred, target):
        v_topk, i_topk = torch.topk(target, self.k)
        v_pred, i_pred = torch.topk(pred, self.k)
        acc = len([i for i_topk in x if i in i_pred])/self.k
        diff = sum(torch.abs(v_pred - v_topk))
        return (acc, diff)
