import torch
from torchmetrics import Metric
import torch.nn as nn

eps = 1e-7


class CustomCrossEntropyLoss:
    def __init__(self, lambd_pres=1, lambd_abs=1):
        super().__init__()
        # print('in my custom')
        self.lambd_abs = lambd_abs
        self.lambd_pres = lambd_pres

    def __call__(self, pred, target):
        """
        target: ground truth
        pred: prediction
        """
        # print('maximum prediction value ',q.max())
        # print('maximum target value',p.max())
        # p=torch.clip(p, min=0, max=0.98)
        # q=torch.clip(q, min=0, max=0.98)
        loss = (-self.lambd_pres * target * torch.log(pred + eps) - self.lambd_abs * (1 - target) * torch.log(1 - pred + eps)).mean()
        # print('inside_loss',loss)
        return loss


class RMSLELoss(nn.Module):
    """
    root mean squared log error
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(target + 1)))


class CustomFocalLoss:
    def __init__(self, alpha=1, gamma=2):
        """
        build on top of binary cross entropy as implemented before
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, pred, target):
        ce_loss = (- target * torch.log(pred + eps) - (1 - target) * torch.log(1 - pred + eps)).mean()
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss


class CustomCrossEntropy(Metric):
    def __init__(self, lambd_pres=1, lambd_abs=1, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.lambd_abs = lambd_abs
        self.lambd_pres = lambd_pres
        self.add_state("correct", default=torch.FloatTensor([0]), dist_reduce_fx="sum")
        self.add_state("total", default=torch.FloatTensor([0]), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        target: target distribution
        pred: predicted distribution
        """
        self.correct += (-self.lambd_pres * target * torch.log(pred) - self.lambd_abs * (1 - target) * torch.log(1 - pred)).sum()
        self.total += target.numel()

    def compute(self):
        return (self.correct / self.total)


class WeightedCustomCrossEntropyLoss:
    def __init__(self, lambd_pres=1, lambd_abs=1):
        super().__init__()
        self.lambd_abs = lambd_abs
        self.lambd_pres = lambd_pres

    def __call__(self, pred, target, weights=1):
        loss = (weights * (
            -self.lambd_pres * target * torch.log(pred + eps)
            - self.lambd_abs * (1 - target) * torch.log(1 - pred + eps))).mean()
        
        return loss