import torch
from torchmetrics import Metric

eps = 1e-7


class CustomCrossEntropyLoss:
    def __init__(self, lambd_pres=1, lambd_abs=1):
        super().__init__()
        # print('in my custom')
        self.lambd_abs = lambd_abs
        self.lambd_pres = lambd_pres

    def __call__(self, p, q):
        # print('maximum prediction value ',q.max())
        # print('maximum target value',p.max())
        # p=torch.clip(p, min=0, max=0.98)
        # q=torch.clip(q, min=0, max=0.98)
        loss = (-self.lambd_pres * p * torch.log(q + eps) - self.lambd_abs * (1 - p) * torch.log(1 - q + eps)).mean()
        # print('inside_loss',loss)
        return loss


'''
class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, lambd_pres = 1, lambd_abs = 1):
        super().__init__()
        self.lambd_abs = lambd_abs
        self.lambd_pres =lambd_pres
    def __call__(self, p, q):
        return (-self.lambd_pres *p * torch.log(q) - self.lambd_abs * (1-p) *torch.log(1 - q)).mean()
'''


class CustomCrossEntropy(Metric):
    def __init__(self, lambd_pres=1, lambd_abs=1, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.lambd_abs = lambd_abs
        self.lambd_pres = lambd_pres
        self.add_state("correct", default=torch.FloatTensor([0]), dist_reduce_fx="sum")
        self.add_state("total", default=torch.FloatTensor([0]), dist_reduce_fx="sum")

    def update(self, p: torch.Tensor, q: torch.Tensor):
        """
        p: target distribution
        q: predicted distribution
        """
        self.correct += (-self.lambd_pres * p * torch.log(q) - self.lambd_abs * (1 - p) * torch.log(1 - q)).sum()
        self.total += p.numel()

    def compute(self):
        return (self.correct / self.total)
