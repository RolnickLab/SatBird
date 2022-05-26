import torch
import torch.nn as nn
import torchmetrics
from torchmetrics import  Metric
#from torchmetrics import PearsonCorrCoef
eps=1e-7
class CustomCrossEntropyLoss:
    def __init__(self, lambd_pres = 1, lambd_abs = 1):
        super().__init__()
        print('in my custom')
        self.lambd_abs = lambd_abs
        self.lambd_pres =lambd_pres
    def __call__(self, p, q):
        print('maximum prediction value ',q.max())
        print('maximum target value',p.max())
        #p=torch.clip(p, min=0, max=0.98)
        #q=torch.clip(q, min=0, max=0.98)
        loss=(-self.lambd_pres *p * torch.log(q+eps) - self.lambd_abs * (1-p + eps) *torch.log(1 - q)).mean()
        print('inside_loss',loss)
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
    def __init__(self, lambd_pres = 1, lambd_abs = 1, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.lambd_abs = lambd_abs
        self.lambd_pres =lambd_pres
        self.add_state("correct", default = torch.FloatTensor([0]), dist_reduce_fx="sum")
        self.add_state("total", default=torch.FloatTensor([0]), dist_reduce_fx="sum")
        
    def update(self, p: torch.Tensor, q: torch.Tensor):
        """
        p: target distribution
        q: predicted distribution
        """
        self.correct += (-self.lambd_pres *p * torch.log(q) - self.lambd_abs * (1-p) *torch.log(1 - q)).sum()
        self.total += p.numel()
    def compute(self):
        return (self.correct/self.total)


class CustomKL(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.FloatTensor([0]), dist_reduce_fx="sum")
        self.add_state("total", default=torch.FloatTensor([0]), dist_reduce_fx="sum")
        
    def update(self, p: torch.Tensor, q: torch.Tensor):
        """
        p: target distribution
        q: predicted distribution
        """
        self.correct += (torch.nansum(p*torch.log(p/q)) + torch.nansum((1-p)*torch.log((1-p)/(1-q))))
        self.total += p.numel()
    def compute(self):
        return (self.correct/self.total)
    
class Presence_k(nn.Module):
    """
    compare accuracy by binarizing targets  1 if species are present with proba > k 
    """
    def __init__(self, k):
        super().__init__()
        self.k = k
    def __call__(self,  target, pred):
        pres = ((pred > self.k) == (target > self.k)).mean()
        return (pres)

class CustomTopK(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.FloatTensor([0]), dist_reduce_fx="sum")
        self.add_state("total", default=torch.FloatTensor([0]), dist_reduce_fx="sum")

    def update(self, target: torch.Tensor, preds: torch.Tensor):
        #preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape
        non_zero_counts = torch.count_nonzero(target, dim=1)
        for i, elem in enumerate(target):
            ki = non_zero_counts[i]
            v_pred, i_pred = torch.topk(preds[i], k = ki)
            v_targ, i_targ = torch.topk(elem, k = ki)
            if ki == 0 :
                self.correct += 1
            else:
                self.correct += len(set(i_pred.cpu().numpy()).intersection(set(i_targ.cpu().numpy()))) / ki
        self.total += target.shape[0]

    def compute(self):
        return (self.correct / self.total).float()
    


def get_metric(metric):
    """Returns the transform function associated to a
    transform_item listed in opts.data.transforms ; transform_item is
    an addict.Dict
    """
    
    if metric.name == "mae" and not metric.ignore is True :
        return torchmetrics.MeanAbsoluteError()

    elif metric.name == "mse" and not metric.ignore is True :
        return torchmetrics.MeanSquaredError()
    
    elif metric.name == "topk" and not metric.ignore is True :
        return CustomTopK()
    
    elif metric.name == "ce" and not metric.ignore is True :
        return CustomCrossEntropy(metric.lambd_pres, metric.lambd_abs)
    elif metric.name =='r2' and not metric.ignore is True:
        return torchmetrics.ExplainedVariance(multioutput='variance_weighted')
        #return  torchmetrics.SpearmanCorrCoef()
    elif metric.name == "kl" and not metric.ignore is True :
        return CustomKL()
    
    elif metric.name == "accuracy" and not metric.ignore is True:
        return torchmetrics.Accuracy()
    elif metric.ignore is True :
        return None

    raise ValueError("Unknown metric_item {}".format(metric))

def get_metrics(opts):
    metrics = []
    
    for m in opts.losses.metrics:
        metrics.append((m.name, get_metric(m), m.scale))
    metrics = [(a,b,c) for (a,b,c) in metrics if b is not None]
    print(metrics)
    return metrics
