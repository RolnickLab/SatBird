import torch
import torch.nn as nn
import torchmetrics

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
        counts = torch.zeros(len(i_topk))
        for i,elem in enumerate(i_topk):
            count = 0
            for it in i_pred[i]:
                if it in elem:
                    count += 1
            counts[i] = count/self.k
        diff = sum(torch.abs(v_pred - v_topk))
        return (counts.mean(), diff.mean())

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
        k = metric.k
        return TopKAccuracy(k)
    
    elif metric.name == "ce" and not metric.ignore is True :
        return CustomCrossEntropyLoss(metric.lambd_pres, metric.lambd_abs)
    
    elif metric.ignore is True :
        return None

    raise ValueError("Unknown metric_item {}".format(metric))

def get_metrics(opts):
    metrics = []

    for m in opts.losses.metrics:
        metrics.append((m.name, get_metric(m), m.scale))
    metrics = [(a,b,c) for (a,b,c) in metrics if b is not None]

    return metrics