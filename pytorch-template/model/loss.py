from torch import tensor
import torch.nn.functional as F
import torch
import torch.nn as nn


def nll_loss(output, target):
    return F.nll_loss(output, target)


def cross_entropy_loss(output, target):
    return F.cross_entropy(output, target)


def binary_cross_entropy_loss(output, target):
    return F.binary_cross_entropy_with_logits(output, target)


def age_cross_entropy_loss(output, target):
    device = torch.device('cuda:0')
    return F.cross_entropy(output, target, weight=torch.tensor([1.5, 3.5, 4.7]).to(device))


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=3, smoothing=0.25, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
