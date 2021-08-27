from torch import tensor
import torch.nn.functional as F
import torch
import torch.nn as nn


def nll_loss(output, target):
    return F.nll_loss(output, target)


def cross_entropy_loss(output, target):
    return F.cross_entropy(output, target)


def cross_entropy_loss_gen(output, target):
    device = 'cuda'
    return F.cross_entropy(output, target, torch.tensor([1.2, 1.]).to(device))


def cross_entropy_loss_age(output, target):
    device = 'cuda'
    return F.cross_entropy(output, target, torch.tensor([1., 1., 5.]).to(device))


def cross_entropy_loss_mask(output, target):
    device = 'cuda'
    return F.cross_entropy(output, target, torch.tensor([1., 1.5, 1.5]).to(device))


# def cross_entropy_loss_age_smooth(output, target):
#     device = 'cuda'
#     loss = nn.MultiLabelMarginLoss()

#     return loss(output, target)
