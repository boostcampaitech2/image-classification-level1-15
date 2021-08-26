from torch import tensor
import torch.nn.functional as F
import torch


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


def cross_entropy_loss_age_smooth(output, target):
    device = 'cuda'
    # <= 50 ->
    # 51 -> 소수점
    #F.cross_entropy(output, target, torch.tensor([1., 1.5, 1.5]).to(device))
    # 51
    # eda 확인후 가중치...
    if 1 < target < 2:
        # 1.1
        # 0.1 -> cross(pred, 2) * 0.1
        weight = target - 1
        ratio = F.cross_entropy(output, torch.tensor([2]).to(device)) * weight
        ratio2 = F.cross_entropy(
            output, torch.tensor([1]).to(device)) * (1 - weight)
        return ratio + ratio2

    return cross_entropy_loss_age(output, target)
