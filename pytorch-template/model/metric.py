import torch


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        # [[,,,],
        # [,,,]] ->

        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def f1(output, target, is_training=False):
    pred = torch.argmax(output, dim=1)

    assert pred.ndim == 1
    assert target.ndim == 1 or target.ndim == 2

    if target.ndim == 2:
        target = target.argmax(dim=1)

    tp = (target * pred).sum().to(torch.float32)
    tn = ((1 - target) * (1 - pred)).sum().to(torch.float32)
    fp = ((1 - target) * pred).sum().to(torch.float32)
    fn = (target * (1 - pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1


def accuracy_multi_label(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        origin_target = torch.argmax(target, dim=1)
        # [0, 1, 1] -> [0, 1, 0] -> 2
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == origin_target).item()
    return correct / len(target)


def f1_multi_label(output, target, is_training=False):
    pred = torch.argmax(output, dim=1)

    assert pred.ndim == 1
    assert target.ndim == 1 or target.ndim == 2

    origin_target = torch.argmax(target, dim=1)

    if origin_target.ndim == 2:
        origin_target = origin_target.argmax(dim=1)

    tp = (pred * origin_target).sum().to(torch.float32)
    fp = ((origin_target - 1) * (origin_target - 1)).sum().to(torch.float32)
    tn = ((pred - 1) * origin_target).sum().to(torch.float32)
    fn = (pred * (origin_target - 1)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1
