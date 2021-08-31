import torch


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
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
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        f1s = []
        epsilon = 1e-7
        for label in range(18):
            label_pred = (pred==label)
            label_target = (target==label)
            label_pred_not = (pred!=label)
            label_target_not = (target!=label)
            tp = (label_target&label_pred).sum().to(torch.float32)
            tn = (label_target_not&label_pred_not).sum().to(torch.float32)
            fp = (label_target_not&label_pred).sum().to(torch.float32)
            fn = (label_target&label_pred_not).sum().to(torch.float32)
            precision = tp / (tp + fp + epsilon)
            recall = tp / (tp + fn + epsilon)
            f1 = 2 * (precision * recall) / (precision + recall + epsilon)
            f1s.append(f1)
        f1s = list(filter(lambda x: x!=0, f1s))
    return sum(f1s)/len(f1s)
