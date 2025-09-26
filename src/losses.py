import torch
import torch.nn as nn

def dice_loss(pred, target, eps = 1e-6):
    probs = torch.softmax(pred, dim = 1)[:,1]
    target = (target == 1).float()
    inter = (probs * target).sum()
    union = probs.sum() + target.sum()
    return 1 - (2 * inter + eps) / (union + eps)

def combined_loss(pred, target, ce_weight = 1.0, dice_weight = 0.5):
    ce = nn.CrossEntropyLoss()(pred, target)
    dice = dice_loss(pred, target)
    return ce_weight * ce + dice_weight * dice

def dice_coeff(pred, target, eps = 1e-6):
    pred = torch.softmax(pred, dim = 1)[:,1] > 0.5
    target = (target == 1)
    if target.sum() == 0:
        return torch.tensor(1.0, device = pred.device)
    inter = (pred & target).float().sum()
    union = pred.float().sum() + target.float().sum()
    return (2 * inter + eps) / (union + eps)

def nodule_acc(pred, target, eps = 1e-6):
    pred = torch.softmax(pred, dim = 1)[:,1] > 0.5
    target = (target == 1)
    nodule_voxels = target.float().sum()
    if nodule_voxels == 0:
        return torch.tensor(1.0, device = pred.device)
    correct = (pred & target).float().sum()
    return (correct + eps) / (nodule_voxels + eps)
