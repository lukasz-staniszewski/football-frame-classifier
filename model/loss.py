import torch.nn.functional as F
import torch.nn as nn
import json
import torch


def nll_loss(output, target, device):
    return F.nll_loss(output, target)


def cross_entropy_loss(output, target, device):
    with open("config.json", "r") as f:
        weights = torch.FloatTensor(json.load(f)["class_weights"]).to(device)
    return nn.CrossEntropyLoss(weight=weights)(output, target)
