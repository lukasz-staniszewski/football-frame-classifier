import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from configparser import ConfigParser


def nll_loss(output, target):
    return F.nll_loss(output, target)


def cross_entropy_loss(output, target):
    weights = np.array(
        ConfigParser.read("config.json")["class_weights"]
    )

    return nn.CrossEntropyLoss(weight=weights)(output, target)
