import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .base import entropy


class MaxEntropyLoss(nn.Module):
    def __init__(self, scale=1.0):
        super(MaxEntropyLoss, self).__init__()

        self.scale = scale
        assert self.scale > 0.0

    def forward(self, cos_theta):
        probs = F.softmax(self.scale * cos_theta, dim=-1)

        entropy_values = entropy(probs, dim=-1)
        losses = np.log(cos_theta.size(-1)) - entropy_values

        return losses.mean()
