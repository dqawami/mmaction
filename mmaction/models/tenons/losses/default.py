import torch.nn.functional as F

from ...registry import CLASSIFICATION_LOSSES
from .base import MetricLearningLossBase


@CLASSIFICATION_LOSSES.register_module
class CrossEntropyLoss(MetricLearningLossBase):
    def __init__(self, **kwargs):
        super(CrossEntropyLoss, self).__init__(**kwargs)

    def _forward(self, cos_theta, target, scale):
        out_losses = F.cross_entropy(scale * cos_theta, target.detach().view(-1), reduction='none')

        return out_losses
