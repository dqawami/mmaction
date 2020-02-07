from os.path import exists

import json
import numpy as np
import torch
import torch.nn.functional as F

from ...registry import CLASSIFICATION_LOSSES
from .base import MetricLearningLossBase


def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    losses = (1 - p) ** gamma * input_values
    return losses


@CLASSIFICATION_LOSSES.register_module
class AMSoftmaxLoss(MetricLearningLossBase):
    """Computes the AM-Softmax loss with cos or arc margin"""
    margin_types = ['cos', 'arc']

    def __init__(self, margin_type='cos', margin=0.5, gamma=0.0, t=1.0,
                 use_adaptive_margins=False, class_counts=None, **kwargs):
        super(AMSoftmaxLoss, self).__init__(**kwargs)

        assert margin_type in AMSoftmaxLoss.margin_types
        self.margin_type = margin_type
        assert gamma >= 0
        self.gamma = gamma
        assert margin > 0
        self.m = margin
        self.cos_m = np.cos(self.m)
        self.sin_m = np.sin(self.m)
        self.th = np.cos(np.pi - self.m)
        assert t >= 1
        self.t = t

        if use_adaptive_margins and class_counts is not None and exists(class_counts):
            with open(class_counts) as input_stream:
                counts_dict = json.load(input_stream)
            counts_dict = {int(k): float(v) for k, v in counts_dict.items()}

            class_ids = list(counts_dict)
            class_ids.sort()

            counts = np.array([counts_dict[class_id] for class_id in class_ids], dtype=np.float32)
            class_margins = (self.m / np.power(counts, 1. / 4.)).reshape((1, -1))

            self.register_buffer('class_margins', torch.from_numpy(class_margins))
            print('[INFO] Enabled adaptive margins for AM-Softmax')
        else:
            self.class_margins = self.m

    def _forward(self, cos_theta, target, scale):
        if self.margin_type == 'cos':
            phi_theta = cos_theta - self.class_margins
        else:
            sine = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
            phi_theta = cos_theta * self.cos_m - sine * self.sin_m
            phi_theta = torch.where(cos_theta > self.th, phi_theta, cos_theta - self.sin_m * self.m)

        index = torch.zeros_like(cos_theta, dtype=torch.uint8)
        index.scatter_(1, target.detach().view(-1, 1), 1)
        output = torch.where(index, phi_theta, cos_theta)

        if self.gamma == 0 and self.t == 1.:
            out_losses = F.cross_entropy(scale * output, target, reduction='none')
        elif self.t > 1:
            h_theta = self.t - 1 + self.t * cos_theta
            support_vecs_mask = (1 - index) * \
                torch.lt(torch.masked_select(phi_theta, index).view(-1, 1).repeat(1, h_theta.shape[1]) - cos_theta, 0)
            output = torch.where(support_vecs_mask, h_theta, output)
            out_losses = F.cross_entropy(scale * output, target, reduction='none')
        else:
            out_losses = focal_loss(F.cross_entropy(scale * output, target, reduction='none'), self.gamma)

        return out_losses
