import json
from abc import ABCMeta, abstractmethod
from os.path import exists

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from ... import builder
from ....core.ops import normalize


def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0.0, p * p.log(), torch.zeros_like(p)).sum(dim=dim, keepdim=keepdim)


class AngleSimpleLinear(nn.Module):
    """Computes cos of angles between input vectors and weights vectors"""

    def __init__(self, in_features, out_features):
        super(AngleSimpleLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.normal_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        if x.ndimension() > 2:
            x = x.squeeze()

        assert x.ndimension() == 2

        return F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0)).clamp(-1, 1)


class AngleMultipleLinear(nn.Module):
    """Based on SoftTriplet loss: https://arxiv.org/pdf/1909.05235.pdf
    """

    def __init__(self, in_features, num_classes, num_centers, scale=10.0, reg_weight=0.2, reg_threshold=0.2):
        super(AngleMultipleLinear, self).__init__()

        self.in_features = in_features
        assert in_features > 0
        self.num_classes = num_classes
        assert num_classes >= 2
        self.num_centers = num_centers
        assert num_centers >= 1
        self.scale = scale
        assert scale > 0.0

        weight_shape = [in_features, num_classes, num_centers] if num_centers > 1 else [in_features, num_classes]
        self.weight = Parameter(torch.Tensor(*weight_shape))
        self.weight.data.normal_().renorm_(2, 1, 1e-5).mul_(1e5)

        self.enable_regularization = reg_weight is not None and reg_weight > 0.0
        if self.enable_regularization:
            self.reg_weight = reg_weight
            if num_centers == 1:
                self.reg_threshold = reg_threshold
                assert self.reg_threshold >= 0.0

                reg_valid_mask = np.triu(np.ones((num_classes, num_classes), dtype=float), k=1)
            else:
                self.reg_weight /= num_classes
                if num_centers > 2:
                    self.reg_weight /= (num_centers - 1) * (num_centers - 2)

                reg_valid_mask = np.tile(np.triu(np.ones((1, num_centers, num_centers), dtype=float), k=1),
                                         (num_classes, 1, 1))

            self.register_buffer('reg_mask', torch.from_numpy(reg_valid_mask))
        else:
            self.reg_weight = None
            self.reg_mask = None

    def forward(self, normalized_x):
        normalized_x = normalized_x.view(-1, self.in_features)
        normalized_weights = normalize(self.weight.view(self.in_features, -1), dim=0)

        prod = normalized_x.mm(normalized_weights)
        if not torch.onnx.is_in_onnx_export():
            prod = prod.clamp(-1.0, 1.0)

        if self.num_centers > 1:
            prod = prod.view(-1, self.num_classes, self.num_centers)

            prod_weights = F.softmax(self.scale * prod, dim=-1)
            scores = torch.sum(prod_weights * prod, dim=-1)
        else:
            scores = prod

        return scores

    def loss(self):
        out_losses = dict()

        if self.enable_regularization:
            normalized_weights = F.normalize(self.weight, dim=0)
            if self.num_centers == 1:
                all_pairwise_scores = normalized_weights.permute(1, 0).matmul(normalized_weights)
                valid_pairwise_scores = all_pairwise_scores[self.reg_mask > 0.0]
                losses = valid_pairwise_scores[valid_pairwise_scores > self.reg_threshold] - self.reg_threshold
                out_losses['loss_cpush'] = self.reg_weight * losses.mean() if losses.numel() > 0 else losses.sum()
            else:
                all_pairwise_scores = normalized_weights.permute(1, 2, 0).matmul(normalized_weights.permute(1, 0, 2))
                valid_pairwise_scores = all_pairwise_scores[self.reg_mask > 0.0]
                losses = 1.0 - valid_pairwise_scores
                out_losses['loss_st_reg'] = self.reg_weight * losses.sum()

        return out_losses


class MetricLearningLossBase(nn.Module, metaclass=ABCMeta):
    _loss_filter_types = ['positives', 'top_k']

    def __init__(self, scale_cfg, pr_product=False, conf_penalty_weight=None,
                 filter_type=None, top_k=None, use_class_weighting=False, class_weights=None):
        super(MetricLearningLossBase, self).__init__()

        self._enable_pr_product = pr_product
        self._conf_penalty_weight = conf_penalty_weight
        self._filter_type = filter_type
        self._top_k = top_k
        if self._filter_type == 'top_k':
            assert self._top_k is not None and self._top_k >= 1

        self.scale_scheduler = builder.build_scheduler(scale_cfg)
        self._last_scale = 0.0

        if use_class_weighting and class_weights is not None and exists(class_weights):
            with open(class_weights) as input_stream:
                weights_dict = json.load(input_stream)
            weights_dict = {int(k): float(v) for k, v in weights_dict.items()}

            class_ids = list(weights_dict)
            class_ids.sort()

            weights = np.array([weights_dict[class_id] for class_id in class_ids], dtype=np.float32)
            self.register_buffer('class_weights', torch.from_numpy(weights))
            print('[INFO] Enabled class weighting')
        else:
            self.class_weights = None

    @property
    def with_regularization(self):
        return self._conf_penalty_weight is not None and self._conf_penalty_weight > 0.0

    @property
    def with_class_weighting(self):
        return self.class_weights is not None

    @property
    def with_filtering(self):
        return self._filter_type is not None and self._filter_type in self._loss_filter_types

    @property
    def with_pr_product(self):
        return self._enable_pr_product

    @property
    def last_scale(self):
        return self._last_scale

    def update_state(self, num_iters_per_epoch):
        assert num_iters_per_epoch > 0
        self.scale_scheduler.iters_per_epoch = num_iters_per_epoch

    @staticmethod
    def _pr_product(prod):
        alpha = torch.sqrt(1.0 - prod.pow(2.0))
        out_prod = alpha.detach() * prod + prod.detach() * (1.0 - alpha)

        return out_prod

    def _regularization(self, cos_theta, scale):
        probs = F.softmax(scale * cos_theta, dim=-1)
        entropy_values = entropy(probs, dim=-1)
        out_values = np.negative(self._conf_penalty_weight) * entropy_values

        return out_values

    def _reweight(self, losses, labels):
        with torch.no_grad():
            loss_weights = torch.gather(self.class_weights, 0, labels.view(-1))

        weighted_losses = loss_weights * losses

        return weighted_losses

    def _filter_losses(self, losses):
        if self._filter_type == 'positives':
            losses = losses[losses > 0.0]
        elif self._filter_type == 'top_k':
            valid_losses = losses[losses > 0.0]

            if valid_losses.numel() > 0:
                num_top_k = int(min(valid_losses.numel(), self._top_k))
                losses, _ = torch.topk(valid_losses, k=num_top_k)
            else:
                losses = valid_losses.new_zeros((0,))

        return losses

    def forward(self, output, labels):
        self._last_scale = self.scale_scheduler.get_scale_and_increment_step()

        if self.with_pr_product:
            output = self._pr_product(output)

        losses = self._forward(output, labels, self._last_scale)

        if self.with_regularization:
            losses += self._regularization(output, self._last_scale)

        if self.with_class_weighting:
            losses = self._reweight(losses, labels)

        if self.with_filtering:
            losses = self._filter_losses(losses)

        return losses.mean() if losses.numel() > 0 else losses.sum()

    @abstractmethod
    def _forward(self, output, labels, scale):
        pass
