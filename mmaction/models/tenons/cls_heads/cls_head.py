import json
from os.path import exists

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ....core.ops import conv_1x1x1_bn, normalize
from ... import builder
from ...registry import HEADS
from ..losses import AngleMultipleLinear


@HEADS.register_module
class ClsHead(nn.Module):
    """Simplest classification head"""

    def __init__(self,
                 with_avg_pool=True,
                 temporal_size=1,
                 spatial_size=7,
                 dropout_ratio=0.8,
                 in_channels=2048,
                 num_classes=101,
                 init_std=0.01,
                 embedding=False,
                 embd_size=128,
                 num_centers=1,
                 st_scale=5.0,
                 reg_weight=1.0,
                 reg_threshold=0.1,
                 angle_std=None,
                 class_counts=None,
                 main_loss_cfg=None,
                 extra_losses_cfg=None):
        super(ClsHead, self).__init__()

        self.with_avg_pool = with_avg_pool
        self.in_channels = in_channels
        self.embd_size = embd_size
        self.temporal_feature_size = temporal_size
        self.spatial_feature_size = spatial_size if not isinstance(spatial_size, int) else (spatial_size, spatial_size)
        self.init_std = init_std
        self.with_embedding = embedding
        self.with_regularization = self.with_embedding and reg_weight is not None and reg_weight > 0.0

        if dropout_ratio is not None and dropout_ratio > 0.0:
            self.dropout = nn.Dropout(p=dropout_ratio)
        else:
            self.dropout = None

        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool3d((temporal_size, ) + spatial_size, stride=1, padding=0)

        if self.with_embedding:
            if in_channels != self.embd_size:
                self.fc_pre_angular = conv_1x1x1_bn(in_channels, self.embd_size, as_list=False)
            else:
                self.fc_pre_angular = None

            self.fc_angular = AngleMultipleLinear(self.embd_size, num_classes,
                                                  num_centers, st_scale,
                                                  reg_weight, reg_threshold)
        else:
            self.fc_cls_out = nn.Linear(in_channels, num_classes)

        if main_loss_cfg is None:
            main_loss_cfg = dict(type='CrossEntropyLoss')
        self.head_loss = builder.build_cl_loss(main_loss_cfg)

        self.extra_losses = dict()
        if extra_losses_cfg is not None:
            assert isinstance(extra_losses_cfg, dict)

            self.extra_losses = {loss_name: builder.build_ml_loss(cfg) for loss_name, cfg in extra_losses_cfg.items()}

        self.enable_sampling = angle_std is not None and angle_std > 0.0
        if self.enable_sampling:
            assert angle_std < np.pi / 2.0

            self.adaptive_sampling = class_counts is not None and exists(class_counts)
            if self.adaptive_sampling:
                with open(class_counts) as input_stream:
                    counts_dict = json.load(input_stream)
                counts_dict = {int(k): float(v) for k, v in counts_dict.items()}

                class_ids = list(counts_dict)
                class_ids.sort()

                counts = np.array([counts_dict[class_id] for class_id in class_ids], dtype=np.float32)
                class_angle_std = angle_std / np.power(counts, 1. / 4.)

                self.register_buffer('angle_std', torch.from_numpy(class_angle_std))
                print('[INFO] Enabled adaptive sampling')
            else:
                self.angle_std = angle_std

    def init_weights(self):
        if not self.with_embedding:
            nn.init.normal_(self.fc_cls_out.weight, 0, self.init_std)
            nn.init.constant_(self.fc_cls_out.bias, 0)

    def reset_weights(self):
        self.init_weights()

    def update_state(self, num_iters_per_epoch):
        self.head_loss.update_state(num_iters_per_epoch)

    def _squash_features(self, x):
        if x.ndimension() == 4:
            x = x.unsqueeze(2)

        if self.with_avg_pool:
            x = self.avg_pool(x)

        return x

    def forward(self, x, labels=None, return_extra_data=False, num_mock_samples=None):
        x = self._squash_features(x)

        if self.dropout is not None:
            x = self.dropout(x)

        if self.with_embedding:
            unnormalized_embd = self.fc_pre_angular(x) if self.fc_pre_angular is not None else x
            unnormalized_embd = unnormalized_embd.view(-1, self.embd_size)

            normalized_embd = normalize(unnormalized_embd, dim=1)
            if self.training and self.enable_sampling:
                with torch.no_grad():
                    unit_directions = F.normalize(torch.randn_like(normalized_embd), dim=1)
                    dot_prod = torch.sum(normalized_embd * unit_directions, dim=1, keepdim=True)
                    orthogonal_directions = unit_directions - dot_prod * normalized_embd

                    if self.adaptive_sampling and labels is not None:
                        batch_size = x.shape[0]
                        all_angle_std = self.angle_std.expand(batch_size, -1)
                        class_indices = torch.arange(batch_size, device=labels.device)
                        angle_std = all_angle_std[class_indices, labels].view(-1, 1)
                        if num_mock_samples is not None and num_mock_samples > 0:
                            mock_std = torch.full((num_mock_samples, 1), self.angle_std,
                                                  dtype=angle_std.dtype, device=angle_std.device)
                            angle_std = torch.cat((angle_std, mock_std), dim=0)
                    else:
                        angle_std = self.angle_std

                    angles = angle_std * torch.randn_like(dot_prod)
                    alpha = torch.clamp_max(torch.where(angles > 0.0, angles, torch.neg(angles)), 0.5 * np.pi)
                    cos_alpha = torch.cos(alpha)
                    sin_alpha = torch.sin(alpha)

                normalized_embd = cos_alpha * normalized_embd + sin_alpha * orthogonal_directions

            cls_score = self.fc_angular(normalized_embd)
        else:
            normalized_embd = None
            cls_score = self.fc_cls_out(x.view(-1, self.in_channels))

        if return_extra_data:
            if num_mock_samples is not None and num_mock_samples > 0:
                normalized_embd = normalized_embd[:-num_mock_samples]

            return cls_score, dict(normalized_embeddings=normalized_embd)
        else:
            return cls_score

    def loss(self, cls_score, labels, normalized_embeddings):
        losses = dict()
        losses['loss_cls'] = self.head_loss(cls_score, labels)
        losses['scale'] = self.head_loss.last_scale

        for extra_loss_name, extra_loss in self.extra_losses.items():
            losses[extra_loss_name] = extra_loss(normalized_embeddings, cls_score, labels)

        if self.with_embedding:
            losses.update(self.fc_angular.loss())

        return losses


@HEADS.register_module
class ClsHead_Inference(ClsHead):
    def __init__(self, *args, **kwargs):
        super(ClsHead_Inference, self).__init__(*args, **kwargs)

        self.dropout = None
        self.extra_losses = dict()

    def forward(self, x):
        x = self._squash_features(x)

        if self.with_embedding:
            unnormalized_embd = self.fc_pre_angular(x).view(-1, self.embd_size)
            normalized_embd = normalize(unnormalized_embd, dim=1)

            cls_score = self.fc_angular(normalized_embd)
        else:
            cls_score = self.fc_cls_out(x)

        return cls_score
