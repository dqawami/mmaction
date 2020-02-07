import json
from os.path import exists

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .. import builder
from ..registry import RECOGNIZERS
from ..tenons.losses import HordeLoss, MaxEntropyLoss, TotalVarianceLoss


@RECOGNIZERS.register_module
class ASLNet3D(nn.Module):
    def __init__(self,
                 backbone,
                 fpn=None,
                 spatial_temporal_module=None,
                 cls_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 extra_losses_cfg=None,
                 masked_num=False,
                 grad_reg_weight=None,
                 enable_extended_labels=False,
                 ext_class_map=None,
                 bn_eval=False,
                 bn_frozen=False):

        super(ASLNet3D, self).__init__()

        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen

        self.backbone = builder.build_backbone(backbone)

        if fpn is not None:
            self.fpn = builder.build_neck(fpn)
        else:
            self.fpn = None

        if spatial_temporal_module is not None:
            self.spatial_temporal_module = builder.build_spatial_temporal_module(
                spatial_temporal_module)
        else:
            raise ValueError('spatial_temporal_module should be specified')

        if cls_head is not None:
            self.cls_head = builder.build_head(cls_head)
        else:
            raise ValueError('cls_head should be specified')

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if extra_losses_cfg is not None:
            if hasattr(extra_losses_cfg, 'with_horde') and extra_losses_cfg.with_horde:
                self.horde_loss = HordeLoss(extra_losses_cfg.in_channels, extra_losses_cfg.embd_size,
                                            extra_losses_cfg.num_classes, extra_losses_cfg.max_moment,
                                            extra_losses_cfg.scale)
                self.horde_weight = extra_losses_cfg.horde_weight

        self.masked_num = masked_num
        if self.with_masked_samples:
            self.max_entropy_loss = MaxEntropyLoss(scale=5.0)
        else:
            self.max_entropy_loss = None

        self.grad_reg_weight = grad_reg_weight
        if self.with_grad_regularization:
            self.grad_reg_loss = TotalVarianceLoss(spatial_kernels=5, temporal_kernels=3, num_channels=3)
        else:
            self.grad_reg_loss = None

        if enable_extended_labels and ext_class_map is not None and exists(ext_class_map):
            with open(ext_class_map) as input_stream:
                class_map = {int(k_): int(v_) for k_, v_ in json.load(input_stream).items()}

            trg_classes, trg_classes_counts = np.unique(list(class_map.values()), return_counts=True)
            assert trg_classes[-1] + 1 == len(trg_classes)

            self.num_real_classes = len(trg_classes)
            self.max_num_clusters = np.max(trg_classes_counts)

            indices = []
            for trg_class_id, trg_class_size in enumerate(trg_classes_counts):
                for local_idx in range(trg_class_size):
                    indices.append(trg_class_id * self.max_num_clusters + local_idx)

            self.register_buffer('class_map_indices', torch.from_numpy(np.array(indices)).long())
        else:
            self.class_map_indices = None

        self.init_weights()

    @property
    def with_horde_loss(self):
        return hasattr(self, 'horde_loss') and self.horde_loss is not None

    @property
    def with_masked_samples(self):
        return self.masked_num is not None and self.masked_num > 0

    @property
    def with_grad_regularization(self):
        return self.grad_reg_weight is not None and self.grad_reg_weight > 0.0

    def init_weights(self):
        self.backbone.init_weights()
        self.spatial_temporal_module.init_weights()
        self.cls_head.init_weights()

    def reset_weights(self):
        self.backbone.reset_weights()
        self.spatial_temporal_module.reset_weights()
        self.cls_head.reset_weights()

    def update_state(self, num_iters_per_epoch):
        self.cls_head.update_state(num_iters_per_epoch)

    def features_test(self, x):
        y = self.backbone(x)

        if self.fpn is not None:
            out = self.fpn(y)
        else:
            assert len(y) == 1
            out = y[0]

        return out

    @staticmethod
    def mask_input(x, logits, num_samples):
        b, _, t, h, w = x.size()

        perm = torch.randperm(b)
        idx = perm[:num_samples]
        x_subset = x[idx]
        logits_subset = logits[idx]

        upsampled_logits = F.interpolate(logits_subset, (t, h, w), mode='nearest')
        new_x = torch.where(upsampled_logits > 0.5, torch.zeros_like(x_subset), x_subset.clone().detach())

        return new_x

    def features_train(self, x, enable_masking=False, num_extra_samples=0):
        y, extra_data = self.backbone(x, return_extra_data=True)

        if self.fpn is not None:
            out = self.fpn(y)
        else:
            assert len(y) == 1
            out = y[0]

        if enable_masking and 'spatial_att_logits' in extra_data and extra_data['spatial_att_logits'] is not None:
            extra_x = self.mask_input(x, extra_data['spatial_att_logits'], num_extra_samples)

            self.backbone.eval()
            y_extra = self.backbone(extra_x, return_extra_data=False, enable_extra_modules=False)
            self.backbone.train()

            if self.fpn is not None:
                out_extra = self.fpn(y_extra)
            else:
                assert len(y_extra) == 1
                out_extra = y_extra[0]

            out = torch.cat((out, out_extra), dim=0)

        return out, extra_data['reg_loss'], extra_data['att_loss']

    @staticmethod
    def extract_gt_logits(logits, labels):
        values = logits[torch.arange(logits.size(0), device=labels.device), labels]
        sum_values = values.sum()
        return sum_values

    def forward_train(self, x, labels):
        x.requires_grad = True
        losses = dict()

        extracted_features, features_reg_loss, features_att_loss =\
            self.features_train(x, self.with_masked_samples, self.masked_num)

        if features_reg_loss is not None:
            losses['loss_fr'] = features_reg_loss

        if features_att_loss is not None:
            losses['loss_att'] = features_att_loss

        if hasattr(self.spatial_temporal_module, 'loss'):
            reduced_features, st_extra_data = self.spatial_temporal_module(extracted_features, return_extra_data=True)
            losses.update(self.spatial_temporal_module.loss(**st_extra_data))
        else:
            reduced_features = self.spatial_temporal_module(extracted_features)

        cls_scores, cls_extra_data = self.cls_head(reduced_features, labels=labels.squeeze(),
                                                   return_extra_data=True, num_mock_samples=self.masked_num)
        if self.with_masked_samples:
            main_cls_scores = cls_scores[:-self.masked_num]
            auxiliary_cls_scores = cls_scores[-self.masked_num:]
        else:
            main_cls_scores = cls_scores
            auxiliary_cls_scores = None

        losses.update(self.cls_head.loss(main_cls_scores, labels.squeeze(), **cls_extra_data))

        if self.with_grad_regularization and 'loss_cls' in losses:
            gt_logits_sum = self.extract_gt_logits(main_cls_scores, labels)
            grad_response = torch.autograd.grad(gt_logits_sum, x, retain_graph=True)[0]
            losses['loss_greg'] = self.grad_reg_weight * self.grad_reg_loss(grad_response)

        if auxiliary_cls_scores is not None and self.max_entropy_loss is not None:
            losses['loss_u'] = self.max_entropy_loss(auxiliary_cls_scores)

        if self.with_horde_loss:
            spatial_features = F.adaptive_avg_pool3d(extracted_features, (1, None, None))
            losses['loss_horde'] = self.horde_weight * self.horde_loss(spatial_features, labels)

        return losses

    def forward_test(self, x):
        y = self.features_test(x)
        y = self.spatial_temporal_module(y)
        y = self.cls_head(y)

        if self.class_map_indices is not None:
            batch_size = x.shape[0]
            negative_values = torch.full((batch_size, self.num_real_classes * self.max_num_clusters), -1.0,
                                         device=y.device, dtype=y.dtype)
            tiled_indices = self.class_map_indices.expand(batch_size, -1)

            y = negative_values.scatter_(1, tiled_indices, y)
            y = y.view(-1, self.num_real_classes, self.max_num_clusters)
            y, _ = torch.max(y, dim=-1)

        return y

    def forward_embd(self, x):
        extracted_features = self.features_test(x)
        reduced_features = self.spatial_temporal_module(extracted_features)

        _, extra_data = self.cls_head(reduced_features, return_extra_data=True)
        unnormalized_embeddings = extra_data['unnormalized_embeddings']
        normalized_embeddings = F.normalize(unnormalized_embeddings, dim=-1)

        return normalized_embeddings

    def forward(self, img_group, gt_label=None, return_loss=True, return_embd=False, **kwargs):
        if return_loss:
            assert gt_label is not None

            return self.forward_train(img_group, gt_label)
        elif return_embd:
            return self.forward_embd(img_group)
        else:
            return self.forward_test(img_group)

    def train(self, train_mode=True):
        super(ASLNet3D, self).train(train_mode)

        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm3d):
                    m.eval()

                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False

        return self


@RECOGNIZERS.register_module
class ASLNet3D_Inference(ASLNet3D):
    def __init__(self, *args, **kwargs):

        super(ASLNet3D_Inference, self).__init__(*args, **kwargs)

    def forward(self, x):
        return self.forward_test(x)
