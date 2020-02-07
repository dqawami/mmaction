import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TotalVarianceLoss(nn.Module):
    def __init__(self, spatial_kernels, temporal_kernels, num_channels,
                 hard_values=False, limits=None, threshold=None):
        super(TotalVarianceLoss, self).__init__()

        self.num_channels = num_channels
        self.padding = (temporal_kernels - 1) // 2, (spatial_kernels - 1) // 2, (spatial_kernels - 1) // 2

        weights = np.ones([num_channels, 1, temporal_kernels, spatial_kernels, spatial_kernels], dtype=np.float32)
        weights /= temporal_kernels * spatial_kernels * spatial_kernels
        self.register_buffer('weights', torch.from_numpy(weights))

        self.hard_values = hard_values
        self.limits = limits
        assert len(self.limits) == 2
        assert self.limits[0] < self.limits[1]
        self.threshold = threshold

    def forward(self, values):
        soft_values = F.conv3d(values, self.weights, None, 1, self.padding, 1, self.num_channels)

        if self.hard_values:
            trg_values = torch.where(soft_values < self.threshold,
                                     torch.full_like(soft_values, self.limits[0]),
                                     torch.full_like(soft_values, self.limits[1]))
        else:
            trg_values = soft_values

        losses = torch.abs(values - trg_values)
        out = losses.mean()

        return out
