import torch
import torch.nn as nn
import numpy as np

from ....core.ops import conv_1x1x1_bn, HSwish, gumbel_sigmoid
from ...registry import SPATIAL_TEMPORAL_MODULES

from mmaction.ops import RoIAlign3D


@SPATIAL_TEMPORAL_MODULES.register_module
class ROIAlignSpatialTemporalModule(nn.Module):
    def __init__(self, in_channels, roi_channels, in_spatial_size=7, in_temporal_size=4,
                 out_spatial_size=5, out_temporal_size=3, gumbel=True):
        super(ROIAlignSpatialTemporalModule, self).__init__()

        self.gumbel = gumbel

        in_spatial_size =\
            in_spatial_size if not isinstance(in_spatial_size, int) else (in_spatial_size, in_spatial_size)
        out_spatial_size =\
            out_spatial_size if not isinstance(out_spatial_size, int) else (out_spatial_size, out_spatial_size)

        scales = np.array([[in_spatial_size[1], in_spatial_size[0], in_temporal_size]], dtype=np.float32)
        self.register_buffer('scales', torch.from_numpy(scales))

        self.roi_conv = nn.Sequential(
            *conv_1x1x1_bn(in_channels, roi_channels),
            HSwish(),
            nn.AvgPool3d((in_temporal_size, ) + in_spatial_size, stride=1, padding=0),
            *conv_1x1x1_bn(roi_channels, 6)
        )
        self.roi_extractor = RoIAlign3D((out_temporal_size, ) + out_spatial_size, 1., 1.)

    def init_weights(self):
        pass

    def reset_weights(self):
        self.init_weights()

    def forward(self, x, return_extra_data=False):
        roi_logits = self.roi_conv(x).view(-1, 6)
        if self.gumbel and self.training:
            raw_roi_limits = gumbel_sigmoid(roi_logits)
        else:
            raw_roi_limits = torch.sigmoid(roi_logits)

        normalized_roi_start = 0.5 * raw_roi_limits[:, :3]
        normalized_roi_end = 1.0 - 0.5 * raw_roi_limits[:, 3:]

        roi_start = normalized_roi_start * self.scales
        roi_end = normalized_roi_end * self.scales

        batch_ids = torch.arange(x.size(0), dtype=x.dtype, device=x.device).view(-1, 1)
        rois = torch.cat((batch_ids, roi_start, roi_end), dim=-1)

        out = self.roi_extractor(x, rois)

        if return_extra_data:
            return out, dict()
        else:
            return out
