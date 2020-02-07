import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, kaiming_init

from ....core.ops import HSwish, conv_1xkxk_bn, conv_kx1x1_bn, conv_1x1x1_bn
from ...registry import NECKS


class SimpleInvertedResidual(nn.Module):
    def __init__(self, in_planes, hidden_dim, out_planes, spatial_kernels, temporal_kernels,
                 identity=True, norm='none'):
        super(SimpleInvertedResidual, self).__init__()

        self.identity = identity and in_planes == out_planes

        self.conv = nn.Sequential(
            # pw
            *conv_1x1x1_bn(in_planes, hidden_dim, norm=norm),
            HSwish(),
            # dw
            *conv_1xkxk_bn(hidden_dim, hidden_dim, spatial_kernels, groups=hidden_dim, norm=norm),
            HSwish(),
            # pw
            *conv_kx1x1_bn(hidden_dim, out_planes, temporal_kernels, norm=norm)
        )

    def forward(self, x):
        y = self.conv(x)
        return x + y if self.identity else y


@NECKS.register_module
class FPNUpscale(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 internal_channels,
                 norm='none'):
        super(FPNUpscale, self).__init__()

        assert isinstance(in_channels, list)
        self.num_inputs = len(in_channels)

        self.lateral_convs = nn.ModuleList()
        for i in range(self.num_inputs):
            self.lateral_convs.append(nn.Sequential(
                *conv_1x1x1_bn(in_channels[i], internal_channels, norm=norm),
                HSwish()
            ))

        self.out_conv = nn.Sequential(
            *conv_1x1x1_bn(internal_channels, out_channels, norm=norm),
            HSwish()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm3d):
                constant_init(m, 1.0, 0.0)

    def forward(self, inputs):
        assert len(inputs) == self.num_inputs

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        for i in range(self.num_inputs - 1, 0, -1):
            src_features = laterals[i]
            trg_features = laterals[i - 1]

            src_features_shape = list(src_features.shape[2:])
            trg_features_shape = list(trg_features.shape[2:])
            scale_factor = tuple([t / s for s, t in zip(src_features_shape, trg_features_shape)])

            laterals[i - 1] += F.interpolate(laterals[i], scale_factor=scale_factor, mode='nearest')

        # build output
        out = self.out_conv(laterals[0])

        return out
