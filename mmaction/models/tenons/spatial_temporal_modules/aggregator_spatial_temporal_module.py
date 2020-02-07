import torch.nn as nn

from ...registry import SPATIAL_TEMPORAL_MODULES
from ...builder import build_spatial_temporal_module


@SPATIAL_TEMPORAL_MODULES.register_module
class AggregatorSpatialTemporalModule(nn.Module):
    def __init__(self, modules):
        super(AggregatorSpatialTemporalModule, self).__init__()

        self.num_modules = len(modules)
        assert self.num_modules > 0

        for i, module_config in enumerate(modules):
            module = build_spatial_temporal_module(module_config)
            self.add_module('module_{}'.format(i), module)

    def init_weights(self):
        for i in range(self.num_modules):
            module = getattr(self, 'module_{}'.format(i))
            module.init_weights()

    def reset_weights(self):
        self.init_weights()

    def forward(self, x, return_extra_data=False):
        y = x
        extra_data = dict()

        for i in range(self.num_modules):
            module = getattr(self, 'module_{}'.format(i))

            if return_extra_data:
                y, module_extra_data = module(y, return_extra_data=return_extra_data)
                extra_data.update(module_extra_data)
            else:
                y = module(y, return_extra_data=return_extra_data)

        if return_extra_data:
            return y, extra_data
        else:
            return y

    def loss(self, **extra_data):
        losses = dict()
        for i in range(self.num_modules):
            module = getattr(self, 'module_{}'.format(i))
            if hasattr(module, 'loss'):
                losses.update(module.loss(**extra_data))

        return losses
