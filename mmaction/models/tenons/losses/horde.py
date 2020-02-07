import numpy as np
import torch
import torch.nn as nn

from .base import AngleSimpleLinear
from .default import CrossEntropyLoss


class HordeLoss(nn.Module):
    """Computes HORDE loss: https://arxiv.org/pdf/1908.02735.pdf
    """

    def __init__(self, in_places, embd_size, num_classes, max_moment=5, scale=30.0):
        super(HordeLoss, self).__init__()

        assert in_places > 0
        assert embd_size > 0
        self.max_moment = max_moment
        assert self.max_moment >= 2
        self.prod_scale = 1.0 / np.sqrt(embd_size)

        for k in range(1, self.max_moment + 1):
            self.add_module('project_{}'.format(k), nn.Conv2d(
                in_places, embd_size, kernel_size=1, padding=0, bias=False))

        for k in range(2, self.max_moment + 1):
            self.add_module('embd_{}'.format(k), nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(embd_size, embd_size, kernel_size=1, padding=0, bias=True),
                AngleSimpleLinear(embd_size, num_classes)
            ))

        for k in range(2, self.max_moment + 1):
            self.add_module('loss_{}'.format(k), CrossEntropyLoss(start_s=scale))

        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d) and name.startswith('project_'):
                weights = np.random.randint(0, 2, m.weight.data.shape, dtype=np.int32)
                weights[weights == 0] = -1

                m.weight.data = torch.from_numpy(weights.astype(np.float32))

                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def forward(self, x, target):
        if x.ndimension() == 5:
            x = x.squeeze(2)

        assert x.ndimension() == 4

        project_module = getattr(self, 'project_{}'.format(1))
        prev_encoded = self.prod_scale * project_module(x)

        moment_losses = []
        for k in range(2, self.max_moment + 1):
            project_module = getattr(self, 'project_{}'.format(k))
            prev_encoded *= project_module(x)

            embd_module = getattr(self, 'embd_{}'.format(k))
            moment_embd = embd_module(prev_encoded)

            loss_module = getattr(self, 'loss_{}'.format(k))
            moment_losses.append(loss_module(moment_embd, target))

        return sum(moment_losses) / float(self.max_moment - 1)
