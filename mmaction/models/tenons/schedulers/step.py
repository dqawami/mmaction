import numpy as np

from ...registry import SCHEDULERS
from .base import SchedulerBase


@SCHEDULERS.register_module
class StepScheduler(SchedulerBase):
    def __init__(self, scales, epochs):
        super(StepScheduler, self).__init__()

        assert len(scales) == len(epochs) + 1
        assert len(scales) > 0

        self._scales = list(scales)
        self._epochs = list(epochs) + [np.iinfo(np.int32).max]

    def _get_scale(self, step, iters_per_epoch):
        out_scale_idx = 0
        for epoch in self._epochs:
            end_step = iters_per_epoch * epoch

            if step < end_step:
                break

            out_scale_idx += 1

        return float(self._scales[out_scale_idx])
