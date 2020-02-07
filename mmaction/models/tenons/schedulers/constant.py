from ...registry import SCHEDULERS
from .base import SchedulerBase


@SCHEDULERS.register_module
class ConstantScheduler(SchedulerBase):
    def __init__(self, scale=30.0):
        super(ConstantScheduler, self).__init__()

        self._end_s = scale
        assert self._end_s > 0.0

    def _get_scale(self, step, iters_per_epoch):
        return self._end_s
