from .base import AngleSimpleLinear, AngleMultipleLinear, entropy
from .default import CrossEntropyLoss
from .am_softmax import AMSoftmaxLoss, focal_loss
from .d_softmax import DSoftmaxLoss
from .arc_softmax import ArcLoss
from .horde import HordeLoss
from .max_entropy import MaxEntropyLoss
from .local_push import LocalPushLoss
from .total_variance_loss import TotalVarianceLoss

__all__ = ['CrossEntropyLoss',
           'AngleSimpleLinear', 'AngleMultipleLinear', 'entropy',
           'AMSoftmaxLoss', 'focal_loss',
           'DSoftmaxLoss',
           'ArcLoss',
           'HordeLoss',
           'MaxEntropyLoss',
           'LocalPushLoss',
           'TotalVarianceLoss']
