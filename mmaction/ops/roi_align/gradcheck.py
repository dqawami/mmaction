import numpy as np
import torch
from torch.autograd import gradcheck

import os.path as osp
import sys
sys.path.append(osp.abspath(osp.join(__file__, '../../')))
from roi_align import RoIAlign2D, RoIAlign3D  # noqa: E402


FEATURE_LENGTH = 4
FEATURE_SIZE = 15
FEATURES_NUM_CHANNELS = 16
NUM_IMAGES = 2
NUM_ROIS = 20
TEMPORAL_SCALE = 1.0 / 4.
SPATIAL_SCALE = 1.0 / 8.


def check_2d():
    img_size = FEATURE_SIZE / SPATIAL_SCALE

    batch_ind = np.random.randint(NUM_IMAGES, size=(NUM_ROIS, 1))
    rois = np.random.rand(NUM_ROIS, 4) * img_size * 0.5
    rois[:, 2:] += img_size * 0.5
    rois = np.hstack((batch_ind, rois))

    feat = torch.randn(NUM_IMAGES, FEATURES_NUM_CHANNELS, FEATURE_SIZE, FEATURE_SIZE,
                       requires_grad=True, device='cuda:0')
    rois = torch.from_numpy(rois).float().cuda()
    inputs = (feat, rois)

    test1 = gradcheck(RoIAlign2D(3, SPATIAL_SCALE), inputs, atol=1e-3, eps=1e-3)
    test2 = gradcheck(RoIAlign2D(3, SPATIAL_SCALE, 2), inputs, atol=1e-3, eps=1e-3)
    print(test1, test2)


def check_3d():
    seq_length = FEATURE_LENGTH / TEMPORAL_SCALE
    img_size = FEATURE_SIZE / SPATIAL_SCALE

    batch_ind = np.random.randint(NUM_IMAGES, size=(NUM_ROIS, 1))
    spatial_rois = np.random.rand(NUM_ROIS, 4) * img_size * 0.5
    spatial_rois[:, 2:] += img_size * 0.5
    temporal_rois = np.random.rand(NUM_ROIS, 2) * seq_length * 0.5
    temporal_rois[:, 1:] += seq_length * 0.5
    rois = np.hstack((batch_ind, spatial_rois[:, :2], temporal_rois[:, :1], spatial_rois[:, 2:], temporal_rois[:, 1:]))

    feat = torch.randn(NUM_IMAGES, FEATURES_NUM_CHANNELS, FEATURE_LENGTH, FEATURE_SIZE, FEATURE_SIZE,
                       requires_grad=True, device='cuda:0')
    rois = torch.from_numpy(rois).float().cuda()
    rois.requires_grad = True
    inputs = (feat, rois)

    test1 = gradcheck(RoIAlign3D(3, TEMPORAL_SCALE, SPATIAL_SCALE), inputs, atol=1e-3, eps=1e-3)
    test2 = gradcheck(RoIAlign3D(3, TEMPORAL_SCALE, SPATIAL_SCALE, 2), inputs, atol=1e-3, eps=1e-3)
    print(test1, test2)


if __name__ == '__main__':
    print('\nChecking ROI Align 2D...')
    check_2d()

    print('\nChecking ROI Align 3D...')
    check_3d()
