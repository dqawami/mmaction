from torch.nn.modules.module import Module
from ..functions.roi_align import RoIAlign2DFunction, RoIAlign3DFunction


class RoIAlign2D(Module):

    def __init__(self, out_size, spatial_scale, sample_num=0):
        super(RoIAlign2D, self).__init__()

        self.out_size = out_size
        self.spatial_scale = float(spatial_scale)
        self.sample_num = int(sample_num)

    def forward(self, features, rois):
        return RoIAlign2DFunction.apply(features, rois, self.out_size,
                                        self.spatial_scale, self.sample_num)


class RoIAlign3D(Module):

    def __init__(self, out_size, temporal_scale, spatial_scale, sample_num=0):
        super(RoIAlign3D, self).__init__()

        self.out_size = out_size
        self.temporal_scale = float(temporal_scale)
        self.spatial_scale = float(spatial_scale)
        self.sample_num = int(sample_num)

    def forward(self, features, rois):
        return RoIAlign3DFunction.apply(features, rois, self.out_size,
                                        self.temporal_scale, self.spatial_scale,
                                        self.sample_num)
