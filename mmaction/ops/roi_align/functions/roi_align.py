from torch.autograd import Function

from .. import roi_align_2d_cuda, roi_align_3d_cuda


class RoIAlign2DFunction(Function):

    @staticmethod
    def forward(ctx, features, rois, out_size, spatial_scale, sample_num=0):
        if isinstance(out_size, int):
            out_h = out_size
            out_w = out_size
        elif isinstance(out_size, tuple):
            assert len(out_size) == 2
            assert isinstance(out_size[0], int)
            assert isinstance(out_size[1], int)
            out_h, out_w = out_size
        else:
            raise TypeError('"out_size" must be an integer or tuple of integers')
        ctx.spatial_scale = spatial_scale
        ctx.sample_num = sample_num
        ctx.save_for_backward(rois)
        ctx.feature_size = features.size()

        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0)

        output = features.new_zeros(num_rois, num_channels, out_h, out_w)
        if features.is_cuda:
            roi_align_2d_cuda.forward(features, rois, out_h, out_w, spatial_scale,
                                      sample_num, output)
        else:
            raise NotImplementedError

        return output

    @staticmethod
    def backward(ctx, grad_output):
        feature_size = ctx.feature_size
        spatial_scale = ctx.spatial_scale
        sample_num = ctx.sample_num
        rois = ctx.saved_tensors[0]
        assert (feature_size is not None and grad_output.is_cuda)

        batch_size, num_channels, data_height, data_width = feature_size
        out_w = grad_output.size(3)
        out_h = grad_output.size(2)

        grad_input = grad_rois = None
        if ctx.needs_input_grad[0]:
            grad_input = rois.new_zeros(batch_size, num_channels, data_height,
                                        data_width)
            roi_align_2d_cuda.backward(grad_output.contiguous(), rois, out_h,
                                       out_w, spatial_scale, sample_num,
                                       grad_input)

        return grad_input, grad_rois, None, None, None


class RoIAlign3DFunction(Function):

    @staticmethod
    def forward(ctx, features, rois, out_size, temporal_scale, spatial_scale, sample_num=0):
        if isinstance(out_size, int):
            out_l = out_size
            out_h = out_size
            out_w = out_size
        elif isinstance(out_size, tuple):
            assert len(out_size) == 3
            assert isinstance(out_size[0], int)
            assert isinstance(out_size[1], int)
            assert isinstance(out_size[2], int)

            out_l, out_h, out_w = out_size
        else:
            raise TypeError('"out_size" must be an integer or tuple of integers')

        ctx.temporal_scale = temporal_scale
        ctx.spatial_scale = spatial_scale
        ctx.sample_num = sample_num
        ctx.save_for_backward(features, rois)

        batch_size, num_channels, data_length, data_height, data_width = features.size()
        num_rois = rois.size(0)

        output = features.new_zeros(num_rois, num_channels, out_l, out_h, out_w)
        if features.is_cuda:
            roi_align_3d_cuda.forward(features, rois,
                                      out_l, out_h, out_w,
                                      temporal_scale, spatial_scale,
                                      sample_num, output)
        else:
            raise NotImplementedError

        return output

    @staticmethod
    def backward(ctx, grad_output):
        temporal_scale = ctx.temporal_scale
        spatial_scale = ctx.spatial_scale
        sample_num = ctx.sample_num
        features, rois = ctx.saved_tensors

        feature_size = features.size()
        rois_size = rois.size()

        assert feature_size is not None and rois_size is not None and grad_output.is_cuda

        out_l = grad_output.size(2)
        out_h = grad_output.size(3)
        out_w = grad_output.size(4)

        grad_input = grad_rois = None

        if ctx.needs_input_grad[0]:
            batch_size, num_channels, data_length, data_height, data_width = feature_size
            grad_input = features.new_zeros(batch_size, num_channels, data_length, data_height, data_width)

            roi_align_3d_cuda.backward(grad_output.contiguous(),
                                       rois.contiguous(),
                                       out_l, out_h, out_w,
                                       temporal_scale, spatial_scale,
                                       sample_num, grad_input)

        if ctx.needs_input_grad[1]:
            num_rois, num_coordinates = rois_size
            grad_rois = rois.new_zeros(num_rois, num_coordinates)

            roi_align_3d_cuda.coord_backward(grad_output.contiguous(),
                                             features.contiguous(), rois.contiguous(),
                                             out_l, out_h, out_w,
                                             temporal_scale, spatial_scale,
                                             sample_num, grad_rois)

        return grad_input, grad_rois, None, None, None, None


roi_align_2d = RoIAlign2DFunction.apply
roi_align_3d = RoIAlign3DFunction.apply
