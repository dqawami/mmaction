#include <torch/extension.h>

#include <cmath>
#include <vector>

int ROIAlignForwardLauncher(const at::Tensor features, const at::Tensor rois,
                            const float temporal_scale, const float spatial_scale,
                            const int sample_num,
                            const int channels, const int length, const int height, const int width,
                            const int num_rois,
                            const int pooled_length, const int pooled_height, const int pooled_width,
                            at::Tensor output);

int ROIAlignBackwardLauncher(const at::Tensor top_grad, const at::Tensor rois,
                             const float temporal_scale, const float spatial_scale,
                             const int sample_num,
                             const int channels, const int length, const int height, const int width,
                             const int num_rois,
                             const int pooled_length, const int pooled_height, const int pooled_width,
                             at::Tensor bottom_grad);

int ROIAlignCoordBackwardLauncher(const at::Tensor top_grad, const at::Tensor features, const at::Tensor rois,
                                  const float temporal_scale, const float spatial_scale,
                                  const int sample_num,
                                  const int channels, const int length, const int height, const int width,
                                  const int num_rois,
                                  const int pooled_length, const int pooled_height, const int pooled_width,
                                  at::Tensor bottom_grad);

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

int roi_align_forward_cuda(at::Tensor features, at::Tensor rois,
                           int pooled_length, int pooled_height, int pooled_width,
                           float temporal_scale, float spatial_scale,
                           int sample_num, at::Tensor output) {
  CHECK_INPUT(features);
  CHECK_INPUT(rois);
  CHECK_INPUT(output);

  // Number of ROIs
  int num_rois = rois.size(0);
  int size_rois = rois.size(1);

  if (size_rois != 7) {
    printf("wrong roi size\n");
    return 0;
  }

  int num_channels = features.size(1);
  int data_length = features.size(2);
  int data_height = features.size(3);
  int data_width = features.size(4);

  ROIAlignForwardLauncher(features, rois, temporal_scale, spatial_scale, sample_num,
                          num_channels, data_length, data_height, data_width, num_rois,
                          pooled_length, pooled_height, pooled_width, output);

  return 1;
}

int roi_align_backward_cuda(at::Tensor top_grad, at::Tensor rois,
                            int pooled_length, int pooled_height, int pooled_width,
                            float temporal_scale, float spatial_scale,
                            int sample_num, at::Tensor bottom_grad) {
  CHECK_INPUT(top_grad);
  CHECK_INPUT(rois);
  CHECK_INPUT(bottom_grad);

  // Number of ROIs
  int num_rois = rois.size(0);
  int size_rois = rois.size(1);
  if (size_rois != 7) {
    printf("wrong roi size\n");
    return 0;
  }

  int num_channels = bottom_grad.size(1);
  int data_length = bottom_grad.size(2);
  int data_height = bottom_grad.size(3);
  int data_width = bottom_grad.size(4);

  ROIAlignBackwardLauncher(top_grad, rois, temporal_scale, spatial_scale, sample_num,
                           num_channels, data_length, data_height, data_width, num_rois,
                           pooled_length, pooled_height, pooled_width, bottom_grad);

  return 1;
}

int roi_align_coord_backward_cuda(at::Tensor top_grad, at::Tensor features, at::Tensor rois,
                                  int pooled_length, int pooled_height, int pooled_width,
                                  float temporal_scale, float spatial_scale,
                                  int sample_num, at::Tensor bottom_grad) {
  CHECK_INPUT(top_grad);
  CHECK_INPUT(features);
  CHECK_INPUT(rois);
  CHECK_INPUT(bottom_grad);

  // Number of ROIs
  int num_rois = rois.size(0);
  int size_rois = rois.size(1);
  if (size_rois != 7) {
    printf("wrong roi size\n");
    return 0;
  }

  int num_channels = features.size(1);
  int data_length = features.size(2);
  int data_height = features.size(3);
  int data_width = features.size(4);

  ROIAlignCoordBackwardLauncher(top_grad, features, rois, temporal_scale, spatial_scale, sample_num,
                                num_channels, data_length, data_height, data_width, num_rois,
                                pooled_length, pooled_height, pooled_width, bottom_grad);

  return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &roi_align_forward_cuda, "Roi_Align forward (CUDA)");
  m.def("backward", &roi_align_backward_cuda, "Roi_Align backward (CUDA)");
  m.def("coord_backward", &roi_align_coord_backward_cuda, "Roi_Align coordinate backward (CUDA)");
}
