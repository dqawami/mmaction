#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 512

inline int GET_BLOCKS(const int N) {
  int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int max_block_num = 65000;
  return min(optimal_block_num, max_block_num);
}

template <typename scalar_t>
__device__ scalar_t bilinear_interpolate(
    const scalar_t *bottom_data,
    const int length, const int height, const int width,
    scalar_t t, scalar_t y, scalar_t x) {
  // deal with cases that inverse elements are out of feature map boundary
  if (t < -1.0 || t > length || y < -1.0 || y > height || x < -1.0 || x > width) {
    return 0;
  }

  if (t <= 0) t = 0;
  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  int t_low = (int)t;
  int y_low = (int)y;
  int x_low = (int)x;

  int t_high;
  if (t_low >= length - 1) {
    t_high = t_low = length - 1;
    t = (scalar_t)t_low;
  } else {
    t_high = t_low + 1;
  }

  int y_high;
  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (scalar_t)y_low;
  } else {
    y_high = y_low + 1;
  }

  int x_high;
  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (scalar_t)x_low;
  } else {
    x_high = x_low + 1;
  }

  scalar_t lt = t - t_low;
  scalar_t ly = y - y_low;
  scalar_t lx = x - x_low;
  scalar_t ht = 1. - lt;
  scalar_t hy = 1. - ly;
  scalar_t hx = 1. - lx;

  // do trilinear interpolation
  scalar_t lt0 = bottom_data[t_low * height * width + y_low * width + x_low];
  scalar_t rt0 = bottom_data[t_low * height * width + y_low * width + x_high];
  scalar_t lb0 = bottom_data[t_low * height * width + y_high * width + x_low];
  scalar_t rb0 = bottom_data[t_low * height * width + y_high * width + x_high];
  scalar_t lt1 = bottom_data[t_high * height * width + y_low * width + x_low];
  scalar_t rt1 = bottom_data[t_high * height * width + y_low * width + x_high];
  scalar_t lb1 = bottom_data[t_high * height * width + y_high * width + x_low];
  scalar_t rb1 = bottom_data[t_high * height * width + y_high * width + x_high];

  scalar_t val = ht * (hy * (hx * lt0 + lx * rt0) +
                       ly * (hx * lb0 + lx * rb0)) +
                 lt * (hy * (hx * lt1 + lx * rt1) +
                       ly * (hx * lb1 + lx * rb1));

  return val;
}

template <typename scalar_t>
__global__ void ROIAlignForward(
    const int num_threads,
    const scalar_t *bottom_data, const scalar_t *bottom_rois,
    const scalar_t temporal_scale, const scalar_t spatial_scale, const int sample_num,
    const int channels, const int length, const int height, const int width,
    const int pooled_length, const int pooled_height, const int pooled_width,
    scalar_t *top_data) {
  CUDA_1D_KERNEL_LOOP(index, num_threads) {
    // (n, c, pt, ph, pw) is an element in the aligned output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int pt = (index / pooled_width / pooled_height) % pooled_length;
    int c = (index / pooled_width / pooled_height / pooled_length) % channels;
    int n = index / pooled_width / pooled_height / pooled_length / channels;

    const scalar_t *offset_bottom_rois = bottom_rois + n * 7;
    int roi_batch_ind = offset_bottom_rois[0];
    scalar_t roi_start_w = offset_bottom_rois[1] * spatial_scale;
    scalar_t roi_start_h = offset_bottom_rois[2] * spatial_scale;
    scalar_t roi_start_t = offset_bottom_rois[3] * temporal_scale;
    scalar_t roi_end_w = offset_bottom_rois[4] * spatial_scale;
    scalar_t roi_end_h = offset_bottom_rois[5] * spatial_scale;
    scalar_t roi_end_t = offset_bottom_rois[6] * temporal_scale;

    // Force malformed ROIs to be 1x1
    scalar_t roi_width = fmaxf((scalar_t)roi_end_w - roi_start_w, 0.);
    scalar_t roi_height = fmaxf((scalar_t)roi_end_h - roi_start_h, 0.);
    scalar_t roi_length = fmaxf((scalar_t)roi_end_t - roi_start_t, 0.);

    scalar_t bin_size_h = roi_height / pooled_height;
    scalar_t bin_size_w = roi_width / pooled_width;
    scalar_t bin_size_t = roi_length / pooled_length;

    const scalar_t *offset_bottom_data =
        bottom_data + (roi_batch_ind * channels + c) * length * height * width;

    int sample_num_t =
        (sample_num > 0) ? sample_num : ceil(roi_length / pooled_length);
    int sample_num_h =
        (sample_num > 0) ? sample_num : ceil(roi_height / pooled_height);
    int sample_num_w =
        (sample_num > 0) ? sample_num : ceil(roi_width / pooled_width);

    scalar_t output_val = 0;
    for (int it = 0; it < sample_num_t; it++) {
      const scalar_t t =
            roi_start_t + pt * bin_size_t + (scalar_t)(it + scalar_t(.5f)) * bin_size_t / (scalar_t)(sample_num_t);

      for (int iy = 0; iy < sample_num_h; iy++) {
        const scalar_t y =
            roi_start_h + ph * bin_size_h + (scalar_t)(iy + scalar_t(.5f)) * bin_size_h / (scalar_t)(sample_num_h);

        for (int ix = 0; ix < sample_num_w; ix++) {
          const scalar_t x =
              roi_start_w + pw * bin_size_w + (scalar_t)(ix + scalar_t(.5f)) * bin_size_w / (scalar_t)(sample_num_w);

          scalar_t val = bilinear_interpolate<scalar_t>(offset_bottom_data, length, height, width, t, y, x);
          output_val += val;
        }
      }
    }

    output_val /= (sample_num_t * sample_num_h * sample_num_w);
    top_data[index] = output_val;
  }
}

int ROIAlignForwardLauncher(
  const at::Tensor features, const at::Tensor rois,
  const float temporal_scale, const float spatial_scale,
  const int sample_num,
  const int channels, const int length, const int height, const int width,
  const int num_rois,
  const int pooled_length, const int pooled_height, const int pooled_width,
  at::Tensor output) {
  const int output_size = num_rois * pooled_length * pooled_height * pooled_width * channels;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      features.type(), "ROIAlignLauncherForward", ([&] {
        const scalar_t *bottom_data = features.data<scalar_t>();
        const scalar_t *rois_data = rois.data<scalar_t>();
        scalar_t *top_data = output.data<scalar_t>();

        ROIAlignForward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, bottom_data, rois_data,
                scalar_t(temporal_scale), scalar_t(spatial_scale),
                sample_num, channels, length, height, width,
                pooled_length, pooled_height, pooled_width,
                top_data);
      }));
  THCudaCheck(cudaGetLastError());

  return 1;
}

template <typename scalar_t>
__device__ void bilinear_interpolate_gradient(
    const int length, const int height, const int width,
    scalar_t t, scalar_t y, scalar_t x,
    scalar_t &w1, scalar_t &w2, scalar_t &w3, scalar_t &w4,
    scalar_t &w5, scalar_t &w6, scalar_t &w7, scalar_t &w8,
    int &t_low, int &t_high, int &y_low, int &y_high, int &x_low, int &x_high) {
  // deal with cases that inverse elements are out of feature map boundary
  if (t < -1.0 || t > length || y < -1.0 || y > height || x < -1.0 || x > width) {
    w1 = w2 = w3 = w4 = w5 = w6 = w7 = w8 = 0.;
    t_low = t_high = y_low = y_high = x_low = x_high =  -1;
    return;
  }

  if (t <= 0) t = 0;
  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  t_low = (int)t;
  y_low = (int)y;
  x_low = (int)x;

  if (t_low >= length - 1) {
    t_high = t_low = length - 1;
    t = (scalar_t)t_low;
  } else {
    t_high = t_low + 1;
  }

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (scalar_t)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (scalar_t)x_low;
  } else {
    x_high = x_low + 1;
  }

  scalar_t lt = t - t_low;
  scalar_t ly = y - y_low;
  scalar_t lx = x - x_low;
  scalar_t ht = 1. - lt;
  scalar_t hy = 1. - ly;
  scalar_t hx = 1. - lx;

  w1 = ht * hy * hx;
  w2 = ht * hy * lx;
  w3 = ht * ly * hx;
  w4 = ht * ly * lx;
  w5 = lt * hy * hx;
  w6 = lt * hy * lx;
  w7 = lt * ly * hx;
  w8 = lt * ly * lx;

  return;
}

template <typename scalar_t>
__global__ void ROIAlignBackward(
    const int nthreads, const scalar_t *top_diff, const scalar_t *bottom_rois,
    const scalar_t temporal_scale, const scalar_t spatial_scale,
    const int sample_num,
    const int channels, const int length, const int height, const int width,
    const int pooled_length, const int pooled_height, const int pooled_width,
    scalar_t *bottom_diff) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, pt, ph, pw) is an element in the aligned output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int pt = (index / pooled_width / pooled_height) % pooled_length;
    int c = (index / pooled_width / pooled_height / pooled_length) % channels;
    int n = index / pooled_width / pooled_height / pooled_length / channels;

    const scalar_t *offset_bottom_rois = bottom_rois + n * 7;
    int roi_batch_ind = offset_bottom_rois[0];
    scalar_t roi_start_w = offset_bottom_rois[1] * spatial_scale;
    scalar_t roi_start_h = offset_bottom_rois[2] * spatial_scale;
    scalar_t roi_start_t = offset_bottom_rois[3] * temporal_scale;
    scalar_t roi_end_w = offset_bottom_rois[4] * spatial_scale;
    scalar_t roi_end_h = offset_bottom_rois[5] * spatial_scale;
    scalar_t roi_end_t = offset_bottom_rois[6] * temporal_scale;

    // Force malformed ROIs to be 1x1
    scalar_t roi_width = fmaxf((scalar_t)roi_end_w - roi_start_w, 0.);
    scalar_t roi_height = fmaxf((scalar_t)roi_end_h - roi_start_h, 0.);
    scalar_t roi_length = fmaxf((scalar_t)roi_end_t - roi_start_t, 0.);

    scalar_t bin_size_h = roi_height / pooled_height;
    scalar_t bin_size_w = roi_width / pooled_width;
    scalar_t bin_size_t = roi_length / pooled_length;

    scalar_t *offset_bottom_diff =
        bottom_diff + (roi_batch_ind * channels + c) * length * height * width;
    int offset_top = (n * channels + c) * pooled_length * pooled_height * pooled_width +
                     pt * pooled_height * pooled_width +
                     ph * pooled_width + pw;
    scalar_t offset_top_diff = top_diff[offset_top];

    int sample_num_t =
        (sample_num > 0) ? sample_num : ceil(roi_length / pooled_length);
    int sample_num_h =
        (sample_num > 0) ? sample_num : ceil(roi_height / pooled_height);
    int sample_num_w =
        (sample_num > 0) ? sample_num : ceil(roi_width / pooled_width);

    const scalar_t normalizer = (scalar_t)(1) / (scalar_t)(sample_num_t * sample_num_h * sample_num_w);
    for (int it = 0; it < sample_num_t; it++) {
      const scalar_t t =
          roi_start_t + pt * bin_size_t + (scalar_t)(it + .5f) * bin_size_t / (scalar_t)(sample_num_t);

      for (int iy = 0; iy < sample_num_h; iy++) {
        const scalar_t y =
            roi_start_h + ph * bin_size_h + (scalar_t)(iy + .5f) * bin_size_h / (scalar_t)(sample_num_h);

        for (int ix = 0; ix < sample_num_w; ix++) {
          const scalar_t x =
              roi_start_w + pw * bin_size_w + (scalar_t)(ix + .5f) * bin_size_w / (scalar_t)(sample_num_w);

          scalar_t w1, w2, w3, w4, w5, w6, w7, w8;
          int x_low, x_high, y_low, y_high, t_low, t_high;
          bilinear_interpolate_gradient<scalar_t>(
              length, height, width, t, y, x,
              w1, w2, w3, w4, w5, w6, w7, w8,
              t_low, t_high, y_low, y_high, x_low, x_high);

          scalar_t g1 = offset_top_diff * w1 * normalizer;
          scalar_t g2 = offset_top_diff * w2 * normalizer;
          scalar_t g3 = offset_top_diff * w3 * normalizer;
          scalar_t g4 = offset_top_diff * w4 * normalizer;
          scalar_t g5 = offset_top_diff * w5 * normalizer;
          scalar_t g6 = offset_top_diff * w6 * normalizer;
          scalar_t g7 = offset_top_diff * w7 * normalizer;
          scalar_t g8 = offset_top_diff * w8 * normalizer;

          if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0 && t_low >= 0 && t_high >= 0) {
            atomicAdd(offset_bottom_diff + t_low * height * width + y_low * width + x_low, g1);
            atomicAdd(offset_bottom_diff + t_low * height * width + y_low * width + x_high, g2);
            atomicAdd(offset_bottom_diff + t_low * height * width + y_high * width + x_low, g3);
            atomicAdd(offset_bottom_diff + t_low * height * width + y_high * width + x_high, g4);
            atomicAdd(offset_bottom_diff + t_high * height * width + y_low * width + x_low, g5);
            atomicAdd(offset_bottom_diff + t_high * height * width + y_low * width + x_high, g6);
            atomicAdd(offset_bottom_diff + t_high * height * width + y_high * width + x_low, g7);
            atomicAdd(offset_bottom_diff + t_high * height * width + y_high * width + x_high, g8);
          }
        }
      }
    }
  }
}

int ROIAlignBackwardLauncher(
    const at::Tensor top_grad, const at::Tensor rois,
    const float temporal_scale, const float spatial_scale,
    const int sample_num,
    const int channels, const int length, const int height, const int width,
    const int num_rois,
    const int pooled_length, const int pooled_height, const int pooled_width,
    at::Tensor bottom_grad) {
  const int output_size = num_rois * pooled_length * pooled_height * pooled_width * channels;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.type(), "ROIAlignLauncherBackward", ([&] {
        const scalar_t *top_diff = top_grad.data<scalar_t>();
        const scalar_t *rois_data = rois.data<scalar_t>();
        scalar_t *bottom_diff = bottom_grad.data<scalar_t>();
        if (sizeof(scalar_t) == sizeof(double)) {
          fprintf(stderr, "double is not supported\n");
          exit(-1);
        }

        ROIAlignBackward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, top_diff, rois_data,
                scalar_t(temporal_scale), scalar_t(spatial_scale),
                sample_num, channels, length, height, width,
                pooled_length, pooled_height, pooled_width,
                bottom_diff);
      }));
  THCudaCheck(cudaGetLastError());

  return 1;
}

template <typename scalar_t>
__device__ void bilinear_interpolate_coord_gradient(
    const scalar_t *bottom_data,
    const int length, const int height, const int width,
    scalar_t t, scalar_t y, scalar_t x,
    scalar_t &w1, scalar_t &w2, scalar_t &w3,
    int &t_low, int &t_high, int &y_low, int &y_high, int &x_low, int &x_high) {
  // deal with cases that inverse elements are out of feature map boundary
  if (t < -1.0 || t > length || y < -1.0 || y > height || x < -1.0 || x > width) {
    w1 = w2 = w3 = 0.;
    t_low = t_high = y_low = y_high = x_low = x_high = -1;
    return;
  }

  if (t <= 0) t = 0;
  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  t_low = (int)t;
  y_low = (int)y;
  x_low = (int)x;

  if (t_low >= length - 1) {
    t_high = t_low = length - 1;
    t = (scalar_t)t_low;
  } else {
    t_high = t_low + 1;
  }

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (scalar_t)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (scalar_t)x_low;
  } else {
    x_high = x_low + 1;
  }

  scalar_t lt = t - t_low;
  scalar_t ly = y - y_low;
  scalar_t lx = x - x_low;
  scalar_t ht = 1. - lt;
  scalar_t hy = 1. - ly;
  scalar_t hx = 1. - lx;

  scalar_t lt0 = bottom_data[t_low * height * width + y_low * width + x_low];
  scalar_t rt0 = bottom_data[t_low * height * width + y_low * width + x_high];
  scalar_t lb0 = bottom_data[t_low * height * width + y_high * width + x_low];
  scalar_t rb0 = bottom_data[t_low * height * width + y_high * width + x_high];
  scalar_t lt1 = bottom_data[t_high * height * width + y_low * width + x_low];
  scalar_t rt1 = bottom_data[t_high * height * width + y_low * width + x_high];
  scalar_t lb1 = bottom_data[t_high * height * width + y_high * width + x_low];
  scalar_t rb1 = bottom_data[t_high * height * width + y_high * width + x_high];

  // = ht * (hy * (hx * lt0 + lx * rt0) + ly * (hx * lb0 + lx * rb0)) +
  //   lt * (hy * (hx * lt1 + lx * rt1) + ly * (hx * lb1 + lx * rb1));

  w1 = ht * (hy * (rt0 - lt0) + ly * (rb0 - lb0)) + lt * (hy * (rt1 - lt1) + ly * (rb1 - lb1));
  w2 = ht * (hx * (lb0 - lt0) + lx * (rb0 - rt0)) + lt * (hx * (lb1 - lt1) + lx * (rb1 - rt1));
  w3 = hy * (hx * (lt1 - lt0) + lx * (rt1 - rt0)) + ly * (hx * (lb1 - lb0) + lx * (rb1 - rb0));

  return;
}

template <typename scalar_t>
__global__ void ROIAlignCoordBackward(
    const int nthreads, const scalar_t *top_diff,
    const scalar_t *bottom_data, const scalar_t *bottom_rois,
    const scalar_t temporal_scale, const scalar_t spatial_scale,
    const int sample_num,
    const int channels, const int length, const int height, const int width,
    const int pooled_length, const int pooled_height, const int pooled_width,
    scalar_t *bottom_diff) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, pt, ph, pw) is an element in the aligned output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int pt = (index / pooled_width / pooled_height) % pooled_length;
    int c = (index / pooled_width / pooled_height / pooled_length) % channels;
    int n = index / pooled_width / pooled_height / pooled_length / channels;

    const scalar_t *offset_bottom_rois = bottom_rois + n * 7;
    int roi_batch_ind = offset_bottom_rois[0];
    scalar_t roi_start_w = offset_bottom_rois[1] * spatial_scale;
    scalar_t roi_start_h = offset_bottom_rois[2] * spatial_scale;
    scalar_t roi_start_t = offset_bottom_rois[3] * temporal_scale;
    scalar_t roi_end_w = offset_bottom_rois[4] * spatial_scale;
    scalar_t roi_end_h = offset_bottom_rois[5] * spatial_scale;
    scalar_t roi_end_t = offset_bottom_rois[6] * temporal_scale;

    scalar_t roi_width = fmaxf((scalar_t)roi_end_w - roi_start_w, 0.);
    scalar_t roi_height = fmaxf((scalar_t)roi_end_h - roi_start_h, 0.);
    scalar_t roi_length = fmaxf((scalar_t)roi_end_t - roi_start_t, 0.);

    scalar_t bin_size_h = roi_height / pooled_height;
    scalar_t bin_size_w = roi_width / pooled_width;
    scalar_t bin_size_t = roi_length / pooled_length;

    const scalar_t *offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * length * height * width;
    scalar_t *offset_bottom_diff = bottom_diff + n * 7;
    scalar_t offset_top_diff = top_diff[(n * channels + c) * pooled_length * pooled_height * pooled_width +
                                        pt * pooled_height * pooled_width + ph * pooled_width + pw];

    int sample_num_t =
        (sample_num > 0) ? sample_num : ceil(roi_length / pooled_length);
    int sample_num_h =
        (sample_num > 0) ? sample_num : ceil(roi_height / pooled_height);
    int sample_num_w =
        (sample_num > 0) ? sample_num : ceil(roi_width / pooled_width);

    const scalar_t scale = offset_top_diff / (scalar_t)(sample_num_t * sample_num_h * sample_num_w);
    for (int it = 0; it < sample_num_t; it++) {
      const scalar_t t =
          roi_start_t + pt * bin_size_t + (scalar_t)(it + .5f) * bin_size_t / (scalar_t)(sample_num_t);

      for (int iy = 0; iy < sample_num_h; iy++) {
        const scalar_t y =
            roi_start_h + ph * bin_size_h + (scalar_t)(iy + .5f) * bin_size_h / (scalar_t)(sample_num_h);

        for (int ix = 0; ix < sample_num_w; ix++) {
          const scalar_t x =
              roi_start_w + pw * bin_size_w + (scalar_t)(ix + .5f) * bin_size_w / (scalar_t)(sample_num_w);

          scalar_t w1, w2, w3;
          int x_low, x_high, y_low, y_high, t_low, t_high;
          bilinear_interpolate_coord_gradient<scalar_t>(
              offset_bottom_data,
              length, height, width, t, y, x,
              w1, w2, w3, t_low, t_high, y_low, y_high, x_low, x_high
          );

          scalar_t g1 = scale * w1 * (1. - (x - roi_start_w) / roi_width) * spatial_scale;
          scalar_t g2 = scale * w2 * (1. - (y - roi_start_h) / roi_height) * spatial_scale;
          scalar_t g3 = scale * w3 * (1. - (t - roi_start_t) / roi_length) * temporal_scale;
          scalar_t g4 = scale * w1 * (x - roi_start_w) / roi_width * spatial_scale;
          scalar_t g5 = scale * w2 * (y - roi_start_h) / roi_height * spatial_scale;
          scalar_t g6 = scale * w3 * (t - roi_start_t) / roi_length * temporal_scale;

          if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0 && t_low >= 0 && t_high >= 0) {
            offset_bottom_diff[0] = (scalar_t)(0);
            atomicAdd(offset_bottom_diff + 1, g1);
            atomicAdd(offset_bottom_diff + 2, g2);
            atomicAdd(offset_bottom_diff + 3, g3);
            atomicAdd(offset_bottom_diff + 4, g4);
            atomicAdd(offset_bottom_diff + 5, g5);
            atomicAdd(offset_bottom_diff + 6, g6);
          }
        }
      }
    }
  }
}

int ROIAlignCoordBackwardLauncher(
    const at::Tensor top_grad, const at::Tensor features, const at::Tensor rois,
    const float temporal_scale, const float spatial_scale,
    const int sample_num,
    const int channels, const int length, const int height, const int width,
    const int num_rois,
    const int pooled_length, const int pooled_height, const int pooled_width,
    at::Tensor bottom_grad) {
  const int output_size = num_rois * pooled_length * pooled_height * pooled_width * channels;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.type(), "ROIAlignLauncherCoordBackward", ([&] {
        const scalar_t *top_diff = top_grad.data<scalar_t>();
        const scalar_t *bottom_data = features.data<scalar_t>();
        const scalar_t *rois_data = rois.data<scalar_t>();
        scalar_t *bottom_diff = bottom_grad.data<scalar_t>();
        if (sizeof(scalar_t) == sizeof(double)) {
          fprintf(stderr, "double is not supported\n");
          exit(-1);
        }

        ROIAlignCoordBackward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, top_diff, bottom_data, rois_data,
                scalar_t(temporal_scale), scalar_t(spatial_scale),
                sample_num, channels, length, height, width,
                pooled_length, pooled_height, pooled_width,
                bottom_diff);
      }));
  THCudaCheck(cudaGetLastError());

  return 1;
}
