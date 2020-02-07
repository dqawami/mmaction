import mmcv
import numpy as np
import random

from mmaction.datasets.extra_aug import PhotoMetricDistortion

__all__ = ['GroupImageTransform', 'ImageTransform', 'BboxTransform', 'GroupMultiScaleCrop']


class GroupCrop(object):
    def __init__(self, crop_quadruple):
        self.crop_quadruple = crop_quadruple

    def __call__(self, img_group, is_flow=False):
        return [mmcv.imcrop(img, self.crop_quadruple)
                for img in img_group], self.crop_quadruple


class GroupCenterCrop(object):
    def __init__(self, size, portrait_mode=False):
        self.size = size if not isinstance(size, int) else (size, size)
        self.portrait_mode = portrait_mode

    def __call__(self, img_group, is_flow=False):
        image_height = img_group[0].shape[0]
        image_width = img_group[0].shape[1]

        crop_width, crop_height = self.size
        x1 = (image_width - crop_width) // 2
        y1 = (image_height - crop_height) // 2

        if self.portrait_mode:
            y1 = min(y1, x1)

        box = np.array([x1, y1, x1 + crop_width - 1, y1 + crop_height - 1])

        return ([mmcv.imcrop(img, box) for img in img_group],
                np.array([x1, y1, crop_width, crop_height], dtype=np.float32))


class Group3CropSample(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size if not isinstance(
            crop_size, int) else (crop_size, crop_size)

    def __call__(self, img_group, is_flow=False):

        image_h = img_group[0].shape[0]
        image_w = img_group[0].shape[1]
        crop_w, crop_h = self.crop_size
        assert crop_h == image_h or crop_w == image_w

        if crop_h == image_h:
            w_step = (image_w - crop_w) // 2
            offsets = list()
            offsets.append((0, 0))  # left
            offsets.append((2 * w_step, 0))  # right
            offsets.append((w_step, 0))  # middle
        elif crop_w == image_w:
            h_step = (image_h - crop_h) // 2
            offsets = list()
            offsets.append((0, 0))  # top
            offsets.append((0, 2 * h_step))  # down
            offsets.append((0, h_step))  # middle

        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = mmcv.imcrop(img, np.array(
                    [o_w, o_h, o_w + crop_w-1, o_h + crop_h-1]))
                normal_group.append(crop)
                flip_crop = mmcv.imflip(crop)

                if is_flow and i % 2 == 0:
                    flip_group.append(mmcv.iminvert(flip_crop))
                else:
                    flip_group.append(flip_crop)

            oversample_group.extend(normal_group)
            # oversample_group.extend(flip_group)
        return oversample_group, None


class GroupOverSample(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size if not isinstance(
            crop_size, int) else (crop_size, crop_size)

    def __call__(self, img_group, is_flow=False):

        image_h = img_group[0].shape[0]
        image_w = img_group[0].shape[1]
        crop_w, crop_h = self.crop_size

        offsets = GroupMultiScaleCrop.fill_fix_offset(
            False, image_w, image_h, crop_w, crop_h)
        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = mmcv.imcrop(img, np.array(
                    [o_w, o_h, o_w + crop_w-1, o_h + crop_h-1]))
                normal_group.append(crop)
                flip_crop = mmcv.imflip(crop)

                if is_flow and i % 2 == 0:
                    flip_group.append(mmcv.iminvert(flip_crop))
                else:
                    flip_group.append(flip_crop)

            oversample_group.extend(normal_group)
            oversample_group.extend(flip_group)
        return oversample_group, None


class GroupMultiScaleCrop(object):

    def __init__(self,
                 input_size,
                 scale_limits=None,
                 scales=None,  # deprecated
                 max_distort=1,  # deprecated
                 fix_crop=True,
                 more_fix_crop=True):
        self.scale_limits = scale_limits
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fixed_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = 'bilinear'

    def __call__(self, img_group, is_flow=False):
        im_h = img_group[0].shape[0]
        im_w = img_group[0].shape[1]

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size((im_w, im_h))
        box = np.array([offset_w, offset_h, offset_w + crop_w - 1, offset_h + crop_h - 1])
        crop_img_group = [mmcv.imcrop(img, box) for img in img_group]

        ret_img_group = [mmcv.imresize(img, (self.input_size[0], self.input_size[1]), interpolation=self.interpolation)
                         for img in crop_img_group]

        return ret_img_group, np.array([offset_w, offset_h, crop_w, crop_h], dtype=np.float32)

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        src_ar = float(image_h) / float(image_w)
        trg_ar = float(self.input_size[1]) / float(self.input_size[0])

        if self.scale_limits is not None:
            scale = np.random.uniform(low=self.scale_limits[1], high=self.scale_limits[0])
        else:
            scale = 1.0

        if src_ar < trg_ar:
            crop_h = scale * image_h
            crop_w = crop_h / trg_ar
        else:
            crop_w = scale * image_w
            crop_h = crop_w * trg_ar
        crop_pair = int(crop_w), int(crop_h)

        if not self.fixed_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret


class GroupImageTransform(object):
    """Preprocess a group of images.
    1. rescale the images to expected size
    2. (for classification networks) crop the images with a given size
    3. flip the images (if needed)
    4(a) divided by 255 (0-255 => 0-1, if needed)
    4. normalize the images
    5. pad the images (if needed)
    6. transpose to (c, h, w)
    7. stack to (N, c, h, w)
    where, N = 1 * N_oversample * N_seg * L
    """

    def __init__(self,
                 mean=None,
                 std=None,
                 to_rgb=False,
                 size_divisor=None,
                 crop_size=None,
                 oversample=None,
                 random_crop=False,
                 more_fix_crop=False,
                 multiscale_crop=False,
                 central_crop=True,
                 scale_limits=None,
                 scales=None,
                 max_distort=1,
                 extra_augm=None,
                 dropout_scale=None):
        self.mean = np.array(mean, dtype=np.float32) if mean is not None else None
        self.std = np.array(std, dtype=np.float32) if std is not None else None
        self.to_rgb = to_rgb
        self.size_divisor = size_divisor

        # croping parameters
        if crop_size is not None:
            if oversample == 'three_crop':
                self.op_crop = Group3CropSample(crop_size)
            elif oversample == 'ten_crop':
                # oversample crop (test)
                self.op_crop = GroupOverSample(crop_size)
            elif multiscale_crop:
                # multiscale crop (train)
                self.op_crop = GroupMultiScaleCrop(
                    crop_size, scale_limits=scale_limits, scales=scales, max_distort=max_distort,
                    fix_crop=not random_crop, more_fix_crop=more_fix_crop)
            else:
                # (val)
                self.op_crop = GroupCenterCrop(crop_size, portrait_mode=not central_crop)
        else:
            self.op_crop = None

        # if use extra augmentation
        if extra_augm is not None:
            self.extra_augm = PhotoMetricDistortion(**extra_augm)
        else:
            self.extra_augm = None

        self.dropout_scale = dropout_scale

    @staticmethod
    def _coarse_dropout_mask(img_size, p, scale):
        assert 0.0 < p <= 1.0
        assert 0.0 < scale <= 1.0

        img_height, img_width = img_size[:2]

        dropout_height = min(img_height, int(1. / scale))
        dropout_weight = min(img_width, int(1. / scale))

        mask = (np.random.random_sample(size=(dropout_height, dropout_weight, 1)) > p).astype(np.uint8)
        mask = mmcv.imresize(mask, (img_width, img_height), interpolation='nearest')
        mask = mask.reshape((img_height, img_width, 1))

        return mask

    def __call__(self, img_group, scale, crop_history=None, flip=False, rotate=None, keep_ratio=True,
                 dropout_prob=None, div_255=False, transpose=True, stack=True):
        # 1. rescale
        if keep_ratio:
            tuple_list = [mmcv.imrescale(img, scale, return_scale=True) for img in img_group]
            img_group, scale_factors = list(zip(*tuple_list))
            scale_factor = scale_factors[0]
        else:
            tuple_list = [mmcv.imresize(img, scale, return_scale=True) for img in img_group]
            img_group, w_scales, h_scales = list(zip(*tuple_list))
            scale_factor = np.array([w_scales[0], h_scales[0], w_scales[0], h_scales[0]], dtype=np.float32)

        # 2. rotate
        if rotate is not None:
            img_group = [mmcv.imrotate(img, rotate) for img in img_group]

        # 3. crop (if necessary)
        if crop_history is not None:
            self.op_crop = GroupCrop(crop_history)
        if self.op_crop is not None:
            img_group, crop_quadruple = self.op_crop(img_group)
        else:
            crop_quadruple = None

        img_shape = img_group[0].shape

        # 4. flip
        if flip:
            img_group = [mmcv.imflip(img) for img in img_group]

        # 5a. extra augmentation
        if self.extra_augm is not None:
            img_group = self.extra_augm(img_group)

        # 5b. coarse dropout
        if self.dropout_scale is not None and dropout_prob is not None and dropout_prob > 0.0:
            dropout_mask = self._coarse_dropout_mask(img_group[0].shape, dropout_prob, self.dropout_scale)
            img_group = [img * dropout_mask for img in img_group]

        # 6a. div_255
        if div_255:
            img_group = [mmcv.imnormalize(img, 0, 255, False) for img in img_group]

        # 6b. normalize
        if self.mean is not None and self.std is not None:
            img_group = [mmcv.imnormalize(img, self.mean, self.std, self.to_rgb) for img in img_group]
        elif self.to_rgb:
            img_group = [mmcv.bgr2rgb(img) for img in img_group]

        # 7. pad
        if self.size_divisor is not None:
            img_group = [mmcv.impad_to_multiple(img, self.size_divisor) for img in img_group]
            pad_shape = img_group[0].shape
        else:
            pad_shape = img_shape

        # 8. transpose
        if transpose:
            img_group = [img.transpose((2, 0, 1)) for img in img_group]

        # 9. stack into numpy.array
        if stack:
            img_group = np.stack(img_group, axis=0)

        return img_group, img_shape, pad_shape, scale_factor, crop_quadruple


class ImageTransform(object):
    """Preprocess an image.
    1. rescale the image to expected size
    2. normalize the image
    3. flip the image (if needed)
    4. pad the image (if needed)
    5. transpose to (c, h, w)
    """

    def __init__(self,
                 mean=(0, 0, 0),
                 std=(1, 1, 1),
                 to_rgb=True,
                 size_divisor=None):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        self.size_divisor = size_divisor

    def __call__(self, img, scale, flip=False, keep_ratio=True):
        if keep_ratio:
            img, scale_factor = mmcv.imrescale(img, scale, return_scale=True)
        else:
            img, w_scale, h_scale = mmcv.imresize(
                img, scale, return_scale=True)
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
        img_shape = img.shape
        img = mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
        if flip:
            img = mmcv.imflip(img)
        if self.size_divisor is not None:
            img = mmcv.impad_to_multiple(img, self.size_divisor)
            pad_shape = img.shape
        else:
            pad_shape = img_shape
        img = img.transpose(2, 0, 1)
        return img, img_shape, pad_shape, scale_factor


def bbox_flip(bboxes, img_shape):
    """Flip bboxes horizontally.
    Args:
        bboxes(ndarray): shape (..., 4*k)
        img_shape(tuple): (height, width)
    """
    assert bboxes.shape[-1] % 4 == 0
    w = img_shape[1]
    flipped = bboxes.copy()
    flipped[..., 0::4] = w - bboxes[..., 2::4] - 1
    flipped[..., 2::4] = w - bboxes[..., 0::4] - 1
    return flipped


def bbox_crop(bboxes, crop_quadruple):
    """Flip bboxes horizontally.
    Args:
        bboxes(ndarray): shape (..., 4*k)
        crop_quadruple(tuple): (x1, y1, tw, th)
    """
    assert bboxes.shape[-1] % 4 == 0
    assert crop_quadruple is not None
    cropped = bboxes.copy()
    x1, y1, tw, th = crop_quadruple
    cropped[..., 0::2] = bboxes[..., 0::2] - x1
    cropped[..., 1::2] = bboxes[..., 1::2] - y1
    return cropped


class BboxTransform(object):
    """Preprocess gt bboxes.
    1. rescale bboxes according to image size
    2. flip bboxes (if needed)
    3. pad the first dimension to `max_num_gts`
    """

    def __init__(self, max_num_gts=None):
        self.max_num_gts = max_num_gts

    def __call__(self, bboxes, img_shape, scale_factor, flip=False, crop=None):
        gt_bboxes = bboxes * scale_factor
        if crop is not None:
            gt_bboxes = bbox_crop(gt_bboxes, crop)
        if flip:
            gt_bboxes = bbox_flip(gt_bboxes, img_shape)
        gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_shape[1] - 1)
        gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_shape[0] - 1)
        if self.max_num_gts is None:
            return gt_bboxes
        else:
            num_gts = gt_bboxes.shape[0]
            padded_bboxes = np.zeros((self.max_num_gts, 4), dtype=np.float32)
            padded_bboxes[:num_gts, :] = gt_bboxes
            return padded_bboxes
