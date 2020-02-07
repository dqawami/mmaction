import mmcv
import numpy as np
import os.path as osp
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset

from .transforms import GroupImageTransform, GroupMultiScaleCrop
from .utils import to_tensor


class RawFramesSegmentedRecord(object):
    def __init__(self, row):
        self._data = row

        assert self.video_num_frames > 0
        assert self.num_frames > 0
        assert self.fps > 0
        assert self.label >= 0
        assert self.clip_start >= self.video_start >= 0
        assert self.video_end >= self.clip_end >= 0

    @property
    def path(self):
        return self._data[0]

    @property
    def label(self):
        return int(self._data[1])

    @property
    def clip_start(self):
        return int(self._data[2])

    @property
    def clip_end(self):
        return int(self._data[3])

    @property
    def video_start(self):
        return int(self._data[4])

    @property
    def video_end(self):
        return int(self._data[5])

    @property
    def fps(self):
        return float(self._data[6])

    @property
    def num_frames(self):
        return self.clip_end - self.clip_start

    @property
    def video_num_frames(self):
        return self.video_end - self.video_start


class ImageRecord(object):
    def __init__(self, row, root_dir):
        self._image_path = osp.join(root_dir, row[0])

    @property
    def path(self):
        return self._image_path


class StreamDataset(Dataset):
    _modalities = ['Grayscale', 'Grayscale_Diff', 'RGB', 'RGB_Diff']

    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_norm_cfg,
                 out_length=16,
                 out_fps=15.0,
                 num_segments=1,
                 temporal_jitter=False,
                 modality='RGB',
                 image_tmpl='img_{}.jpg',
                 img_scale=256,
                 img_scale_file=None,
                 input_size=224,
                 min_intersection=1.0,
                 div_255=False,
                 flip_ratio=0.5,
                 rotate_delta=None,
                 resize_keep_ratio=True,
                 random_crop=False,
                 scale_limits=None,
                 extra_augm=None,
                 test_mode=False,
                 central_crop=True,
                 dropout_prob=None,
                 dropout_scale=None,
                 mixup_alpha=None,
                 mixup_images_file=None,
                 mixup_images_root=None):
        self.img_prefix = img_prefix
        self.video_infos = self.load_annotations(ann_file)

        self.output_length = out_length
        self.output_fps = out_fps
        self.num_segments = num_segments
        self.temporal_jitter = temporal_jitter
        self.min_intersection = min_intersection
        assert 0.0 < self.min_intersection <= 1.0

        # parameters for modalities
        if isinstance(modality, (list, tuple)):
            assert len(modality) == 1
            self.modality = modality[0]
        else:
            self.modality = modality
        self.enable_diff = 'Diff' in self.modality
        assert self.modality in StreamDataset._modalities

        self.img_norm_cfg = img_norm_cfg
        self.to_grayscale = 'Grayscale' in self.modality
        if self.to_grayscale:
            self.img_norm_cfg['to_rgb'] = False

        self.norm_mean = np.array(self.img_norm_cfg['mean'], dtype=np.float32)\
            if 'mean' in self.img_norm_cfg and self.img_norm_cfg.mean is not None else None
        self.norm_std = np.array(self.img_norm_cfg['std'], dtype=np.float32)\
            if 'std' in self.img_norm_cfg and self.img_norm_cfg.std is not None else None
        self.to_rgb = self.img_norm_cfg['to_rgb'] if 'to_rgb' in self.img_norm_cfg else False

        if isinstance(image_tmpl, (list, tuple)):
            assert len(image_tmpl) == 1
            self.image_template = image_tmpl[0]
        else:
            self.image_template = image_tmpl

        # parameters for image pre-processing
        # img_scale
        if isinstance(img_scale, int):
            img_scale = (np.Inf, img_scale)
        self.img_scale = img_scale
        if img_scale_file is not None:
            self.img_scale_dict = {line.split(' ')[0]: (int(line.split(' ')[1]), int(line.split(' ')[2]))
                                   for line in open(img_scale_file)}
        else:
            self.img_scale_dict = None
        # network input size
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.input_size = input_size

        # parameters for specification from pre-trained networks (legacy issue)
        self.div_255 = div_255

        # parameters for data augmentation
        self.flip_ratio = flip_ratio
        self.rotate_delta = rotate_delta
        self.resize_keep_ratio = resize_keep_ratio

        # test mode or not
        self.test_mode = test_mode

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

        # transforms
        self.img_group_transform = GroupImageTransform(
            crop_size=self.input_size, random_crop=random_crop,
            multiscale_crop=not self.test_mode, scale_limits=scale_limits,
            central_crop=central_crop, extra_augm=extra_augm, dropout_scale=dropout_scale)

        self.with_dropout = dropout_scale is not None and dropout_prob is not None and dropout_prob > 0.0
        self.dropout_prob = dropout_prob

        self.mixup_alpha = mixup_alpha
        self.with_mixup = False
        if not self.test_mode and self.mixup_alpha is not None and self.mixup_alpha > 0.0 and\
           mixup_images_file is not None and osp.exists(mixup_images_file) and\
           mixup_images_root is not None and osp.exists(mixup_images_root):
            self.mixup_image_paths = self.load_mixup_image_paths(mixup_images_file, mixup_images_root)
            self.with_mixup = len(self.mixup_image_paths) >= 1
        if self.with_mixup:
            print('[INFO] Enabled Mixup augmentation with {} extra images'.format(len(self.mixup_image_paths)))
            self.mixup_op_crop = GroupMultiScaleCrop(
                self.input_size, scale_limits=scale_limits, fix_crop=False, more_fix_crop=False)

    def __len__(self):
        return len(self.video_infos)

    @staticmethod
    def load_mixup_image_paths(images_path, images_root_dir):
        return [ImageRecord(x.strip().split(' '), images_root_dir) for x in open(images_path)]

    @staticmethod
    def load_annotations(ann_file):
        return [RawFramesSegmentedRecord(x.strip().split(' ')) for x in open(ann_file)]

    def get_ann_info(self, idx):
        record = self.get_record_by_idx(idx)

        return dict(path=record.path,
                    label=record.label,
                    start=record.clip_start,
                    end=record.clip_end,
                    num_frames=record.num_frames)

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            self.flag[i] = 1

    @staticmethod
    def _load_image(directory, image_template, idx):
        return mmcv.imread(osp.join(directory, image_template.format(idx)))

    def _estimate_time_step(self, video_fps):
        time_step = int(np.round(float(video_fps) / float(self.output_fps)))
        time_step = time_step if time_step >= 1 else 1

        if self.enable_diff:
            time_step = time_step // 2 if time_step >= 2 else 1

        return time_step

    def _estimate_clip_lengths(self, time_step):
        output_length = 2 * self.output_length if self.enable_diff else self.output_length
        return time_step * output_length, output_length

    def _sample_indices(self, record):
        time_step = self._estimate_time_step(record.fps)
        input_length, output_length = self._estimate_clip_lengths(time_step)

        if record.video_num_frames < input_length:
            num_valid_frames = record.video_num_frames // time_step

            if self.temporal_jitter and not self.enable_diff:
                offsets = np.random.randint(low=0, high=time_step, size=num_valid_frames, dtype=np.int32)
            else:
                offsets = np.zeros(num_valid_frames, dtype=np.int32)

            indices = np.array([i * time_step + offsets[i] for i in range(num_valid_frames)])

            num_rest = output_length - num_valid_frames
            if num_rest > 0:
                num_before = np.random.randint(num_rest + 1)
                num_after = num_rest - num_before
                indices = np.concatenate((np.full(num_before, indices[0], dtype=np.int32),
                                          indices,
                                          np.full(num_after, indices[-1], dtype=np.int32)))
        else:
            if record.num_frames < input_length:
                bumpy_num_frames = int(float(1.0 - self.min_intersection) * float(record.num_frames))
                shift_start = max(record.video_start, record.clip_end - bumpy_num_frames - input_length)
                shift_end = min(record.video_end - input_length + 1, record.clip_start + bumpy_num_frames + 1)
            else:
                shift_start = record.clip_start
                shift_end = record.clip_end - input_length + 1

            if self.temporal_jitter and not self.enable_diff:
                offsets = np.random.randint(low=0, high=time_step, size=output_length, dtype=np.int32)
            else:
                offsets = np.zeros(output_length, dtype=np.int32)

            start_pos = np.random.randint(low=shift_start, high=shift_end)
            indices = np.array([start_pos + i * time_step + offsets[i] for i in range(output_length)])

        return indices + 1  # frame index starts from 1

    def _get_test_indices(self, record):
        time_step = self._estimate_time_step(record.fps)
        input_length, output_length = self._estimate_clip_lengths(time_step)

        if record.video_num_frames < input_length:
            indices = np.array([i * time_step for i in range(record.video_num_frames // time_step)])

            num_rest = output_length - len(indices)
            if num_rest > 0:
                num_before = num_rest // 2
                num_after = num_rest - num_before
                indices = np.concatenate((np.full(num_before, indices[0], dtype=np.int32),
                                          indices,
                                          np.full(num_after, indices[-1], dtype=np.int32)))
        else:
            if record.num_frames < input_length:
                # shift_start = max(record.video_start, record.clip_end - input_length)
                shift_end = min(record.video_end - input_length + 1, record.clip_start + 1)
                # start_pos = shift_start
                start_pos = shift_end - 1
                # start_pos = (shift_start + shift_end) // 2
            else:
                shift_start = record.clip_start
                shift_end = record.clip_end - input_length + 1
                # start_pos = shift_start
                # start_pos = shift_end - 1
                start_pos = (shift_start + shift_end) // 2

            indices = np.array([start_pos + i * time_step for i in range(output_length)])

        return indices + 1  # frame index starts from 1

    def _get_frames(self, record, image_template, indices):
        images_dir = osp.join(self.img_prefix, record.path)
        images = [self._load_image(images_dir, image_template, p) for p in indices]
        return images

    def get_record_by_idx(self, idx):
        return self.video_infos[idx]

    @staticmethod
    def convert_color(img_group, to_grayscale, to_rgb):
        if to_grayscale:
            return [mmcv.bgr2gray(img, keepdim=True) for img in img_group]
        elif to_rgb:
            return [mmcv.bgr2rgb(img) for img in img_group]
        else:
            return img_group

    @staticmethod
    def prepare_clip_data(img_group, enable_diff, mean=None, std=None):
        def _make_diff_pair(_img_a, _img_b):
            _float_img_a = _img_a.astype(np.float32)
            _float_img_b = _img_b.astype(np.float32)
            _out = np.concatenate((_float_img_a, _float_img_b - _float_img_a), axis=2)
            return _out

        if enable_diff:
            data_group = [_make_diff_pair(img_group[i], img_group[i + 1]) for i in range(0, len(img_group), 2)]
        else:
            data_group = np.stack(img_group, axis=0).astype(np.float32)

        if mean is not None and std is not None:
            data_group = (data_group - mean.reshape((1, 1, 1, -1))) / std.reshape((1, 1, 1, -1))

        return np.transpose(data_group, (3, 0, 1, 2))

    def _prepare_images(self, idx):
        record = self.get_record_by_idx(idx)
        segment_indices = self._get_test_indices(record) if self.test_mode else self._sample_indices(record)

        img_group = self._get_frames(record, self.image_template, segment_indices)

        if self.img_scale_dict is not None and record.path in self.img_scale_dict:
            img_scale = self.img_scale_dict[record.path]
        else:
            img_scale = self.img_scale

        flip = True if self.flip_ratio is not None and np.random.rand() < self.flip_ratio else False
        rotate = np.random.uniform(-self.rotate_delta, self.rotate_delta) if self.rotate_delta is not None else None
        dropout = self.dropout_prob if self.with_dropout and np.random.randint(2) else None

        img_group, img_shape, pad_shape, scale_factor, crop_quadruple = self.img_group_transform(
            img_group, img_scale, crop_history=None, flip=flip, rotate=rotate,
            keep_ratio=self.resize_keep_ratio, dropout_prob=dropout, div_255=self.div_255,
            transpose=False, stack=False)
        img_group = self.convert_color(img_group, self.to_grayscale, self.to_rgb)

        return img_group, record.label

    def _prepare_mixup_image(self, idx):
        image = mmcv.imread(self.mixup_image_paths[idx].path)

        rescaled_image = mmcv.imrescale(image, self.img_scale)

        cropped_images, _ = self.mixup_op_crop([rescaled_image])
        cropped_image = cropped_images[0]

        if np.random.randint(2):
            cropped_image = mmcv.imflip(cropped_image)

        out_images = self.convert_color([cropped_image], self.to_grayscale, self.to_rgb)
        out_image = out_images[0]

        return out_image.astype(np.float32)

    def __getitem__(self, idx):
        images, label = self._prepare_images(idx)

        if self.with_mixup:
            alpha = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            alpha = alpha if alpha < 0.5 else 1.0 - alpha

            mixup_image_idx = np.random.randint(len(self.mixup_image_paths))
            mixup_image = self._prepare_mixup_image(mixup_image_idx)

            images = [images[i].astype(np.float32) * (1.0 - alpha) + mixup_image * alpha for i in range(len(images))]

        images_data = self.prepare_clip_data(images, self.enable_diff, self.norm_mean, self.norm_std)

        out_data = dict(gt_label=DC(to_tensor(label), stack=True, pad_dims=None),
                        img_group=DC(to_tensor(images_data), stack=True, pad_dims=2))

        return out_data
