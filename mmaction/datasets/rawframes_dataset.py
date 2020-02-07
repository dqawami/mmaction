import mmcv
import numpy as np
import os.path as osp
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset

from .transforms import GroupImageTransform
from .utils import to_tensor


class RawFramesRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class RawFramesDataset(Dataset):
    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_norm_cfg,
                 num_segments=1,
                 new_length=16,
                 new_step=4,
                 random_shift=True,
                 proxy_generator=False,
                 temporal_jitter=False,
                 modality='RGB',
                 image_tmpl='img_{}.jpg',
                 img_scale=256,
                 img_scale_file=None,
                 input_size=224,
                 div_255=False,
                 flip_ratio=0.5,
                 rotate_delta=None,
                 resize_keep_ratio=True,
                 test_mode=False,
                 random_crop=False,
                 more_fix_crop=False,
                 multiscale_crop=False,
                 scale_limits=None,
                 scales=None,
                 max_distort=1,
                 extra_augm=None,
                 input_format='NCHW'):
        super(RawFramesDataset, self).__init__()

        # prefix of images path
        self.img_prefix = img_prefix

        # load annotations
        self.video_infos = self.load_annotations(ann_file)
        # normalization config
        self.img_norm_cfg = img_norm_cfg

        # parameters for frame fetching
        # number of segments
        self.num_segments = num_segments
        # number of consecutive frames
        self.old_length = new_length * new_step
        self.new_length = new_length
        # number of steps (sparse sampling for efficiency of io)
        self.new_step = new_step
        # whether to temporally random shift when training
        self.random_shift = random_shift
        # whether to create proxy video clip
        self.proxy_generator = proxy_generator
        # whether to temporally jitter if new_step > 1
        self.temporal_jitter = temporal_jitter

        # parameters for modalities
        if isinstance(modality, (list, tuple)):
            assert len(modality) == 1
            self.modality = modality[0]
        else:
            self.modality = modality
        assert 'RGB' == self.modality

        if isinstance(image_tmpl, (list, tuple)):
            assert len(image_tmpl) == 1
            self.image_template = image_tmpl[0]
        else:
            self.image_template = image_tmpl

        # parameters for image preprocessing
        # img_scale
        if isinstance(img_scale, int):
            img_scale = (np.Inf, img_scale)
        self.img_scale = img_scale
        if img_scale_file is not None:
            self.img_scale_dict = {line.split(' ')[0]:
                                   (int(line.split(' ')[1]),
                                    int(line.split(' ')[2]))
                                   for line in open(img_scale_file)}
        else:
            self.img_scale_dict = None
        # network input size
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.input_size = input_size

        # parameters for specification from pre-trained networks (lecacy issue)
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
            crop_size=self.input_size,
            random_crop=random_crop,
            more_fix_crop=more_fix_crop,
            multiscale_crop=multiscale_crop, scale_limits=scale_limits,
            scales=scales, max_distort=max_distort, extra_augm=extra_augm,
            **self.img_norm_cfg)

        # input format
        assert input_format in ['NCHW', 'NCTHW']
        self.input_format = input_format

    def __len__(self):
        return len(self.video_infos)

    @staticmethod
    def load_annotations(ann_file):
        return [RawFramesRecord(x.strip().split(' ')) for x in open(ann_file)]

    def get_ann_info(self, idx):
        return {'path': self.video_infos[idx].path,
                'num_frames': self.video_infos[idx].num_frames,
                'label': self.video_infos[idx].label}

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            self.flag[i] = 1

    def _get_record(self, idx):
        return self.video_infos[idx], None

    def _load_images(self, record_path, indices, modality, video_reader):
        images_dir = osp.join(self.img_prefix, record_path)

        if modality in ['RGB', 'RGBDiff']:
            return [mmcv.imread(osp.join(images_dir, self.image_template.format(idx)))
                    for idx in indices]
        elif modality == 'Flow':
            x_imgs = [mmcv.imread(osp.join(images_dir, self.image_template.format('x', idx)), flag='grayscale')
                      for idx in indices]
            y_imgs = [mmcv.imread(osp.join(images_dir, self.image_template.format('y', idx)), flag='grayscale')
                      for idx in indices]
            return [x_imgs, y_imgs]
        else:
            raise ValueError('Not implemented yet; modality should be ["RGB", "RGBDiff", "Flow"]')

    def _sample_indices(self, record):
        average_duration = (record.num_frames - self.old_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration)
            if not self.proxy_generator:
                offsets += np.random.randint(average_duration, size=self.num_segments)
        elif record.num_frames > max(self.num_segments, self.old_length):
            if not self.proxy_generator:
                offsets = np.sort(np.random.randint(record.num_frames - self.old_length + 1, size=self.num_segments))
            else:
                offsets = np.zeros((self.num_segments,), dtype=int)
        else:
            offsets = np.zeros((self.num_segments,), dtype=int)

        if self.temporal_jitter:
            skip_offsets = np.random.randint(self.new_step, size=self.old_length // self.new_step)
        else:
            skip_offsets = np.zeros(self.old_length // self.new_step, dtype=int)

        return offsets + 1, skip_offsets  # frame index starts from 1

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.old_length - 1:
            tick = (record.num_frames - self.old_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,), dtype=int)

        if self.temporal_jitter:
            skip_offsets = np.random.randint(self.new_step, size=self.old_length // self.new_step)
        else:
            skip_offsets = np.zeros(self.old_length // self.new_step, dtype=int)

        return offsets + 1, skip_offsets

    def _get_test_indices(self, record):
        if self.proxy_generator or record.num_frames <= self.old_length - 1:
            offsets = np.zeros((self.num_segments,), dtype=int)
        else:
            tick = (record.num_frames - self.old_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        if self.temporal_jitter:
            skip_offsets = np.random.randint(self.new_step, size=self.old_length // self.new_step)
        else:
            skip_offsets = np.zeros(self.old_length // self.new_step, dtype=int)

        return offsets + 1, skip_offsets

    def _get_regular_frame_ids(self, record, indices, skip_offsets):
        frame_indices = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i, _ in enumerate(range(0, self.old_length, self.new_step)):
                if p + skip_offsets[i] <= record.num_frames:
                    frame_indices.append(p + skip_offsets[i])
                else:
                    frame_indices.append(p)

                if p + self.new_step < record.num_frames:
                    p += self.new_step

        return frame_indices

    def _get_proxy_frame_ids(self, record, indices, eval_mode):
        def _scatter_elements(_trg_size, _num):
            return np.arange(_trg_size - 1, _trg_size - 1 - _num, -1)

        num_final_frames = len(range(0, self.old_length, self.new_step))

        frame_indices = list()
        for seg_offset in indices:
            num_free_frames = record.num_frames - seg_offset + 1
            range_size = num_free_frames // num_final_frames

            if range_size > 0:
                num_free_elem = num_free_frames - range_size * num_final_frames
                range_sizes = np.full(num_final_frames, range_size)
                if num_free_elem > 0:
                    extra_positions = _scatter_elements(num_final_frames, num_free_elem)\
                        if eval_mode else np.random.choice(num_final_frames, num_free_elem, replace=False)
                    range_sizes[extra_positions] += 1
                range_start_pos = np.cumsum(np.concatenate(([seg_offset], range_sizes)))[:-1]

                range_internal_shifts = range_sizes // 2\
                    if eval_mode else [np.random.randint(range_sizes[i]) for i in range(num_final_frames)]
                img_offsets = np.array(range_internal_shifts) + range_start_pos
            else:
                if eval_mode:
                    inv_range_size = num_final_frames // num_free_frames
                    num_free_elem = num_final_frames - inv_range_size * num_free_frames
                    inv_range_sizes = np.full(num_free_frames, inv_range_size)
                    if num_free_elem > 0:
                        extra_positions = _scatter_elements(num_free_frames, num_free_elem)
                        inv_range_sizes[extra_positions] += 1
                    img_offsets = [seg_offset + i for i in range(num_free_frames) for _ in range(inv_range_sizes[i])]
                else:
                    img_offsets = np.sort(np.random.randint(seg_offset, record.num_frames + 1, num_final_frames))

            for img_offset in img_offsets:
                frame_indices.append(img_offset)

        return frame_indices

    def __getitem__(self, idx):
        record, reader = self._get_record(idx)
        if self.test_mode:
            segment_indices, skip_offsets = self._get_test_indices(record)
        else:
            segment_indices, skip_offsets = self._sample_indices(record)\
                if self.random_shift else self._get_val_indices(record)
        eval_enabled = self.test_mode or not self.random_shift

        # handle the first modality
        if self.proxy_generator:
            record_indices = self._get_proxy_frame_ids(record, segment_indices, eval_enabled)
        else:
            record_indices = self._get_regular_frame_ids(record, segment_indices, skip_offsets)
        img_group = self._load_images(record.path, record_indices, self.modality, reader)

        flip = True if self.flip_ratio is not None and np.random.rand() < self.flip_ratio else False
        rotate = np.random.uniform(-self.rotate_delta, self.rotate_delta) if self.rotate_delta is not None else None

        if self.img_scale_dict is not None and record.path in self.img_scale_dict:
            img_scale = self.img_scale_dict[record.path]
        else:
            img_scale = self.img_scale

        img_group, img_shape, pad_shape, scale_factor, crop_quadruple = self.img_group_transform(
            img_group, img_scale, flip=flip, rotate=rotate, keep_ratio=self.resize_keep_ratio,
            div_255=self.div_255, transpose=True, stack=True)
        img_group = np.transpose(img_group, (1, 0, 2, 3))

        data = dict(img_group=DC(to_tensor(img_group), stack=True, pad_dims=2),
                    gt_label=DC(to_tensor(record.label), stack=True, pad_dims=None))

        return data
