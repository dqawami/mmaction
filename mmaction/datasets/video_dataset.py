import os.path as osp

try:
    import decord
except ImportError:
    pass

from .rawframes_dataset import RawFramesDataset


class RawFramesRecord(object):

    def __init__(self, row):
        self._data = row
        self.num_frames = -1

    @property
    def path(self):
        return self._data[0]

    @property
    def label(self):
        return int(self._data[1])


class VideoDataset(RawFramesDataset):
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
        super(VideoDataset, self).__init__(ann_file, img_prefix, img_norm_cfg, num_segments, new_length, new_step,
                                           random_shift, proxy_generator, temporal_jitter, modality, image_tmpl,
                                           img_scale, img_scale_file, input_size, div_255, flip_ratio, rotate_delta,
                                           resize_keep_ratio, test_mode, random_crop, more_fix_crop, multiscale_crop,
                                           scale_limits, scales, max_distort, extra_augm, input_format)

    @staticmethod
    def load_annotations(ann_file):
        return [RawFramesRecord(x.strip().split(' ')) for x in open(ann_file)]

    def get_ann_info(self, idx):
        return {'path': self.video_infos[idx].path,
                'label': self.video_infos[idx].label}

    def _get_record(self, idx):
        record = self.video_infos[idx]

        video_reader = decord.VideoReader(osp.join(self.img_prefix, record.path))
        record.num_frames = len(video_reader)

        return record, video_reader

    def _load_images(self, record_path, indices, modality, video_reader):
        if modality in ['RGB', 'RGBDiff']:
            indices = [i - 1 for i in indices]
            images = video_reader.get_batch(indices).asnumpy()
            return images
        elif modality == 'Flow':
            raise NotImplementedError
        else:
            raise ValueError('Not implemented yet; modality should be ["RGB", "RGBDiff", "Flow"]')
