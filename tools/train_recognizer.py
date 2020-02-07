from __future__ import division

import argparse
from os.path import exists

from mmcv import Config

import torch

from mmaction import __version__
from mmaction.datasets import get_trimmed_dataset
from mmaction.apis import train_network, init_dist, get_root_logger, set_random_seed
from mmaction.models import build_recognizer
from mmaction.core import update_data_paths


MODEL_SOURCES = 'modelzoo://', 'torchvision://', 'open-mmlab://', 'http://', 'https://'


def parse_args():
    parser = argparse.ArgumentParser(description='Train an action recognizer')
    parser.add_argument('config',
                        help='train config file path')
    parser.add_argument('--data_dir',
                        help='the dir with dataset')
    parser.add_argument('--work_dir',
                        help='the dir to save logs and models')
    parser.add_argument('--resume_from',
                        help='the checkpoint file to resume from')
    parser.add_argument('--load_from',
                        help='the checkpoint file to init weights from')
    parser.add_argument('--load2d_from',
                        help='the checkpoint file to init 2D weights from')
    parser.add_argument('--validate', action='store_true',
                        help='whether to evaluate the checkpoint during training')
    parser.add_argument('--gpus', type=int, default=1,
                        help='number of gpus to use (only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed')
    parser.add_argument('--num_videos', type=int,
                        help='number of videos per GPU')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--sync_bn', action='store_true')
    parser.add_argument('--ignores', nargs='+', type=str, required=False)
    args = parser.parse_args()

    return args


def is_valid(model_path):
    if model_path is None:
        return False

    return exists(model_path) or model_path.startswith(MODEL_SOURCES)


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    if is_valid(args.resume_from):
        cfg.resume_from = args.resume_from

    if is_valid(args.load_from):
        cfg.load_from = args.load_from

    if is_valid(args.load2d_from):
        cfg.model.backbone.pretrained = args.load2d_from
        cfg.model.backbone.pretrained2d = True

    if args.num_videos is not None:
        assert args.num_videos > 0
        cfg.data.videos_per_gpu = args.num_videos

    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(mmact_version=__version__, config=cfg.text)

    if args.data_dir is not None:
        cfg = update_data_paths(cfg, args.data_dir)

    if hasattr(cfg.model, 'masked_num') and cfg.model.masked_num is not None and cfg.model.masked_num > 0:
        assert cfg.data.videos_per_gpu > cfg.model.masked_num

        cfg.data.videos_per_gpu -= cfg.model.masked_num

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    train_dataset = get_trimmed_dataset(cfg.data.train)
    ignores = ['num_batches_tracked']
    if args.ignores is not None and len(args.ignores) > 0:
        ignores += args.ignores

    model = build_recognizer(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    if args.sync_bn:
        logger.info('Enabled SyncBatchNorm')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    train_network(model, train_dataset, cfg,
                  distributed=distributed, validate=args.validate,
                  logger=logger, ignores=tuple(ignores))


if __name__ == '__main__':
    main()
