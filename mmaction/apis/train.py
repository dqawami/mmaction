from __future__ import division

from collections import OrderedDict

import torch
from mmcv.runner import Runner, DistSamplerSeedHook
from mmcv.parallel import MMDataParallel

from mmaction.core.utils.checkpoint import load_checkpoint
from mmaction.core import DistOptimizerHook, DistEvalTopKAccuracyHook, AVADistEvalmAPHook
from mmaction.datasets import build_dataloader

from .env import get_root_logger
from .model import MMDistributedDataParallel


def parse_losses(losses):
    log_vars = OrderedDict()
    for candidate_name, candidate_value in losses.items():
        if isinstance(candidate_value, torch.Tensor):
            log_vars[candidate_name] = candidate_value.mean()
        elif isinstance(candidate_value, list):
            log_vars[candidate_name] = sum(_entrance.mean() for _entrance in candidate_value)
        elif isinstance(candidate_value, float):
            log_vars[candidate_name] = candidate_value
        else:
            raise TypeError('{} is not a tensor or list of tensors'.format(candidate_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for name, value in log_vars.items():
        if isinstance(value, torch.Tensor):
            log_vars[name] = value.item()
        else:
            log_vars[name] = value

    return loss, log_vars


def batch_processor(model, data, train_mode):
    losses = model(**data)
    loss, log_vars = parse_losses(losses)

    outputs = dict(
        loss=loss, log_vars=log_vars,
        num_samples=len(data['img_group'].data))

    return outputs


def train_network(model,
                  dataset,
                  cfg,
                  distributed=False,
                  validate=False,
                  logger=None,
                  ignores=None):
    if logger is None:
        logger = get_root_logger(cfg.log_level)

    if distributed:
        _dist_train(model, dataset, cfg, validate=validate, logger=logger, ignores=ignores)
    else:
        _non_dist_train(model, dataset, cfg, validate=validate, logger=logger, ignores=ignores)


def _dist_train(model, dataset, cfg, validate=False, logger=None, ignores=None):
    # prepare data loaders
    data_loaders = [
        build_dataloader(
            dataset,
            cfg.data.videos_per_gpu,
            cfg.data.workers_per_gpu,
            dist=True)
    ]
    num_steps_per_epoch = len(data_loaders[0])

    if hasattr(model, 'update_state'):
        model.update_state(num_steps_per_epoch)

    if cfg.load_from:
        load_checkpoint(model, cfg.load_from,
                        strict=False, logger=logger,
                        show_converted=True, ignores=ignores)

        if hasattr(cfg, 'model_partial_init') and cfg.model_partial_init:
            model.reset_weights()

    # put model on gpus
    model = MMDistributedDataParallel(model.cuda())

    # build runner
    runner = Runner(model, batch_processor, cfg.optimizer, cfg.work_dir, cfg.log_level)

    # fix warm-up bug
    if hasattr(cfg.lr_config, 'warmup_iters'):
        if not hasattr(cfg.lr_config, 'by_epoch') or cfg.lr_config.by_epoch:
            cfg.lr_config.warmup_iters *= num_steps_per_epoch

    # register hooks
    optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
    runner.register_training_hooks(cfg.lr_config, optimizer_config, cfg.checkpoint_config, cfg.log_config)
    runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        eval_epoch = cfg.eval_epoch if hasattr(cfg, 'eval_epoch') else 1
        if cfg.data.val.type in ['RawFramesDataset', 'StreamDataset', 'VideoDataset']:
            runner.register_hook(DistEvalTopKAccuracyHook(
                cfg.data.val, eval_epoch, k=(1, 5), num_valid_classes=cfg.data.num_test_classes))
        elif cfg.data.val.type == 'AVADataset':
            runner.register_hook(AVADistEvalmAPHook(cfg.data.val, eval_epoch))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)

    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


def _non_dist_train(model, dataset, cfg, validate=False, logger=None, ignores=None):
    # prepare data loaders
    data_loaders = [
        build_dataloader(
            dataset,
            cfg.data.videos_per_gpu,
            cfg.data.workers_per_gpu,
            cfg.gpus,
            dist=False)
    ]
    num_steps_per_epoch = len(data_loaders[0])

    if hasattr(model, 'update_state'):
        model.update_state(num_steps_per_epoch)

    if cfg.load_from:
        load_checkpoint(model, cfg.load_from,
                        strict=False, logger=logger,
                        show_converted=True, ignores=ignores)

        if hasattr(cfg, 'model_partial_init') and cfg.model_partial_init:
            model.reset_weights()

    # put model on gpus
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()

    # build runner
    runner = Runner(model, batch_processor, cfg.optimizer, cfg.work_dir, cfg.log_level)

    # fix warm-up bug
    if hasattr(cfg.lr_config, 'warmup_iters'):
        if not hasattr(cfg.lr_config, 'by_epoch') or cfg.lr_config.by_epoch:
            cfg.lr_config.warmup_iters *= len(data_loaders[0])

    # register hooks
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config, cfg.checkpoint_config, cfg.log_config)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)

    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
