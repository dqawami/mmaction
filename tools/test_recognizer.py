import argparse

import torch
import mmcv
import numpy as np

from mmcv.runner import parallel_test, obj_from_dict
from mmcv.parallel import scatter, collate, MMDataParallel

from mmaction import datasets
from mmaction.core import load_checkpoint, update_data_paths
from mmaction.datasets import build_dataloader
from mmaction.models import build_recognizer, recognizers
from mmaction.core.evaluation.accuracy import invalid_filtered, mean_top_k_accuracy, mean_average_precision


def single_test(model, data_loader):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(**data, return_loss=False)
        results.append(result)

        batch_size = data['img_group'].data[0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def _data_func(data, device_id):
    data = scatter(collate([data], samples_per_gpu=1), [device_id])[0]
    return dict(return_loss=False, rescale=True, **data)


def parse_args():
    parser = argparse.ArgumentParser(description='Test an action recognizer')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--data_dir', help='the dir with dataset')
    parser.add_argument('--gpus', default=1, type=int, help='GPU number used for testing')
    parser.add_argument('--proc_per_gpu', default=1, type=int, help='Number of processes per GPU')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--mode', choices=['train', 'val', 'test'], default='test', help='target dataset for evaluation')
    parser.add_argument('--num_classes', default=-1, type=int, help='Number of test classes')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.num_classes is not None and args.num_classes > 0:
        cfg.num_test_classes = args.num_classes
    if args.data_dir is not None:
        cfg = update_data_paths(cfg, args.data_dir)

    assert args.mode in cfg.data
    data_cfg = getattr(cfg.data, args.mode)
    data_cfg.test_mode = True

    dataset = obj_from_dict(data_cfg, datasets, dict(test_mode=True))
    if args.gpus == 1:
        model = build_recognizer(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        load_checkpoint(model, args.checkpoint, strict=False)
        model = MMDataParallel(model, device_ids=[0])

        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            num_gpus=1,
            dist=False,
            shuffle=False)
        outputs = single_test(model, data_loader)
    else:
        model_args = cfg.model.copy()
        model_args.update(train_cfg=None, test_cfg=cfg.test_cfg)
        model_type = getattr(recognizers, model_args.pop('type'))
        outputs = parallel_test(
            model_type,
            model_args,
            args.checkpoint,
            dataset,
            _data_func,
            range(args.gpus),
            workers_per_gpu=args.proc_per_gpu)

    gt_labels = []
    for i in range(len(dataset)):
        ann = dataset.get_ann_info(i)
        gt_labels.append(ann['label'])

    results = np.array([res.cpu().numpy().mean(axis=0) for res in outputs], dtype=np.float32)

    if cfg.data.num_test_classes is not None and cfg.data.num_test_classes > 0:
        results = results[:, :cfg.data.num_test_classes]

    top1_value = mean_top_k_accuracy(results, gt_labels, k=1)
    top5_value = mean_top_k_accuracy(results, gt_labels, k=5)

    print("\nMean Top-1 Accuracy = {:.03f}%".format(top1_value * 100))
    print("Mean Top-5 Accuracy = {:.03f}%".format(top5_value * 100))

    map_value = mean_average_precision(results, gt_labels)
    print("mAP = {:.03f}%".format(map_value * 100))

    invalid_ids = invalid_filtered(results, gt_labels)
    print('\nNum invalid classes: {} / {}'.format(len(invalid_ids), cfg.data.num_test_classes))

    num_invalid_samples = sum([len(ids) for ids in invalid_ids.values()])
    print('Num invalid samples: {} / {}'.format(num_invalid_samples, len(gt_labels)))


if __name__ == '__main__':
    main()
