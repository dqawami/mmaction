from argparse import ArgumentParser

import torch
import mmcv
from ptflops import get_model_complexity_info

from mmaction.models import build_recognizer


def main():
    parser = ArgumentParser(description='Measure number of FLOPs')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--layer_stat', '-ls', action='store_true', help='Whether to print per layer stat')
    args = parser.parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.videos_per_gpu = 1
    cfg.model.type += '_Inference'
    cfg.model.backbone.type += '_Inference'
    cfg.model.backbone.inference = True
    cfg.model.cls_head.type += '_Inference'

    time_length = cfg.data.test.out_length if hasattr(cfg.data.test, 'out_length') else cfg.data.test.new_length
    input_size = (cfg.model.backbone.num_input_layers, time_length) + cfg.data.test.input_size

    with torch.no_grad():
        net = build_recognizer(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg).cuda()

        flops, params = get_model_complexity_info(
            net, input_size, as_strings=True, print_per_layer_stat=args.layer_stat)

        print('Flops:  ' + flops)
        print('Params: ' + params)


if __name__ == '__main__':
    main()
