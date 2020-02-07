import argparse

import torch
import onnx
import mmcv

from mmaction.models import build_recognizer
from mmaction.core import load_checkpoint


def convert_to_onnx(net, input_size, output_file_path, check):
    dummy_input = torch.randn((1, *input_size))
    input_names = ['input']
    output_names = ['output']

    dynamic_axes = {'input': {0: 'batch_size', 1: 'channels', 2: 'length', 3: 'height', 4: 'width'},
                    'output': {0: 'batch_size', 1: 'scores'}}

    torch.onnx.export(net, dummy_input, output_file_path, verbose=True,
                      input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes,
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX)

    net_from_onnx = onnx.load(output_file_path)
    if check:
        try:
            onnx.checker.check_model(net_from_onnx)
            print('ONNX check passed.')
        except onnx.onnx_cpp2py_export.checker.ValidationError as ex:
            print('ONNX check failed: {}.'.format(ex))

    return onnx.helper.printable_graph(net_from_onnx.graph)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('output_name', help='Output file')
    parser.add_argument('--num_classes', default=-1, type=int, help='Number of test classes')
    parser.add_argument('--check', action='store_true', help='Output file')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    cfg.data.videos_per_gpu = 1
    cfg.model.type += '_Inference'
    cfg.model.backbone.type += '_Inference'
    cfg.model.backbone.inference = True
    cfg.model.cls_head.type += '_Inference'

    if args.num_classes is not None and args.num_classes > 0:
        cfg.num_test_classes = args.num_classes
        cfg.model.cls_head.num_classes = args.num_classes

    net = build_recognizer(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    net.eval()
    load_checkpoint(net, args.checkpoint, force_matching=True)

    time_length = cfg.data.test.out_length if hasattr(cfg.data.test, 'out_length') else cfg.data.test.new_length
    input_size = (3, time_length) + cfg.data.test.input_size

    convert_to_onnx(net, input_size, args.output_name, check=args.check)


if __name__ == '__main__':
    main()
