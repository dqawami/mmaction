from os.path import exists, isfile
from collections import OrderedDict
import warnings

import torch
from terminaltables import AsciiTable
from mmcv.runner.checkpoint import open_mmlab_model_urls, get_torchvision_models, load_url_dist
from mmcv.runner.utils import get_dist_info


def load_state_dict(module, state_dict, strict=False, logger=None, force_matching=False,
                    show_converted=False, ignores=None):
    rank, _ = get_dist_info()

    unexpected_keys = []
    converted_pairs = []
    shape_mismatch_pairs = []
    shape_casted_pairs = []

    own_state = module.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            unexpected_keys.append(name)
            continue

        if isinstance(param, torch.nn.Parameter):
            param = param.data

        src_shape = param.size()
        trg_shape = own_state[name].size()
        if src_shape != trg_shape:
            is_valid = False
            if force_matching:
                is_valid = len(src_shape) == len(trg_shape)
                for i in range(len(src_shape)):
                    is_valid &= src_shape[i] >= trg_shape[i]

            if is_valid:
                ind = [slice(0, d) for d in list(trg_shape)]
                own_state[name].copy_(param[ind])

                shape_casted_pairs.append([name, list(own_state[name].size()), list(param.size())])
            else:
                shape_mismatch_pairs.append([name, list(own_state[name].size()), list(param.size())])
        elif ignores is None or not name.endswith(ignores):
            own_state[name].copy_(param)

            if show_converted:
                converted_pairs.append([name, list(own_state[name].size())])

    all_missing_keys = set(own_state.keys()) - set(state_dict.keys())

    missing_keys = [key for key in all_missing_keys if 'num_batches_tracked' not in key]

    err_msg = []
    if unexpected_keys:
        err_msg.append('unexpected key in source state_dict: {}\n'.format(', '.join(unexpected_keys)))
    if missing_keys:
        err_msg.append('missing keys in source state_dict: {}\n'.format(', '.join(missing_keys)))

    if shape_mismatch_pairs:
        casted_info = 'these keys have mismatched shape:\n'
        header = ['key', 'expected shape', 'loaded shape']
        table_data = [header] + shape_mismatch_pairs
        table = AsciiTable(table_data)
        err_msg.append(casted_info + table.table)

    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)

    ok_message = []
    if converted_pairs:
        converted_info = 'These keys have been converted correctly:\n'
        header = ['key', 'shape']
        table_data = [header] + converted_pairs
        table = AsciiTable(table_data)
        ok_message.append(converted_info + table.table)

    if len(ok_message) > 0 and rank == 0:
        warning_msg = '\n'.join(ok_message)
        if logger is not None:
            logger.warning(warning_msg)
        else:
            print(warning_msg)

    warning_msg = []
    if shape_casted_pairs:
        casted_info = 'these keys have been shape casted:\n'
        header = ['key', 'expected shape', 'loaded shape']
        table_data = [header] + shape_casted_pairs
        table = AsciiTable(table_data)
        warning_msg.append(casted_info + table.table)

    if len(warning_msg) > 0 and rank == 0:
        warning_msg.insert(0, 'The model and loaded state dict do not match exactly\n')
        warning_msg = '\n'.join(warning_msg)
        if logger is not None:
            logger.warning(warning_msg)
        else:
            print(warning_msg)


def load_checkpoint(model, filename, map_location=None, strict=False, logger=None,
                    force_matching=False, show_converted=False, ignores=None):
    if filename.startswith('modelzoo://'):
        warnings.warn('The URL scheme of "modelzoo://" is deprecated, please use "torchvision://" instead')
        model_urls = get_torchvision_models()
        model_name = filename[11:]
        checkpoint = load_url_dist(model_urls[model_name])
    elif filename.startswith('torchvision://'):
        model_urls = get_torchvision_models()
        model_name = filename[14:]
        checkpoint = load_url_dist(model_urls[model_name])
    elif filename.startswith('open-mmlab://'):
        model_name = filename[13:]
        checkpoint = load_url_dist(open_mmlab_model_urls[model_name])
    elif filename.startswith(('http://', 'https://')):
        checkpoint = load_url_dist(filename)
    else:
        if not exists(filename):
            raise IOError('{} does not exist'.format(filename))
        elif not isfile(filename):
            raise IOError('{} is not a checkpoint file'.format(filename))

        checkpoint = torch.load(filename, map_location=map_location)

    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise RuntimeError('No state_dict found in checkpoint file {}'.format(filename))

    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}

    if hasattr(model, 'module'):
        load_state_dict(model.module, state_dict, strict, logger, force_matching, show_converted, ignores)
    else:
        load_state_dict(model, state_dict, strict, logger, force_matching, show_converted, ignores)

    return checkpoint
