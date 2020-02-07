from os.path import exists


def update_data_paths(cfg, data_dir):
    cfg.data.num_test_classes = cfg.num_test_classes

    cfg.root_dir = data_dir
    cfg.data_root_dir = '{}/{}'.format(cfg.root_dir, cfg.subdir_name)
    cfg.images_dir = '{}/{}'.format(cfg.data_root_dir, cfg.frames_type)

    if hasattr(cfg, 'enable_extended_labels') and cfg.enable_extended_labels:
        cfg.model.enable_extended_labels = True
        cfg.model.ext_class_map = '{}/classmap{}.json'.format(cfg.data_root_dir, cfg.num_train_classes)

    if hasattr(cfg.model, 'cls_head') and hasattr(cfg.model.cls_head, 'main_loss_cfg'):
        loss_cfg = cfg.model.cls_head.main_loss_cfg

        if hasattr(loss_cfg, 'use_class_weighting') and loss_cfg.use_class_weighting:
            weights_file = '{}/weights{}.json'.format(cfg.data_root_dir, cfg.num_train_classes)
            if exists(weights_file):
                cfg.model.cls_head.main_loss_cfg.class_weights = weights_file
            else:
                cfg.model.cls_head.main_loss_cfg.use_class_weighting = False
                cfg.model.cls_head.main_loss_cfg.class_weights = None

        counts_file = '{}/counts{}.json'.format(cfg.data_root_dir, cfg.num_train_classes)
        if hasattr(cfg.model.cls_head, 'class_counts'):
            if exists(counts_file):
                cfg.model.cls_head.class_counts = counts_file
            else:
                cfg.model.cls_head.class_counts = None
        if hasattr(loss_cfg, 'use_adaptive_margins') and loss_cfg.use_adaptive_margins:
            if exists(counts_file):
                cfg.model.cls_head.main_loss_cfg.class_counts = counts_file
            else:
                cfg.model.cls_head.main_loss_cfg.use_adaptive_margins = False
                cfg.model.cls_head.main_loss_cfg.class_counts = None

    for entry in ['train', 'val', 'test']:
        if entry not in cfg.data:
            continue

        if entry == 'train':
            entry_name = '{}{}'.format(entry, cfg.num_train_classes)
        else:
            entry_name = '{}{}'.format(entry, cfg.num_test_classes)

        cfg.data[entry].img_prefix = cfg.images_dir
        cfg.data[entry].ann_file = '{}/{}.txt'.format(cfg.data_root_dir, entry_name)

        if entry == 'train' and\
           hasattr(cfg, 'mixup_images_file') and cfg.mixup_images_file is not None and\
           hasattr(cfg, 'mixup_images_root') and cfg.mixup_images_root is not None:
            if 'mixup_images_file' in cfg.data[entry] and 'mixup_images_root' in cfg.data[entry]:
                cfg.data[entry].mixup_images_file = '{}/{}'.format(cfg.root_dir, cfg.mixup_images_file)
                cfg.data[entry].mixup_images_root = '{}/{}'.format(cfg.root_dir, cfg.mixup_images_root)

    return cfg
