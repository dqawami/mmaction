# global parameters
num_test_classes = 700
num_train_classes = 700
subdir_name = 'kinetics700'
frames_type = 'rawframes'
root_dir = 'data'
data_root_dir = '{}/{}'.format(root_dir, subdir_name)
work_dir = None
load_from = None
resume_from = None

# model settings
model_partial_init = False
model = dict(
    type='ASLNet3D',
    backbone=dict(
        type='MobileNetV3_S3D',
        num_input_layers=3,
        mode='large',
        pretrained=None,
        pretrained2d=False,
        width_mult=1.0,
        pool1_stride_t=1,
        # block ids:      0  1  2  3  4  5  6  7  8  9  10 11 12 13 14
        temporal_strides=(1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1),
        temporal_kernels=(5, 3, 3, 3, 3, 5, 5, 3, 3, 5, 3, 3, 3, 3, 3),
        use_st_att=      (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0),
        use_temporal_avg_pool=True,
        input_bn=False,
        out_conv=True,
        out_attention=False,
        weight_norm='none',
        dropout_cfg=dict(
            p=0.1,
            mu=0.1,
            sigma=0.03,
            dist='gaussian'
        ),
    ),
    spatial_temporal_module=dict(
        type='AggregatorSpatialTemporalModule',
        modules=[
            dict(type='AverageSpatialTemporalModule',
                 temporal_size=4,
                 spatial_size=7),
        ],
    ),
    cls_head=dict(
        type='ClsHead',
        with_avg_pool=False,
        temporal_size=1,
        spatial_size=1,
        dropout_ratio=None,
        in_channels=960,
        num_classes=num_train_classes,
        embedding=True,
        embd_size=256,
        num_centers=1,
        st_scale=10.0,
        reg_weight=1.0,
        reg_threshold=0.1,
        main_loss_cfg=dict(
            type='AMSoftmaxLoss',
            pr_product=True,
            start_s=10.0,
            margin_type='cos',
            margin=0.35,
            gamma=0.0,
            t=1.0,
            conf_penalty_weight=0.085,
            filter_type='positives',
            top_k=None,
        ),
        extra_losses_cfg=dict(
            loss_lpush=dict(
                type='LocalPushLoss',
                margin=0.1,
                weight=1.0,
                smart_margin=True,
            ),
        ),
    ),
    masked_num=None,
)
train_cfg = None
test_cfg = None

# dataset settings
train_dataset_type = 'RawFramesDataset'
test_dataset_type = 'RawFramesDataset'
images_dir = '{}/{}'.format(data_root_dir, frames_type)
img_norm_cfg = dict(
    mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
    std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    to_rgb=True)

data = dict(
    videos_per_gpu=14,
    workers_per_gpu=3,
    num_test_classes=num_test_classes,
    train=dict(
        type=train_dataset_type,
        ann_file='{}/train{}.txt'.format(data_root_dir, num_train_classes),
        img_prefix=images_dir,
        img_norm_cfg=img_norm_cfg,
        num_segments=1,
        new_length=16,
        new_step=1,
        random_shift=True,
        proxy_generator=False,
        temporal_jitter=False,
        modality='RGB',
        image_tmpl='img_{:05d}.jpg',
        img_scale=256,
        input_size=(224, 224),
        div_255=False,
        flip_ratio=0.5,
        rotate_delta=10.0,
        resize_keep_ratio=True,
        random_crop=True,
        scale_limits=[1.0, 0.875],
        extra_augm=dict(
            brightness_range=(65, 190),
            contrast_range=(0.5, 1.5),
            saturation_range=(0.5, 1.5),
            hue_delta=18),
        test_mode=False),
    val=dict(
        type=test_dataset_type,
        ann_file='{}/val{}.txt'.format(data_root_dir, num_test_classes),
        img_prefix=images_dir,
        img_norm_cfg=img_norm_cfg,
        num_segments=1,
        new_length=16,
        new_step=1,
        proxy_generator=False,
        temporal_jitter=False,
        modality='RGB',
        image_tmpl='img_{:05d}.jpg',
        img_scale=256,
        input_size=(224, 224),
        div_255=False,
        flip_ratio=None,
        rotate_delta=None,
        resize_keep_ratio=True,
        random_crop=False,
        test_mode=True),
    test=dict(
        type=test_dataset_type,
        ann_file='{}/test{}.txt'.format(data_root_dir, num_test_classes),
        img_prefix=images_dir,
        img_norm_cfg=img_norm_cfg,
        num_segments=1,
        new_length=16,
        new_step=1,
        proxy_generator=False,
        temporal_jitter=False,
        modality='RGB',
        image_tmpl='img_{:05d}.jpg',
        img_scale=256,
        input_size=(224, 224),
        div_255=False,
        flip_ratio=None,
        rotate_delta=None,
        resize_keep_ratio=True,
        random_crop=False,
        test_mode=True)
)

# optimizer
optimizer = dict(type='SGD', lr=1e-2, momentum=0.9, weight_decay=1e-4)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
cudnn_benchmark = True

# learning policy
lr_config = dict(
    policy='step',
    step=[40, 80])
checkpoint_config = dict(interval=1)
workflow = [('train', 1)]
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 100
eval_epoch = 1
dist_params = dict(backend='nccl')
log_level = 'INFO'
