# dataset settings
dataset_type = 'STARDataset'
data_root = '/data/STAR/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1.0,
        flip=False,
        transforms=[
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + "split_ss_star/trainval/annfiles", #'train/object-TXT/',
        img_prefix=data_root + "split_ss_star/trainval/images", #'train/img/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + "split_ss_star/val/annfiles", #'val/object-TXT/',
        img_prefix=data_root + "split_ss_star/val/images", #'val/img/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "split_ss_star/test/images", #"'val/object-TXT/',
        img_prefix=data_root + "split_ss_star/test/images", # 'val/img/',
        pipeline=test_pipeline))
