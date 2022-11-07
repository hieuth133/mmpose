default_scope = 'mmpose'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='coco/AP',
        rule='greater',
        max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='PoseVisualizationHook', enable=False))
custom_hooks = [dict(type='SyncBuffersHook')]
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(
    type='LogProcessor', window_size=50, by_epoch=True, num_digits=6)
log_level = 'INFO'
load_from = None
resume = False
file_client_args = dict(backend='disk')
train_cfg = dict(by_epoch=True, max_epochs=50, val_interval=10)
val_cfg = dict()
test_cfg = dict()
optim_wrapper = dict(optimizer=dict(type='Adam', lr=0.0005))
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001, by_epoch=False),
    dict(
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True)
]
auto_scale_lr = dict(base_batch_size=32)
codec = dict(
    type='MSRAHeatmap', input_size=(288, 384), heatmap_size=(72, 96), sigma=3)
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='LiteHRNet',
        in_channels=3,
        extra=dict(
            stem=dict(stem_channels=32, out_channels=32, expand_ratio=1),
            num_stages=3,
            stages_spec=dict(
                num_modules=(2, 4, 2),
                num_branches=(2, 3, 4),
                num_blocks=(2, 2, 2),
                module_type=('LITE', 'LITE', 'LITE'),
                with_fuse=(True, True, True),
                reduce_ratios=(8, 8, 8),
                num_channels=((40, 80), (40, 80, 160), (40, 80, 160, 320))),
            with_head=True)),
    head=dict(
        type='HeatmapHead',
        in_channels=40,
        out_channels=4,
        deconv_out_channels=None,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=dict(
            type='MSRAHeatmap',
            input_size=(288, 384),
            heatmap_size=(72, 96),
            sigma=3)),
    test_cfg=dict(flip_test=True, flip_mode='heatmap', shift_heatmap=True))
dataset_type = 'PlateDataset'
data_mode = 'topdown'
data_root = '../datasets/'
train_pipeline = [
    dict(type='LoadImage', file_client_args=dict(backend='disk')),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform',
        rotate_factor=60,
        scale_factor=(0.75, 1.25)),
    dict(type='TopdownAffine', input_size=(288, 384)),
    dict(
        type='GenerateTarget',
        target_type='heatmap',
        encoder=dict(
            type='MSRAHeatmap',
            input_size=(288, 384),
            heatmap_size=(72, 96),
            sigma=3)),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage', file_client_args=dict(backend='disk')),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(288, 384)),
    dict(type='PackPoseInputs')
]
train_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='PlateDataset',
        data_root='../datasets/',
        data_mode='topdown',
        ann_file='train/annotations/joined_5000.json',
        data_prefix=dict(img='train/images/'),
        pipeline=[
            dict(type='LoadImage', file_client_args=dict(backend='disk')),
            dict(type='GetBBoxCenterScale'),
            dict(type='RandomFlip', direction='horizontal'),
            dict(type='RandomHalfBody'),
            dict(
                type='RandomBBoxTransform',
                rotate_factor=60,
                scale_factor=(0.75, 1.25)),
            dict(type='TopdownAffine', input_size=(288, 384)),
            dict(
                type='GenerateTarget',
                target_type='heatmap',
                encoder=dict(
                    type='MSRAHeatmap',
                    input_size=(288, 384),
                    heatmap_size=(72, 96),
                    sigma=3)),
            dict(type='PackPoseInputs')
        ]))
val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='PlateDataset',
        data_root='../datasets/',
        data_mode='topdown',
        ann_file='val/annotations/joined_500.json',
        bbox_file=None,
        data_prefix=dict(img='val/images/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImage', file_client_args=dict(backend='disk')),
            dict(type='GetBBoxCenterScale'),
            dict(type='TopdownAffine', input_size=(288, 384)),
            dict(type='PackPoseInputs')
        ]))
test_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='PlateDataset',
        data_root='../datasets/',
        data_mode='topdown',
        ann_file='val/annotations/joined_500.json',
        bbox_file=None,
        data_prefix=dict(img='val/images/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImage', file_client_args=dict(backend='disk')),
            dict(type='GetBBoxCenterScale'),
            dict(type='TopdownAffine', input_size=(288, 384)),
            dict(type='PackPoseInputs')
        ]))
val_evaluator = dict(
    type='CocoMetric', ann_file='../datasets/val/annotations/joined_500.json')
test_evaluator = dict(
    type='CocoMetric', ann_file='../datasets/val/annotations/joined_500.json')
launcher = 'none'
work_dir = 'work_dir/train'
