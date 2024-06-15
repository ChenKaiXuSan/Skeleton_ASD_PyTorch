# _base_ = '/workspace/skeleton/compare_experiment/_base_/default_runtime.py'

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN', graph_cfg=dict(layout='coco', mode='stgcn_spatial')),
    cls_head=dict(type='GCNHead', num_classes=3, in_channels=256))

dataset_type = 'PoseDataset'
ann_file = '/workspace/data/seg_skeleton_pkl/whole_annotations.pkl'
train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSampleFrames', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(
        type='UniformSampleFrames', clip_len=100, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(
        type='UniformSampleFrames', clip_len=100, num_clips=10,
        test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file,
            pipeline=train_pipeline,
            split='train')))
val_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=val_pipeline,
        split='val',
        test_mode=True))
test_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=test_pipeline,
        split='val',
        test_mode=True))

val_evaluator = [dict(type='AccMetric'), dict(type='ConfusionMatrix')]
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=100, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=0,
        T_max=16,
        by_epoch=True,
        convert_to_iter_based=True)
]

optim_wrapper = dict(
    optimizer=dict(
        type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True))

default_hooks = dict(checkpoint=dict(interval=1), logger=dict(interval=100))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (16 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=128)

# ! write by mine 
load_from = '/workspace/skeleton/compare_experiment/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221129-484a394a.pth'
# resume = True

# runtime settings
default_scope = 'mmaction'  # The default registry scope to find modules. Refer to https://mmengine.readthedocs.io/en/latest/tutorials/registry.html
default_hooks = dict(  # Hooks to execute default actions like updating model parameters and saving checkpoints.
    runtime_info=dict(type='RuntimeInfoHook'),  # The hook to updates runtime information into message hub
    timer=dict(type='IterTimerHook'),  # The logger used to record time spent during iteration
    logger=dict(
        type='LoggerHook',  # The logger used to record logs during training/validation/testing phase
        interval=20,  # Interval to print the log
        ignore_last=False), # Ignore the log of last iterations in each epoch
    param_scheduler=dict(type='ParamSchedulerHook'),  # The hook to update some hyper-parameters in optimizer
    checkpoint=dict(
        type='CheckpointHook',  # The hook to save checkpoints periodically
        interval=3,  # The saving period
        save_best='auto',  # Specified metric to mearsure the best checkpoint during evaluation
        max_keep_ckpts=3),  # The maximum checkpoints to keep
    sampler_seed=dict(type='DistSamplerSeedHook'),  # Data-loading sampler for distributed training
    sync_buffers=dict(type='SyncBuffersHook'))  # Synchronize model buffers at the end of each epoch
env_cfg = dict(  # Dict for setting environment
    cudnn_benchmark=False,  # Whether to enable cudnn benchmark
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0), # Parameters to setup multiprocessing
    dist_cfg=dict(backend='nccl')) # Parameters to setup distributed environment, the port can also be set

work_dir = '/workspace/skeleton/logs/stgcn/'