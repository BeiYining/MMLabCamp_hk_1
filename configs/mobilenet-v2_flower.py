model = dict(
    type='ImageClassifier',
    backbone=dict(type='MobileNetV2', widen_factor=1.0),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=5,
        in_channels=1280,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,),
    ))

dataset_type = 'ImageNet'
data_root = "/home/yangshuo/past_comp/data/flower/work_tmp"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type='ImageNet',
        data_prefix= data_root + '/train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='RandomResizedCrop', size=224),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ],
        ann_file= data_root  + '/train.txt',
        classes=data_root + '/classes.txt'),
    val=dict(
        type='ImageNet',
        data_prefix=data_root + '/val',
        ann_file=data_root + '/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(256, -1)),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ],
        classes=data_root + '/classes.txt'),
    test=dict(
        type='ImageNet',
        data_prefix='data/imagenet/val',
        ann_file='data/imagenet/meta/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(256, -1)),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]))
evaluation = dict(interval=5, metric='accuracy', metric_options=dict(topk=(1, )))
checkpoint_config = dict(interval=10)
log_config = dict(interval=10, 
                  hooks=[dict(type='TextLoggerHook') , 
                    dict(type='TensorboardLoggerHook')]
                  )
dist_params = dict(backend='nccl')
log_level = 'INFO'
# load_from = '/home/yangshuo/past_comp/mmclassification/param_model/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth'
load_from = '/home/yangshuo/past_comp/MMLabCamp_hk_1/work_dir/flower/latest.pth'
# resume_from = '/home/yangshuo/past_comp/MMLabCamp_hk_1/work_dir/flower/latest.pth'
resume_from = None
workflow = [('train', 1)]
optimizer = dict(type='SGD', lr=0.0001, momentum=0.2, weight_decay=0.001)
optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy='step',
    step=[5,15])

runner = dict(type='EpochBasedRunner', max_epochs=20)
work_dir = '/home/yangshuo/past_comp/MMLabCamp_hk_1/work_dir/flower'