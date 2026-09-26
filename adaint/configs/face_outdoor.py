exp_name = 'ailut_face_outdoor'

custom_imports = dict(
    imports=['adaint'],
    allow_failed_imports=False)

# model settings
model = dict(
    type='AiLUT',
    n_ranks=3,
    n_vertices=33,
    en_adaint=True,
    en_adaint_share=False,
    backbone='tpami',  # 'res18'
    pretrained=False,
    n_colors=3,
    sparse_factor=0.0001,
    smooth_factor=0,
    monotonicity_factor=10.0,
    recons_loss=dict(type='L1Loss', loss_weight=1.0))
# model training and testing settings
# 85 images/epoch (see adaint/annfiles/Face/outdoor_train.txt) with batch size 1
# => 1 epoch == 85 iters. Fix AdaInt for the first 5 epochs.
train_cfg = dict(n_fix_iters=85 * 5)
test_cfg = dict(metrics=['PSNR', 'deltaE'], crop_border=0)

# dataset settings
train_dataset_type = 'Face'
val_dataset_type = 'Face'

# NOTE: input images are 3000x4000 while groundtruth images are exported at
# 750x1000 (exactly 1/4 scale). `ResizeToMatch` downsizes `lq` to `gt`'s
# resolution right after loading, so all subsequent paired transforms
# (crop/flip) operate on equally-shaped images.
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        backend='pillow',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        backend='pillow',
        channel_order='rgb'),
    dict(type='ResizeToMatch', keys=['lq'], reference_key='gt', interpolation='area'),
    dict(type='RandomRatioCrop', keys=['lq', 'gt'], crop_ratio=(0.6, 1.0)),
    dict(type='Flip', keys=['lq', 'gt'],
         flip_ratio=0.5, direction='horizontal'),
    dict(type='RandomColorJitter', keys=[
         'lq'], brightness=0.2, saturation=0.2),
    dict(type='FlexibleRescaleToZeroOne', keys=['lq', 'gt'], precision=32),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        backend='pillow',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        backend='pillow',
        channel_order='rgb'),
    dict(type='ResizeToMatch', keys=['lq'], reference_key='gt', interpolation='area'),
    dict(type='FlexibleRescaleToZeroOne', keys=['lq', 'gt'], precision=32),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path'])
]

# the actual face dataset directory (LQ and GT share the same folder,
# distinguished by filename suffix)
data_root = '/DB/data/yujietang/IQ-LUT/face_dataset/all/2_outdoors_data'

data = dict(
    workers_per_gpu=4,
    train_dataloader=dict(samples_per_gpu=1),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

    # train (90% of the outdoor set, i.e. everything except the last 10%)
    train=dict(
        type=train_dataset_type,
        dir_lq=data_root,
        dir_gt=data_root,
        ann_file='adaint/annfiles/Face/outdoor_train.txt',
        pipeline=train_pipeline,
        test_mode=False,
        filetmpl_lq='{}_input.jpg',
        filetmpl_gt='{}_gt.jpg'),
    # val (held-out last 10% of the outdoor set, used for PSNR/deltaE during training)
    val=dict(
        type=val_dataset_type,
        dir_lq=data_root,
        dir_gt=data_root,
        ann_file='adaint/annfiles/Face/outdoor_val.txt',
        pipeline=test_pipeline,
        test_mode=True,
        filetmpl_lq='{}_input.jpg',
        filetmpl_gt='{}_gt.jpg'),
    # test (same held-out split as val; the untouched blind test set lives in
    # adaint/configs/face_test.py and is never referenced here)
    test=dict(
        type=val_dataset_type,
        dir_lq=data_root,
        dir_gt=data_root,
        ann_file='adaint/annfiles/Face/outdoor_val.txt',
        pipeline=test_pipeline,
        test_mode=True,
        filetmpl_lq='{}_input.jpg',
        filetmpl_gt='{}_gt.jpg'),
)

# optimizer
optimizers = dict(
    type='Adam',
    lr=1e-4,
    weight_decay=0,
    betas=(0.9, 0.999),
    eps=1e-8,
    paramwise_cfg=dict(custom_keys={'adaint': dict(lr_mult=0.1)}))
lr_config = None

# learning policy
total_iters = 85 * 400

checkpoint_config = dict(interval=85, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=85, save_image=False)
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
