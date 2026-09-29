exp_name = 'indoor_output2'

custom_imports = dict(
    imports=['adaint'],
    allow_failed_imports=False)

# number of self-distillation generations (gen 0 = plain training, gen>0 =
# born-again student distilled from gen-1's best checkpoint). Reduced from
# the paper's default (5-7) since the indoor set only has 67 training images
# and more generations bring diminishing/risky returns on such little data.
gen = 4
teacher_config_path = 'adaint/configs/face_indoor.py'
kdloss_type = 'L1Loss'
kdloss_weight = 1.0
# model settings
model = dict(
    type='AiLUT_kd_unified',
    n_ranks=3,
    n_vertices=33,
    en_adaint=True,
    en_adaint_share=False,
    backbone='res18',  # 'res18''tpami'
    pretrained=False,
    n_colors=3,
    sparse_factor=0.0001,
    smooth_factor=0,
    monotonicity_factor=10.0,
    recons_loss=dict(type='L1Loss', loss_weight=1.0))
# model training and testing settings
# 67 images/epoch (see adaint/annfiles/Face/indoor_train.txt) with batch size 1
# => 1 epoch == 67 iters. Fix AdaInt for the first 5 epochs.
train_cfg = dict(n_fix_iters=67 * 5)
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
data_root = '/DB/data/yujietang/IQ-LUT/face_dataset/all/1_indoors_data'

data = dict(
    workers_per_gpu=4,
    train_dataloader=dict(samples_per_gpu=1),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

    # train (90% of the indoor set, i.e. everything except the last 10%)
    train=dict(
        type=train_dataset_type,
        dir_lq=data_root,
        dir_gt=data_root,
        ann_file='adaint/annfiles/Face/indoor_train.txt',
        pipeline=train_pipeline,
        test_mode=False,
        filetmpl_lq='{}_input.jpg',
        filetmpl_gt='{}_gt.jpg'),
    # val (held-out last 10% of the indoor set, used for PSNR/deltaE during training)
    val=dict(
        type=val_dataset_type,
        dir_lq=data_root,
        dir_gt=data_root,
        ann_file='adaint/annfiles/Face/indoor_val.txt',
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
        ann_file='adaint/annfiles/Face/indoor_val.txt',
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
total_iters = 67 * 100

checkpoint_config = dict(interval=67, save_optimizer=False, by_epoch=False)
evaluation = dict(interval=67, save_image=False)
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
