exp_name = 'ailut_face_test'

custom_imports = dict(
    imports=['adaint'],
    allow_failed_imports=False)

# Blind inference on the untouched test split (`3_test_data`, 64 images,
# NO groundtruth is provided for this split). This config is ONLY meant to be
# used with `tools/test.py` (or `tools/deploy_test.py`) together with a
# checkpoint trained by either `face_indoor_kd_unified.py` or
# `face_outdoor_kd_unified.py` -- pick whichever domain best matches what you
# want to run inference on. The model architecture below must match the
# checkpoint you load (it does, since indoor/outdoor share the same
# n_ranks/n_vertices/backbone).
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
# metrics=None (or missing) => forward_test() does NOT require a groundtruth
# image, since none exists for this blind test split.
test_cfg = dict(metrics=None, crop_border=0)

dataset_type = 'Face'

# no ResizeToMatch here: there is no GT to match resolution against, so we
# run inference at the input's native full resolution (same as `adaint/demo.py`).
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        backend='pillow',
        channel_order='rgb'),
    dict(type='FlexibleRescaleToZeroOne', keys=['lq'], precision=32),
    dict(type='ImageToTensor', keys=['lq']),
    dict(type='Collect', keys=['lq'], meta_keys=['lq_path']),
]

data_root = '/DB/data/yujietang/IQ-LUT/face_dataset/all/3_test_data'

data = dict(
    workers_per_gpu=1,
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

    test=dict(
        type=dataset_type,
        dir_lq=data_root,
        # dir_gt/filetmpl_gt are required arguments of the dataset class but
        # are never read from disk since `test_pipeline` has no GT-loading
        # step; pointing them at dir_lq is harmless.
        dir_gt=data_root,
        ann_file='adaint/annfiles/Face/test.txt',
        pipeline=test_pipeline,
        test_mode=True,
        filetmpl_lq='{}_input.jpg',
        filetmpl_gt='{}_gt.jpg'),
)

dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
