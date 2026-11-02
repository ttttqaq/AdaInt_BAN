# Controlled shared-tone ablation; generated from the pinned safe protocol.
exp_name = 'face_indoor_shared_tone_2'

custom_imports = {'imports': ['adaint'], 'allow_failed_imports': False}

gen = 1

train_epochs = 100

adaint_fix_epochs = 0

checkpoint_interval_epochs = 5

evaluation_interval_epochs = 5

infer_after_generation = True

enable_kd = False

kdloss_type = 'MSELoss'

kdloss_weight = 0.1

model = {'type': 'AiLUTDual',
 'lut_mode': 'basis',
 'postprocess': {'result_condition': False,
                 'global_dc': False,
                 'regional_dc': False,
                 'global_dc_limit': 0.08,
                 'regional_dc_limit': 0.04,
                 'calibration_hidden': 64,
                 'multiscale': False,
                 'guidance': 'luminance',
                 'freeze_global': False,
                 'fixed_rgb_bias': None},
 'n_ranks': 5,
 'n_vertices': 33,
 'curve_points': 16,
 'stats_bins': 16,
 'stats_pool_size': 64,
 'condition_dim': 256,
 'head_hidden_dim': 128,
 'local_resolution': 128,
 'local_channels': 16,
 'context_size': 256,
 'min_interval': 0.001,
 'guided_filter_radius': 2,
 'guided_filter_epsilon': 0.001,
 'backbone': 'tpami',
 'pretrained': False,
 'smooth_factor': 1e-05,
 'monotonicity_factor': 0.01,
 'global_magnitude_factor': 1e-05,
 'local_magnitude_factor': 0.001,
 'local_smoothness_factor': 0.0001,
 'edge_smoothness_scale': 10.0,
 'color_loss_weight': 0.3,
 'color_grid_size': 16,
 'color_top_fraction': 0.1,
 'color_hard_weight': 0.25,
 'kd_decay_start_ratio': 0.3,
 'kd_decay_end_ratio': 0.8,
 'kd_final_weight': 0.0,
 'local_enabled': False,
 'use_region_condition': True,
 'region_samples_size': 32,
 'region_sample_mix': 0.5,
 'region_min_area': 0.0001,
 'foreground_rgb_weight': 0.1,
 'background_rgb_weight': 0.1,
 'foreground_color_weight': 0.1,
 'background_color_weight': 0.1,
 'face_color_weight': 0.0,
 'boundary_gradient_weight': 0.0,
 'safe_affine': {'enabled': True,
                 'knee': 0.92,
                 'raw_weight': 0.25,
                 'highlight_weight': 1.0,
                 'gradient_weight': 0.1,
                 'top_fraction': 0.1}}

test_cfg = {'metrics': ['PSNR',
             'deltaE_P3',
             'deltaE_P3_top',
             'deltaE_P3_foreground',
             'deltaE_P3_background',
             'deltaL_P3_foreground',
             'deltaAB_P3_foreground'],
 'crop_border': 0}

data = {'workers_per_gpu': 2,
 'train_dataloader': {'samples_per_gpu': 1, 'workers_per_gpu': 2, 'persistent_workers': True},
 'val_dataloader': {'samples_per_gpu': 1, 'workers_per_gpu': 2, 'persistent_workers': True},
 'test_dataloader': {'samples_per_gpu': 1, 'workers_per_gpu': 2, 'persistent_workers': True},
 'train': {'type': 'Face',
           'dir_lq': '/DB/data/yujietang/IQ-LUT/face_dataset/all/1_indoors_data',
           'dir_gt': '/DB/data/yujietang/IQ-LUT/face_dataset/all/1_indoors_data',
           'ann_file': 'adaint/annfiles/Face/indoor_train.txt',
           'pipeline': [{'type': 'LoadImageFromFile',
                         'io_backend': 'disk',
                         'key': 'lq',
                         'backend': 'pillow',
                         'channel_order': 'rgb'},
                        {'type': 'LoadImageFromFile',
                         'io_backend': 'disk',
                         'key': 'gt',
                         'backend': 'pillow',
                         'channel_order': 'rgb'},
                        {'type': 'ResizeToMatch',
                         'keys': ['lq'],
                         'reference_key': 'gt',
                         'interpolation': 'area'},
                        {'type': 'RandomRatioCrop', 'keys': ['lq', 'gt'], 'crop_ratio': [0.6, 1.0]},
                        {'type': 'Flip',
                         'keys': ['lq', 'gt'],
                         'flip_ratio': 0.5,
                         'direction': 'horizontal'},
                        {'type': 'RandomColorJitter',
                         'keys': ['lq'],
                         'brightness': 0.2,
                         'saturation': 0.2},
                        {'type': 'FlexibleRescaleToZeroOne', 'keys': ['lq', 'gt'], 'precision': 32},
                        {'type': 'ImageToTensor', 'keys': ['lq', 'gt']},
                        {'type': 'Collect',
                         'keys': ['lq', 'gt'],
                         'meta_keys': ['lq_path', 'gt_path']}],
           'test_mode': False,
           'filetmpl_lq': '{}_input.jpg',
           'filetmpl_gt': '{}_gt.jpg'},
 'val': {'type': 'Face',
         'dir_lq': '/DB/data/yujietang/IQ-LUT/face_dataset/all/1_indoors_data',
         'dir_gt': '/DB/data/yujietang/IQ-LUT/face_dataset/all/1_indoors_data',
         'ann_file': 'adaint/annfiles/Face/indoor_val.txt',
         'pipeline': [{'type': 'LoadImageFromFile',
                       'io_backend': 'disk',
                       'key': 'lq',
                       'backend': 'pillow',
                       'channel_order': 'rgb'},
                      {'type': 'LoadImageFromFile',
                       'io_backend': 'disk',
                       'key': 'gt',
                       'backend': 'pillow',
                       'channel_order': 'rgb'},
                      {'type': 'ResizeToMatch',
                       'keys': ['lq'],
                       'reference_key': 'gt',
                       'interpolation': 'area'},
                      {'type': 'FlexibleRescaleToZeroOne', 'keys': ['lq', 'gt'], 'precision': 32},
                      {'type': 'ImageToTensor', 'keys': ['lq', 'gt']},
                      {'type': 'Collect',
                       'keys': ['lq', 'gt'],
                       'meta_keys': ['lq_path', 'gt_path']}],
         'test_mode': True,
         'filetmpl_lq': '{}_input.jpg',
         'filetmpl_gt': '{}_gt.jpg'},
 'test': {'type': 'Face',
          'dir_lq': '/DB/data/yujietang/IQ-LUT/face_dataset/all/3_test_data',
          'dir_gt': '/DB/data/yujietang/IQ-LUT/face_dataset/all/3_test_data',
          'ann_file': 'adaint/annfiles/Face/test.txt',
          'pipeline': [{'type': 'LoadImageFromFile',
                        'io_backend': 'disk',
                        'key': 'lq',
                        'backend': 'pillow',
                        'channel_order': 'rgb'},
                       {'type': 'FlexibleRescaleToZeroOne', 'keys': ['lq'], 'precision': 32},
                       {'type': 'ImageToTensor', 'keys': ['lq']},
                       {'type': 'Collect', 'keys': ['lq'], 'meta_keys': ['lq_path']}],
          'test_mode': True,
          'filetmpl_lq': '{}_input.jpg',
          'filetmpl_gt': '{}_gt.jpg'}}

optimizers = {'type': 'Adam',
 'lr': 0.0001,
 'weight_decay': 0.0,
 'betas': [0.9, 0.999],
 'eps': 1e-08,
 'paramwise_cfg': {'custom_keys': {'adaint': {'lr_mult': 0.1}}}}

lr_config = {'policy': 'CosineAnnealing', 'min_lr': 1e-06, 'by_epoch': False}

checkpoint_config = {'save_optimizer': True}

evaluation = {'save_image': False}

log_config = {'hooks': [{'type': 'LUTEpochTextLoggerHook'}]}

visual_config = None

dist_params = {'backend': 'nccl'}

log_level = 'INFO'

postprocess_init = None

work_dir = '/DB/data/yujietang/IQ-LUT/image_enhance/AdaInt_BAN/work_dirs/face_indoor_shared_tone_2'

load_from = None

resume_from = None

find_unused_parameters = True

log_interval_epochs = 1

dataset = 'face'

semantic_masks = {'region_label': 'face_hair_neck',
 'train_foreground_root': '/DB/data/yujietang/IQ-LUT/face_dataset/all/indoor_face_masks/masks',
 'val_foreground_root': '/DB/data/yujietang/IQ-LUT/face_dataset/all/indoor_face_masks/masks',
 'test_foreground_root': '/DB/data/yujietang/IQ-LUT/face_dataset/all/test_face_masks/masks',
 'train_face_root': None,
 'val_face_root': None,
 'test_face_root': None,
 'foreground_suffix': '_face_mask',
 'face_suffix': '_face_mask',
 'feather_radius': 1,
 'perturb_probability': 0.0,
 'perturb_pixels': 1}

best_checkpoint_metric = 'PSNR'

best_checkpoint_rule = 'greater'

ablation = {'index': 2,
 'label': 'global_only',
 'seed': 2026,
 'baseline_commit': '1239ab4',
 'parent_commit': '0757473',
 'checkpoint_epochs': 5}
