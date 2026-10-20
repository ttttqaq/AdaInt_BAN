# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import glob
import os
import os.path as osp
import time
import json
import shutil
import sys
import subprocess

# Prefer this checkout over any editable installation of the project.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mmcv
from mmcv import Config
import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmedit import __version__
from mmedit.apis import set_random_seed, single_gpu_test, train_model
from mmedit.datasets import build_dataloader, build_dataset
from mmedit.models import build_model
from mmedit.utils import collect_env, get_root_logger
from adaint.config_utils import load_config
from adaint.experiment_utils import (
    ENHANCER_TYPES, inference_config, load_enhancer_checkpoint,
    resolve_training_schedule, save_generation_configs, configure_epoch_display,
    select_best_checkpoint)


def parse_args():
    parser = argparse.ArgumentParser(description='Train an editor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--start-gen', type=int, default=0,
                        help='start at an existing generation after interruption')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def infer_and_save_split(cfg, dataset_cfg, checkpoint_path, save_dir, split=None):
    r"""Run inference with `checkpoint_path` over the images described by
    `dataset_cfg` (e.g. `cfg.data.train` or `cfg.data.val`) and save the
    enhanced outputs as `{id}_result.png` under `save_dir`.

    This deliberately builds a LQ-only pipeline (no `ResizeToMatch`, no GT
    loading) so every image is enhanced at its own native full resolution,
    exactly like `adaint/demo.py` / the blind-test config (`face_test.py`) do
    -- NOT downsized to the GT's resolution the way training/validation
    metrics computation does. Always runs on a single GPU (cuda:0),
    regardless of `--gpus`/`--launcher`.
    """
    full_res_pipeline = [
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
    if cfg.model.type in ('AiLUTDual', 'AiLUTBilateral'):
        from adaint.semantic_pipeline import semantic_mask_step
        if split is None:
            split = next((name for name in ('train', 'val', 'test')
                if cfg.data.get(name) and
                dataset_cfg.ann_file == cfg.data[name].ann_file and
                dataset_cfg.dir_lq == cfg.data[name].dir_lq), None)
        if split not in ('train', 'val', 'test'):
            raise ValueError('specify train/val/test to select the matching mask root')
        step = semantic_mask_step(cfg, split, training=False)
        full_res_pipeline.insert(1, step)
        mask_keys = ['foreground_mask', 'supervision_mask']
        if step.get('face_root'):
            mask_keys.append('face_mask')
        for transform in full_res_pipeline:
            if transform['type'] in ('FlexibleRescaleToZeroOne', 'ImageToTensor', 'Collect'):
                transform['keys'] = list(transform['keys']) + mask_keys
        full_res_pipeline[-1]['meta_keys'].append('foreground_mask_path')
    infer_cfg = copy.deepcopy(dataset_cfg)
    infer_cfg.pipeline = full_res_pipeline
    infer_cfg.test_mode = True
    dataset = build_dataset(infer_cfg)

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        persistent_workers=cfg.data.get('workers_per_gpu', 1) > 0,
        dist=False,
        shuffle=False)

    # no GT is fed to the model (see `full_res_pipeline` above), so metrics
    # must be disabled here -- otherwise forward_test() would assert on a
    # missing GT image.
    infer_test_cfg = copy.deepcopy(cfg.test_cfg)
    infer_test_cfg['metrics'] = None
    model_cfg = inference_config(cfg).model
    model = build_model(model_cfg, train_cfg=None, test_cfg=infer_test_cfg)
    if model_cfg.type in ENHANCER_TYPES:
        load_enhancer_checkpoint(model, checkpoint_path)
    else:
        load_checkpoint(model, checkpoint_path, map_location='cpu')
    model = MMDataParallel(model.cuda(), device_ids=[0])

    mmcv.mkdir_or_exist(save_dir)
    single_gpu_test(model, data_loader, save_image=True, save_path=save_dir)

    # forward_test() names the saved file after the LQ image's own basename
    # (i.e. `{id}_input.png`, since `filetmpl_lq='{}_input.jpg'`); rename to
    # `{id}_result.png`.
    for fname in os.listdir(save_dir):
        if fname.endswith('_input.png'):
            new_name = fname[:-len('_input.png')] + '_result.png'
            os.replace(osp.join(save_dir, fname), osp.join(save_dir, new_name))


def build_generation_config(base_cfg, gen, work_dir, args):
    """Create an isolated config for one self-distillation generation.

    MMCV's runner mutates hook configs while registering them (for example,
    it removes ``lr_config['policy']``). Each generation therefore needs its
    own deep copy instead of reusing the config passed to the previous run.
    """
    cfg = Config(
        copy.deepcopy(base_cfg._cfg_dict),
        filename=getattr(base_cfg, 'filename', None))
    cfg.work_dir = osp.join(work_dir, f'gen_{gen}')
    cfg.gpus = args.gpus

    # A resume checkpoint belongs only to the first selected generation.
    # Following generations start afresh and distil the previous best model.
    cfg.resume_from = base_cfg.get('resume_from') if gen == args.start_gen else None
    if args.resume_from is not None and gen == args.start_gen:
        cfg.resume_from = args.resume_from
    if cfg.resume_from and cfg.model.type in ENHANCER_TYPES:
        hooks = cfg.get('custom_hooks', [])
        if not any(hook['type'] == 'LUTResumeCounterHook' for hook in hooks):
            hooks.append(dict(type='LUTResumeCounterHook', priority='HIGHEST'))
        cfg.custom_hooks = hooks

    if args.autoscale_lr:
        # Scale from the original base learning rate for every generation.
        cfg.optimizers['lr'] = cfg.optimizers['lr'] * cfg.gpus / 8

    if gen > 0:
        cfg.model.gen = gen
        cfg.model.kd_loss = None
        if cfg.get('enable_kd', True):
            model_path = osp.join(work_dir, f'gen_{gen-1}', 'best.pth')
            cfg.model.kd_loss = dict(
                type='KD_Loss',
                config_path=osp.join(work_dir, f'gen_{gen-1}', 'inference_config.py'),
                model_path=model_path,
                loss_type=cfg.kdloss_type,
                loss_weight=cfg.kdloss_weight)

    return cfg


def main():
    args = parse_args()

    base_cfg = load_config(args.config)
    if args.seed is None:
        args.seed = base_cfg.get('seed')
    if args.work_dir is not None:
        work_dir = args.work_dir
    else:
        work_dir = base_cfg.work_dir
    if args.no_validate:
        raise ValueError('train_kd_unified requires validation to select best.pth')
    if not 0 <= args.start_gen < base_cfg.gen:
        raise ValueError('--start-gen must be within configured generations')
    for gen in range(args.start_gen, base_cfg.gen):
        cfg = build_generation_config(base_cfg, gen, work_dir, args)
        if cfg.model.type == 'AiLUTBilateral':
            from adaint.calibration_checkpoint import validate_scratch_start
            validate_scratch_start(cfg)
        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True

        # init distributed env first, since logger depends on the dist info.
        if args.launcher == 'none':
            distributed = False
        else:
            distributed = True
            init_dist(args.launcher, **cfg.dist_params)

        # create work_dir
        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
        # init the logger before other steps
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
        logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

        # log env info
        env_info_dict = collect_env.collect_env()
        env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                    dash_line)

        # log some basic info
        logger.info('Distributed training: {}'.format(distributed))
        logger.info('mmedit Version: {}'.format(__version__))
        logger.info('Config:\n{}'.format(cfg.text))

        # set random seeds
        if args.seed is not None:
            logger.info('Set random seed to {}, deterministic: {}'.format(
                args.seed, args.deterministic))
            set_random_seed(args.seed, deterministic=args.deterministic)
        cfg.seed = args.seed

        datasets = [build_dataset(cfg.data.train)]
        resolve_training_schedule(cfg, len(datasets[0]), distributed)
        configure_epoch_display(logger, cfg)
        if not cfg.get('checkpoint_config') or not cfg.get('evaluation'):
            raise ValueError('generation training needs checkpoint and evaluation hooks')
        checkpoint_interval = cfg.checkpoint_config.interval
        evaluation_interval = cfg.evaluation.interval
        if (checkpoint_interval < 1 or evaluation_interval < 1 or
                evaluation_interval % checkpoint_interval != 0):
            raise ValueError('evaluation interval must be a positive multiple of '
                             'checkpoint interval to select an existing best checkpoint')
        if cfg.get('log_interval_epochs') is not None:
            logger.info('Training schedule: %s images, %s epochs/generation, '
                        'AdaInt fixed for %s epochs; save every %s epochs, '
                        'evaluate every %s epochs, log every %s epochs',
                        len(datasets[0]), cfg.train_epochs, cfg.adaint_fix_epochs,
                        cfg.checkpoint_interval_epochs, cfg.evaluation_interval_epochs,
                        cfg.log_interval_epochs)
        else:
            logger.info('Resolved training budget: %s images, %s iters/generation, '
                        '%s AdaInt freeze iters', len(datasets[0]), cfg.total_iters,
                        cfg.train_cfg.get('n_fix_iters', 0))
        save_generation_configs(cfg)
        model = build_model(
            cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

        if cfg.model.type == 'AiLUTBilateral':
            if model.n_fix_iters or not all(p.requires_grad for p in model.parameters()):
                raise ValueError('scratch training requires all student parameters to be trainable')
            logger.info('Bilateral calibration: %s trainable parameters; %s',
                sum(p.numel() for p in model.parameters()),
                'resume interrupted training' if cfg.get('resume_from') else 'initialize from scratch')

        if len(cfg.workflow) == 2:
            val_dataset = copy.deepcopy(cfg.data.val)
            val_dataset.pipeline = cfg.data.train.pipeline
            datasets.append(build_dataset(val_dataset))
        if cfg.checkpoint_config is not None:
            # save version, config file content and class names in
            # checkpoints as meta data
            cfg.checkpoint_config.meta = dict(
                mmedit_version=__version__,
                config=cfg.text,
            )

        # Reject incompatible old architectures before MMCV's non-strict resume/load.
        for checkpoint_path in (cfg.get('resume_from'), cfg.get('load_from')):
            if checkpoint_path and cfg.model.type in ENHANCER_TYPES:
                load_enhancer_checkpoint(model, checkpoint_path)

        # meta information
        meta = dict()
        if cfg.get('exp_name', None) is None:
            cfg['exp_name'] = osp.splitext(osp.basename(cfg.work_dir))[0]
        meta['exp_name'] = cfg.exp_name
        meta['mmedit Version'] = __version__
        meta['seed'] = args.seed
        meta['env_info'] = env_info
        if cfg.model.type == 'AiLUTBilateral':
            meta['training_initialization'] = 'scratch'
            cfg.checkpoint_config.meta['training_initialization'] = 'scratch'
        try:
            meta['code_revision'] = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                stderr=subprocess.DEVNULL).decode().strip()
        except (OSError, subprocess.CalledProcessError):
            meta['code_revision'] = 'unavailable'
        cfg.checkpoint_config.meta['code_revision'] = meta['code_revision']

        # add an attribute for visualization convenience
        train_model(
            model,
            datasets,
            cfg,
            distributed=distributed,
            validate=(not args.no_validate),
            timestamp=timestamp,
            meta=meta)

        # Include this generation's earlier validation history after a resume.
        data_list = []
        for json_file_path in sorted(glob.glob(osp.join(cfg.work_dir, '*.log.json'))):
            with open(json_file_path) as file:
                data_list.extend(json.loads(line) for line in file if line.strip())
        epoch_steps = cfg.iters_per_epoch if cfg.get('log_interval_epochs') is not None else None
        best_metric = cfg.get('best_checkpoint_metric', 'PSNR')
        best_rule = cfg.get('best_checkpoint_rule', 'greater')
        destination_file = osp.join(cfg.work_dir, 'best.pth')
        if cfg.model.type == 'AiLUTBilateral':
            from adaint.calibration_checkpoint import save_calibration_selection
            selection = save_calibration_selection(cfg, data_list, epoch_steps)
            print('Scratch checkpoint selection: {}'.format(selection))
        else:
            best_score, progress, source_file = select_best_checkpoint(
                data_list, cfg.work_dir, epoch_steps, best_metric, best_rule)
            unit = 'epoch' if epoch_steps else 'iter'
            print('gen: {}, {} 最佳 {}: {:g}, score: {}'.format(
                gen, best_metric, unit, progress, best_score))

            # 复制按配置指标选出的 checkpoint
            # 源文件路径
            # 目标文件路径
            destination_file = osp.join(cfg.work_dir, 'best.pth')
            shutil.copy(source_file, destination_file)

        # 用刚生成的 best.pth 对本代的训练集 + 验证集全部图片跑一遍推理（每张
        # 图都在其原图满分辨率上跑，不做 ResizeToMatch/裁剪），结果图片保存在
        # 同一个 work_dir（即 work_dirs/{exp_name}/gen_{gen}）下的 train_infer/
        # 子目录，文件名为 {id}_result.png。分布式训练时只在 rank 0 上跑一次，
        # 避免重复推理。
        rank, _ = get_dist_info()
        if cfg.get('infer_after_generation', True) and (not distributed or rank == 0):
            infer_save_dir = osp.join(cfg.work_dir, 'train_infer')
            infer_and_save_split(cfg, cfg.data.train, destination_file, infer_save_dir, split='train')
            infer_and_save_split(cfg, cfg.data.val, destination_file, infer_save_dir, split='val')
            print(f"gen: {gen}, 已用 best.pth 对训练集+验证集全部图片"
                 f"（满分辨率）推理完毕，结果保存在 {infer_save_dir}")


if __name__ == '__main__':
    main()
