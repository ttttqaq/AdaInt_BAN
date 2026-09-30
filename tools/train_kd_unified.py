# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import json
import shutil

import mmcv
import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmedit import __version__
from mmedit.apis import set_random_seed, single_gpu_test, train_model
from mmedit.datasets import build_dataloader, build_dataset
from mmedit.models import build_model
from mmedit.utils import collect_env, get_root_logger
from adaint.config_utils import load_config


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


def infer_and_save_split(cfg, dataset_cfg, checkpoint_path, save_dir):
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
    infer_cfg = copy.deepcopy(dataset_cfg)
    infer_cfg.pipeline = full_res_pipeline
    infer_cfg.test_mode = True
    dataset = build_dataset(infer_cfg)

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        dist=False,
        shuffle=False)

    # no GT is fed to the model (see `full_res_pipeline` above), so metrics
    # must be disabled here -- otherwise forward_test() would assert on a
    # missing GT image.
    infer_test_cfg = copy.deepcopy(cfg.test_cfg)
    infer_test_cfg['metrics'] = None
    model = build_model(cfg.model, train_cfg=None, test_cfg=infer_test_cfg)
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


def main():
    args = parse_args()

    cfg = load_config(args.config)
    if args.work_dir is not None:
        work_dir = args.work_dir
    else:
        work_dir = cfg.work_dir
    for gen in range(cfg.gen):
        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        # update configs according to CLI args
        gen_dir = f'gen_{gen}'
        cfg.work_dir = osp.join(work_dir, gen_dir)
        if args.resume_from is not None:
            cfg.resume_from = args.resume_from
        cfg.gpus = args.gpus

        if args.autoscale_lr:
            # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
            cfg.optimizer['lr'] = cfg.optimizer['lr'] * cfg.gpus / 8

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

        if gen > 0:
            config_path = cfg.teacher_config_path
            model_path = osp.join(work_dir, f'gen_{gen-1}', 'best.pth')
            loss_type = cfg.kdloss_type
            loss_weight = cfg.kdloss_weight
            cfg.model.kd_loss = dict(type='KD_Loss', config_path=config_path, model_path=model_path,
                                     loss_type=loss_type, loss_weight=loss_weight)
            cfg.model.gen = gen
        model = build_model(
            cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

        datasets = [build_dataset(cfg.data.train)]
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

        # meta information
        meta = dict()
        if cfg.get('exp_name', None) is None:
            cfg['exp_name'] = osp.splitext(osp.basename(cfg.work_dir))[0]
        meta['exp_name'] = cfg.exp_name
        meta['mmedit Version'] = __version__
        meta['seed'] = args.seed
        meta['env_info'] = env_info

        # add an attribute for visualization convenience
        train_model(
            model,
            datasets,
            cfg,
            distributed=distributed,
            validate=(not args.no_validate),
            timestamp=timestamp,
            meta=meta)

        json_file_path = osp.join(cfg.work_dir, f'{timestamp}.log.json')
        # 读取JSON文件内容
        with open(json_file_path, 'r') as file:
            json_data = file.read()

        # 将JSON数据解析为Python对象
        data_list = [json.loads(line)
                     for line in json_data.split('\n') if line.strip()]
        # 找到所有mode为val的PSNR
        val_psnr_list = [(entry['iter'], entry['PSNR'])
                         for entry in data_list if entry.get('mode') == 'val']

        # 找到PSNR最大的iter
        max_psnr_entry = max(val_psnr_list, key=lambda x: x[1])
        max_iter = max_psnr_entry[0]
        print(
            f"gen: {gen}, PSNR最大时的iter：{max_psnr_entry[0]}, 对应的PSNR值：{max_psnr_entry[1]}")

        # 复制一份PSNR最大的checkpoint
        # 源文件路径
        source_file = osp.join(cfg.work_dir, f'iter_{max_iter}.pth')
        # 目标文件路径
        destination_file = osp.join(cfg.work_dir, 'best.pth')
        shutil.copy(source_file, destination_file)

        # 用刚生成的 best.pth 对本代的训练集 + 验证集全部图片跑一遍推理（每张
        # 图都在其原图满分辨率上跑，不做 ResizeToMatch/裁剪），结果图片保存在
        # 同一个 work_dir（即 work_dirs/{exp_name}/gen_{gen}）下的 train_infer/
        # 子目录，文件名为 {id}_result.png。分布式训练时只在 rank 0 上跑一次，
        # 避免重复推理。
        rank, _ = get_dist_info()
        if not distributed or rank == 0:
            infer_save_dir = osp.join(cfg.work_dir, 'train_infer')
            infer_and_save_split(cfg, cfg.data.train, destination_file, infer_save_dir)
            infer_and_save_split(cfg, cfg.data.val, destination_file, infer_save_dir)
            print(f"gen: {gen}, 已用 best.pth 对训练集+验证集全部图片"
                 f"（满分辨率）推理完毕，结果保存在 {infer_save_dir}")


if __name__ == '__main__':
    main()
