"""Evaluate the original basis checkpoint with the ROI experiment's SAME masks."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from mmcv.parallel import MMDataParallel
from mmedit.apis import single_gpu_test
from mmedit.datasets import build_dataset, build_dataloader
from mmedit.models import build_model
from adaint.config_utils import load_config
from adaint.experiment_utils import load_enhancer_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='new ROI-Lab experiment configuration')
    parser.add_argument('--basis-config', default='adaint/configs/face_niguang_semantic_basis.yaml')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    cfg, basis = load_config(args.config), load_config(args.basis_config)
    checkpoint = args.checkpoint or cfg.get('roi_init_from')
    if not checkpoint:
        raise ValueError('a basis checkpoint is required')
    torch.set_num_threads(2)
    torch.manual_seed(cfg.get('seed', 2026))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dataset = build_dataset(cfg.data.val)
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0,
                             dist=False, shuffle=False, persistent_workers=False)
    model = build_model(basis.model, train_cfg=None, test_cfg=cfg.test_cfg)
    load_enhancer_checkpoint(model, checkpoint)
    results = single_gpu_test(MMDataParallel(model.cuda(), device_ids=[0]), loader)
    report = dict(checkpoint=checkpoint, validation_images=len(dataset),
        foreground_ignore_boxes=dict(cfg.semantic_masks.get('foreground_ignore_boxes', {})),
        metrics=dataset.evaluate(results), per_image=[r['eval_result'] for r in results])
    output = Path(args.output or os.path.join(cfg.work_dir, 'baseline_same_masks.json'))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(output=str(output), metrics=report['metrics']), indent=2))


if __name__ == '__main__':
    main()
