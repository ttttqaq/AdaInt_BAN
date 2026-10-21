"""Run a short real-data optimizer/validation check without starting a full experiment."""
import argparse
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from mmedit.datasets import build_dataset
from mmedit.models import build_model
from adaint.config_utils import load_config
from adaint.semantic_pipeline import LoadSemanticMasks
from adaint.experiment_utils import initialize_postprocess


def tensor_batch(sample, device, max_side):
    data = {key: value.unsqueeze(0).to(device) for key, value in sample.items()
            if torch.is_tensor(value)}
    height, width = data['lq'].shape[-2:]
    if max(height, width) > max_side:
        scale = max_side / max(height, width)
        size = (max(1, round(height * scale)), max(1, round(width * scale)))
        for key in ('lq', 'gt', 'foreground_mask', 'supervision_mask', 'face_mask'):
            if key in data:
                mode = 'nearest' if key in ('supervision_mask', 'face_mask') else 'bilinear'
                options = dict(align_corners=False) if mode == 'bilinear' else {}
                data[key] = F.interpolate(data[key], size=size, mode=mode, **options)
    return data


def check_coverage(dataset):
    # The first item creates each loader's filename index and verifies image alignment.
    dataset[0]
    loaders = [step for step in dataset.pipeline.transforms if isinstance(step, LoadSemanticMasks)]
    if len(loaders) != 1:
        raise ValueError('expected one mask loader per split')
    loader = loaders[0]
    for entry in dataset.data_infos:
        stem = Path(entry['lq_path']).stem
        if stem.endswith('_input'):
            stem = stem[:-len('_input')]
        for index in loader.indices.values():
            if stem not in index:
                raise FileNotFoundError('missing mask for ' + entry['lq_path'])
    return len(dataset)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--max-side', type=int, default=256)
    parser.add_argument('--check-mask-coverage', action='store_true')
    args = parser.parse_args()
    if args.steps < 1 or args.max_side < 32:
        raise ValueError('steps >= 1 and max-side >= 32 required')
    cfg = load_config(args.config)
    if cfg.model.type != 'AiLUTDual':
        raise ValueError('this smoke check requires AiLUTDual')
    torch.manual_seed(2026)
    datasets = {split: build_dataset(cfg.data[split]) for split in ('train', 'val', 'test')}
    if args.check_mask_coverage:
        print(json.dumps(dict(mask_coverage={split: check_coverage(dataset)
                                           for split, dataset in datasets.items()})))
    model = build_model(cfg.model, train_cfg=dict(total_iters=args.steps, n_fix_iters=0),
                        test_cfg=cfg.test_cfg).to(args.device)
    if cfg.get('postprocess_init'):
        initialize_postprocess(model, cfg.postprocess_init)
    elif getattr(model, 'global_frozen', False):
        raise ValueError('freeze_global smoke check requires postprocess_init')
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizers.lr)
    for step in range(args.steps):
        model.train()
        data = tensor_batch(datasets['train'][step % len(datasets['train'])], args.device, args.max_side)
        result = model.train_step(data, optimizer)
        if any(not math.isfinite(value) for value in result['log_vars'].values()):
            raise RuntimeError('nonfinite smoke training log')
        print(json.dumps(dict(step=step + 1, train=result['log_vars'])))
    model.eval()
    data = tensor_batch(datasets['val'][0], args.device, args.max_side)
    with torch.no_grad():
        result = model.forward_test(**data)
    metrics = result['eval_result']
    if any(not math.isfinite(value) for value in metrics.values()):
        raise RuntimeError('nonfinite validation metric')
    print(json.dumps(dict(validation=metrics, parameters=sum(p.numel() for p in model.parameters()))))
    print('Semantic dual smoke check passed.')


if __name__ == '__main__':
    main()
