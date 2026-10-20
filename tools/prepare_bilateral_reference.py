"""Cache TRAINING-only basis L*/100 and evaluate the SAME-mask validation baseline."""
import argparse
import copy
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image, ImageCms
import torch
from mmedit.datasets import build_dataset
from mmedit.models import build_model
from adaint.config_utils import load_config
from adaint.color import display_p3_to_lab
from adaint.experiment_utils import load_enhancer_checkpoint
from adaint.calibration_reference import (fingerprint, mask_signature, sha256_file,
                                          validate_reference_cache)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg.model.type != 'AiLUTBilateral':
        raise ValueError('reference preparation expects AiLUTBilateral configuration')
    root = Path(cfg.calibration_reference_dir)
    if root.exists():
        _, metrics = validate_reference_cache(cfg)
        print('Existing cache is valid: ' + str(root))
        print(json.dumps(metrics, indent=2))
        return
    basis = load_config(cfg.calibration_basis_config)
    torch.set_num_threads(2)
    torch.manual_seed(cfg.get('seed', 2026))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = build_model(basis.model, train_cfg=None, test_cfg=cfg.test_cfg).cuda().eval()
    load_enhancer_checkpoint(model, cfg.calibration_init_from)
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=root.name + '-building-', dir=str(root.parent)))
    manifest = dict(version=1, checkpoint=str(Path(cfg.calibration_init_from).resolve()),
        checkpoint_sha256=sha256_file(cfg.calibration_init_from), basis_model=dict(basis.model),
        semantic_masks=mask_signature(cfg), train_records={}, val_records={})
    try:
        with torch.no_grad():
            for split in ('train', 'val'):
                # Paired, full-image, no jitter/crop/exposure/reference-cache loading.
                options = copy.deepcopy(cfg.data.val)
                options.ann_file = cfg.data[split].ann_file
                options.dir_lq, options.dir_gt = cfg.data[split].dir_lq, cfg.data[split].dir_gt
                options.filetmpl_lq = cfg.data[split].filetmpl_lq
                options.filetmpl_gt = cfg.data[split].filetmpl_gt
                # Train and val can use different mask roots.
                for step in options.pipeline:
                    if step['type'] == 'LoadSemanticMasks':
                        step['foreground_root'] = cfg.semantic_masks[split + '_foreground_root']
                dataset = build_dataset(options)
                results = []
                for index in range(len(dataset)):
                    item = dataset[index]
                    meta = item['meta'].data
                    source = item['lq'][None].cuda()
                    gt = item['gt'][None].cuda()
                    mask = item['foreground_mask'][None].cuda()
                    sample = Path(meta['lq_path']).stem
                    sample = sample[:-6] if sample.endswith('_input') else sample
                    with Image.open(meta['lq_path']) as image:
                        profile = image.info.get('icc_profile')
                    if not profile or 'p3' not in ImageCms.getProfileDescription(
                            ImageCms.ImageCmsProfile(io.BytesIO(profile))).lower():
                        raise ValueError('calibration requires Display P3 source: ' + meta['lq_path'])
                    output = model.forward_dummy(source, foreground_mask=mask)[0]
                    record = dict(lq_path=fingerprint(meta['lq_path']), gt_path=fingerprint(meta['gt_path']),
                        mask_path=fingerprint(meta['foreground_mask_path']), shape=list(source.shape[-2:]))
                    if split == 'train':
                        lightness = display_p3_to_lab(output.clamp(0, 1))[:, :1].div(100).clamp(0, 1)
                        record['cache'] = sample + '.npy'
                        np.save(str(temporary / record['cache']),
                            lightness[0].permute(1, 2, 0).cpu().numpy().astype(np.float32), allow_pickle=False)
                    else:
                        evaluation = model.evaluate(output, gt, foreground_mask=mask,
                            supervision_mask=item['supervision_mask'][None].cuda())
                        results.append(dict(eval_result=evaluation))
                        record['metrics'] = evaluation
                    manifest[split + '_records'][sample] = record
                    print('{} {}/{} ID {}'.format(split, index + 1, len(dataset), sample), flush=True)
                if split == 'val':
                    (temporary / 'baseline_same_masks.json').write_text(json.dumps(dataset.evaluate(results), indent=2))
        (temporary / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        os.rename(str(temporary), str(root))
    except Exception:
        shutil.rmtree(str(temporary))
        raise
    print('Prepared TRAINING references and same-mask validation baseline: ' + str(root))
    print((root / 'baseline_same_masks.json').read_text())


if __name__ == '__main__':
    main()
