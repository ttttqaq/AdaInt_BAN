"""Provenance validation for offline basis outputs; no reference network in student."""
import hashlib
import json
import math
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with open(str(path), 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def mask_signature(cfg):
    return json.loads(json.dumps(dict(cfg.semantic_masks), sort_keys=True))


def validate_reference_cache(cfg):
    root = Path(cfg.calibration_reference_dir)
    path = root / 'manifest.json'
    if not path.is_file():
        raise FileNotFoundError('run tools/prepare_bilateral_reference.py before training')
    manifest = json.loads(path.read_text())
    if manifest.get('version') != 1 or manifest.get('checkpoint_sha256') != sha256_file(cfg.calibration_init_from):
        raise ValueError('reference checkpoint changed or cache schema is incompatible')
    if manifest['semantic_masks'] != mask_signature(cfg):
        raise ValueError('reference masks/config differ; use a new cache directory')
    from .config_utils import load_config
    basis_model = json.loads(json.dumps(dict(load_config(cfg.calibration_basis_config).model), sort_keys=True))
    if manifest['basis_model'] != basis_model:
        raise ValueError('reference basis architecture/config changed')
    for split, key in [('train', 'train_records'), ('val', 'val_records')]:
        options = cfg.data[split]
        expected = Path(options.ann_file).read_text().split()
        if not expected or len(set(expected)) != len(expected) or set(expected) != set(manifest[key]):
            raise ValueError('reference {} split changed'.format(split))
        for sample, record in manifest[key].items():
            for kind in ('lq', 'gt'):
                source = Path(options['dir_' + kind]) / options['filetmpl_' + kind].format(sample)
                if str(source.resolve()) != record[kind + '_path']['path']:
                    raise ValueError('reference {} {} path changed'.format(split, kind))
            for field in ('lq_path', 'gt_path', 'mask_path'):
                if fingerprint(record[field]['path']) != record[field]:
                    raise ValueError('reference source changed: ' + record[field]['path'])
            if split == 'train' and not (root / record['cache']).is_file():
                raise FileNotFoundError('missing training reference: ' + record['cache'])
    metrics = json.loads((root / 'baseline_same_masks.json').read_text())
    for key in ('PSNR', 'deltaE_P3_foreground', 'deltaL_P3_foreground',
                'deltaL_P3_background', 'deltaL_P3_highlights'):
        if not math.isfinite(metrics.get(key, float('nan'))):
            raise ValueError('missing/nonfinite same-mask baseline metric: ' + key)
    for region in ('foreground', 'background', 'highlights'):
        if not 0 < metrics.get(region + '_valid_count', 0) <= len(manifest['val_records']):
            raise ValueError('no valid same-mask baseline region: ' + region)
    return manifest, metrics
