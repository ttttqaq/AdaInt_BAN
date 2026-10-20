"""Aligned cached references and mild foreground-only linear-P3 exposure variation."""
import json
from pathlib import Path
import random

import numpy as np
from mmedit.datasets.registry import PIPELINES


@PIPELINES.register_module()
class LoadCalibrationReference:
    def __init__(self, root):
        self.root = Path(root)
        manifest = self.root / 'manifest.json'
        if not manifest.is_file():
            raise FileNotFoundError('prepare reference cache first: python tools/prepare_bilateral_reference.py CONFIG')
        self.records = json.loads(manifest.read_text())['train_records']

    def __call__(self, results):
        sample = Path(results['lq_path']).stem
        sample = sample[:-6] if sample.endswith('_input') else sample
        record = self.records.get(sample)
        if record is None:
            raise ValueError('sample not in TRAINING reference cache: ' + sample)
        for key in ('lq_path', 'gt_path'):
            path = Path(results[key]).resolve()
            if str(path) != record[key]['path'] or path.stat().st_size != record[key]['size'] or path.stat().st_mtime_ns != record[key]['mtime_ns']:
                raise ValueError('reference source changed; prepare a new cache: ' + str(path))
        value = np.load(str(self.root / record['cache']), allow_pickle=False)
        if value.dtype != np.float32 or value.shape != results['lq'].shape[:2] + (1,) or not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
            raise ValueError('invalid or misaligned L*/100 reference for ' + sample)
        results['reference_lightness'] = value.copy()
        return results


def perturb_foreground_exposure(rgb, mask, max_ev=.3):
    """Float encoded P3 -> float encoded P3. Zero-mask pixels are BITWISE unchanged."""
    if rgb.dtype != np.float32 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('foreground exposure expects float32 encoded RGB')
    yy, xx = np.where(mask[..., 0] > .5)
    if not len(xx) or max_ev == 0:
        return rgb.copy()
    height, width = rgb.shape[:2]
    y, x = np.mgrid[:height, :width].astype(np.float32)
    u = (x - (xx.min() + xx.max()) / 2) / max(float(xx.max() - xx.min()) / 2, 1)
    v = (y - (yy.min() + yy.max()) / 2) / max(float(yy.max() - yy.min()) / 2, 1)
    style = random.randrange(3)
    amplitude = random.uniform(-max_ev, max_ev)
    if style == 0:
        pattern = np.ones_like(u)
    elif style == 1:
        angle = random.uniform(-np.pi, np.pi)
        pattern = np.clip(np.cos(angle) * u + np.sin(angle) * v, -1, 1)
    else:
        cx, cy, sigma = random.uniform(-.5, .5), random.uniform(-.5, .5), random.uniform(.4, 1.)
        pattern = np.exp(-((u - cx) ** 2 + (v - cy) ** 2) / (2 * sigma ** 2))
    gain = np.exp2(amplitude * pattern[..., None] * mask).astype(np.float32)
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    # Cap positive exposure by physical channel headroom, without introducing
    # additional clipping or changing already-white pixels when gain >= 1.
    headroom = np.maximum(1., 1. / np.maximum(linear.max(2, keepdims=True), 1e-8))
    changed = linear * np.minimum(gain, headroom)
    encoded = np.where(changed <= .0031308, changed * 12.92,
                       1.055 * np.maximum(changed, .0031308) ** (1 / 2.4) - .055)
    return np.where(mask > 0, encoded.clip(0, 1), rgb).astype(np.float32)
