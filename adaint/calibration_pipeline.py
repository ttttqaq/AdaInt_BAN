"""Mild foreground-only linear-P3 exposure variation."""
import random

import numpy as np


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
