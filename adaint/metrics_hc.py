"""Human-centred metrics from PPR10K/code_3DLUT/calculate_metrics.m.

Human pixels (any nonzero mask value) have weight 1, background weight 0.5.
The official protocol averages over ALL pixels, without normalizing by the
mask sum. RGB errors are weighted before squaring; Lab errors before their
Euclidean norm. Images follow tensor2img's uint8 BGR convention.
"""
import numpy as np
from skimage.color import rgb2lab


def _cubic(distance):
    distance = np.abs(distance)
    return ((1.5 * distance ** 3 - 2.5 * distance ** 2 + 1) * (distance <= 1) +
            (-0.5 * distance ** 3 + 2.5 * distance ** 2 - 4 * distance + 2) *
            ((distance > 1) & (distance <= 2)))


def _resize_mask(mask, shape):
    """MATLAB imresize-style bicubic, antialiasing and symmetric boundaries.

    Keep the input mask's integer type after resizing, as imresize does.
    In particular, small positive interpolation tails must round to zero
    before the official mask==0 foreground/background decision.
    """
    result = mask.astype(np.float64)
    scales = np.array(shape, dtype=np.float64) / mask.shape
    for axis in np.argsort(scales):
        length, output_length = result.shape[axis], shape[axis]
        if length == output_length:
            continue
        scale = output_length / length
        width = 4 / min(scale, 1)
        positions = (np.arange(output_length) + 0.5) / scale + 0.5
        left = np.floor(positions - width / 2)
        indices = left[:, None].astype(np.int64) + np.arange(int(np.ceil(width)) + 2)
        distance = positions[:, None] - indices
        weights = scale * _cubic(scale * distance) if scale < 1 else _cubic(distance)
        weights /= weights.sum(axis=1, keepdims=True)
        # Pixel-edge symmetric extension, including the repeated end pixels.
        reflection = np.r_[np.arange(length), np.arange(length - 1, -1, -1)]
        indices = reflection[(indices - 1) % (2 * length)]
        pixels = np.take(result, indices, axis=axis)
        if axis == 0:
            result = (pixels * weights[..., None]).sum(axis=1)
        else:
            result = (pixels * weights[None, ...]).sum(axis=2)
    if np.issubdtype(mask.dtype, np.integer):
        limits = np.iinfo(mask.dtype)
        result = np.floor(np.clip(result, limits.min, limits.max) + 0.5).astype(mask.dtype)
    return result


def _prepare(img1, img2, mask, crop_border):
    img1, img2 = np.asarray(img1), np.asarray(img2)
    if img1.shape != img2.shape or img1.ndim != 3 or img1.shape[2] != 3:
        raise ValueError('HC metrics require matching HWC three-channel images')
    mask = np.asarray(mask)
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[..., 0]
    if mask.ndim != 2:
        raise ValueError('HC metrics require a grayscale human-region mask')
    if mask.shape != img1.shape[:2]:
        mask = _resize_mask(mask, img1.shape[:2])
    if crop_border < 0 or 2 * crop_border >= min(img1.shape[:2]):
        raise ValueError('crop_border leaves no pixels for HC evaluation')
    if crop_border:
        region = (slice(crop_border, -crop_border),) * 2
        img1, img2, mask = img1[region], img2[region], mask[region]
    weights = np.where(mask == 0, 0.5, 1.0)
    return img1.astype(np.float64), img2.astype(np.float64), weights


def psnr_hc(img1, img2, mask, crop_border=0):
    """PSNR of weighted RGB images, with the original peak value 255."""
    img1, img2, weights = _prepare(img1, img2, mask, crop_border)
    mse = np.mean(((img1 - img2) * weights[..., None]) ** 2)
    return float('inf') if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def delta_e_hc(img1, img2, mask, crop_border=0):
    """Weighted CIE76 Lab distance (the official Eab_hc metric)."""
    img1, img2, weights = _prepare(img1, img2, mask, crop_border)
    lab1 = rgb2lab(img1[..., ::-1] / 255.0)
    lab2 = rgb2lab(img2[..., ::-1] / 255.0)
    difference = (lab1 - lab2) * weights[..., None]
    return float(np.sqrt(np.sum(difference ** 2, axis=-1)).mean())
