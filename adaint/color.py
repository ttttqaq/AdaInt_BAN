"""Display P3/D65 color supervision; no learned parameters or inference layers."""
import math
import numbers

import torch
import torch.nn as nn

from .modules import stratified_pixels


def display_p3_to_lab(rgb):
    """Encoded RGB NCHW -> CIE Lab with a D65 reference white.

    Display P3 uses the sRGB transfer curve, but different primaries. Matrix
    and transfer definitions: https://www.w3.org/TR/css-color-4/#color-conversion-code
    Both XYZ and Lab use D65 here; this is not CSS's D50-adapted Lab.
    Extend the transfer curve by reflection outside the gamut, and the Lab
    linear branch below its threshold, so predictions retain useful gradients.
    """
    if rgb.ndim != 4 or rgb.size(1) != 3 or not rgb.is_floating_point():
        raise ValueError('Display P3 conversion requires floating-point NCHW RGB')
    # Keep double precision for numeric checks; use float32 for half inputs.
    if rgb.dtype in (torch.float16, torch.bfloat16):
        rgb = rgb.float()
    magnitude = rgb.abs()
    nonlinear = rgb.sign() * ((magnitude + 0.055) / 1.055).pow(2.4)
    linear = torch.where(magnitude <= 0.04045, rgb / 12.92, nonlinear)
    matrix = rgb.new_tensor([
        [608311 / 1250200, 189793 / 714400, 198249 / 1000160],
        [35783 / 156275, 247089 / 357200, 198249 / 2500400],
        [0., 32229 / 714400, 5220557 / 5000800]])
    xyz = torch.einsum('ij,bjhw->bihw', matrix, linear)
    # D65 chromaticity x=0.3127, y=0.3290, consistent with the P3 matrix.
    white = rgb.new_tensor([0.3127 / 0.3290, 1.,
                            (1. - 0.3127 - 0.3290) / 0.3290])
    relative = xyz / white.view(1, 3, 1, 1)
    threshold = (6. / 29.) ** 3
    # Evaluate a safe inactive power branch too: torch.where evaluates both.
    f = torch.where(relative > threshold,
                    relative.clamp_min(threshold).pow(1. / 3.),
                    relative * (841. / 108.) + 4. / 29.)
    x, y, z = f.unbind(1)
    return torch.stack((116. * y - 16., 500. * (x - y),
                        200. * (y - z)), dim=1)


def p3_delta_e_map(pred, target, epsilon=0.):
    """Per-pixel Delta E 76 in Lab units, optionally smoothed at equality."""
    if pred.shape != target.shape:
        raise ValueError('color supervision requires aligned prediction and GT')
    difference = display_p3_to_lab(pred) - display_p3_to_lab(target)
    return (difference.square().sum(1) + epsilon ** 2).sqrt() - epsilon


def grid_top_mean(error_map, grid_size=16, top_fraction=0.1):
    """Per-image mean of the worst q non-overlapping rectangular grid cells.

    Partition the actual paired crop, not the full-image context. Uneven
    dimensions are split at floor(i * size / count); every pixel belongs to
    exactly one cell. Small images use fewer cells rather than duplicate them.
    """
    height, width = error_map.shape[-2:]
    rows, columns = min(grid_size, height), min(grid_size, width)
    row_sizes = [(i + 1) * height // rows - i * height // rows
                 for i in range(rows)]
    col_sizes = [(i + 1) * width // columns - i * width // columns
                 for i in range(columns)]
    row_means = torch.stack([part.mean(-2) for part in
                             torch.split(error_map, row_sizes, dim=-2)], dim=-2)
    cells = torch.stack([part.mean(-1) for part in
                         torch.split(row_means, col_sizes, dim=-1)], dim=-1)
    count = max(1, math.ceil(rows * columns * top_fraction))
    return cells.flatten(1).topk(count, dim=1).values.mean(1)


class P3LabColorLoss(nn.Module):
    """Global color distance plus a modest worst-grid term, normalized by 100."""
    def __init__(self, grid_size=16, top_fraction=0.1, hard_weight=0.25):
        super().__init__()
        if (isinstance(grid_size, bool) or
                not isinstance(grid_size, numbers.Integral) or grid_size < 1):
            raise ValueError('color_grid_size must be a positive integer')
        if not 0 < top_fraction <= 1 or not 0 <= hard_weight <= 1:
            raise ValueError('invalid color top fraction or hard-region weight')
        self.grid_size = int(grid_size)
        self.top_fraction = float(top_fraction)
        self.hard_weight = float(hard_weight)

    def statistics(self, pred, target, epsilon=0.):
        error = p3_delta_e_map(pred, target, epsilon)
        mean = error.mean((-1, -2))
        hard = grid_top_mean(error, self.grid_size, self.top_fraction)
        return mean, hard

    def forward(self, pred, target):
        mean, hard = self.statistics(pred, target, epsilon=1e-3)
        loss = ((1. - self.hard_weight) * mean + self.hard_weight * hard).mean() / 100.
        return loss, dict(color_delta_e=mean.detach().mean(),
                          color_delta_e_top=hard.detach().mean())


class P3LuminanceCDFLoss(nn.Module):
    """Match the L* distribution of a paired crop, optionally within each region."""

    def __init__(self, mode, weight, num_thresholds=32, bandwidth=None,
                 sample_size=64, min_region_samples=16):
        super().__init__()
        if mode not in ('whole', 'regional'):
            raise ValueError("luminance_histogram.mode must be 'whole' or 'regional'")
        if (isinstance(weight, bool) or not isinstance(weight, numbers.Real) or
                not math.isfinite(weight) or weight < 0):
            raise ValueError('luminance_histogram.weight must be finite and nonnegative')
        for name, value in (('num_thresholds', num_thresholds), ('sample_size', sample_size),
                            ('min_region_samples', min_region_samples)):
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
                raise ValueError('luminance_histogram.' + name + ' must be a positive integer')
        if bandwidth is None:
            bandwidth = 1. / num_thresholds
        if (isinstance(bandwidth, bool) or not isinstance(bandwidth, numbers.Real) or
                not math.isfinite(bandwidth) or bandwidth <= 0):
            raise ValueError('luminance_histogram.bandwidth must be finite and positive')
        self.mode = mode
        self.weight = float(weight)
        self.num_thresholds = num_thresholds
        self.bandwidth = float(bandwidth)
        self.sample_size = sample_size
        self.min_region_samples = min_region_samples

    def _soft_counts(self, image):
        pixels = stratified_pixels(image, self.sample_size).clamp(0, 1)
        lightness = display_p3_to_lab(pixels)[:, 0].flatten(1) / 100.
        thresholds = (torch.arange(self.num_thresholds, device=lightness.device,
                                   dtype=lightness.dtype) + .5) / self.num_thresholds
        return ((thresholds - lightness[..., None]) / self.bandwidth).sigmoid()

    def forward(self, prediction, target, supervision_mask):
        if prediction.shape != target.shape:
            raise ValueError('luminance histogram supervision requires aligned prediction and GT')
        # Using the same sample locations and weights on both sides lets us
        # subtract the soft counts before taking each region's average.
        difference = self._soft_counts(prediction) - self._soft_counts(target.detach())
        if self.mode == 'whole':
            return self.weight * difference.mean(1).abs().mean()
        foreground = stratified_pixels(supervision_mask.detach(), self.sample_size)
        foreground = foreground.flatten(2).to(difference)
        weights = torch.cat((foreground, 1 - foreground), dim=1)
        mass = weights.sum(-1)
        valid = mass >= self.min_region_samples
        error = (weights.bmm(difference) / mass.clamp_min(1)[..., None]).abs().mean(-1)
        valid_count = valid.sum(-1)
        per_image = (error * valid).sum(-1) / valid_count.clamp_min(1)
        return self.weight * per_image.sum() / (valid_count > 0).sum().clamp_min(1)
