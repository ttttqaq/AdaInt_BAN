"""Signed foreground L* grids; full-resolution queries use position AND lightness."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .color import display_p3_to_lab
from .calibration_geometry import (_check_boxes, encode_p3, lab_to_linear_p3,
                                    compress_gamut)


def apply_lightness_residual(rgb, delta):
    """Difference writeback: delta=0 preserves even out-of-gamut RGB EXACTLY.

    Compress chroma towards same-Y gray for both conversions. Subtracting the
    zero-delta conversion avoids changing inherited output through a Lab roundtrip.
    The final display clamp remains the existing evaluator/exporter's responsibility.
    """
    if delta.shape != rgb[:, :1].shape:
        raise ValueError('L* residual must align with RGB')
    lab = display_p3_to_lab(rgb)
    def converted(lightness):
        desired = torch.cat((lightness.clamp(0, 100), lab[:, 1:]), 1)
        return encode_p3(compress_gamut(lab_to_linear_p3(desired)))
    result = rgb + (converted(lab[:, :1] + delta) - converted(lab[:, :1]))
    return result, display_p3_to_lab(result)[:, :1] - lab[:, :1]


def slice_lightness_grid(grid, source, roi_box=None, crop_box=None):
    """B,1,L,V,U grid -> B,1,H,W; node endpoints are ROI image EDGES.

    Input L*/100 is a fixed, non-learned guide. Crops retain full-image position;
    never resize a sliced 2D map to the native image. Outside-ROI residuals are zero.
    """
    if grid.ndim != 5 or grid.shape[:2] != (source.size(0), 1):
        raise ValueError('grid must have B,1,D,H,W shape')
    batch, _, height, width = source.shape
    unit = source.new_tensor([0., 0., 1., 1.])[None].expand(batch, -1)
    boxes = unit if roi_box is None else roi_box.to(source)
    crop = unit if crop_box is None else crop_box.to(source)
    _check_boxes(boxes, batch)
    _check_boxes(crop, batch)
    x = (torch.arange(width, device=source.device, dtype=source.dtype) + .5) / width
    y = (torch.arange(height, device=source.device, dtype=source.dtype) + .5) / height
    x = crop[:, 0:1] + x[None] * (crop[:, 2:3] - crop[:, 0:1])
    y = crop[:, 1:2] + y[None] * (crop[:, 3:4] - crop[:, 1:2])
    x = (x - boxes[:, 0:1]) / (boxes[:, 2:3] - boxes[:, 0:1])
    y = (y - boxes[:, 1:2]) / (boxes[:, 3:4] - boxes[:, 1:2])
    xx, yy = x[:, None].expand(-1, height, -1), y[:, :, None].expand(-1, -1, width)
    guide = display_p3_to_lab(source)[:, 0].div(100).clamp(0, 1)
    query = torch.stack((xx, yy, guide), -1).unsqueeze(1) * 2 - 1
    field = F.grid_sample(grid, query, mode='bilinear', padding_mode='border',
                          align_corners=True).squeeze(2)
    inside = ((xx >= 0) & (xx <= 1) & (yy >= 0) & (yy <= 1))[:, None]
    return field * inside.to(field)


class ForegroundLightnessGrid(nn.Module):
    def __init__(self, condition_dim=256, channels=16, spatial_nodes=8,
                 lightness_nodes=8, lightness_limit=8.):
        super().__init__()
        if channels < 4 or spatial_nodes < 2 or lightness_nodes < 2:
            raise ValueError('invalid calibration grid dimensions')
        if not math.isfinite(lightness_limit) or lightness_limit <= 0:
            raise ValueError('lightness_limit must be positive and finite')
        self.spatial_nodes, self.lightness_nodes = int(spatial_nodes), int(lightness_nodes)
        self.lightness_limit = float(lightness_limit)
        self.shallow = nn.Sequential(nn.Conv2d(13, channels, 3, padding=1), nn.LeakyReLU(.2),
                                     nn.Conv2d(channels, channels, 3, padding=1), nn.LeakyReLU(.2))
        self.down = nn.Sequential(nn.Conv2d(channels, channels * 2, 3, 2, 1), nn.LeakyReLU(.2),
            nn.Conv2d(channels * 2, channels * 3, 3, 2, 1), nn.LeakyReLU(.2),
            nn.Conv2d(channels * 3, channels * 4, 3, 2, 1), nn.LeakyReLU(.2))
        self.context = nn.Sequential(nn.Linear(condition_dim * 2, channels), nn.LeakyReLU(.2))
        self.output = nn.Conv2d(channels * 5, lightness_nodes, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, source, base, mask, conditions, active):
        lightness = display_p3_to_lab(source)[:, :1] / 100
        base_lightness = display_p3_to_lab(base.clamp(0, 1))[:, :1] / 100
        differences = [lightness - F.avg_pool2d(lightness, k, 1, k // 2,
                                               count_include_pad=False) for k in (3, 7)]
        height, width = source.shape[-2:]
        u = ((torch.arange(width, device=source.device, dtype=source.dtype) + .5) / width * 2 - 1)
        v = ((torch.arange(height, device=source.device, dtype=source.dtype) + .5) / height * 2 - 1)
        coordinates = (u[None,None,None].expand(source.size(0),1,height,-1),
                       v[None,None,:,None].expand(source.size(0),1,-1,width))
        features = self.shallow(torch.cat((source, base.clamp(0, 1), mask,
                                          lightness, base_lightness, *differences, *coordinates), 1))
        features = F.adaptive_avg_pool2d(self.down(features), (self.spatial_nodes,) * 2)
        context = self.context(torch.cat(conditions, 1))[:, :, None, None].expand(
            -1, -1, self.spatial_nodes, self.spatial_nodes)
        raw = self.output(torch.cat((features, context), 1)).unsqueeze(1)
        # No region-mean subtraction: constant, signed exposure changes are representable.
        return self.lightness_limit * raw.tanh() * active.unsqueeze(-1)


def multiscale_calibration_loss(actual, target, mask, scales=(1, 4, 8), min_coverage=.05):
    """Compare masked cell MEANS, image-first; target is a stopped-gradient residual."""
    losses, errors = [], {}
    for scale in scales:
        mass = F.adaptive_avg_pool2d(mask, (scale, scale))
        difference = F.adaptive_avg_pool2d((actual - target) * mask, (scale, scale)) / mass.clamp_min(1e-8)
        valid = (mass >= min_coverage).to(mask)
        error = ((difference / 100).square() + 1e-6).sqrt() - .001
        count = valid.sum((1, 2, 3))
        values = (error * valid).sum((1, 2, 3)) / count.clamp_min(1)
        images = (count > 0).to(values)
        losses.append((values * images).sum() / images.sum().clamp_min(1))
        absolute = (difference.abs() * valid).sum((1, 2, 3)) / count.clamp_min(1)
        errors[scale] = ((absolute * images).sum() / images.sum().clamp_min(1)).detach()
    return torch.stack(losses).mean(), errors
