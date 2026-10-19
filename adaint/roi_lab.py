"""Native-image foreground ROIs and bounded, centered P3/D65 Lab residuals."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .color import display_p3_to_lab
from .semantic_modules import checked_mask, masked_per_image


def roi_boxes(mask, margin=.15):
    """Square boxes in normalized IMAGE-EDGE coordinates; allow outside padding."""
    if not math.isfinite(margin) or not 0 <= margin <= 1:
        raise ValueError('ROI margin must be finite and between zero and one')
    height, width = mask.shape[-2:]
    boxes = []
    for image_mask in mask.detach():
        positions = (image_mask[0] > 0).nonzero(as_tuple=False)
        if not len(positions):
            boxes.append(mask.new_tensor([0., 0., 1., 1.]))
            continue
        lo, hi = positions.min(0)[0].to(mask), positions.max(0)[0].to(mask) + 1
        center = (lo + hi) / 2
        side = (hi - lo).max() * (1 + 2 * margin)
        boxes.append(torch.stack(((center[1] - side / 2) / width,
                                  (center[0] - side / 2) / height,
                                  (center[1] + side / 2) / width,
                                  (center[0] + side / 2) / height)))
    return torch.stack(boxes)


def _check_boxes(boxes, batch):
    if boxes.shape != (batch, 4) or not torch.isfinite(boxes).all():
        raise ValueError('ROI/crop boxes must have finite B,4 coordinates')
    if (boxes[:, 2:] <= boxes[:, :2]).any():
        raise ValueError('ROI/crop boxes must have positive width and height')


def sample_roi(image, boxes, size, mask=False):
    """Sample from original pixels, preserving aspect by square-box padding."""
    _check_boxes(boxes, image.size(0))
    positions = (torch.arange(size, device=image.device, dtype=image.dtype) + .5) / size
    x = boxes[:, 0:1] + positions[None] * (boxes[:, 2:3] - boxes[:, 0:1])
    y = boxes[:, 1:2] + positions[None] * (boxes[:, 3:4] - boxes[:, 1:2])
    grid = torch.stack((x[:, None].expand(-1, size, -1),
                        y[:, :, None].expand(-1, -1, size)), -1) * 2 - 1
    return F.grid_sample(image, grid, mode='bilinear',
                         padding_mode='zeros' if mask else 'border', align_corners=False)


def extract_roi(image, mask, size=128, margin=.15):
    mask = checked_mask(mask, image)
    boxes = roi_boxes(mask, margin)
    return sample_roi(image, boxes, size), sample_roi(mask, boxes, size, True), boxes


def paste_roi(field, boxes, size, crop_box=None):
    """Evaluate ONE full-ROI field at native/full-image or aligned crop pixels."""
    batch, _, _, _ = field.shape
    boxes = boxes.to(field)
    _check_boxes(boxes, batch)
    crop = (field.new_tensor([0., 0., 1., 1.])[None].expand(batch, -1)
            if crop_box is None else crop_box.to(field))
    _check_boxes(crop, batch)
    height, width = size
    px = (torch.arange(width, device=field.device, dtype=field.dtype) + .5) / width
    py = (torch.arange(height, device=field.device, dtype=field.dtype) + .5) / height
    x = crop[:, 0:1] + px[None] * (crop[:, 2:3] - crop[:, 0:1])
    y = crop[:, 1:2] + py[None] * (crop[:, 3:4] - crop[:, 1:2])
    x = (x - boxes[:, 0:1]) / (boxes[:, 2:3] - boxes[:, 0:1])
    y = (y - boxes[:, 1:2]) / (boxes[:, 3:4] - boxes[:, 1:2])
    grid = torch.stack((x[:, None].expand(-1, height, -1),
                        y[:, :, None].expand(-1, -1, width)), -1) * 2 - 1
    return F.grid_sample(field, grid, mode='bilinear', padding_mode='zeros', align_corners=False)


def decode_p3(rgb):
    magnitude = rgb.abs()
    return torch.where(magnitude <= .04045, rgb / 12.92,
                       rgb.sign() * ((magnitude + .055) / 1.055).pow(2.4))


def encode_p3(linear):
    magnitude = linear.abs()
    nonlinear = linear.sign() * (1.055 * magnitude.clamp_min(.0031308).pow(1 / 2.4) - .055)
    return torch.where(magnitude <= .0031308, linear * 12.92, nonlinear)


def lab_to_linear_p3(lab):
    """Inverse of this project's D65 Lab, without a D50 chromatic adaptation."""
    fy = (lab[:, 0] + 16) / 116
    fx, fz = fy + lab[:, 1] / 500, fy - lab[:, 2] / 200
    f = torch.stack((fx, fy, fz), 1)
    relative = torch.where(f > 6 / 29, f.pow(3), (f - 4 / 29) * (108 / 841))
    white = lab.new_tensor([.3127 / .3290, 1., (1 - .3127 - .3290) / .3290])
    xyz = relative * white[None, :, None, None]
    inverse = lab.new_tensor([
        [2.493496911941425, -.9313836179191239, -.40271078445071684],
        [-.8294889695615747, 1.7626640603183463, .023624685841943577],
        [.03584583024378447, -.07617238926804182, .9568845240076872]])
    return torch.einsum('ij,bjhw->bihw', inverse, xyz)


def compress_gamut(linear):
    """Compress chroma towards same-Y gray; preserve Y, avoid channel clipping."""
    weights = linear.new_tensor([35783 / 156275, 247089 / 357200, 198249 / 2500400])
    y = (linear * weights[None, :, None, None]).sum(1, keepdim=True).clamp(0, 1)
    chroma = linear - y
    upper = (1 - y) / chroma.clamp_min(1e-8)
    lower = y / (-chroma).clamp_min(1e-8)
    limits = torch.where(chroma > 0, upper, torch.where(chroma < 0, lower, torch.ones_like(chroma)))
    scale = limits.min(1, keepdim=True)[0].clamp(0, 1)
    return y + scale * chroma


def apply_lab_residual(rgb, delta):
    lab = display_p3_to_lab(rgb)
    desired = lab + delta
    # Physical luminance is kept within the displayable interval; chroma is
    # compressed jointly in LINEAR P3, preserving that luminance at the gamut edge.
    desired = torch.cat((desired[:, :1].clamp(0, 100), desired[:, 1:]), 1)
    result = encode_p3(compress_gamut(lab_to_linear_p3(desired)))
    return result, display_p3_to_lab(result) - lab


class ForegroundROILab(nn.Module):
    """Small native-ROI encoder shared by LUT conditioning and local Lab heads."""
    def __init__(self, condition_dim=256, channels=16, lightness_limit=8., chroma_limit=4.):
        super().__init__()
        if channels < 4 or condition_dim < 1:
            raise ValueError('invalid ROI feature dimensions')
        if any(not math.isfinite(v) or v <= 0 for v in (lightness_limit, chroma_limit)):
            raise ValueError('ROI Lab residual limits must be positive and finite')
        self.register_buffer('limits', torch.tensor([lightness_limit, chroma_limit, chroma_limit])[None, :, None, None])
        self.shallow = nn.Sequential(nn.Conv2d(7, channels, 3, padding=1), nn.LeakyReLU(.2),
                                     nn.Conv2d(channels, channels, 3, padding=1), nn.LeakyReLU(.2))
        self.deep = nn.Sequential(nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1), nn.LeakyReLU(.2),
                                  nn.Conv2d(channels * 2, channels * 2, 3, padding=1), nn.LeakyReLU(.2))
        self.fusion = nn.Sequential(nn.Conv2d(channels * 3, channels, 3, padding=1), nn.LeakyReLU(.2))
        self.condition = nn.Linear(channels, condition_dim)
        self.adapter = nn.Sequential(nn.Conv2d(channels + 3, channels, 3, padding=1), nn.LeakyReLU(.2))
        self.film = nn.Linear(condition_dim, channels * 2)
        self.output = nn.Conv2d(channels, 3, 1, bias=False)
        for layer in (self.condition, self.film, self.output):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def encode(self, rgb, mask, active):
        lightness = display_p3_to_lab(rgb)[:, :1] / 100
        mean3 = F.avg_pool2d(lightness, 3, 1, 1, count_include_pad=False)
        mean7 = F.avg_pool2d(lightness, 7, 1, 3, count_include_pad=False)
        shallow = self.shallow(torch.cat((rgb, mask, lightness, lightness - mean3, lightness - mean7), 1))
        deep = F.interpolate(self.deep(shallow), shallow.shape[-2:], mode='bilinear', align_corners=False)
        features = self.fusion(torch.cat((shallow, deep), 1))
        pooled = (features * mask).sum((-1, -2)) / mask.sum((-1, -2)).clamp_min(1e-8)
        return features, self.condition(pooled) * active.flatten(1)

    def decode(self, features, base_rgb, condition, mask, active):
        base_lab = display_p3_to_lab(base_rgb) / base_rgb.new_tensor([100., 128., 128.])[None, :, None, None]
        features = self.adapter(torch.cat((features, base_lab), 1))
        gamma, beta = self.film(condition).chunk(2, 1)
        features = F.leaky_relu((1 + gamma[:, :, None, None]) * features + beta[:, :, None, None], .2)
        bounded = self.output(features).tanh()
        mean = (bounded * mask).sum((-1, -2), keepdim=True) / mask.sum((-1, -2), keepdim=True).clamp_min(1e-8)
        # Half-limits make the CENTERED field stay within +/- limits too.
        return (bounded - mean) * (self.limits / 2) * active

    def regularizers(self, actual_delta, mask, source, min_area, edge_scale=10.):
        normalized = actual_delta / self.limits
        magnitude = masked_per_image(normalized.square().mean(1, keepdim=True), mask, min_area)[0]
        smooth = magnitude.new_zeros(())
        lightness = display_p3_to_lab(source)[:, :1] / 100
        for axis in (-1, -2):
            difference = torch.diff(normalized, dim=axis).square().mean(1, keepdim=True)
            edge = torch.diff(lightness, dim=axis).abs()
            lo, hi = [slice(None)] * 4, [slice(None)] * 4
            lo[axis], hi[axis] = slice(None, -1), slice(1, None)
            edge_mask = (mask[tuple(lo)] + mask[tuple(hi)]) / 2
            smooth += masked_per_image(difference * torch.exp(-edge_scale * edge), edge_mask, min_area)[0]
        return magnitude, smooth
