"""Native foreground ROI geometry and P3/D65 conversion for calibration."""
import math

import torch
import torch.nn.functional as F

from .semantic_modules import checked_mask


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
