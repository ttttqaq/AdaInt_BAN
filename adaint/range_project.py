"""Identity-in-gamut endpoint protection and GT-relative error supervision."""
import math

import torch
import torch.nn.functional as F


DEFAULTS = dict(enabled=False, range_weight=.1, hard_weight=.1,
                gradient_weight=.01, hard_fraction=.1)


def range_options(value=None):
    options = dict(DEFAULTS)
    if value is not None:
        if not isinstance(value, dict) or set(value) - set(options):
            raise ValueError('unknown or non-mapping range_project options')
        options.update(value)
    if not isinstance(options['enabled'], bool):
        raise ValueError('range_project.enabled must be boolean')
    for key in ('range_weight', 'hard_weight', 'gradient_weight', 'hard_fraction'):
        number = options[key]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
            raise ValueError('range_project.' + key + ' must be finite')
    if not 0 < options['hard_fraction'] <= 1:
        raise ValueError('range_project.hard_fraction must be in (0,1]')
    if any(options[key] < 0 for key in ('range_weight', 'hard_weight', 'gradient_weight')):
        raise ValueError('range_project loss weights must be nonnegative')
    return options


def project_rgb(raw):
    """The RGB cube's Euclidean projection; all legal channels are unchanged."""
    if not raw.is_floating_point() or raw.ndim != 4 or raw.size(1) != 3:
        raise ValueError('expected floating B,3,H,W RGB')
    raw = raw.float() if raw.dtype in (torch.float16, torch.bfloat16) else raw
    if not torch.isfinite(raw).all():
        raise FloatingPointError('nonfinite enhancer proposal')
    return raw.clamp(0, 1)


def range_error(raw):
    return (F.relu(raw - 1).square() + F.relu(-raw).square()).mean(1, keepdim=True)


def regional_mean(error, weights):
    mass = weights.sum((1, 2, 3))
    valid = mass > 0
    value = (error * weights).sum((1, 2, 3)) / mass.clamp_min(1e-12)
    return (value * valid.to(error)).sum() / valid.sum().clamp_min(1).to(error)


@torch.no_grad()
def difficult_weights(raw, target, mask, fraction=.1, min_area=.0001):
    """Select actual RGB errors/violations, independently per image and region.

    Selection has no gradient. Whole-image and whole-region reconstruction
    remain active, so crossing this selection boundary cannot remove supervision.
    """
    score = (raw.detach() - target.detach()).square().mean(1, keepdim=True)
    score = score + range_error(raw.detach())
    selected = torch.zeros_like(mask)
    for index in range(len(raw)):
        weights = mask[index, 0].detach().flatten()
        support = torch.nonzero(weights > 0, as_tuple=False).flatten()
        if weights.mean() < min_area or support.numel() == 0:
            continue
        count = max(1, int(math.ceil(fraction * support.numel())))
        local = score[index, 0].flatten()[support]
        chosen = support[torch.topk(local, count, sorted=False).indices]
        selected[index, 0].view(-1)[chosen] = weights[chosen]
    return selected


def difficult_losses(raw, output, target, masks, fraction=.1, min_area=.0001):
    """Raw RGB gradients survive projection; final edge errors match GT.

    Edges with either endpoint selected are included, while both endpoints must
    still belong to the same semantic region. No gradient magnitude is rewarded.
    """
    target = target.detach()
    rgb = (raw.sum() + output.sum()) * 0
    gradient = rgb
    rgb_count = gradient_count = 0
    error = (raw - target).square().mean(1, keepdim=True)
    for mask in masks:
        selected = difficult_weights(raw, target, mask, fraction, min_area)
        if not (selected.sum((1, 2, 3)) > 0).any():
            continue
        rgb = rgb + regional_mean(error, selected)
        rgb_count += 1
        for axis in (-1, -2):
            if output.size(axis) < 2:
                continue
            before, after = [slice(None)] * 4, [slice(None)] * 4
            before[axis], after[axis] = slice(None, -1), slice(1, None)
            region_edge = torch.minimum(mask[tuple(before)], mask[tuple(after)])
            edge = torch.maximum(selected[tuple(before)], selected[tuple(after)]) * region_edge
            if not (edge.sum((1, 2, 3)) > 0).any():
                continue
            difference = (torch.diff(output, dim=axis) - torch.diff(target, dim=axis)).square().mean(1, keepdim=True)
            gradient = gradient + regional_mean(difference, edge)
            gradient_count += 1
    return rgb / max(rgb_count, 1), gradient / max(gradient_count, 1)
