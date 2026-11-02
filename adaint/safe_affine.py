"""Selective Display-P3 highlight shoulder and feasible RGB correction.

The legacy enhancer remains the proposal generator. All protection is applied
after local centering, semantic fusion and DC. Protected pixels bypass every
color conversion exactly; no learned parameters or teacher are introduced.
"""
import math

import torch
import torch.nn.functional as F


DEFAULTS = dict(enabled=False, knee=.92, raw_weight=.25,
                highlight_weight=1., gradient_weight=.1, top_fraction=.1)


def safe_options(value=None):
    result = dict(DEFAULTS)
    if value is not None:
        if not isinstance(value, dict) or set(value) - set(result):
            raise ValueError('unknown or non-mapping safe_affine options')
        result.update(value)
    if not isinstance(result['enabled'], bool):
        raise ValueError('safe_affine.enabled must be boolean')
    for key in ('knee', 'raw_weight', 'highlight_weight', 'gradient_weight', 'top_fraction'):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError('safe_affine.' + key + ' must be finite')
    if not .5 < result['knee'] < 1 or not 0 < result['top_fraction'] <= 1:
        raise ValueError('invalid shoulder knee or highlight fraction')
    if any(result[key] < 0 for key in ('raw_weight', 'highlight_weight', 'gradient_weight')):
        raise ValueError('safe_affine loss weights must be nonnegative')
    return result


def _working(value):
    if not torch.is_tensor(value) or not value.is_floating_point() or value.ndim != 4 or value.size(1) != 3:
        raise ValueError('expected floating B,3,H,W RGB')
    return value.float() if value.dtype in (torch.float16, torch.bfloat16) else value


def decode_p3(value):
    """Extended signed Display-P3 transfer, as in adaint.color."""
    magnitude = value.abs()
    return torch.where(magnitude <= .04045, value / 12.92,
                       value.sign() * ((magnitude + .055) / 1.055).pow(2.4))


def encode_p3(value):
    # Keep the inactive power branch differentiable at zero too.
    magnitude = value.abs()
    curved = 1.055 * magnitude.clamp_min(1e-12).pow(1. / 2.4) - .055
    return torch.where(magnitude <= .0031308, value * 12.92, value.sign() * curved)


def feasible_rgb(candidate, source):
    """Largest common RGB step from a feasible source towards candidate.

    In-range proposals are returned bit-for-bit. A conservative FP margin on
    active steps prevents cancellation crossing a cube face. Input-boundary
    outward steps correctly fall back to the original pixel (alpha=0).
    """
    proposal, anchor = _working(candidate), _working(source)
    if proposal.shape != anchor.shape:
        raise ValueError('candidate/source shapes differ')
    if not torch.isfinite(proposal).all() or not torch.isfinite(anchor).all():
        raise FloatingPointError('nonfinite RGB proposal/source')
    if ((anchor < 0) | (anchor > 1)).any():
        raise ValueError('source must be in [0,1]')
    delta = proposal - anchor
    positive, negative = delta > 0, delta < 0
    # Zero denominators use one only in inactive branches; no epsilon changes
    # the feasible interval for a genuinely small nonzero correction.
    upper = (1 - anchor) / torch.where(positive, delta, torch.ones_like(delta))
    lower = anchor / torch.where(negative, -delta, torch.ones_like(delta))
    bounds = torch.where(positive, upper, torch.where(negative, lower, torch.ones_like(delta)))
    alpha = bounds.amin(1, keepdim=True).clamp(0, 1)
    active = ((proposal < 0) | (proposal > 1)).any(1, keepdim=True)
    alpha = alpha * (1 - 8 * torch.finfo(proposal.dtype).eps)
    protected = anchor + alpha * delta
    return torch.where(active, protected, proposal), active


def protect_rgb(raw, source, knee=.92):
    """Smoothly compress dangerous highlights, preserving linear RGB ratios.

    The shoulder is identity up to knee. Above it, the maximum linear channel
    follows k+(1-k)*(m-k)/(1-k+m-k), a C1, increasing asymptote below one.
    All three linear channels share the same gain. Negative colors are handled
    by the final source-anchored constraint. Low, already legal pixels bypass
    the transfer functions exactly. The shoulder uses raw, unclipped RGB.
    """
    raw, source = _working(raw), _working(source)
    if not torch.isfinite(raw).all():
        raise FloatingPointError('nonfinite enhancer proposal')
    # An exploding model should fail explicitly rather than hiding overflow in
    # a range clamp. This limit is far outside the enhancer's useful RGB domain.
    if raw.abs().amax() > 1e6:
        raise FloatingPointError('RGB proposal magnitude exceeds numerical domain')
    linear = decode_p3(raw)
    peak = linear.amax(1, keepdim=True)
    k = decode_p3(raw.new_tensor(knee))
    excess = (peak - k).clamp_min(0)
    compressed = k + (1 - k) * excess / ((1 - k) + excess)
    gain = compressed / peak.clamp_min(1e-12)
    shoulder = (raw.amax(1, keepdim=True) > knee)
    candidate = torch.where(shoulder, encode_p3(linear * gain), raw)
    output, projected = feasible_rgb(candidate, source)
    risk = shoulder | ((raw < 0) | (raw > 1)).any(1, keepdim=True)
    return output, dict(risk=risk, shoulder=shoulder, projected=projected)


def regional_mean(error, weights):
    mass = weights.sum((1, 2, 3))
    valid = mass > 0
    per_image = (error * weights).sum((1, 2, 3)) / mass.clamp_min(1e-12)
    return (per_image * valid.to(error)).sum() / valid.sum().clamp_min(1).to(error)


def highlight_mask(target, mask, fraction=.1, min_area=.0001):
    """Detached GT-relative encoded-luma quantile, separately per region/image."""
    luma = (target.detach() * target.new_tensor([.2126, .7152, .0722])[None, :, None, None]).sum(1, keepdim=True)
    selected = torch.zeros_like(mask)
    for i in range(len(target)):
        weights = mask[i, 0].detach()
        if weights.mean() < min_area or weights.sum() <= 0:
            continue
        threshold = torch.quantile(luma[i, 0][weights > 0], 1 - fraction)
        selected[i, 0] = weights * (luma[i, 0] >= threshold).to(weights)
    return selected


def highlight_losses(output, target, masks, fraction=.1, min_area=.0001):
    """Symmetric RGB and RGB-gradient errors, normalized per valid region."""
    target = target.detach()
    rgb, gradient, valid_rgb, valid_gradient = [output.sum() * 0 for _ in range(4)]
    error = (output - target).square().mean(1, keepdim=True)
    for mask in masks:
        selected = highlight_mask(target, mask, fraction, min_area)
        if not (selected.sum((1, 2, 3)) > 0).any():
            continue
        rgb = rgb + regional_mean(error, selected)
        valid_rgb = valid_rgb + 1
        for axis in (-1, -2):
            if output.size(axis) < 2:
                continue
            before, after = [slice(None)] * 4, [slice(None)] * 4
            before[axis], after[axis] = slice(None, -1), slice(1, None)
            edge_mask = torch.minimum(selected[tuple(before)], selected[tuple(after)])
            if not (edge_mask.sum((1, 2, 3)) > 0).any():
                continue
            difference = (torch.diff(output, dim=axis) - torch.diff(target, dim=axis)).square().mean(1, keepdim=True)
            gradient = gradient + regional_mean(difference, edge_mask)
            valid_gradient = valid_gradient + 1
    return rgb / valid_rgb.clamp_min(1), gradient / valid_gradient.clamp_min(1)
