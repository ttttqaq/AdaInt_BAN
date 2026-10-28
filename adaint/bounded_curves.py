"""Bounded RGB LUTs and spatial monotone curves, without output clipping.

The small numerical margin is separate from the near-identity initialization.
It leaves room for floating-point interpolation/fusion; it does not fix the
learned black/white endpoints at ``curve_epsilon`` and ``1-curve_epsilon``.
"""
import math
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F

from .highlight_guard import gt_highlight_mask
from .modules import crop_map, identity_lut, luminance
from .postprocess import ResultAwareLocalDecoder


NUMERICAL_MARGIN = 1.e-6


def _epsilon(value, name):
    if (isinstance(value, bool) or not isinstance(value, numbers.Real) or
            not math.isfinite(value) or not 0 < value < .5):
        raise ValueError(name + ' must be a finite number in (0, .5)')
    return float(value)


def _floating(value, name):
    if (not torch.is_tensor(value) or not value.is_floating_point() or
            not value.numel() or not torch.isfinite(value).all()):
        raise ValueError(name + ' must be a nonempty finite floating-point tensor')
    # Retain the numerical margin and stable normalization under mixed precision.
    return value.float() if value.dtype in (torch.float16, torch.bfloat16) else value


def _image(value, name, bounded=False):
    value = _floating(value, name)
    if value.ndim != 4 or value.size(1) != 3:
        raise ValueError(name + ' must have shape B,3,H,W')
    if bounded and ((value < 0).any() or (value > 1).any()):
        raise ValueError(name + ' must lie in [0,1] before curve evaluation')
    return value


def bounded_lut(raw_residual, vertices, epsilon=.001):
    """Map a signed residual around the current-node identity into bounded RGB.

    ``epsilon`` controls only the near-identity prior. The final affine interior
    margin is 1e-6, so even saturated sigmoid values leave interpolation headroom.
    No raw prediction is clipped; all finite residuals are valid latent values.
    """
    epsilon = _epsilon(epsilon, 'epsilon')
    residual = _floating(raw_residual, 'raw_residual')
    vertex = _floating(vertices, 'vertices').to(residual)
    if (residual.ndim != 5 or residual.size(1) != 3 or
            len(set(residual.shape[-3:])) != 1):
        raise ValueError('raw_residual must have shape B,3,D,D,D')
    if vertex.shape != (residual.size(0), 3, residual.size(-1)):
        raise ValueError('vertices must have shape B,3,D matching the LUT')
    if (vertex < 0).any() or (vertex > 1).any() or (torch.diff(vertex, dim=-1) <= 0).any():
        raise ValueError('vertices must increase strictly within [0,1]')
    prior = epsilon + (1. - 2. * epsilon) * identity_lut(vertex)
    logits = prior.log() - torch.log1p(-prior) + residual
    return NUMERICAL_MARGIN + (1. - 2. * NUMERICAL_MARGIN) * logits.sigmoid()


class BoundedCurveDecoder(ResultAwareLocalDecoder):
    """Reuse spatial features, replacing RGB affine deltas by monotone curves.

    Every RGB channel has ``segments+2`` latent masses: a black endpoint, positive
    segment increments, and white endpoint reserve. Guided restoration happens on
    logits, followed by softmax/cumulative normalization and bounded interpolation.
    """
    def __init__(self, condition_dim, resolution, channels, radius, epsilon,
                 result_condition=False, multiscale=False, guidance='luminance',
                 segments=4, curve_epsilon=.001):
        if isinstance(segments, bool) or not isinstance(segments, int) or segments < 1:
            raise ValueError('segments must be a positive integer')
        if guidance not in ('luminance', 'rgb'):
            raise ValueError("guidance must be 'luminance' or 'rgb'")
        curve_epsilon = _epsilon(curve_epsilon, 'curve_epsilon')
        super().__init__(condition_dim, resolution, channels, radius, epsilon,
                         result_condition, multiscale, guidance)
        self.segments, self.curve_epsilon = segments, curve_epsilon
        self.curve_channels = 3 * (segments + 2)
        # The replacement heads do not perturb the common feature RNG sequence.
        with torch.random.fork_rng(devices=[]):
            self.outputs = nn.ModuleList([
                nn.Conv2d(channels * 2, self.curve_channels, 1, bias=False)
                for _ in range(2)])
            if multiscale:
                self.refinement_outputs = nn.ModuleList([
                    nn.Conv2d(channels * 2, self.curve_channels, 1, bias=False)
                    for _ in range(2)])
        for output in list(self.outputs) + list(getattr(self, 'refinement_outputs', [])):
            nn.init.zeros_(output.weight)
        masses = [curve_epsilon] + [(1. - 2. * curve_epsilon) / segments] * segments
        masses.append(curve_epsilon)
        self.register_buffer('curve_base_logits', torch.tensor(masses).log().view(
            1, segments + 2, 1, 1))

    def filter_coefficients(self, low, source):
        """Guided filtering for arbitrary latent-channel counts."""
        low = _floating(low, 'curve coefficients')
        reference = F.interpolate(source.to(low), low.shape[-2:],
                                  mode='bilinear', align_corners=False)
        guide = luminance(reference) if self.guidance == 'luminance' else reference
        mean_g, mean_p = self.box_mean(guide), self.box_mean(low)
        if self.guidance == 'luminance':
            variance = self.box_mean(guide.square()) - mean_g.square()
            covariance = self.box_mean(guide * low) - mean_g * mean_p
            slope = covariance / (variance.clamp_min(0) + self.epsilon)
            return low, self.box_mean(slope), self.box_mean(mean_p - slope * mean_g)
        batch, count, height, width = low.shape
        gg = (guide[:, :, None] * guide[:, None, :]).flatten(1, 2)
        gp = (guide[:, :, None] * low[:, None, :]).flatten(1, 2)
        covariance = self.box_mean(gg).view(batch, 3, 3, height, width)
        covariance = covariance - mean_g[:, :, None] * mean_g[:, None, :]
        cross = self.box_mean(gp).view(batch, 3, count, height, width)
        cross = cross - mean_g[:, :, None] * mean_p[:, None, :]
        matrix = covariance.permute(0, 3, 4, 1, 2)
        identity = torch.eye(3, device=low.device, dtype=low.dtype)
        slopes = torch.linalg.solve(matrix + self.epsilon * identity,
                                    cross.permute(0, 3, 4, 1, 2))
        slopes = slopes.permute(0, 4, 3, 1, 2).contiguous()
        offset = mean_p - (slopes * mean_g[:, None]).sum(2)
        return low, self.box_mean(slopes.flatten(1, 2)), self.box_mean(offset)

    def upsample(self, slope, offset, target, crop_box=None):
        """Restore a latent subset, supporting scalar or RGB vector guidance."""
        count = offset.size(1)
        target = target.to(offset)
        result = crop_map(offset, crop_box, target.shape[-2:])
        if slope.size(1) == count:
            return result + crop_map(slope, crop_box, target.shape[-2:]) * luminance(target)
        if slope.size(1) != 3 * count:
            raise ValueError('guided slopes must have C or 3*C channels')
        slopes = slope.view(slope.size(0), count, 3, *slope.shape[-2:])
        for channel in range(3):
            result = result + crop_map(slopes[:, :, channel], crop_box,
                                      target.shape[-2:]) * target[:, channel:channel + 1]
        return result

    def curve_knots(self, logits):
        """Return monotone knots for one RGB channel's restored residual logits."""
        logits = _floating(logits, 'restored curve logits')
        if logits.ndim != 4 or logits.size(1) != self.segments + 2:
            raise ValueError('one-channel curve logits must have segments+2 channels')
        masses = (logits + self.curve_base_logits.to(logits)).softmax(1)
        cumulative = masses.cumsum(1)
        # Divide by the SAME cumulative sum used for the last node. This retains
        # ordering and [0,1] even when floating-point softmax masses do not sum to 1.
        knots = cumulative[:, :-1] / cumulative[:, -1:]
        return NUMERICAL_MARGIN + (1. - 2. * NUMERICAL_MARGIN) * knots

    def apply_fields(self, fields, guide, global_output, crop_box=None):
        """Restore and apply one branch, one RGB curve at a time to limit memory."""
        guide = _image(guide, 'guide', bounded=True)
        global_output = _image(global_output, 'global_output', bounded=True)
        if guide.shape != global_output.shape:
            raise ValueError('guide and global_output must share B,3,H,W coordinates')
        if not fields:
            raise ValueError('at least one coefficient field is required')
        count, outputs = self.segments + 2, []
        for channel in range(3):
            restored = None
            start, end = channel * count, (channel + 1) * count
            for low, slope, offset in fields:
                if low.size(1) != self.curve_channels or offset.size(1) != self.curve_channels:
                    raise ValueError('coefficient fields have an incompatible curve channel count')
                if slope.size(1) == self.curve_channels:
                    selected_slope = slope[:, start:end]
                elif slope.size(1) == 3 * self.curve_channels:
                    selected_slope = slope.view(slope.size(0), self.curve_channels, 3,
                                                *slope.shape[-2:])[:, start:end].flatten(1, 2)
                else:
                    raise ValueError('coefficient fields have an incompatible slope channel count')
                value = self.upsample(selected_slope, offset[:, start:end], guide, crop_box)
                restored = value if restored is None else restored + value
            knots = self.curve_knots(restored)
            position = global_output[:, channel:channel + 1].to(knots) * self.segments
            lower = position.floor().long().clamp(max=self.segments - 1)
            fraction = position - lower.to(position)
            left, right = knots.gather(1, lower), knots.gather(1, lower + 1)
            outputs.append(left + (right - left) * fraction)
        return torch.cat(outputs, 1)


def _weighted_mean(error, weights):
    mass = weights.sum((1, 2, 3))
    valid = mass > 0
    per_image = (error * weights).sum((1, 2, 3)) / mass.clamp_min(1e-12)
    value = (per_image * valid.to(error)).sum() / valid.sum().clamp_min(1).to(error)
    return value, valid.any().to(error)


def symmetric_highlight_losses(output, gt, masks, top_fraction=.1, min_area=.0001):
    """Unweighted GT-bright RGB MSE and two-ended bright-edge gradient MSE.

    Positive and negative RGB errors receive equal cost. Select each region's
    highlights from GT only, normalize images first, and average valid regions.
    Edges contribute only when BOTH endpoints belong to that selected region.
    """
    output, target = _image(output, 'output'), _image(gt, 'gt', bounded=True).detach()
    target = target.to(output)
    if output.shape != target.shape:
        raise ValueError('output and gt must share B,3,H,W coordinates')
    if torch.is_tensor(masks):
        masks = (masks,)
    if not isinstance(masks, (tuple, list)) or not masks:
        raise ValueError('masks must be a nonempty sequence of regional masks')
    zero = output.sum() * 0.
    rgb_total, rgb_count, gradient_total, gradient_count = zero, zero, zero, zero
    rgb_error = (output - target).square().mean(1, keepdim=True)
    for mask in masks:
        selected = gt_highlight_mask(target, mask, top_fraction, min_area).to(output)
        value, valid = _weighted_mean(rgb_error, selected)
        rgb_total, rgb_count = rgb_total + value, rgb_count + valid
        for axis in (-1, -2):
            if output.size(axis) < 2:
                continue
            before, after = [slice(None)] * 4, [slice(None)] * 4
            before[axis], after[axis] = slice(None, -1), slice(1, None)
            edge_mask = torch.minimum(selected[tuple(before)], selected[tuple(after)])
            error = (torch.diff(output, dim=axis) - torch.diff(target, dim=axis)).square().mean(1, keepdim=True)
            value, valid = _weighted_mean(error, edge_mask)
            gradient_total, gradient_count = gradient_total + value, gradient_count + valid
    return rgb_total / rgb_count.clamp_min(1.), gradient_total / gradient_count.clamp_min(1.)
