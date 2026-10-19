"""Post-LUT spatial tone in Display P3 linear luminance.

The transfer curve and P3/D65 Y coefficients match color.display_p3_to_lab.
Extended signed RGB values from the original LUT are kept differentiable.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import PredictionHead, crop_map
from .semantic_modules import masked_per_image


def decode_p3(rgb):
    magnitude = rgb.abs()
    return torch.where(magnitude <= .04045, rgb / 12.92,
                       rgb.sign() * ((magnitude + .055) / 1.055).pow(2.4))


def encode_p3(linear):
    # A safe inactive power branch is necessary for finite gradients at black.
    magnitude = linear.abs()
    return torch.where(magnitude <= .0031308, linear * 12.92,
                       linear.sign() * (1.055 * magnitude.clamp_min(.0031308).pow(1 / 2.4) - .055))


def p3_luminance(linear):
    weights = linear.new_tensor([35783 / 156275, 247089 / 357200, 198249 / 2500400])
    return (linear * weights.view(1, 3, 1, 1)).sum(1, keepdim=True)


def region_lightness_loss(difference, input_lightness, mask, min_area,
                          shadow_weight=2., shadow_threshold=.5):
    """Bounded input-only shadow weights; validity uses the unweighted mask."""
    weight = 1 + shadow_weight * (1 - input_lightness.detach() / shadow_threshold).clamp(0, 1)
    error = (difference.square() + 1e-6).sqrt() - .001
    mass = (mask * weight).sum((1, 2, 3))
    valid = (mask.mean((1, 2, 3)) >= min_area).to(error)
    value = (mask * weight * error).sum((1, 2, 3)) / mass.clamp_min(1e-8)
    return (value * valid).sum() / valid.sum().clamp_min(1)


class SpatialLuminanceTone(nn.Module):
    """Independent regional curves/FiLM heads on shared RGB+Y+mask features."""
    def __init__(self, condition_dim, hidden_dim=128, curves=3, segments=16,
                 resolution=128, channels=16, radius=2, epsilon=.001,
                 max_gain=4., max_slope=4.):
        super().__init__()
        if curves < 1 or segments < 2 or resolution < 16 or channels < 4 or radius < 1:
            raise ValueError('invalid spatial tone dimensions')
        if (not all(math.isfinite(v) for v in (epsilon, max_gain, max_slope)) or
                epsilon <= 0 or max_gain < 1 or not 1 < max_slope <= segments):
            raise ValueError('invalid spatial tone numerical limits')
        self.curves, self.segments, self.resolution = int(curves), int(segments), int(resolution)
        self.radius, self.epsilon = int(radius), float(epsilon)
        self.max_gain, self.max_slope = float(max_gain), float(max_slope)
        # Mixing uniform increments and a simplex bounds every slope by max_slope.
        self.curve_strength = (max_slope - 1) / (segments - 1)
        self.curve_heads = nn.ModuleList([
            PredictionHead(condition_dim, hidden_dim, curves * segments) for _ in range(2)])
        self.shared = nn.Sequential(
            nn.Conv2d(7, channels, 3, 2, 1), nn.LeakyReLU(.2),
            nn.Conv2d(channels, channels * 2, 3, 1, 1), nn.LeakyReLU(.2),
            nn.Conv2d(channels * 2, channels * 2, 3, 1, 1), nn.LeakyReLU(.2))
        self.films = nn.ModuleList([nn.Linear(condition_dim, channels * 4) for _ in range(2)])
        self.spatial_heads = nn.ModuleList([nn.Conv2d(channels * 2, curves, 1) for _ in range(2)])
        for head in self.curve_heads:
            nn.init.zeros_(head.output.weight)
            nn.init.zeros_(head.output.bias)
        for film in self.films:
            nn.init.zeros_(film.weight)
            nn.init.zeros_(film.bias)
        # Exact identity curves, but unequal spatial weights break curve symmetry.
        for head in self.spatial_heads:
            nn.init.normal_(head.weight, std=.001)
            nn.init.normal_(head.bias, std=.001)

    def box_mean(self, value):
        return F.avg_pool2d(value, 2 * self.radius + 1, stride=1,
                            padding=self.radius, count_include_pad=False)

    def predict(self, source, mask, conditions):
        small = F.interpolate(source, (self.resolution,) * 2, mode='bilinear', align_corners=False)
        small_mask = F.interpolate(mask, small.shape[-2:], mode='area')
        light = p3_luminance(decode_p3(small))
        features = self.shared(torch.cat((small, light,
            F.avg_pool2d(light, 3, 1, 1, count_include_pad=False),
            F.avg_pool2d(light, 7, 1, 3, count_include_pad=False), small_mask), 1))
        reference = F.interpolate(small, features.shape[-2:], mode='bilinear', align_corners=False)
        guidance = p3_luminance(decode_p3(reference))
        mean_g = self.box_mean(guidance)
        variance = (self.box_mean(guidance.square()) - mean_g.square()).clamp_min(0)
        fields = []
        for condition, curve_head, film, spatial_head in zip(
                conditions, self.curve_heads, self.films, self.spatial_heads):
            raw = curve_head(condition).view(-1, self.curves, self.segments)
            increments = ((1 - self.curve_strength) / self.segments +
                          self.curve_strength * raw.softmax(-1))
            knots = F.pad(increments.cumsum(-1), (1, 0))
            gamma, beta = film(condition).chunk(2, 1)
            local = F.leaky_relu((1 + gamma[:, :, None, None]) * features +
                                 beta[:, :, None, None], negative_slope=.2)
            logits = spatial_head(local)
            mean_p = self.box_mean(logits)
            covariance = self.box_mean(guidance * logits) - mean_g * mean_p
            slope = covariance / (variance + self.epsilon)
            offset = mean_p - slope * mean_g
            fields.append(dict(knots=knots, slopes=increments * self.segments,
                low_logits=logits, slope=self.box_mean(slope), offset=self.box_mean(offset)))
        return fields, reference

    def apply_tone(self, rgb, field, guidance_rgb, crop_box=None):
        """Filter/interpolate logits in full-context coordinates, then softmax."""
        size = rgb.shape[-2:]
        logits = (crop_map(field['slope'], crop_box, size) * p3_luminance(decode_p3(guidance_rgb)) +
                  crop_map(field['offset'], crop_box, size))
        alpha = logits.softmax(1)
        linear = decode_p3(rgb)
        light = p3_luminance(linear)
        coordinate = light.clamp(0, 1) * self.segments
        index = coordinate.long().clamp(max=self.segments - 1)
        fraction = coordinate - index.to(coordinate)
        target = torch.zeros_like(light)
        first_slope = torch.zeros_like(light)
        # Avoid storing K full-resolution curve outputs in addition to the weights.
        for k in range(self.curves):
            knots = field['knots'][:, k]
            lo = knots.gather(1, index.flatten(1)).view_as(light)
            hi = knots.gather(1, (index + 1).flatten(1)).view_as(light)
            target = target + alpha[:, k:k+1] * (lo + fraction * (hi - lo))
            first_slope = first_slope + alpha[:, k:k+1] * field['slopes'][:, k, :1, None, None]
        gain = torch.where(light > 1e-6, target / light.clamp_min(1e-6), first_slope)
        # Extend the identity outside the curve's physical [0,1] domain.
        gain = torch.where((light >= 0) & (light <= 1), gain, torch.ones_like(gain))
        gain = gain.clamp(max=self.max_gain)
        # Ray compression: reduce a COMMON gain instead of clipping RGB channels.
        # Existing out-of-gamut LUT values stay unchanged at identity, and cannot
        # be amplified further. The original final export still handles those.
        headroom = linear.max(1, keepdim=True)[0].clamp_min(1e-6).reciprocal().clamp_min(1.)
        gain = torch.minimum(gain, headroom)
        return encode_p3(linear * gain), gain

    def regularizers(self, fields, reference, mask, min_area, slope_limit=2., edge_scale=10.):
        low_mask = F.interpolate(mask, reference.shape[-2:], mode='area')
        curve_penalty, spatial_penalty = reference.new_zeros(()), reference.new_zeros(())
        for index, field in enumerate(fields):
            region = low_mask if index == 0 else 1 - low_mask
            valid = (region.mean((1, 2, 3)) >= min_area).to(reference)
            per_image = F.relu(field['slopes'] - slope_limit).square().mean((1, 2))
            curve_penalty = curve_penalty + (per_image * valid).sum() / valid.sum().clamp_min(1) / 2
            alpha = field['low_logits'].softmax(1)
            for axis in (-1, -2):
                difference = torch.diff(alpha, dim=axis).square().mean(1, keepdim=True)
                edge = torch.diff(reference, dim=axis).abs().mean(1, keepdim=True)
                lo, hi = [slice(None)] * 4, [slice(None)] * 4
                lo[axis], hi[axis] = slice(None, -1), slice(1, None)
                edge_mask = (region[tuple(lo)] + region[tuple(hi)]) / 2
                spatial_penalty = spatial_penalty + masked_per_image(
                    difference * torch.exp(-edge_scale * edge), edge_mask, min_area)[0] / 2
        return curve_penalty, spatial_penalty
