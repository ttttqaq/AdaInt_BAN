"""Foreground-only signed log gain on frozen, full-context local features."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import crop_map
from .semantic_modules import masked_per_image
from .spatial_tone import decode_p3, encode_p3, p3_luminance


class ForegroundSpatialGain(nn.Module):
    """A lightweight adapter and ONE gain head; no curves or background head.

    Reuse the frozen local decoder's features. The adapter also sees original
    and post-LUT/local luminance, their local means/contrast, the foreground
    mask and the existing foreground condition. Bound the log gain AFTER
    full-context guided upsampling so filtering cannot violate the bounds.
    """
    def __init__(self, feature_channels, condition_dim, channels=16,
                 max_gain=2., radius=2, epsilon=.001):
        super().__init__()
        if feature_channels < 1 or condition_dim < 1 or channels < 4 or radius < 1:
            raise ValueError('invalid foreground gain dimensions')
        if not math.isfinite(max_gain) or max_gain <= 1:
            raise ValueError('gain_max must be finite and greater than one')
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError('gain guided-filter epsilon must be positive and finite')
        self.max_gain, self.log_bound = float(max_gain), math.log(max_gain)
        self.radius, self.epsilon = int(radius), float(epsilon)
        self.adapter = nn.Sequential(
            nn.Conv2d(feature_channels + 10, channels, 3, padding=1), nn.LeakyReLU(.2),
            nn.Conv2d(channels, channels, 3, padding=1), nn.LeakyReLU(.2))
        self.film = nn.Linear(condition_dim, channels * 2)
        self.output = nn.Conv2d(channels, 1, 1)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def box_mean(self, value):
        return F.avg_pool2d(value, 2 * self.radius + 1, stride=1,
                            padding=self.radius, count_include_pad=False)

    def predict(self, features, source, base, mask, condition, active):
        if source.shape != base.shape or features.shape[-2:] != source.shape[-2:]:
            raise ValueError('gain features/source/base must use the same full-context grid')
        original_y = p3_luminance(decode_p3(source.clamp(0, 1)))
        base_y = p3_luminance(decode_p3(base.clamp(0, 1)))
        original_mean = F.avg_pool2d(original_y, 7, 1, 3, count_include_pad=False)
        base_mean = F.avg_pool2d(base_y, 7, 1, 3, count_include_pad=False)
        descriptor = torch.cat((original_y, base_y, base_y - original_y,
            F.avg_pool2d(original_y, 3, 1, 1, count_include_pad=False),
            F.avg_pool2d(base_y, 3, 1, 1, count_include_pad=False),
            original_mean, base_mean, original_y - original_mean,
            base_y - base_mean, mask), 1)
        adapted = self.adapter(torch.cat((features, descriptor), 1))
        gamma, beta = self.film(condition).chunk(2, 1)
        adapted = F.leaky_relu((1 + gamma[:, :, None, None]) * adapted +
                               beta[:, :, None, None], negative_slope=.2)
        raw = self.output(adapted)
        mean_g, mean_p = self.box_mean(original_y), self.box_mean(raw)
        variance = (self.box_mean(original_y.square()) - mean_g.square()).clamp_min(0)
        covariance = self.box_mean(original_y * raw) - mean_g * mean_p
        slope = covariance / (variance + self.epsilon)
        offset = mean_p - slope * mean_g
        return dict(slope=self.box_mean(slope), offset=self.box_mean(offset), active=active)

    def apply_gain(self, rgb, field, guidance_rgb, crop_box=None):
        size = rgb.shape[-2:]
        raw = (crop_map(field['slope'], crop_box, size) *
               p3_luminance(decode_p3(guidance_rgb)) + crop_map(field['offset'], crop_box, size))
        requested_log_gain = self.log_bound * raw.tanh() * field['active']
        linear = decode_p3(rgb)
        # Common-gain headroom avoids adding per-channel clipping/colour shifts.
        # Preserve existing out-of-gamut baseline values at identity, while
        # preventing further amplification. Dimming remains possible.
        headroom = linear.max(1, keepdim=True)[0].clamp_min(1e-6).reciprocal().clamp_min(1.)
        applied_log_gain = torch.minimum(requested_log_gain, headroom.log())
        gain = applied_log_gain.exp()
        # Residual form makes zero initialization EXACTLY preserve the baseline,
        # including signed/extended RGB values and transfer round-trip rounding.
        output = rgb + (encode_p3(linear * gain) - encode_p3(linear))
        return output, gain, applied_log_gain

    @staticmethod
    def regularizers(log_gain, mask, context_log_gain, context_source,
                     context_mask, min_area, edge_scale=10.):
        magnitude = masked_per_image(log_gain.square(), mask, min_area)[0]
        smoothness = log_gain.new_zeros(())
        # Regularize the ACTUALLY applied full-context gain on its fixed grid,
        # rather than raw logits or curve weights. This is crop independent.
        for axis in (-1, -2):
            difference = torch.diff(context_log_gain, dim=axis).square()
            edge = torch.diff(context_source, dim=axis).abs().mean(1, keepdim=True)
            lo, hi = [slice(None)] * 4, [slice(None)] * 4
            lo[axis], hi[axis] = slice(None, -1), slice(1, None)
            edge_mask = (context_mask[tuple(lo)] + context_mask[tuple(hi)]) / 2
            smoothness = smoothness + masked_per_image(
                difference * torch.exp(-edge_scale * edge), edge_mask, min_area)[0]
        return magnitude, smoothness
