"""Optional result-conditioned postprocessing; the legacy path remains available."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import crop_map, luminance
from .semantic_modules import DualLocalAffineDecoder


DEFAULTS = dict(result_condition=False, global_dc=False, regional_dc=False,
                global_dc_limit=.08, regional_dc_limit=.04,
                calibration_hidden=64, multiscale=False,
                guidance='luminance', freeze_global=False, fixed_rgb_bias=None)


def postprocess_options(value):
    options = dict(DEFAULTS)
    if value is not None:
        if not isinstance(value, dict):
            raise ValueError('model.postprocess must be a mapping')
        unknown = set(value) - set(options)
        if unknown:
            raise ValueError('unknown postprocess options: ' + ', '.join(sorted(unknown)))
        options.update(value)
    for name in ('result_condition', 'global_dc', 'regional_dc', 'multiscale', 'freeze_global'):
        if not isinstance(options[name], bool):
            raise ValueError('postprocess.' + name + ' must be boolean')
    for name in ('global_dc_limit', 'regional_dc_limit'):
        if not math.isfinite(options[name]) or options[name] <= 0:
            raise ValueError('postprocess.' + name + ' must be finite and positive')
    hidden = options['calibration_hidden']
    if isinstance(hidden, bool) or not isinstance(hidden, int) or hidden < 4:
        raise ValueError('calibration_hidden must be an integer >= 4')
    if options['guidance'] not in ('luminance', 'rgb'):
        raise ValueError("postprocess.guidance must be 'luminance' or 'rgb'")
    bias = options['fixed_rgb_bias']
    if bias is not None:
        if len(bias) != 3 or not all(math.isfinite(x) for x in bias):
            raise ValueError('fixed_rgb_bias must contain three finite RGB values')
        if options['global_dc']:
            raise ValueError('fixed_rgb_bias and global_dc are alternative ablations')
    return options


class ResultAwareLocalDecoder(DualLocalAffineDecoder):
    """Preserve the old coefficient path and optionally add result/scale features.

    With luminance guidance, zero adapters/refinement heads preserve a migrated
    checkpoint's output. RGB guidance deliberately changes coefficient filtering.
    """
    def __init__(self, condition_dim, resolution, channels, radius, epsilon,
                 result_condition=False, multiscale=False, guidance='luminance'):
        super().__init__(condition_dim, resolution, channels, radius, epsilon)
        self.guidance = guidance
        self.result_condition = result_condition
        self.multiscale = multiscale
        if result_condition:
            # Optional modules must not shift common ablation initializations.
            with torch.random.fork_rng(devices=[]):
                self.result_adapter = nn.Conv2d(7, channels, 3, 2, 1, bias=False)
                nn.init.zeros_(self.result_adapter.weight)
        if multiscale:
            with torch.random.fork_rng(devices=[]):
                self.pyramid_fine = nn.Sequential(nn.Conv2d(3, channels * 2, 3, 1, 1),
                                                  nn.LeakyReLU(.2))
                self.pyramid_coarse = nn.Sequential(
                    nn.Conv2d(channels * 2, channels * 2, 3, 2, 1), nn.LeakyReLU(.2))
                self.refinement_outputs = nn.ModuleList([
                    nn.Conv2d(channels * 2, 4, 1, bias=False) for _ in range(2)])
                for output in self.refinement_outputs:
                    nn.init.zeros_(output.weight)

    def filter_coefficients(self, low, source):
        reference = F.interpolate(source, low.shape[-2:], mode='bilinear', align_corners=False)
        guide = luminance(reference) if self.guidance == 'luminance' else reference
        mean_g, mean_p = self.box_mean(guide), self.box_mean(low)
        if self.guidance == 'luminance':
            variance = self.box_mean(guide.square()) - mean_g.square()
            covariance = self.box_mean(guide * low) - mean_g * mean_p
            slope = covariance / (variance.clamp_min(0) + self.epsilon)
            offset = mean_p - slope * mean_g
            return low, self.box_mean(slope), self.box_mean(offset)
        # Vector guided filtering: solve a regularized 3x3 RGB covariance system.
        batch, _, height, width = guide.shape
        gg = (guide[:, :, None] * guide[:, None, :]).flatten(1, 2)
        gp = (guide[:, :, None] * low[:, None, :]).flatten(1, 2)
        covariance = self.box_mean(gg).view(batch, 3, 3, height, width)
        covariance = covariance - mean_g[:, :, None] * mean_g[:, None, :]
        cross = self.box_mean(gp).view(batch, 3, 4, height, width)
        cross = cross - mean_g[:, :, None] * mean_p[:, None, :]
        matrix = covariance.permute(0, 3, 4, 1, 2)
        identity = torch.eye(3, device=low.device, dtype=low.dtype)
        slopes = torch.linalg.solve(matrix + self.epsilon * identity,
                                    cross.permute(0, 3, 4, 1, 2))
        slopes = slopes.permute(0, 4, 3, 1, 2).contiguous()  # B,4,3,H,W
        offset = mean_p - (slopes * mean_g[:, None]).sum(2)
        slopes = self.box_mean(slopes.flatten(1, 2))
        return low, slopes, self.box_mean(offset)

    def upsample(self, slope, offset, target, crop_box=None):
        if slope.size(1) == 4:
            return super().upsample(slope, offset, target, crop_box)
        result = crop_map(offset, crop_box, target.shape[-2:])
        slopes = slope.view(slope.size(0), 4, 3, *slope.shape[-2:])
        # Avoid materializing twelve full-resolution slope maps simultaneously.
        for channel in range(3):
            result = result + crop_map(slopes[:, :, channel], crop_box,
                                      target.shape[-2:]) * target[:, channel:channel + 1]
        return result

    def forward(self, source, conditions, global_context=None, foreground_context=None):
        small = F.interpolate(source, (self.resolution,) * 2, mode='bilinear', align_corners=False)
        shallow_logits = self.shallow[0](small)
        if self.result_condition:
            if global_context is None or foreground_context is None:
                raise ValueError('result-conditioned decoder requires full global/mask context')
            extra = torch.cat((global_context[0], global_context[1], foreground_context), 1)
            extra = F.interpolate(extra, small.shape[-2:], mode='bilinear', align_corners=False)
            shallow_logits = shallow_logits + self.result_adapter(extra)
        shallow = self.shallow[1](shallow_logits)
        deep = self.deep(shallow)
        shared = self.fusion(torch.cat((shallow, F.interpolate(
            deep, shallow.shape[-2:], mode='bilinear', align_corners=False)), 1))
        if self.multiscale:
            pyramid = self.pyramid_fine(small) + F.interpolate(
                self.pyramid_coarse(deep), small.shape[-2:], mode='bilinear', align_corners=False)
        fields = []
        for index, (condition, film, output) in enumerate(zip(conditions, self.films, self.outputs)):
            gamma, beta = film(condition).chunk(2, 1)
            features = F.leaky_relu((1 + gamma[:, :, None, None]) * shared +
                                   beta[:, :, None, None], negative_slope=.2)
            branch = [self.filter_coefficients(output(features), source)]
            if self.multiscale:
                fine = F.leaky_relu(pyramid + F.interpolate(features, small.shape[-2:],
                                    mode='bilinear', align_corners=False), negative_slope=.2)
                branch.append(self.filter_coefficients(self.refinement_outputs[index](fine), source))
            fields.append(branch)
        reference = F.interpolate(source, shared.shape[-2:], mode='bilinear', align_corners=False)
        return fields, reference


class ResidualCalibration(nn.Module):
    """Small bounded signed RGB DC heads; every new output starts at zero."""
    def __init__(self, global_dc, regional_dc, hidden, global_limit, regional_limit):
        super().__init__()
        self.global_limit, self.regional_limit = global_limit, regional_limit
        self.encoder = nn.Sequential(nn.Conv2d(10, 16, 3, 2, 1), nn.LeakyReLU(.2),
                                     nn.Conv2d(16, 32, 3, 2, 1), nn.LeakyReLU(.2))
        if global_dc:
            with torch.random.fork_rng(devices=[]):
                self.global_head = self.head(32 * 3 + 7, hidden)
        if regional_dc:
            with torch.random.fork_rng(devices=[]):
                self.regional_head = self.head(32 * 3 + 7, hidden)

    @staticmethod
    def head(features, hidden):
        head = nn.Sequential(nn.Linear(features, hidden), nn.LeakyReLU(.2), nn.Linear(hidden, 3))
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
        return head

    def forward(self, source, global_context, mask, min_area):
        small = F.interpolate(torch.cat((source, global_context[0], global_context[1], mask), 1),
                              (64, 64), mode='bilinear', align_corners=False)
        features = self.encoder(small)
        weights = F.interpolate(mask, features.shape[-2:], mode='area')
        pool = lambda m: (features * m).sum((-1, -2)) / m.sum((-1, -2)).clamp_min(1e-8)
        pi = mask.mean((-1, -2))
        mixed = mask * global_context[0] + (1 - mask) * global_context[1]
        statistics = torch.cat((features.mean((-1, -2)), pool(weights), pool(1 - weights),
                                source.mean((-1, -2)), mixed.mean((-1, -2)), pi), 1)
        zero = source.new_zeros((source.size(0), 3, 1, 1))
        d0 = (self.global_limit * self.global_head(statistics).tanh()[:, :, None, None]
              if hasattr(self, 'global_head') else zero)
        valid = ((pi >= min_area) & (pi <= 1 - min_area)).to(source)[:, :, None, None]
        d1 = (self.regional_limit * self.regional_head(statistics).tanh()[:, :, None, None] * valid
              if hasattr(self, 'regional_head') else zero)
        return d0, d1, pi[:, :, None, None]
