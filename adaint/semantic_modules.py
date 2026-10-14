"""Region-conditioned independent color mappings and shared local features."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import (ColorStatistics, ConditionEncoder, ToneCurve,
                      AdaptiveIntervals, NormalizedResidualLUT, LocalAffineDecoder,
                      luminance, stratified_pixels)


def checked_mask(mask, image, name='foreground_mask'):
    if mask is None:
        raise ValueError(name + ' is required for semantic enhancement')
    mask = mask.to(device=image.device, dtype=image.dtype)
    if mask.ndim != 4 or mask.shape[:2] != (image.size(0), 1):
        raise ValueError(name + ' must have shape B,1,H,W')
    if mask.shape[-2:] != image.shape[-2:]:
        raise ValueError(name + ' must align with its image')
    if not torch.isfinite(mask).all() or (mask < 0).any() or (mask > 1).any():
        raise ValueError(name + ' must contain finite weights in [0,1]')
    return mask


def masked_per_image(error, mask, min_area=0.0001):
    """Normalize each image first; empty/tiny images do not dilute the mean."""
    if error.ndim == 3:
        error = error[:, None]
    weights = mask.to(error)
    mass = weights.sum((-1, -2))
    valid = weights.mean((-1, -2)) >= min_area
    values = (error * weights).sum((-1, -2)) / mass.clamp_min(1e-8)
    values = values.mean(1)
    valid = valid[:, 0].to(error.dtype)
    return (values * valid).sum() / valid.sum().clamp_min(1), values, valid


class RegionColorStatistics(ColorStatistics):
    def forward(self, imgs, samples=None, mask=None):
        if mask is None:
            return super().forward(imgs, samples)
        small = (stratified_pixels(imgs, self.pool_size)
                 if samples is None else samples).clamp(0, 1)
        weights = checked_mask(mask, small, 'statistics mask')
        values = torch.cat((small, luminance(small)), 1)
        denominator = weights.sum((-1, -2)).clamp_min(1e-8)
        centers = self.hist_centers.to(values).view(1, 1, -1, 1, 1)
        hist = F.relu(1 - (values.unsqueeze(2) - centers).abs() * (self.bins - 1))
        hist = (hist * weights.unsqueeze(2)).sum((-1, -2)) / denominator[:, :, None]
        hist = hist / hist.sum(-1, keepdim=True).clamp_min(1e-8)
        mean = (values * weights).sum((-1, -2)) / denominator
        variance = ((values - mean[:, :, None, None]).square() * weights).sum((-1, -2)) / denominator
        # A zero-variance region must also have finite backward derivatives.
        std = variance.clamp_min(1e-12).sqrt()
        dark = ((values < .05).to(values.dtype) * weights).sum((-1, -2)) / denominator
        bright = ((values > .95).to(values.dtype) * weights).sum((-1, -2)) / denominator
        saturation = small.max(1, keepdim=True)[0] - small.min(1, keepdim=True)[0]
        sat_mean = (saturation * weights).sum((-1, -2)) / denominator
        sat_var = ((saturation - sat_mean[:, :, None, None]).square() * weights).sum((-1, -2)) / denominator
        return torch.cat((hist.flatten(1), mean, std, dark, bright,
                          sat_mean, sat_var.clamp_min(1e-12).sqrt()), 1)


class RegionMapping(nn.Module):
    def __init__(self, cnn_features, stats_features, condition_dim, hidden_dim,
                 n_vertices, n_ranks, curve_points, min_interval):
        super().__init__()
        self.condition_encoder = ConditionEncoder(cnn_features, stats_features * 2 + 2, condition_dim)
        self.tone_curve = ToneCurve(curve_points, condition_dim, hidden_dim)
        self.adaint = AdaptiveIntervals(n_vertices, condition_dim, hidden_dim, min_interval)
        self.lut_generator = NormalizedResidualLUT(n_vertices, n_ranks, condition_dim, hidden_dim)

    def forward(self, cnn_code, global_stats, region_stats, area, reliability):
        condition = self.condition_encoder(cnn_code, torch.cat(
            (global_stats, region_stats, area, reliability), 1))
        curves = self.tone_curve.predict_curves(condition)
        vertices = self.adaint(condition)
        tone_vertices = self.tone_curve.apply_curves(vertices, curves)
        weights, residual = self.lut_generator(condition)
        from .modules import identity_lut
        return condition, weights, vertices, identity_lut(tone_vertices) + residual


class DualLocalAffineDecoder(LocalAffineDecoder):
    """Extract spatial features once, then use independent FiLM/output heads."""
    def __init__(self, condition_dim, resolution=128, channels=16, radius=2, epsilon=.001):
        super().__init__(condition_dim, resolution, channels, radius, epsilon)
        del self.film
        del self.output
        self.films = nn.ModuleList([nn.Linear(condition_dim, channels * 4) for _ in range(2)])
        self.outputs = nn.ModuleList([nn.Conv2d(channels * 2, 4, 1, bias=False) for _ in range(2)])
        for module in list(self.films) + list(self.outputs):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, source, conditions):
        small = F.interpolate(source, (self.resolution,) * 2, mode='bilinear', align_corners=False)
        shallow = self.shallow(small)
        deep = F.interpolate(self.deep(shallow), shallow.shape[-2:], mode='bilinear', align_corners=False)
        shared = self.fusion(torch.cat((shallow, deep), 1))
        low_source = F.interpolate(source, shared.shape[-2:], mode='bilinear', align_corners=False)
        guidance = luminance(low_source)
        mean_g = self.box_mean(guidance)
        variance = self.box_mean(guidance.square()) - mean_g.square()
        results = []
        for condition, film, output in zip(conditions, self.films, self.outputs):
            gamma, beta = film(condition).chunk(2, 1)
            features = F.leaky_relu((1 + gamma[:, :, None, None]) * shared +
                                   beta[:, :, None, None], negative_slope=.2)
            low = output(features)
            mean_p = self.box_mean(low)
            covariance = self.box_mean(guidance * low) - mean_g * mean_p
            slope = covariance / (variance.clamp_min(0) + self.epsilon)
            offset = mean_p - slope * mean_g
            results.append((low, self.box_mean(slope), self.box_mean(offset)))
        return results, low_source
