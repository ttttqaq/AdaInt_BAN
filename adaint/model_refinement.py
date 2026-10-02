"""Fixed LUT architectures and independently controlled training methods.

RefinementBase has the V2 architecture. ArchitectureV3 always adds post-tone
conditioning and a multiscale, guided spatial decoder. GlobalLocalV3 is a
separate architecture with one global and two independently gated local LUTs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16

from mmedit.models.registry import MODELS
from .model_architecture_v2 import (
    AiLUTArchitectureV2, ColorStatistics, ConditionEncoder)


def normalized_lab(rgb):
    """Differentiable sRGB/D65 Lab; also provides gradients outside [0, 1]."""
    rgb = rgb.float()
    linear = torch.where(
        rgb <= 0.04045, rgb / 12.92,
        ((rgb.clamp_min(0.04045) + 0.055) / 1.055).pow(2.4))
    matrix = rgb.new_tensor([[0.4124564, 0.3575761, 0.1804375],
                             [0.2126729, 0.7151522, 0.0721750],
                             [0.0193339, 0.1191920, 0.9503041]])
    xyz = torch.einsum('oc,bchw->bohw', matrix, linear)
    xyz = xyz / rgb.new_tensor([0.95047, 1.0, 1.08883]).view(1, 3, 1, 1)
    delta = 6.0 / 29.0
    transformed = torch.where(
        xyz > delta ** 3, xyz.clamp_min(delta ** 3).pow(1.0 / 3.0),
        xyz / (3 * delta ** 2) + 4.0 / 29.0)
    fx, fy, fz = transformed.unbind(1)
    return torch.stack(((116 * fy - 16) / 100,
                        500 * (fx - fy) / 128,
                        200 * (fy - fz) / 128), dim=1)


def stratified_pixels(imgs, size):
    """One deterministic centre sample per stratum, without averaging pixels."""
    height, width = imgs.shape[-2:]
    ys = ((torch.arange(size, device=imgs.device).float() + 0.5)
          * height / size).long().clamp(max=height - 1)
    xs = ((torch.arange(size, device=imgs.device).float() + 0.5)
          * width / size).long().clamp(max=width - 1)
    return imgs.index_select(-2, ys).index_select(-1, xs)


class RefinementStatistics(ColorStatistics):
    def __init__(self, bins, pool_size, use_stratified=False):
        super().__init__(bins, pool_size)
        self.use_stratified = bool(use_stratified)

    def forward(self, imgs, samples=None):
        if self.use_stratified:
            imgs = samples if samples is not None else stratified_pixels(
                imgs, self.pool_size)
        return super().forward(imgs)


def crop_map(values, crop_box, size):
    """Crop normalized pixel-edge bounds using align_corners=False centres."""
    if crop_box is None:
        return F.interpolate(values, size=size, mode='bilinear',
                             align_corners=False)
    crop_box = crop_box.to(device=values.device, dtype=values.dtype).view(-1, 4)
    height, width = size
    ys = (torch.arange(height, device=values.device, dtype=values.dtype) + 0.5) / height
    xs = (torch.arange(width, device=values.device, dtype=values.dtype) + 0.5) / width
    yy = crop_box[:, 1:2] + ys[None] * (crop_box[:, 3:4] - crop_box[:, 1:2])
    xx = crop_box[:, 0:1] + xs[None] * (crop_box[:, 2:3] - crop_box[:, 0:1])
    grid = torch.stack((xx[:, None, :].expand(-1, height, -1),
                        yy[:, :, None].expand(-1, -1, width)), dim=-1)
    return F.grid_sample(values, grid * 2 - 1, mode='bilinear',
                         padding_mode='border', align_corners=False)


def luminance(imgs):
    return (imgs * imgs.new_tensor([0.2126, 0.7152, 0.0722])
            .view(1, 3, 1, 1)).sum(1, keepdim=True)


class GuidedSpatialDecoder(nn.Module):
    """Always fuses shallow/deep features and performs guided logit upsampling."""
    def __init__(self, condition_dim, outputs, resolution=128, channels=16,
                 temperature=1.0, radius=2, epsilon=0.001):
        super().__init__()
        if resolution < 32 or channels < 4 or radius < 1 or epsilon <= 0:
            raise ValueError('invalid spatial decoder sizes or guided-filter settings')
        self.resolution = int(resolution)
        self.temperature = max(float(temperature), 1e-4)
        self.radius = int(radius)
        self.epsilon = float(epsilon)
        self.shallow = nn.Sequential(
            nn.Conv2d(3, channels, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True))
        self.deep = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels * 2, channels * 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels * 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))
        self.condition_projection = nn.Linear(condition_dim, channels * 2)
        self.output = nn.Conv2d(channels * 2, outputs, 1)
        nn.init.normal_(self.output.weight, std=1e-3)
        nn.init.zeros_(self.output.bias)

    def box_mean(self, x):
        return F.avg_pool2d(x, 2 * self.radius + 1, stride=1,
                            padding=self.radius, count_include_pad=False)

    def forward_logits(self, source, condition, target, crop_box=None):
        small = F.interpolate(source, size=(self.resolution,) * 2,
                              mode='bilinear', align_corners=False)
        shallow = self.shallow(small)
        deep = F.interpolate(self.deep(shallow), size=shallow.shape[-2:],
                             mode='bilinear', align_corners=False)
        fused = self.fusion(torch.cat((shallow, deep), dim=1))
        fused = fused + self.condition_projection(condition).view(
            condition.shape[0], -1, 1, 1)
        low_logits = crop_map(self.output(fused), crop_box, shallow.shape[-2:])
        # Filtering the logits preserves positivity/unit sum after softmax.
        high_guidance = luminance(F.avg_pool2d(
            target, 3, stride=1, padding=1, count_include_pad=False))
        low_guidance = F.interpolate(high_guidance, size=low_logits.shape[-2:],
                                     mode='bilinear', align_corners=False)
        mean_g, mean_p = self.box_mean(low_guidance), self.box_mean(low_logits)
        variance = self.box_mean(low_guidance.square()) - mean_g.square()
        covariance = self.box_mean(low_guidance * low_logits) - mean_g * mean_p
        a = covariance / (variance.clamp_min(0) + self.epsilon)
        b = mean_p - a * mean_g
        a, b = self.box_mean(a), self.box_mean(b)
        high_logits = (F.interpolate(a, size=target.shape[-2:], mode='bilinear',
                                     align_corners=False) * high_guidance +
                       F.interpolate(b, size=target.shape[-2:], mode='bilinear',
                                     align_corners=False))
        return high_logits / self.temperature, low_logits / self.temperature

    def forward(self, source, condition, target=None, crop_box=None):
        target = source if target is None else target
        high, low = self.forward_logits(source, condition, target, crop_box)
        return high.softmax(1), low.softmax(1)


@MODELS.register_module()
class AiLUTRefinementBase(AiLUTArchitectureV2):
    """V2 architecture with isolated losses, statistics and training controls."""
    def __init__(self, use_stratified_statistics=False,
                 use_kd_decay=False, kd_decay_start_ratio=0.3,
                 kd_decay_end_ratio=1.0, kd_final_weight=0.1,
                 use_confidence_kd=False, confidence_temperature=0.05,
                 use_lab_loss=False, lab_loss_weight=0.05,
                 use_rgb_mse_loss=False, rgb_mse_loss_weight=0.1,
                 use_low_frequency_loss=False, low_frequency_loss_weight=0.1,
                 low_frequency_size=32, use_edge_aware_smoothness=False,
                 edge_smoothness_scale=10.0, use_full_image_context=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.color_statistics = RefinementStatistics(
            kwargs.get('stats_bins', 16), kwargs.get('stats_pool_size', 64),
            use_stratified_statistics)
        self.use_kd_decay = bool(use_kd_decay)
        self.kd_decay_start_ratio = float(kd_decay_start_ratio)
        self.kd_decay_end_ratio = float(kd_decay_end_ratio)
        self.kd_final_weight = float(kd_final_weight)
        if not 0 <= self.kd_decay_start_ratio < self.kd_decay_end_ratio <= 1:
            raise ValueError('KD decay ratios must satisfy 0 <= start < end <= 1')
        if not 0 <= self.kd_final_weight <= 1 or confidence_temperature <= 0:
            raise ValueError('invalid KD final weight or confidence temperature')
        self.use_confidence_kd = bool(use_confidence_kd)
        self.confidence_temperature = float(confidence_temperature)
        self.use_lab_loss = bool(use_lab_loss)
        self.lab_loss_weight = float(lab_loss_weight)
        self.use_rgb_mse_loss = bool(use_rgb_mse_loss)
        self.rgb_mse_loss_weight = float(rgb_mse_loss_weight)
        self.use_low_frequency_loss = bool(use_low_frequency_loss)
        self.low_frequency_loss_weight = float(low_frequency_loss_weight)
        self.low_frequency_size = int(low_frequency_size)
        self.use_edge_aware_smoothness = bool(use_edge_aware_smoothness)
        self.edge_smoothness_scale = float(edge_smoothness_scale)
        self.use_full_image_context = bool(use_full_image_context)
        if min(self.lab_loss_weight, self.rgb_mse_loss_weight,
               self.low_frequency_loss_weight, self.edge_smoothness_scale) < 0:
            raise ValueError('loss weights must be non-negative')
        if self.low_frequency_size < 1:
            raise ValueError('low_frequency_size must be positive')

    def kd_multiplier(self):
        if not self.use_kd_decay:
            return 1.0
        total = (self.train_cfg or {}).get('total_iters', 0)
        if total <= 0:
            raise ValueError('KD decay requires resolved train_cfg.total_iters')
        progress = float(self.cnt_iters.item()) / max(total - 1, 1)
        fraction = max(0.0, min(1.0,
            (progress - self.kd_decay_start_ratio) /
            (self.kd_decay_end_ratio - self.kd_decay_start_ratio)))
        return 1.0 + fraction * (self.kd_final_weight - 1.0)

    def _conditions(self, source, samples=None):
        statistics = self.color_statistics(source, samples)
        condition = self.condition_encoder(self.backbone(source), statistics)
        curves = self.tone_curve.predict_curves(condition)
        return condition, curves, statistics

    def _guides(self, source, condition, imgs, crop_box):
        guide, low = self.spatial_guide(source, condition)
        if crop_box is not None:
            guide = crop_map(guide, crop_box, imgs.shape[-2:])
            low = crop_map(low, crop_box, low.shape[-2:])
        return guide, low

    def forward_dummy(self, imgs, return_aux=False, lq_context=None,
                      context_samples=None, crop_box=None):
        full_context = self.use_full_image_context and lq_context is not None
        source = lq_context if full_context else imgs
        samples = context_samples if full_context else None
        box = crop_box if full_context else None
        condition, curves, statistics = self._conditions(source, samples)
        tone_output = self.tone_curve.apply_curves(imgs, curves)
        vertices = (self.adaint(condition) if self.en_adaint else
                    self.uniform_vertices.expand(imgs.shape[0], -1, -1))
        weights, residual_luts = self.lut_generator(condition, vertices)
        guide, low = self._guides(source, condition, imgs, box)
        residual = self._apply_spatial_luts(
            tone_output.clamp(0, 1), residual_luts, vertices, guide)
        output = tone_output + residual
        if return_aux:
            return output, weights, vertices, dict(
                residual_luts=residual_luts, guide=guide,
                low_resolution_guide=low, curves=curves,
                statistics=statistics, tone_output=tone_output)
        return output, weights, vertices

    @auto_fp16(apply_to=('lq', 'lq_context', 'context_samples'))
    def forward(self, lq, gt=None, test_mode=False, **kwargs):
        if test_mode:
            return self.forward_test(lq, gt, **kwargs)
        return self.forward_train(lq, gt, **kwargs)

    def _regularizations(self, vertices, residual_luts):
        # V2's all-direction loss averages over 3 output channels, while its
        # diagonal variant sums channel-specific means. Match surviving terms.
        mono_factor = self.monotonicity_factor
        if self.use_diagonal_monotonicity:
            mono_factor /= self.n_colors
        return self.lut_generator.regularizations(
            vertices, residual_luts, self.smooth_factor, mono_factor,
            diagonal_monotonicity=self.use_diagonal_monotonicity)

    def forward_train(self, lq, gt, lq_context=None, context_samples=None,
                      crop_box=None, **kwargs):
        if self.use_full_image_context and (lq_context is None or crop_box is None):
            raise ValueError('full-image context requires ContextAwareAugment pipeline')
        output, weights, vertices, aux = self.forward_dummy(
            lq, return_aux=True, lq_context=lq_context,
            context_samples=context_samples, crop_box=crop_box)
        losses = dict(loss_recons=self.recons_loss(output, gt))
        if self.kd_loss is not None:
            multiplier = self.kd_multiplier()
            losses['loss_kd'] = multiplier * self.kd_loss(
                output, lq, gt=gt, use_confidence=self.use_confidence_kd,
                confidence_temperature=self.confidence_temperature,
                lq_context=lq_context, context_samples=context_samples,
                crop_box=crop_box)
            losses['kd_multiplier'] = output.new_tensor(multiplier)
        if self.use_lab_loss:
            losses['loss_lab'] = self.lab_loss_weight * F.l1_loss(
                normalized_lab(output), normalized_lab(gt))
        if self.use_rgb_mse_loss:
            losses['loss_rgb_mse'] = self.rgb_mse_loss_weight * F.mse_loss(output, gt)
        if self.use_low_frequency_loss:
            size = (self.low_frequency_size,) * 2
            losses['loss_low_frequency'] = self.low_frequency_loss_weight * F.l1_loss(
                normalized_lab(F.adaptive_avg_pool2d(output, size)),
                normalized_lab(F.adaptive_avg_pool2d(gt, size)))
        if self.sparse_factor > 0:
            losses['loss_sparse'] = self.sparse_factor * weights.abs().mean()
        smooth, mono = self._regularizations(vertices, aux['residual_luts'])
        if self.smooth_factor > 0:
            losses['loss_smooth'] = smooth
        if self.monotonicity_factor > 0:
            losses['loss_mono'] = mono
        if self.use_spatial_smoothness and self.spatial_smoothness_factor > 0:
            low = aux['low_resolution_guide']
            reference = F.interpolate(
                F.avg_pool2d(lq, 3, stride=1, padding=1, count_include_pad=False),
                size=low.shape[-2:], mode='bilinear', align_corners=False)
            tv = output.new_zeros(())
            for axis in (-1, -2):
                difference = torch.diff(low, dim=axis).abs()
                if self.use_edge_aware_smoothness:
                    edge = torch.diff(reference, dim=axis).abs().mean(1, keepdim=True)
                    difference = difference * torch.exp(-self.edge_smoothness_scale * edge)
                tv = tv + difference.mean()
            losses['loss_guide_smooth'] = self.spatial_smoothness_factor * tv
        losses.update(
            guide_spatial_std=aux['low_resolution_guide'].detach().flatten(2).std(
                dim=-1, unbiased=False).mean(),
            residual_abs_mean=(output - aux['tone_output']).detach().abs().mean(),
            output_change=(output - lq).detach().abs().mean(),
            target_change=(gt - lq).detach().abs().mean())
        return dict(losses=losses, num_samples=len(gt),
                    results=dict(lq=lq.detach().cpu(), gt=gt.detach().cpu(),
                                 output=output.detach().cpu()))


@MODELS.register_module()
class AiLUTArchitectureV3(AiLUTRefinementBase):
    """Fixed post-tone conditioning plus multiscale guided spatial architecture."""
    def __init__(self, guided_filter_radius=2, guided_filter_epsilon=0.001, **kwargs):
        super().__init__(**kwargs)
        dimension = kwargs.get('condition_dim', 256)
        self.post_tone_encoder = ConditionEncoder(
            dimension, self.color_statistics.output_features, dimension)
        self.spatial_guide = GuidedSpatialDecoder(
            dimension, self.n_spatial_luts,
            resolution=kwargs.get('guide_resolution', 128),
            channels=kwargs.get('guide_channels', 16),
            temperature=kwargs.get('guide_temperature', 1.0),
            radius=guided_filter_radius, epsilon=guided_filter_epsilon)

    def _conditions(self, source, samples=None):
        condition, curves, statistics = super()._conditions(source, samples)
        # Curve before statistics: T(mean(x)) and mean(T(x)) are different.
        selected = (samples if self.color_statistics.use_stratified and samples is not None
                    else stratified_pixels(source, self.color_statistics.pool_size)
                    if self.color_statistics.use_stratified else source)
        tone_values = self.tone_curve.apply_curves(selected, curves)
        post_statistics = self.color_statistics(tone_values)
        return self.post_tone_encoder(condition, post_statistics), curves, statistics

    def _guides(self, source, condition, imgs, crop_box):
        return self.spatial_guide(source, condition, imgs, crop_box)


@MODELS.register_module()
class AiLUTGlobalLocalV3(AiLUTArchitectureV3):
    """Fixed one-global/two-local architecture, with independent signed gates."""
    def __init__(self, local_gate_scale=0.5, **kwargs):
        if kwargs.get('n_spatial_luts', 3) != 3:
            raise ValueError('GlobalLocalV3 requires exactly 3 LUTs: global + 2 local')
        kwargs['n_spatial_luts'] = 3
        super().__init__(**kwargs)
        if local_gate_scale <= 0:
            raise ValueError('local_gate_scale must be positive')
        self.local_gate_scale = float(local_gate_scale)
        old = self.spatial_guide.output
        self.spatial_guide.output = nn.Conv2d(old.in_channels, 2, 1)
        nn.init.normal_(self.spatial_guide.output.weight, std=1e-3)
        nn.init.zeros_(self.spatial_guide.output.bias)

    def _guides(self, source, condition, imgs, crop_box):
        high, low = self.spatial_guide.forward_logits(source, condition, imgs, crop_box)
        high = self.local_gate_scale * torch.tanh(high)
        low = self.local_gate_scale * torch.tanh(low)
        return torch.cat((torch.ones_like(high[:, :1]), high), dim=1), low

    def _regularizations(self, vertices, residual_luts):
        # Local residual LUTs need not themselves be monotonic. Constrain the
        # four extreme maps global +/- scale*local_a +/- scale*local_b instead.
        global_lut, local_a, local_b = residual_luts.unbind(1)
        effective = torch.stack([
            global_lut + self.local_gate_scale * (a * local_a + b * local_b)
            for a, b in ((-1, -1), (-1, 1), (1, -1), (1, 1))], dim=1)
        return super()._regularizations(vertices, effective)
