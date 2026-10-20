"""Two independent AdaInt mappings, local corrections, then spatial fusion."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ailut import ailut_transform
from mmedit.models.registry import MODELS
from mmedit.utils import get_root_logger

from .model import AiLUT
from .color import p3_delta_e_map, display_p3_to_lab
from .modules import stratified_pixels, lut_regularizations
from .semantic_modules import (checked_mask, masked_per_image, RegionColorStatistics,
                               RegionMapping, DualLocalAffineDecoder)


def region_samples(image, mask, size):
    """Weighted systematic sampling; no masked-out black background is counted."""
    weights = mask.flatten(1).detach().contiguous()
    cumulative = weights.cumsum(1)
    fraction = (torch.arange(size * size, device=image.device, dtype=image.dtype) + .5) / (size * size)
    queries = cumulative[:, -1:] * fraction[None]
    indices = torch.searchsorted(cumulative, queries.contiguous()).clamp(max=weights.size(1) - 1)
    return image.flatten(2).gather(2, indices[:, None].expand(-1, 3, -1)).view(-1, 3, size, size)


@MODELS.register_module()
class AiLUTDual(AiLUT):
    def __init__(self, local_enabled=True, use_region_condition=True,
                 region_samples_size=32, region_sample_mix=.5, region_min_area=.0001,
                 foreground_rgb_weight=.1, background_rgb_weight=.1,
                 foreground_color_weight=.1, background_color_weight=.1,
                 face_color_weight=0., boundary_gradient_weight=0.,
                 lut_mode='tone_residual', **kwargs):
        if lut_mode not in ('tone_residual', 'basis'):
            raise ValueError("lut_mode must be 'tone_residual' or 'basis'")
        self.lut_mode = lut_mode
        if kwargs.get('gen', 0) > 0 or kwargs.get('kd_loss') is not None:
            raise ValueError('AiLUTDual currently trains without BAN; use gen=1, enable_kd=false in config')
        super().__init__(**kwargs)
        self.local_enabled, self.use_region_condition = bool(local_enabled), bool(use_region_condition)
        if region_samples_size < 1 or not 0 <= region_sample_mix <= 1 or not 0 < region_min_area < .5:
            raise ValueError('invalid regional statistics settings')
        self.region_samples_size = int(region_samples_size)
        self.region_sample_mix, self.region_min_area = float(region_sample_mix), float(region_min_area)
        for name, value in dict(foreground_rgb_weight=foreground_rgb_weight,
                                background_rgb_weight=background_rgb_weight,
                                foreground_color_weight=foreground_color_weight,
                                background_color_weight=background_color_weight,
                                face_color_weight=face_color_weight,
                                boundary_gradient_weight=boundary_gradient_weight).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(name + ' must be finite and non-negative')
            setattr(self, name, float(value))
        self.color_statistics = RegionColorStatistics(kwargs.get('stats_bins', 16),
                                                     kwargs.get('stats_pool_size', 64))
        dimension, hidden = kwargs.get('condition_dim', 256), kwargs.get('head_hidden_dim', 128)
        for name in ('condition_encoder', 'tone_curve', 'adaint', 'lut_generator', 'local_decoder'):
            delattr(self, name)
        options = (self.backbone.out_channels, self.color_statistics.output_features, dimension,
                   hidden, self.n_vertices, kwargs.get('n_ranks', 5),
                   kwargs.get('curve_points', 16), kwargs.get('min_interval', .001), lut_mode)
        self.branches = nn.ModuleDict({name: RegionMapping(*options)
                                       for name in ('foreground', 'background')})
        if self.local_enabled:
            self.local_decoder = DualLocalAffineDecoder(dimension,
                kwargs.get('local_resolution', 128), kwargs.get('local_channels', 16),
                kwargs.get('guided_filter_radius', 2), kwargs.get('guided_filter_epsilon', .001))

    def forward_dummy(self, imgs, return_aux=False, lq_context=None, context_samples=None,
                      crop_box=None, foreground_mask=None, foreground_context=None,
                      foreground_samples=None, foreground_region_samples=None,
                      background_region_samples=None, region_area=None):
        foreground_mask = checked_mask(foreground_mask, imgs)
        if crop_box is not None and (lq_context is None or foreground_context is None):
            raise ValueError('crop training needs aligned full-image and full-mask context')
        source_image = imgs if lq_context is None else lq_context
        source_mask = foreground_mask if lq_context is None else checked_mask(
            foreground_context, lq_context, 'foreground_context')
        source = F.interpolate(source_image, (self.context_size,) * 2,
                               mode='bilinear', align_corners=False)
        full_mask = F.interpolate(source_mask, source.shape[-2:], mode='area')
        samples = stratified_pixels(source_image, self.color_statistics.pool_size) if context_samples is None else context_samples
        sample_mask = (stratified_pixels(source_mask, self.color_statistics.pool_size)
                       if foreground_samples is None else checked_mask(foreground_samples, samples, 'foreground_samples'))
        global_stats = self.color_statistics(source_image, samples)
        cnn_code = self.backbone(source)
        area = source_mask.mean((-1, -2)) if region_area is None else region_area[:, :1].to(imgs)
        if not torch.isfinite(area).all() or (area < 0).any() or (area > 1).any():
            raise ValueError('region_area must be finite in [0,1]')
        supplied_samples = (foreground_region_samples, background_region_samples)
        names = ('foreground', 'background')
        conditions, weights_list, vertices_list, luts, global_outputs, stats_list = [], [], [], [], [], []
        for index, name in enumerate(names):
            mask = sample_mask if index == 0 else 1 - sample_mask
            region = area if index == 0 else 1 - area
            reliability = (region / self.region_min_area).clamp(0, 1)
            stats = self.color_statistics(source_image, samples, mask)
            supplemental = supplied_samples[index]
            if supplemental is None:
                supplemental = region_samples(source_image,
                    source_mask if index == 0 else 1 - source_mask, self.region_samples_size)
            roi_stats = self.color_statistics(supplemental, supplemental)
            # If the coarse grid missed the region entirely, trust its ROI samples.
            mix = torch.where(mask.sum((-1, -2)) > 1e-8,
                              region.new_full(region.shape, self.region_sample_mix),
                              torch.ones_like(region))
            stats = (1 - mix) * stats + mix * roi_stats
            stats = reliability * stats + (1 - reliability) * global_stats
            if not self.use_region_condition:
                stats, region, reliability = global_stats, torch.zeros_like(region), torch.ones_like(region)
            condition, weights, vertices, lut = self.branches[name](cnn_code, global_stats, stats, region, reliability)
            conditions.append(condition)
            weights_list.append(weights)
            vertices_list.append(vertices)
            luts.append(lut)
            stats_list.append(stats)
            global_outputs.append(ailut_transform(imgs, lut, vertices))
        context_deltas, deltas, lows, outputs, means = [], [], [], [], []
        if self.local_enabled:
            local_fields, low_source = self.local_decoder(source, conditions)
            low_mask = F.interpolate(full_mask, low_source.shape[-2:], mode='area')
            for index, (low, slope, offset) in enumerate(local_fields):
                coefficients = self.local_decoder.upsample(slope, offset, imgs, crop_box)
                raw_delta = coefficients[:, :1] * global_outputs[index] + coefficients[:, 1:]
                context_coefficients = self.local_decoder.upsample(slope, offset, low_source)
                context_global = ailut_transform(low_source, luts[index], vertices_list[index])
                context_raw = context_coefficients[:, :1] * context_global + context_coefficients[:, 1:]
                mask = low_mask if index == 0 else 1 - low_mask
                mass = mask.sum((-1, -2), keepdim=True)
                active = ((area if index == 0 else 1 - area) >= self.region_min_area).to(imgs)[:, :, None, None]
                mean = (context_raw * mask).sum((-1, -2), keepdim=True) / mass.clamp_min(1e-8)
                delta = (raw_delta - mean) * active
                context_deltas.append((context_raw - mean) * active)
                deltas.append(delta)
                means.append(mean * active)
                lows.append(low)
                outputs.append(global_outputs[index] + delta)
        else:
            low_source, low_mask = source, full_mask
            for global_output in global_outputs:
                outputs.append(global_output)
                deltas.append(torch.zeros_like(global_output))
                context_deltas.append(torch.zeros_like(source))
                lows.append(source.new_zeros((source.size(0), 4, *source.shape[-2:])))
                means.append(source.new_zeros((source.size(0), 3, 1, 1)))
        output = foreground_mask * outputs[0] + (1 - foreground_mask) * outputs[1]
        weights, vertices = torch.stack(weights_list, 1), torch.stack(vertices_list, 1)
        if return_aux:
            return output, weights, vertices, dict(
                luts=luts, vertices=vertices_list, branch_outputs=outputs,
                global_outputs=global_outputs, local_deltas=deltas,
                context_deltas=context_deltas, delta_means=means,
                low_coefficients=lows, local_reference=low_source,
                local_masks=(low_mask, 1 - low_mask), statistics=stats_list, conditions=conditions,
                affine_fields=[(slope, offset) for _, slope, offset in local_fields] if self.local_enabled else [])
        return output, weights, vertices

    def extra_forward_kwargs(self, kwargs):
        return {}

    def extra_training_losses(self, aux, **kwargs):
        return {}

    def forward_train(self, lq, gt, lq_context=None, context_samples=None, crop_box=None,
                      foreground_mask=None, supervision_mask=None, face_mask=None, **kwargs):
        required = ('foreground_context', 'foreground_samples', 'foreground_region_samples',
                    'background_region_samples', 'region_area')
        if lq_context is None or context_samples is None or crop_box is None or any(kwargs.get(key) is None for key in required):
            raise ValueError('training requires SemanticContextAugment with complete context and regional samples')
        output, weights, vertices, aux = self.forward_dummy(lq, True, lq_context, context_samples,
            crop_box, foreground_mask, **{key: kwargs[key] for key in required},
            **self.extra_forward_kwargs(kwargs))
        supervision_mask = checked_mask(supervision_mask, lq, 'supervision_mask')
        losses = dict(loss_recons=self.recons_loss(output, gt))
        if self.color_loss_weight:
            color, statistics = self.color_loss(output, gt)
            losses['loss_color'] = self.color_loss_weight * color
            losses.update(statistics)
        rgb_error = (output - gt).square().mean(1, keepdim=True)
        color_error = p3_delta_e_map(output, gt, epsilon=1e-3)
        for name, mask in (('foreground', supervision_mask), ('background', 1 - supervision_mask)):
            rgb, _, valid = masked_per_image(rgb_error, mask, self.region_min_area)
            color, _, _ = masked_per_image(color_error, mask, self.region_min_area)
            rgb_weight, color_weight = getattr(self, name + '_rgb_weight'), getattr(self, name + '_color_weight')
            if rgb_weight:
                losses['loss_' + name + '_rgb'] = rgb_weight * rgb
            if color_weight:
                losses['loss_' + name + '_color'] = color_weight * color / 100.
            losses[name + '_delta_e'] = color.detach()
            losses[name + '_valid_fraction'] = valid.detach().mean()
        if self.face_color_weight:
            face_mask = checked_mask(face_mask, lq, 'face_mask')
            face, _, _ = masked_per_image(color_error, face_mask, self.region_min_area)
            losses['loss_face_color'] = self.face_color_weight * face / 100.
        lab_difference = display_p3_to_lab(output) - display_p3_to_lab(gt)
        losses.update(self.extra_training_losses(aux, output=output, gt=gt,
            supervision_mask=supervision_mask, **kwargs))
        for key, error in (('foreground_delta_L', lab_difference[:, :1].abs()),
                           ('foreground_delta_ab', lab_difference[:, 1:].square().sum(1).clamp_min(1e-12).sqrt())):
            losses[key] = masked_per_image(error, supervision_mask, self.region_min_area)[0].detach()
        regularizers = [lut_regularizations(lut, vertex) for lut, vertex in zip(aux['luts'], aux['vertices'])]
        for index, (name, factor) in enumerate((('loss_lut_smooth', self.smooth_factor),
            ('loss_lut_mono', self.monotonicity_factor), ('loss_global_magnitude', self.global_magnitude_factor))):
            if factor:
                losses[name] = factor * (regularizers[0][index] + regularizers[1][index]) / 2
        if self.local_enabled and self.local_magnitude_factor:
            magnitude = sum(masked_per_image(delta.square().mean(1, keepdim=True), mask,
                                            self.region_min_area)[0]
                            for delta, mask in zip(aux['context_deltas'], aux['local_masks'])) / 2
            losses['loss_local_magnitude'] = self.local_magnitude_factor * magnitude
        if self.local_enabled and self.local_smoothness_factor:
            smooth = output.new_zeros(())
            for low, mask in zip(aux['low_coefficients'], aux['local_masks']):
                for axis in (-1, -2):
                    difference = torch.diff(low, dim=axis).square().mean(1, keepdim=True)
                    edge = torch.diff(aux['local_reference'], dim=axis).abs().mean(1, keepdim=True)
                    lo, hi = [slice(None)] * 4, [slice(None)] * 4
                    lo[axis], hi[axis] = slice(None, -1), slice(1, None)
                    edge_mask = (mask[tuple(lo)] + mask[tuple(hi)]) / 2
                    smooth += masked_per_image(difference * torch.exp(-self.edge_smoothness_scale * edge),
                                                edge_mask, self.region_min_area)[0] / 2
            losses['loss_local_smooth'] = self.local_smoothness_factor * smooth
        if self.boundary_gradient_weight:
            band = F.max_pool2d(supervision_mask, 3, 1, 1) + F.max_pool2d(-supervision_mask, 3, 1, 1)
            boundary = output.new_zeros(())
            for axis in (-1, -2):
                error = (torch.diff(output, dim=axis) - torch.diff(gt, dim=axis)).square().mean(1, keepdim=True)
                lo, hi = [slice(None)] * 4, [slice(None)] * 4
                lo[axis], hi[axis] = slice(None, -1), slice(1, None)
                boundary += masked_per_image(error, torch.maximum(band[tuple(lo)], band[tuple(hi)]),
                                             self.region_min_area)[0]
            losses['loss_boundary_gradient'] = self.boundary_gradient_weight * boundary
        losses.update(lut_weight_abs_mean=weights.detach().abs().mean(),
                      lut_branch_weight_difference=(weights[:, 0] - weights[:, 1]).detach().abs().mean(),
                      vertex_branch_difference=(vertices[:, 0] - vertices[:, 1]).detach().abs().mean(),
                      local_correction_abs_mean=(foreground_mask * aux['local_deltas'][0] +
                          (1 - foreground_mask) * aux['local_deltas'][1]).detach().abs().mean(),
                      output_change=(output - lq).detach().abs().mean())
        return dict(losses=losses, num_samples=len(gt),
                    results=dict(lq=lq.detach().cpu(), gt=gt.detach().cpu(), output=output.detach().cpu()))

    def train_step(self, data_batch, optimizer):
        frozen = self.cnt_iters.item() < self.n_fix_iters
        if frozen != self.adaint_fixed:
            for branch in self.branches.values():
                branch.adaint.requires_grad_(not frozen)
            self.adaint_fixed = frozen
            get_root_logger().info('%s both AdaInt heads at iteration %s',
                                   'Freeze' if frozen else 'Unfreeze', int(self.cnt_iters.item()))
        outputs = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(outputs.pop('losses'))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.cnt_iters += 1
        outputs.update(log_vars=log_vars)
        return outputs

    def evaluate(self, output, gt, meta=None, foreground_mask=None, face_mask=None,
                 supervision_mask=None):
        extra_names = {'deltaE_P3_foreground', 'deltaE_P3_background', 'deltaE_P3_face',
                       'deltaL_P3_foreground', 'deltaAB_P3_foreground',
                       'deltaL_P3_background', 'deltaL_P3_highlights'}
        # Use an independent ConfigDict, since inherited evaluate knows global metrics.
        original = self.test_cfg
        self.test_cfg = type(original)(dict(original))
        self.test_cfg.metrics = [name for name in original.metrics if name not in extra_names]
        try:
            result = super().evaluate(output, gt, meta)
        finally:
            self.test_cfg = original
        if not extra_names.intersection(original.metrics):
            return result
        mask = checked_mask(supervision_mask if supervision_mask is not None else foreground_mask, output)
        pred, target = output.detach().clamp(0, 1), gt.detach().clamp(0, 1)
        border = original.crop_border
        if border:
            pred, target, mask = [value[..., border:-border, border:-border] for value in (pred, target, mask)]
            if face_mask is not None:
                face_mask = face_mask[..., border:-border, border:-border]
        color = p3_delta_e_map(pred, target)
        lab = display_p3_to_lab(pred) - display_p3_to_lab(target)
        for name, region_mask, error in (
            ('deltaE_P3_foreground', mask, color),
            ('deltaE_P3_background', 1 - mask, color),
            ('deltaL_P3_foreground', mask, lab[:, :1].abs()),
            ('deltaL_P3_background', 1 - mask, lab[:, :1].abs()),
            ('deltaL_P3_highlights', (display_p3_to_lab(target)[:, :1] >= 80).to(mask), lab[:, :1].abs()),
            ('deltaAB_P3_foreground', mask, lab[:, 1:].norm(dim=1))):
            if name in original.metrics:
                value, _, valid = masked_per_image(error, region_mask, self.region_min_area)
                result[name] = value.item()
                region = ('background' if name.endswith('background') else
                          'highlights' if name.endswith('highlights') else 'foreground')
                result['_valid_' + region] = valid.item()
        if 'deltaE_P3_face' in original.metrics:
            face_mask = checked_mask(face_mask, pred, 'face_mask')
            value, _, valid = masked_per_image(color, face_mask, self.region_min_area)
            result['deltaE_P3_face'], result['_valid_face'] = value.item(), valid.item()
        return result
