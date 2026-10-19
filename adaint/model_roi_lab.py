"""Jointly trained dual LUTs with native foreground-ROI Lab local correction."""
import math

import torch
from ailut import ailut_transform
from mmedit.models.registry import MODELS

from .color import display_p3_to_lab
from .model_semantic import AiLUTDual
from .roi_lab import ForegroundROILab, extract_roi, paste_roi, apply_lab_residual
from .semantic_modules import checked_mask, masked_per_image


@MODELS.register_module()
class AiLUTROILab(AiLUTDual):
    def __init__(self, roi_size=128, roi_margin=.15, roi_channels=16,
                 roi_lightness_limit=8., roi_chroma_limit=4.,
                 roi_mean_lightness_weight=.1, roi_spatial_lightness_weight=.1,
                 roi_magnitude_factor=.001, roi_smoothness_factor=.0001, **kwargs):
        if kwargs.get('lut_mode', 'basis') != 'basis' or not kwargs.get('local_enabled', True):
            raise ValueError('ROI Lab requires basis LUTs and the background local decoder')
        kwargs.update(lut_mode='basis', foreground_local_enabled=False)
        super().__init__(**kwargs)
        if self.n_fix_iters:
            raise ValueError('ROI Lab is joint training: set adaint_fix_epochs/n_fix_iters=0')
        if roi_size < 32 or not math.isfinite(roi_margin) or not 0 <= roi_margin <= 1:
            raise ValueError('invalid ROI size/margin')
        self.roi_size, self.roi_margin = int(roi_size), float(roi_margin)
        for name, value in dict(roi_mean_lightness_weight=roi_mean_lightness_weight,
            roi_spatial_lightness_weight=roi_spatial_lightness_weight,
            roi_magnitude_factor=roi_magnitude_factor, roi_smoothness_factor=roi_smoothness_factor).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(name + ' must be finite and non-negative')
            setattr(self, name, float(value))
        self.foreground_roi = ForegroundROILab(kwargs.get('condition_dim', 256),
            roi_channels, roi_lightness_limit, roi_chroma_limit)
        # No teacher, freeze stage, train()/eval() override or detached features.
        if not all(parameter.requires_grad for parameter in self.parameters()):
            raise ValueError('all ROI-Lab model parameters must be trainable')

    def forward_dummy(self, imgs, return_aux=False, lq_context=None, context_samples=None,
                      crop_box=None, foreground_mask=None, foreground_context=None,
                      foreground_samples=None, foreground_region_samples=None,
                      background_region_samples=None, region_area=None,
                      foreground_roi=None, foreground_roi_mask=None, foreground_roi_box=None):
        foreground_mask = checked_mask(foreground_mask, imgs)
        supplied = (foreground_roi, foreground_roi_mask, foreground_roi_box)
        if any(value is not None for value in supplied) and not all(value is not None for value in supplied):
            raise ValueError('ROI image, mask and full-image box must be supplied together')
        if foreground_roi is None:
            if crop_box is not None or lq_context is not None:
                raise ValueError('crop training requires ROI extracted from the ORIGINAL full image')
            foreground_roi, foreground_roi_mask, foreground_roi_box = extract_roi(
                imgs, foreground_mask, self.roi_size, self.roi_margin)
        if foreground_roi.shape != (imgs.size(0), 3, self.roi_size, self.roi_size):
            raise ValueError('native foreground ROI must have B,3,roi_size,roi_size shape')
        foreground_roi_mask = checked_mask(foreground_roi_mask, foreground_roi, 'foreground_roi_mask')
        area = (foreground_mask.mean((-1, -2)) if region_area is None else region_area[:, :1].to(imgs))
        active = (area >= self.region_min_area).to(imgs)[:, :, None, None]
        features, addition = self.foreground_roi.encode(foreground_roi, foreground_roi_mask, active)
        _, weights, vertices, aux = super().forward_dummy(imgs, True, lq_context,
            context_samples, crop_box, foreground_mask, foreground_context, foreground_samples,
            foreground_region_samples, background_region_samples, region_area,
            foreground_condition_offset=addition)
        roi_base = ailut_transform(foreground_roi, aux['luts'][0], aux['vertices'][0])
        requested = self.foreground_roi.decode(features, roi_base, aux['conditions'][0],
                                                foreground_roi_mask, active)
        roi_output, roi_actual = apply_lab_residual(roi_base, requested)
        field = paste_roi(requested, foreground_roi_box, imgs.shape[-2:], crop_box)
        foreground, actual = apply_lab_residual(aux['global_outputs'][0], field)
        background = aux['branch_outputs'][1]
        output = foreground_mask * foreground + (1 - foreground_mask) * background
        if return_aux:
            aux['branch_outputs'] = [foreground, background]
            aux['local_deltas'] = [foreground - aux['global_outputs'][0], aux['local_deltas'][1]]
            aux.update(roi_rgb=roi_output, roi_pre_rgb=roi_base, roi_source=foreground_roi,
                roi_mask=foreground_roi_mask, roi_box=foreground_roi_box, roi_active=active,
                roi_requested_delta=requested, roi_actual_delta=roi_actual,
                foreground_lab_delta=actual, roi_condition_offset=addition)
            return output, weights, vertices, aux
        return output, weights, vertices

    def forward_train(self, *args, **kwargs):
        required = ('foreground_roi', 'foreground_roi_mask', 'foreground_roi_box',
                    'foreground_roi_gt', 'foreground_roi_supervision')
        if any(kwargs.get(key) is None for key in required):
            raise ValueError('ROI training needs full-image ROI inputs and separate ROI GT supervision')
        return super().forward_train(*args, **kwargs)

    def extra_training_losses(self, aux, **kwargs):
        target = kwargs['foreground_roi_gt']
        if target.shape != aux['roi_rgb'].shape:
            raise ValueError('ROI GT must align with the original-image ROI')
        mask = checked_mask(kwargs['foreground_roi_supervision'], target, 'ROI supervision') * aux['roi_active']
        # GT is only used here, after prediction; it never enters ROI features/LUT conditions.
        difference = (display_p3_to_lab(aux['roi_rgb'])[:, :1] -
                      display_p3_to_lab(target)[:, :1]) / 100
        mean = (difference * mask).sum((-1, -2), keepdim=True) / mask.sum((-1, -2), keepdim=True).clamp_min(1e-8)
        centered = difference - mean
        mean_loss = masked_per_image(mean.abs().expand_as(difference), mask, self.region_min_area)[0]
        spatial_loss = masked_per_image((centered.square() + 1e-6).sqrt() - .001,
                                        mask, self.region_min_area)[0]
        active_mask = aux['roi_mask'] * aux['roi_active']
        magnitude, smooth = self.foreground_roi.regularizers(aux['roi_actual_delta'],
            active_mask, aux['roi_source'], self.region_min_area, self.edge_smoothness_scale)
        losses = {}
        for name, value, weight in (
            ('loss_roi_mean_lightness', mean_loss, self.roi_mean_lightness_weight),
            ('loss_roi_spatial_lightness', spatial_loss, self.roi_spatial_lightness_weight),
            ('loss_roi_magnitude', magnitude, self.roi_magnitude_factor),
            ('loss_roi_smooth', smooth, self.roi_smoothness_factor)):
            if weight:
                losses[name] = value * weight
        def average(error):
            return masked_per_image(error, active_mask, self.region_min_area)[0].detach()
        requested_mean = (aux['roi_requested_delta'] * aux['roi_mask']).sum((-1, -2)) / aux['roi_mask'].sum((-1, -2)).clamp_min(1e-8)
        actual_mean = (aux['roi_actual_delta'] * aux['roi_mask']).sum((-1, -2)) / aux['roi_mask'].sum((-1, -2)).clamp_min(1e-8)
        losses.update(roi_mean_lightness_error=(mean_loss.detach() * 100),
            roi_spatial_lightness_error=(spatial_loss.detach() * 100),
            roi_requested_mean_abs=requested_mean.detach().abs().mean(),
            roi_actual_mean_abs=actual_mean.detach().abs().mean(),
            roi_delta_L_abs=average(aux['roi_actual_delta'][:, :1].abs()),
            roi_delta_ab_abs=average(aux['roi_actual_delta'][:, 1:].norm(dim=1, keepdim=True)),
            roi_delta_L_spatial_std=average((aux['roi_actual_delta'][:, :1] - actual_mean[:, :1, None, None]).square()).sqrt(),
            roi_requested_delta_L_std=average(aux['roi_requested_delta'][:, :1].square()).sqrt(),
            roi_condition_abs=aux['roi_condition_offset'].detach().abs().mean())
        return losses
