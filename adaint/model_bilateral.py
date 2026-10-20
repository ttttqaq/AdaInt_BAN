"""Complete basis enhancer + signed ROI lightness grid, jointly trained from step 1."""
import math

import torch
from ailut import ailut_transform
from mmedit.models.registry import MODELS

from .model_semantic import AiLUTDual
from .semantic_modules import checked_mask, masked_per_image
from .color import display_p3_to_lab
from .calibration_geometry import extract_roi
from .bilateral_calibration import (ForegroundLightnessGrid, slice_lightness_grid,
                                    apply_lightness_residual, multiscale_calibration_loss)


@MODELS.register_module()
class AiLUTBilateral(AiLUTDual):
    def __init__(self, roi_size=128, roi_margin=.15, calibration_channels=16,
                 grid_spatial_nodes=8, grid_lightness_nodes=8, calibration_limit=8.,
                 calibration_loss_weight=.2, calibration_scales=(1, 4, 8),
                 calibration_min_coverage=.05, calibration_magnitude_factor=.0001,
                 calibration_smoothness_factor=.0001, **kwargs):
        if kwargs.get('lut_mode', 'basis') != 'basis' or not kwargs.get('local_enabled', True):
            raise ValueError('bilateral calibration retains complete basis LUTs and BOTH local heads')
        kwargs['lut_mode'] = 'basis'
        super().__init__(**kwargs)
        if self.n_fix_iters:
            raise ValueError('all parameters train from step 1: set adaint_fix_epochs=0')
        if roi_size < 32 or not math.isfinite(roi_margin) or not 0 <= roi_margin <= 1:
            raise ValueError('invalid native ROI geometry')
        self.roi_size, self.roi_margin = int(roi_size), float(roi_margin)
        if not calibration_scales or any(not isinstance(s, int) or s < 1 or s > roi_size for s in calibration_scales):
            raise ValueError('invalid multiscale calibration partitions')
        if not 0 < calibration_min_coverage <= 1:
            raise ValueError('invalid cell coverage')
        self.calibration_scales = tuple(calibration_scales)
        self.calibration_min_coverage = float(calibration_min_coverage)
        for key, value in dict(calibration_loss_weight=calibration_loss_weight,
            calibration_magnitude_factor=calibration_magnitude_factor,
            calibration_smoothness_factor=calibration_smoothness_factor).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(key + ' must be finite and nonnegative')
            setattr(self, key, float(value))
        self.foreground_calibration = ForegroundLightnessGrid(kwargs.get('condition_dim', 256),
            calibration_channels, grid_spatial_nodes, grid_lightness_nodes, calibration_limit)

    def extra_forward_kwargs(self, kwargs):
        return {key: kwargs[key] for key in ('foreground_roi', 'foreground_roi_mask',
                'foreground_roi_box') if key in kwargs}

    def forward_dummy(self, imgs, return_aux=False, lq_context=None, context_samples=None,
                      crop_box=None, foreground_mask=None, foreground_context=None,
                      foreground_samples=None, foreground_region_samples=None,
                      background_region_samples=None, region_area=None,
                      foreground_roi=None, foreground_roi_mask=None, foreground_roi_box=None):
        mask = checked_mask(foreground_mask, imgs)
        supplied = (foreground_roi, foreground_roi_mask, foreground_roi_box)
        if any(v is not None for v in supplied) and not all(v is not None for v in supplied):
            raise ValueError('ROI RGB, mask and full-image box are required together')
        if foreground_roi is None:
            if crop_box is not None or lq_context is not None:
                raise ValueError('crop training requires ROI extracted BEFORE crop/downsampling')
            foreground_roi, foreground_roi_mask, foreground_roi_box = extract_roi(
                imgs, mask, self.roi_size, self.roi_margin)
        if foreground_roi.shape != (imgs.size(0), 3, self.roi_size, self.roi_size):
            raise ValueError('ROI has the wrong native sampling resolution')
        roi_mask = checked_mask(foreground_roi_mask, foreground_roi)
        _, weights, vertices, aux = super().forward_dummy(imgs, True, lq_context, context_samples,
            crop_box, mask, foreground_context, foreground_samples, foreground_region_samples,
            background_region_samples, region_area)
        area = mask.mean((-1, -2)) if region_area is None else region_area[:, :1].to(imgs)
        active = (area >= self.region_min_area).to(imgs)[:, :, None, None]
        # Render the current complete FG mapping on native ROI coordinates,
        # using the SAME LUT, vertices, guided coefficients and full-region mean.
        # The calibration auxiliary objective cannot change this target or its inputs.
        with torch.no_grad():
            roi_global = ailut_transform(foreground_roi, aux['luts'][0], aux['vertices'][0])
            slope, offset = aux['affine_fields'][0]
            coeff = self.local_decoder.upsample(slope, offset, foreground_roi, foreground_roi_box)
            roi_base = roi_global + (coeff[:, :1] * roi_global + coeff[:, 1:]) * active - aux['delta_means'][0]
        grid = self.foreground_calibration(foreground_roi, roi_base, roi_mask,
            [value.detach() for value in aux['conditions']], active)
        requested = slice_lightness_grid(grid, imgs, foreground_roi_box, crop_box)
        pre_foreground = aux['branch_outputs'][0]
        foreground, actual = apply_lightness_residual(pre_foreground, requested)
        output = mask * foreground + (1 - mask) * aux['branch_outputs'][1]
        if return_aux:
            roi_requested = slice_lightness_grid(grid, foreground_roi)
            roi_output, roi_actual = apply_lightness_residual(roi_base, roi_requested)
            aux['branch_outputs'] = [foreground, aux['branch_outputs'][1]]
            aux['local_deltas'][0] = aux['local_deltas'][0] + foreground - pre_foreground
            aux.update(calibration_grid=grid, calibration_base_roi=roi_base,
                calibration_roi_output=roi_output, calibration_actual_roi=roi_actual,
                calibration_requested_roi=roi_requested, calibration_roi_mask=roi_mask * active,
                calibration_native_actual=actual, calibration_pre_foreground=pre_foreground)
            return output, weights, vertices, aux
        return output, weights, vertices

    def forward_train(self, *args, **kwargs):
        required = ('foreground_roi', 'foreground_roi_mask', 'foreground_roi_box',
                    'foreground_roi_gt', 'foreground_roi_supervision')
        if any(kwargs.get(key) is None for key in required):
            raise ValueError('bilateral training needs aligned full-image ROI supervision')
        return super().forward_train(*args, **kwargs)

    def extra_training_losses(self, aux, output, gt, supervision_mask, **kwargs):
        roi_gt = kwargs['foreground_roi_gt']
        if roi_gt.shape != aux['calibration_base_roi'].shape:
            raise ValueError('ROI GT must align with the original full-image ROI')
        roi_mask = checked_mask(kwargs['foreground_roi_supervision'], roi_gt)
        roi_mask = roi_mask * (aux['calibration_roi_mask'] > 0).to(roi_mask)
        target = (display_p3_to_lab(roi_gt)[:, :1] -
                  display_p3_to_lab(aux['calibration_base_roi'])[:, :1]).detach()
        actual = aux['calibration_actual_roi']
        value, errors = multiscale_calibration_loss(actual, target, roi_mask,
            self.calibration_scales, self.calibration_min_coverage)
        losses = dict(loss_calibration_multiscale=self.calibration_loss_weight * value)
        limit = self.foreground_calibration.lightness_limit
        magnitude = masked_per_image((actual / limit).square(), roi_mask, self.region_min_area)[0]
        grid = aux['calibration_grid'] / limit
        smooth = sum(torch.diff(grid, dim=axis).square().mean() for axis in (-1, -2, -3))
        losses.update(loss_calibration_magnitude=self.calibration_magnitude_factor * magnitude,
                      loss_calibration_smooth=self.calibration_smoothness_factor * smooth)
        for scale, error in errors.items():
            losses['calibration_L_error_' + str(scale)] = error
        mean = (actual * roi_mask).sum((-1, -2), keepdim=True) / roi_mask.sum((-1, -2), keepdim=True).clamp_min(1e-8)
        losses.update(calibration_delta_L_mean=mean.detach().mean(),
            calibration_delta_L_abs=masked_per_image(actual.abs(), roi_mask, self.region_min_area)[0].detach(),
            calibration_delta_L_std=masked_per_image((actual - mean).square(), roi_mask, self.region_min_area)[0].sqrt().detach(),
            calibration_grid_abs=aux['calibration_grid'].detach().abs().mean(),
            calibration_bound_fraction=(aux['calibration_grid'].detach().abs() > .99 * limit).float().mean())
        return losses
