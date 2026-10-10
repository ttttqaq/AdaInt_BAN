"""The branch's only enhancer: fused global tone/AdaInt LUT + local affine correction."""
import numbers
import os.path as osp

import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16
from ailut import ailut_transform
from mmedit.core import delta_e, psnr, ssim, tensor2img
from mmedit.models.base import BaseModel
from mmedit.models.builder import build_loss
from mmedit.models.registry import MODELS
from mmedit.utils import get_root_logger

from .backbones import TPAMIBackbone
from .color import P3LabColorLoss
from .metrics_hc import delta_e_hc, psnr_hc
from .modules import (ColorStatistics, ConditionEncoder, ToneCurve,
                      AdaptiveIntervals, NormalizedResidualLUT, LocalAffineDecoder,
                      identity_lut, lut_regularizations)


@MODELS.register_module()
class AiLUT(BaseModel):
    allowed_metrics = {'PSNR': psnr, 'SSIM': ssim, 'deltaE': delta_e,
                       'PSNR_hc': psnr_hc, 'deltaE_hc': delta_e_hc}

    def __init__(self, n_ranks=5, n_vertices=33, curve_points=16,
                 stats_bins=16, stats_pool_size=64, condition_dim=256,
                 head_hidden_dim=128, local_resolution=128, local_channels=16,
                 context_size=256, min_interval=0.001, guided_filter_radius=2,
                 guided_filter_epsilon=0.001, backbone='tpami', pretrained=False,
                 smooth_factor=1e-5, monotonicity_factor=0.01,
                 global_magnitude_factor=1e-5, local_magnitude_factor=0.001,
                 local_smoothness_factor=0.0001, edge_smoothness_scale=10.,
                 kd_decay_start_ratio=0.3, kd_decay_end_ratio=1.,
                 kd_final_weight=0., color_loss_weight=0., color_grid_size=16,
                 color_top_fraction=0.1, color_hard_weight=0.25,
                 gen=0, kd_loss=None,
                 train_cfg=None, test_cfg=None):
        super().__init__()
        if backbone.lower() != 'tpami' or pretrained:
            raise ValueError('this branch retains the unpretrained TPAMI backbone')
        if context_size < 32:
            raise ValueError('context_size must be at least 32')
        if not 0 <= kd_decay_start_ratio < kd_decay_end_ratio <= 1:
            raise ValueError('invalid KD decay ratios')
        if not 0 <= kd_final_weight <= 1:
            raise ValueError('kd_final_weight must be between zero and one')
        self.backbone_name = backbone.lower()
        self.backbone = TPAMIBackbone(extra_pooling=True)
        self.color_statistics = ColorStatistics(stats_bins, stats_pool_size)
        self.condition_encoder = ConditionEncoder(
            self.backbone.out_channels, self.color_statistics.output_features, condition_dim)
        self.tone_curve = ToneCurve(curve_points, condition_dim, head_hidden_dim)
        self.adaint = AdaptiveIntervals(n_vertices, condition_dim, head_hidden_dim, min_interval)
        self.lut_generator = NormalizedResidualLUT(n_vertices, n_ranks, condition_dim, head_hidden_dim)
        self.local_decoder = LocalAffineDecoder(condition_dim, local_resolution, local_channels,
                                                guided_filter_radius, guided_filter_epsilon)
        self.n_vertices = int(n_vertices)
        self.context_size = int(context_size)
        # Used by the context-aware pipeline and frozen BAN teacher.
        self.use_full_image_context = True
        for key, value in dict(smooth_factor=smooth_factor,
                               monotonicity_factor=monotonicity_factor,
                               global_magnitude_factor=global_magnitude_factor,
                               local_magnitude_factor=local_magnitude_factor,
                               local_smoothness_factor=local_smoothness_factor,
                               edge_smoothness_scale=edge_smoothness_scale).items():
            value = float(value)
            if value < 0:
                raise ValueError(key+' must be non-negative')
            setattr(self, key, float(value))
        self.kd_decay_start_ratio = float(kd_decay_start_ratio)
        self.kd_decay_end_ratio = float(kd_decay_end_ratio)
        self.kd_final_weight = float(kd_final_weight)
        self.gen, self.train_cfg, self.test_cfg = int(gen), train_cfg or {}, test_cfg
        self.n_fix_iters = self.train_cfg.get('n_fix_iters', 0)
        self.adaint_fixed = False
        self.fp16_enabled = False
        self.register_buffer('cnt_iters', torch.zeros(1))
        self.recons_loss = build_loss(dict(type='MSELoss', loss_weight=1., reduction='mean'))
        if not 0 <= color_loss_weight < float('inf'):
            raise ValueError('color_loss_weight must be finite and non-negative')
        self.color_loss_weight = float(color_loss_weight)
        self.color_loss = P3LabColorLoss(color_grid_size, color_top_fraction,
                                       color_hard_weight)
        self.kd_loss = build_loss(kd_loss) if self.gen > 0 and kd_loss is not None else None
        self.init_weights()

    def init_weights(self):
        def initialize(module):
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.InstanceNorm2d) and module.affine:
                nn.init.normal_(module.weight, 1., .02)
                nn.init.zeros_(module.bias)
        self.backbone.apply(initialize)

    def kd_multiplier(self):
        total = self.train_cfg.get('total_iters', 0)
        if total <= 0:
            raise ValueError('BAN decay needs resolved train_cfg.total_iters')
        progress = float(self.cnt_iters.item()) / max(total-1, 1)
        fraction = max(0., min(1., (progress-self.kd_decay_start_ratio)/
                             (self.kd_decay_end_ratio-self.kd_decay_start_ratio)))
        return 1.+fraction*(self.kd_final_weight-1.)

    def forward_dummy(self, imgs, return_aux=False, lq_context=None,
                      context_samples=None, crop_box=None):
        if crop_box is not None and lq_context is None:
            raise ValueError('crop_box requires full-image context')
        source = imgs if lq_context is None else lq_context
        source = F.interpolate(source, (self.context_size,)*2, mode='bilinear', align_corners=False)
        statistics = self.color_statistics(imgs if lq_context is None else source,
                                           context_samples)
        condition = self.condition_encoder(self.backbone(source), statistics)
        curves = self.tone_curve.predict_curves(condition)
        vertices = self.adaint(condition)
        tone_vertices = self.tone_curve.apply_curves(vertices, curves)
        weights, residual_lut = self.lut_generator(condition)
        global_lut = identity_lut(tone_vertices)+residual_lut
        # The original RGB is queried once at target resolution; tone changes
        # the LUT values, not the input coordinate system.
        global_output = ailut_transform(imgs, global_lut, vertices)
        low, slope, offset, low_source = self.local_decoder(source, condition)
        coefficients = self.local_decoder.upsample(slope, offset, imgs, crop_box)
        raw_delta = coefficients[:, :1]*global_output+coefficients[:, 1:]
        # Define the constant component on the SAME full-context quadrature
        # grid for crop training, validation, inference and frozen teachers.
        # This additional LUT query is small (default 64x64), not full-resolution.
        context_coefficients = self.local_decoder.upsample(slope, offset, low_source)
        context_global = ailut_transform(low_source, global_lut, vertices)
        context_delta = (context_coefficients[:, :1]*context_global+
                         context_coefficients[:, 1:])
        delta_mean = context_delta.mean((-1, -2), keepdim=True)
        local_delta = raw_delta-delta_mean
        output = global_output+local_delta
        if return_aux:
            return output, weights, vertices, dict(
                global_lut=global_lut, global_output=global_output, curves=curves,
                statistics=statistics, local_delta=local_delta,
                context_delta=context_delta-delta_mean, delta_mean=delta_mean,
                low_coefficients=low, local_reference=low_source)
        return output, weights, vertices

    @auto_fp16(apply_to=('lq', 'lq_context', 'context_samples'))
    def forward(self, lq, gt=None, test_mode=False, **kwargs):
        if test_mode:
            return self.forward_test(lq, gt, **kwargs)
        return self.forward_train(lq, gt, **kwargs)

    def forward_train(self, lq, gt, lq_context=None, context_samples=None,
                      crop_box=None, **kwargs):
        if lq_context is None or context_samples is None or crop_box is None:
            raise ValueError('training requires ContextAwareAugment whole-image context')
        output, weights, vertices, aux = self.forward_dummy(
            lq, True, lq_context, context_samples, crop_box)
        losses = dict(loss_recons=self.recons_loss(output, gt))
        if self.color_loss_weight:
            color, statistics = self.color_loss(output, gt)
            losses['loss_color'] = self.color_loss_weight * color
            losses.update(statistics)
        if self.kd_loss is not None:
            multiplier = self.kd_multiplier()
            losses['loss_kd'] = (multiplier*self.kd_loss(
                output, lq, lq_context=lq_context,
                context_samples=context_samples, crop_box=crop_box)
                if multiplier > 0 else output.new_zeros(()))
            losses['kd_multiplier'] = output.new_tensor(multiplier)
        if self.smooth_factor or self.monotonicity_factor or self.global_magnitude_factor:
            smooth, mono, magnitude = lut_regularizations(aux['global_lut'], vertices)
            for name, factor, value in (
                    ('loss_lut_smooth', self.smooth_factor, smooth),
                    ('loss_lut_mono', self.monotonicity_factor, mono),
                    ('loss_global_magnitude', self.global_magnitude_factor, magnitude)):
                if factor:
                    losses[name] = factor*value
        if self.local_magnitude_factor:
            losses['loss_local_magnitude'] = self.local_magnitude_factor*aux['context_delta'].square().mean()
        if self.local_smoothness_factor:
            smooth = output.new_zeros(())
            for axis in (-1, -2):
                difference = torch.diff(aux['low_coefficients'], dim=axis)
                edge = torch.diff(aux['local_reference'], dim=axis).abs().mean(1, keepdim=True)
                smooth = smooth+(difference.square()*torch.exp(-self.edge_smoothness_scale*edge)).mean()
            losses['loss_local_smooth'] = self.local_smoothness_factor*smooth
        losses.update(
            lut_weight_abs_mean=weights.detach().abs().mean(),
            lut_weight_batch_std=weights.detach().std(0, unbiased=False).mean(),
            local_coefficient_spatial_std=aux['low_coefficients'].detach().flatten(2).std(-1, unbiased=False).mean(),
            local_correction_abs_mean=aux['local_delta'].detach().abs().mean(),
            global_change=(aux['global_output']-lq).detach().abs().mean(),
            output_change=(output-lq).detach().abs().mean(),
            target_change=(gt-lq).detach().abs().mean())
        return dict(losses=losses, num_samples=len(gt),
                    results=dict(lq=lq.detach().cpu(), gt=gt.detach().cpu(), output=output.detach().cpu()))

    def forward_test(self, lq, gt=None, meta=None, save_image=False,
                     save_path=None, iteration=None):
        if isinstance(lq, (list, tuple)):
            if not lq:
                raise ValueError('evaluation batch must contain images')
            if gt is not None and len(gt) != len(lq):
                raise ValueError('evaluation requires one GT per input image')
            if meta is not None and len(meta) != len(lq):
                raise ValueError('evaluation requires one metadata entry per image')

            def image_tensor(image):
                # MMCV's CPU scatter adds a singleton batch dimension.
                if image.ndim == 4 and image.size(0) == 1:
                    image = image[0]
                if image.ndim != 3:
                    raise ValueError('evaluation images must have CHW shape')
                return image

            images = [image_tensor(image) for image in lq]
            targets = [image_tensor(image) for image in gt] if gt is not None else None
            groups = {}
            for index, image in enumerate(images):
                if targets is not None and targets[index].shape != image.shape:
                    raise ValueError('evaluation input and GT sizes must match')
                groups.setdefault(tuple(image.shape), []).append(index)
            results = [None] * len(images)
            for indices in groups.values():
                group_results = self.forward_test(
                    torch.stack([images[index] for index in indices]),
                    gt=(torch.stack([targets[index] for index in indices])
                        if targets is not None else None),
                    meta=([meta[index] for index in indices] if meta is not None else None),
                    save_image=save_image, save_path=save_path, iteration=iteration)
                if isinstance(group_results, dict):
                    group_results = [group_results]
                for index, result in zip(indices, group_results):
                    results[index] = result
            return results

        if meta is not None and len(meta) != lq.size(0):
            raise ValueError('evaluation requires one metadata entry per image')
        if gt is not None and gt.shape != lq.shape:
            raise ValueError('evaluation input and GT sizes must match')
        output, _, _ = self.forward_dummy(lq)
        results = []
        for index in range(lq.size(0)):
            sample_output = output[index:index + 1]
            sample_gt = gt[index:index + 1] if gt is not None else None
            sample_meta = meta[index:index + 1] if meta is not None else None
            if self.test_cfg is not None and self.test_cfg.get('metrics', None):
                if sample_gt is None:
                    raise ValueError('evaluation with metrics requires gt images')
                result = dict(eval_result=self.evaluate(
                    sample_output, sample_gt, meta=sample_meta))
            else:
                result = dict(lq=lq[index:index + 1].cpu(), output=sample_output.cpu())
                if sample_gt is not None:
                    result['gt'] = sample_gt.cpu()

            if save_image:
                if not sample_meta or not sample_meta[0].get('lq_path'):
                    raise ValueError('saving evaluation images requires lq_path metadata')
                folder_name = osp.splitext(osp.basename(sample_meta[0]['lq_path']))[0]
                if isinstance(iteration, numbers.Number):
                    image_path = osp.join(
                        save_path, folder_name,
                        '{}-{:06d}.png'.format(folder_name, iteration + 1))
                elif iteration is None:
                    image_path = osp.join(save_path, '{}.png'.format(folder_name))
                else:
                    raise ValueError('iteration must be a number or None')
                mmcv.imwrite(tensor2img(sample_output), image_path)
            results.append(result)
        return results[0] if len(results) == 1 else results

    def train_step(self, data_batch, optimizer):
        frozen = self.cnt_iters.item() < self.n_fix_iters
        if frozen != self.adaint_fixed:
            self.adaint.requires_grad_(not frozen)
            self.adaint_fixed = frozen
            get_root_logger().info('%s AdaInt at iteration %s',
                                   'Freeze' if frozen else 'Unfreeze', int(self.cnt_iters.item()))
        outputs = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(outputs.pop('losses'))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.cnt_iters += 1
        outputs.update(log_vars=log_vars)
        return outputs

    def val_step(self, data_batch, **kwargs):
        return self.forward_test(**data_batch, **kwargs)

    def evaluate(self, output, gt, meta=None):
        crop_border = self.test_cfg.crop_border
        eval_result = dict()
        color_metrics = {'deltaE_P3', 'deltaE_P3_top'}
        if color_metrics.intersection(self.test_cfg.metrics):
            # Evaluate the displayed gamut; train on the raw output without
            # clamping. Keep floating-point RGB and avoid an 8-bit round trip.
            pred, target = output.detach().clamp(0, 1), gt.detach().clamp(0, 1)
            if crop_border:
                pred = pred[..., crop_border:-crop_border, crop_border:-crop_border]
                target = target[..., crop_border:-crop_border, crop_border:-crop_border]
            mean, hard = self.color_loss.statistics(pred, target)
            for metric, value in (('deltaE_P3', mean), ('deltaE_P3_top', hard)):
                if metric in self.test_cfg.metrics:
                    eval_result[metric] = value.mean().item()
        output = tensor2img(output)
        gt = tensor2img(gt)
        mask = None
        if any(metric in ('PSNR_hc', 'deltaE_hc') for metric in self.test_cfg.metrics):
            if not meta or len(meta) != 1 or not meta[0].get('mask_path'):
                raise ValueError('HC metrics require mask_path metadata for each image')
            mask = mmcv.imread(meta[0]['mask_path'], flag='grayscale', backend='cv2')
            if mask is None:
                raise FileNotFoundError(meta[0]['mask_path'])
        for metric in self.test_cfg.metrics:
            if metric in color_metrics:
                continue
            if metric in ('PSNR_hc', 'deltaE_hc'):
                eval_result[metric] = self.allowed_metrics[metric](
                    output, gt, mask, crop_border)
            else:
                eval_result[metric] = self.allowed_metrics[metric](
                    output, gt, crop_border)
        return eval_result
