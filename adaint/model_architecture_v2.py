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

from .model import Res18Backbone, TPAMIBackbone
from .metrics_hc import delta_e_hc, psnr_hc


class PredictionHead(nn.Module):
    """A small task-specific head used after the shared condition encoder."""

    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LeakyReLU(0.2, inplace=True))
        self.output = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.output(self.hidden(x))


class ColorStatistics(nn.Module):
    """Extract differentiable color-distribution features from the input."""

    def __init__(self, bins=16, pool_size=64):
        super().__init__()
        if bins < 2:
            raise ValueError('stats_bins must be at least 2')
        if pool_size < 8:
            raise ValueError('stats_pool_size must be at least 8')
        self.bins = int(bins)
        self.pool_size = int(pool_size)
        self.output_features = 4 * self.bins + 18
        self.register_buffer('hist_centers', torch.linspace(0, 1, self.bins))

    def forward(self, imgs):
        small = F.adaptive_avg_pool2d(
            imgs, (self.pool_size, self.pool_size)).clamp(0, 1)
        luminance = (0.2126 * small[:, 0:1] +
                     0.7152 * small[:, 1:2] +
                     0.0722 * small[:, 2:3])
        values = torch.cat((small, luminance), dim=1)

        centers = self.hist_centers.to(dtype=values.dtype).view(1, 1, -1, 1, 1)
        bin_width = 1.0 / (self.bins - 1)
        hist = F.relu(1.0 - (values.unsqueeze(2) - centers).abs() / bin_width)
        hist = hist.mean(dim=(-1, -2))
        hist = hist / hist.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        mean = values.mean(dim=(-1, -2))
        std = values.std(dim=(-1, -2), unbiased=False)
        dark = (values < 0.05).to(values.dtype).mean(dim=(-1, -2))
        bright = (values > 0.95).to(values.dtype).mean(dim=(-1, -2))
        saturation = small.max(dim=1)[0] - small.min(dim=1)[0]
        saturation_stats = torch.stack(
            (saturation.mean(dim=(-1, -2)),
             saturation.std(dim=(-1, -2), unbiased=False)), dim=-1)
        return torch.cat((hist.flatten(1), mean, std, dark, bright,
                          saturation_stats), dim=1)


class ConditionEncoder(nn.Module):
    """Fuse CNN context with raw color statistics without erasing either."""

    def __init__(self, cnn_features, stats_features, condition_dim):
        super().__init__()
        self.cnn_projection = nn.Linear(cnn_features, condition_dim)
        self.stats_projection = nn.Linear(stats_features, condition_dim)
        self.fusion = nn.Sequential(
            nn.Linear(condition_dim * 2, condition_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(condition_dim, condition_dim))
        self.norm = nn.LayerNorm(condition_dim)

    def forward(self, cnn_code, stats):
        cnn_code = self.cnn_projection(cnn_code)
        stats = self.stats_projection(stats)
        fused = self.fusion(torch.cat((cnn_code, stats), dim=1))
        return self.norm(cnn_code + stats + fused)


class ToneCurve(nn.Module):
    """Image-adaptive monotone 1D curves applied before the residual 3D LUT."""

    def __init__(self, n_colors, curve_points, condition_dim, hidden_dim,
                 flexible_endpoints=True, endpoint_range=0.05):
        super().__init__()
        if curve_points < 2:
            raise ValueError('curve_points must be at least 2')
        if not 0 <= endpoint_range < 0.49:
            raise ValueError('tone_endpoint_range must be in [0, 0.49)')
        self.n_colors = int(n_colors)
        self.curve_points = int(curve_points)
        self.flexible_endpoints = bool(flexible_endpoints)
        self.endpoint_range = float(endpoint_range)
        endpoint_params = 2 if self.flexible_endpoints else 0
        self.head = PredictionHead(
            condition_dim, hidden_dim,
            self.n_colors * (self.curve_points + endpoint_params))
        nn.init.zeros_(self.head.output.weight)
        nn.init.zeros_(self.head.output.bias)

    def predict_curves(self, condition):
        raw = self.head(condition).view(condition.shape[0], self.n_colors, -1)
        increments = raw[..., :self.curve_points].softmax(dim=-1)
        normalized = F.pad(increments.cumsum(dim=-1), (1, 0), value=0.0)
        if not self.flexible_endpoints:
            return normalized

        endpoints = raw[..., self.curve_points:]
        low = self.endpoint_range * torch.tanh(endpoints[..., 0:1])
        high = 1.0 + self.endpoint_range * torch.tanh(endpoints[..., 1:2])
        return low + normalized * (high - low)

    def apply_curves(self, imgs, curves):
        position = imgs.clamp(0, 1) * self.curve_points
        left = position.floor().long().clamp(max=self.curve_points - 1)
        fraction = position - left.to(position.dtype)
        flat_left = left.flatten(2)
        value_left = curves.gather(2, flat_left).view_as(imgs)
        value_right = curves.gather(2, flat_left + 1).view_as(imgs)
        return value_left * (1.0 - fraction) + value_right * fraction

    def forward(self, condition, imgs):
        curves = self.predict_curves(condition)
        return self.apply_curves(imgs, curves), curves


class AdaIntV2(nn.Module):
    """Adaptive intervals predicted by a nonlinear head with optional floor."""

    def __init__(self, n_colors, n_vertices, condition_dim, hidden_dim,
                 adaint_share=False, use_interval_floor=True,
                 min_interval=0.001):
        super().__init__()
        repeat_factor = 1 if adaint_share else n_colors
        self.n_colors = int(n_colors)
        self.n_vertices = int(n_vertices)
        self.adaint_share = bool(adaint_share)
        self.use_interval_floor = bool(use_interval_floor)
        self.min_interval = float(min_interval)
        n_intervals = self.n_vertices - 1
        if self.min_interval < 0:
            raise ValueError('min_interval must be non-negative')
        if self.use_interval_floor and self.min_interval * n_intervals >= 1:
            raise ValueError('min_interval is too large for n_vertices')
        self.head = PredictionHead(
            condition_dim, hidden_dim, n_intervals * repeat_factor)
        nn.init.zeros_(self.head.output.weight)
        nn.init.zeros_(self.head.output.bias)

    def forward(self, condition):
        intervals = self.head(condition).view(
            condition.shape[0], -1, self.n_vertices - 1).softmax(dim=-1)
        if self.use_interval_floor:
            remaining = 1.0 - self.min_interval * (self.n_vertices - 1)
            intervals = self.min_interval + remaining * intervals
        if self.adaint_share:
            intervals = intervals.repeat_interleave(self.n_colors, dim=1)
        return F.pad(intervals.cumsum(dim=-1), (1, 0), value=0.0)


class SignedLowRankLUT(nn.Module):
    """Generate several residual LUTs from a bounded signed basis mixture."""

    def __init__(self, n_colors, n_vertices, coarse_vertices, n_ranks,
                 n_spatial_luts, condition_dim, hidden_dim,
                 vertex_embedding_dim, signed_weight_scale=1.0,
                 expert_init_scale=0.01):
        super().__init__()
        self.n_colors = int(n_colors)
        self.n_vertices = int(n_vertices)
        self.coarse_vertices = int(coarse_vertices or n_vertices)
        self.n_ranks = int(n_ranks)
        self.n_spatial_luts = int(n_spatial_luts)
        self.signed_weight_scale = float(signed_weight_scale)
        self.expert_init_scale = float(expert_init_scale)
        if not 2 <= self.coarse_vertices <= self.n_vertices:
            raise ValueError('coarse_vertices must be between 2 and n_vertices')
        if self.n_ranks < 1:
            raise ValueError('n_ranks must be positive')
        if self.n_spatial_luts < 2:
            raise ValueError('n_spatial_luts must be at least 2')
        if self.signed_weight_scale <= 0 or self.expert_init_scale <= 0:
            raise ValueError('signed_weight_scale and expert_init_scale must be positive')

        self.vertex_encoder = nn.Sequential(
            nn.Linear(self.n_colors * self.n_vertices, vertex_embedding_dim),
            nn.LeakyReLU(0.2, inplace=True))
        self.router = PredictionHead(
            condition_dim + vertex_embedding_dim, hidden_dim,
            self.n_spatial_luts * self.n_ranks)
        nn.init.normal_(self.router.output.weight, std=1e-3)
        nn.init.zeros_(self.router.output.bias)
        self.basis_luts = nn.Parameter(torch.empty(
            self.n_ranks, self.n_colors,
            self.coarse_vertices, self.coarse_vertices, self.coarse_vertices))
        self.init_basis()

    def init_basis(self):
        axis = torch.linspace(-1, 1, self.coarse_vertices)
        r, g, b = torch.meshgrid(axis, axis, axis)
        patterns = torch.stack((r, g, b, r * g, r * b, g * b, r * g * b))
        coeff = torch.randn(self.n_ranks, self.n_colors, patterns.shape[0])
        if self.n_ranks > 1:
            coeff -= coeff.mean(dim=0, keepdim=True)
        residual = torch.einsum('mcp,pxyz->mcxyz', coeff, patterns)
        residual *= self.expert_init_scale / residual.abs().max().clamp_min(1e-8)
        with torch.no_grad():
            self.basis_luts.copy_(residual)

    def forward(self, condition, vertices):
        vertex_code = self.vertex_encoder(vertices.flatten(1))
        logits = self.router(torch.cat((condition, vertex_code), dim=1))
        weights = self.signed_weight_scale * torch.tanh(
            logits.view(condition.shape[0], self.n_spatial_luts, self.n_ranks))
        residual_luts = torch.einsum(
            'bkm,mcxyz->bkcxyz', weights, self.basis_luts)
        if self.coarse_vertices != self.n_vertices:
            residual_luts = F.interpolate(
                residual_luts.reshape(
                    -1, self.n_colors,
                    self.coarse_vertices, self.coarse_vertices,
                    self.coarse_vertices),
                size=(self.n_vertices,) * 3,
                mode='trilinear', align_corners=True)
            residual_luts = residual_luts.view(
                condition.shape[0], self.n_spatial_luts, self.n_colors,
                self.n_vertices, self.n_vertices, self.n_vertices)
        return weights, residual_luts

    def regularizations(self, vertices, residual_luts, smoothness,
                        monotonicity, diagonal_monotonicity=True):
        tv = residual_luts.new_zeros(())
        for axis in (3, 4, 5):
            tv = tv + torch.diff(residual_luts, dim=axis).square().mean()

        batch = vertices.shape[0]
        identity = residual_luts.new_empty(
            batch, self.n_colors,
            self.n_vertices, self.n_vertices, self.n_vertices)
        identity[:, 0] = vertices[:, 0].view(batch, 1, 1, self.n_vertices)
        identity[:, 1] = vertices[:, 1].view(batch, 1, self.n_vertices, 1)
        identity[:, 2] = vertices[:, 2].view(batch, self.n_vertices, 1, 1)
        final_luts = identity.unsqueeze(1) + residual_luts

        mono = residual_luts.new_zeros(())
        if diagonal_monotonicity:
            for color in range(self.n_colors):
                axis = 5 - color
                derivative = torch.diff(
                    final_luts[:, :, color:color + 1], dim=axis)
                mono = mono + F.relu(-derivative).mean()
        else:
            for axis in (3, 4, 5):
                mono = mono + F.relu(
                    -torch.diff(final_luts, dim=axis)).mean()
        return smoothness * tv, monotonicity * mono


class SpatialGuide(nn.Module):
    """Predict a low-resolution context map for spatial LUT blending."""

    def __init__(self, condition_dim, n_spatial_luts, resolution=128,
                 channels=16, temperature=1.0):
        super().__init__()
        if resolution < 32:
            raise ValueError('guide_resolution must be at least 32')
        if channels < 4:
            raise ValueError('guide_channels must be at least 4')
        self.resolution = int(resolution)
        self.temperature = max(float(temperature), 1e-4)
        self.features = nn.Sequential(
            nn.Conv2d(3, channels, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels * 2, channels * 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))
        self.condition_projection = nn.Linear(condition_dim, channels * 2)
        self.output = nn.Conv2d(channels * 2, n_spatial_luts, 1)
        nn.init.normal_(self.output.weight, std=1e-3)
        nn.init.zeros_(self.output.bias)

    def forward(self, imgs, condition):
        guide_input = F.interpolate(
            imgs, size=(self.resolution, self.resolution),
            mode='bilinear', align_corners=False)
        features = self.features(guide_input)
        features = features + self.condition_projection(condition).view(
            condition.shape[0], -1, 1, 1)
        low_resolution = torch.softmax(
            self.output(features) / self.temperature, dim=1)
        full_resolution = F.interpolate(
            low_resolution, size=imgs.shape[-2:],
            mode='bilinear', align_corners=False)
        return full_resolution, low_resolution


@MODELS.register_module()
class AiLUTArchitectureV2(BaseModel):
    """Statistics-conditioned, spatially adaptive cascaded LUT enhancer."""

    allowed_metrics = {'PSNR': psnr, 'SSIM': ssim, 'deltaE': delta_e,
                       'PSNR_hc': psnr_hc, 'deltaE_hc': delta_e_hc}

    def __init__(self,
                 n_ranks=4,
                 n_vertices=33,
                 coarse_vertices=17,
                 curve_points=16,
                 stats_bins=16,
                 stats_pool_size=64,
                 condition_dim=256,
                 head_hidden_dim=128,
                 vertex_embedding_dim=64,
                 n_spatial_luts=2,
                 guide_resolution=128,
                 guide_channels=16,
                 guide_temperature=1.0,
                 signed_weight_scale=1.0,
                 expert_init_scale=0.01,
                 use_interval_floor=True,
                 min_interval=0.001,
                 use_flexible_tone_endpoints=True,
                 tone_endpoint_range=0.05,
                 use_diagonal_monotonicity=True,
                 use_spatial_smoothness=True,
                 spatial_smoothness_factor=0.001,
                 en_adaint=True,
                 en_adaint_share=False,
                 backbone='tpami',
                 pretrained=False,
                 n_colors=3,
                 sparse_factor=0.0,
                 smooth_factor=0.0,
                 monotonicity_factor=10.0,
                 recons_loss=dict(
                     type='L1Loss', loss_weight=1.0, reduction='mean'),
                 gen=0,
                 kd_loss=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        if n_colors != 3:
            raise ValueError('AiLUTArchitectureV2 currently requires RGB input')
        if backbone.lower() not in ('tpami', 'res18'):
            raise ValueError('backbone must be tpami or res18')

        self.backbone_name = backbone.lower()
        self.backbone = dict(
            tpami=TPAMIBackbone,
            res18=Res18Backbone)[self.backbone_name](
                pretrained, extra_pooling=en_adaint)
        self.color_statistics = ColorStatistics(stats_bins, stats_pool_size)
        self.condition_encoder = ConditionEncoder(
            self.backbone.out_channels,
            self.color_statistics.output_features,
            condition_dim)
        self.tone_curve = ToneCurve(
            n_colors, curve_points, condition_dim, head_hidden_dim,
            flexible_endpoints=use_flexible_tone_endpoints,
            endpoint_range=tone_endpoint_range)

        self.en_adaint = bool(en_adaint)
        if self.en_adaint:
            self.adaint = AdaIntV2(
                n_colors, n_vertices, condition_dim, head_hidden_dim,
                adaint_share=en_adaint_share,
                use_interval_floor=use_interval_floor,
                min_interval=min_interval)
        else:
            uniform_vertices = torch.arange(n_vertices).div(n_vertices - 1)
            self.register_buffer(
                'uniform_vertices',
                uniform_vertices.repeat(n_colors, 1).unsqueeze(0))

        self.lut_generator = SignedLowRankLUT(
            n_colors, n_vertices, coarse_vertices, n_ranks,
            n_spatial_luts, condition_dim, head_hidden_dim,
            vertex_embedding_dim,
            signed_weight_scale=signed_weight_scale,
            expert_init_scale=expert_init_scale)
        self.spatial_guide = SpatialGuide(
            condition_dim, n_spatial_luts,
            resolution=guide_resolution,
            channels=guide_channels,
            temperature=guide_temperature)

        self.n_colors = int(n_colors)
        self.n_vertices = int(n_vertices)
        self.n_spatial_luts = int(n_spatial_luts)
        self.sparse_factor = float(sparse_factor)
        self.smooth_factor = float(smooth_factor)
        self.monotonicity_factor = float(monotonicity_factor)
        self.use_diagonal_monotonicity = bool(use_diagonal_monotonicity)
        self.use_spatial_smoothness = bool(use_spatial_smoothness)
        self.spatial_smoothness_factor = float(spatial_smoothness_factor)
        self.gen = int(gen)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False

        self.init_weights()
        self.recons_loss = build_loss(recons_loss)
        self.kd_loss = None
        if self.gen > 0 and kd_loss is not None:
            self.kd_loss = build_loss(kd_loss)

        self.n_fix_iters = train_cfg.get('n_fix_iters', 0) if train_cfg else 0
        self.adaint_fixed = False
        self.register_buffer('cnt_iters', torch.zeros(1))

    def init_weights(self):
        def initialize_backbone(module):
            classname = module.__class__.__name__
            if 'Conv' in classname:
                nn.init.xavier_normal_(module.weight.data)
                if module.bias is not None:
                    nn.init.zeros_(module.bias.data)
            elif 'InstanceNorm' in classname and module.affine:
                nn.init.normal_(module.weight.data, 1.0, 0.02)
                nn.init.zeros_(module.bias.data)

        if self.backbone_name != 'res18':
            self.backbone.apply(initialize_backbone)

    def _apply_spatial_luts(self, imgs, residual_luts, vertices, guide):
        batch, _, height, width = imgs.shape
        expanded_imgs = imgs.unsqueeze(1).expand(
            -1, self.n_spatial_luts, -1, -1, -1).contiguous().view(
                batch * self.n_spatial_luts, self.n_colors, height, width)
        expanded_vertices = vertices.unsqueeze(1).expand(
            -1, self.n_spatial_luts, -1, -1).contiguous().view(
                batch * self.n_spatial_luts,
                self.n_colors, self.n_vertices)
        flat_luts = residual_luts.contiguous().view(
            batch * self.n_spatial_luts, self.n_colors,
            self.n_vertices, self.n_vertices, self.n_vertices)
        residuals = ailut_transform(
            expanded_imgs, flat_luts, expanded_vertices).view(
                batch, self.n_spatial_luts,
                self.n_colors, height, width)
        return (residuals * guide.unsqueeze(2)).sum(dim=1)

    def forward_dummy(self, imgs, return_aux=False):
        cnn_code = self.backbone(imgs)
        statistics = self.color_statistics(imgs)
        condition = self.condition_encoder(cnn_code, statistics)
        tone_output, curves = self.tone_curve(condition, imgs)

        if self.en_adaint:
            vertices = self.adaint(condition)
        else:
            vertices = self.uniform_vertices.expand(imgs.shape[0], -1, -1)
        weights, residual_luts = self.lut_generator(condition, vertices)
        guide, low_resolution_guide = self.spatial_guide(imgs, condition)
        residual = self._apply_spatial_luts(
            tone_output.clamp(0, 1), residual_luts, vertices, guide)
        output = tone_output + residual

        if return_aux:
            aux = dict(
                residual_luts=residual_luts,
                guide=guide,
                low_resolution_guide=low_resolution_guide,
                curves=curves,
                statistics=statistics)
            return output, weights, vertices, aux
        return output, weights, vertices

    @auto_fp16(apply_to=('lq', ))
    def forward(self, lq, gt=None, test_mode=False, **kwargs):
        if test_mode:
            return self.forward_test(lq, gt, **kwargs)
        return self.forward_train(lq, gt)

    def forward_train(self, lq, gt):
        losses = dict()
        output, weights, vertices, aux = self.forward_dummy(
            lq, return_aux=True)
        losses['loss_recons'] = self.recons_loss(output, gt)
        if self.kd_loss is not None:
            losses['loss_kd'] = self.kd_loss(output, lq)
        if self.sparse_factor > 0:
            losses['loss_sparse'] = self.sparse_factor * weights.abs().mean()

        reg_smoothness, reg_monotonicity = self.lut_generator.regularizations(
            vertices, aux['residual_luts'],
            self.smooth_factor, self.monotonicity_factor,
            diagonal_monotonicity=self.use_diagonal_monotonicity)
        if self.smooth_factor > 0:
            losses['loss_smooth'] = reg_smoothness
        if self.monotonicity_factor > 0:
            losses['loss_mono'] = reg_monotonicity
        if self.use_spatial_smoothness and self.spatial_smoothness_factor > 0:
            guide = aux['low_resolution_guide']
            guide_tv = (torch.diff(guide, dim=-1).abs().mean() +
                        torch.diff(guide, dim=-2).abs().mean())
            losses['loss_guide_smooth'] = (
                self.spatial_smoothness_factor * guide_tv)

        return dict(
            losses=losses,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), gt=gt.cpu(), output=output.cpu()))

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
        if self.en_adaint and self.cnt_iters < self.n_fix_iters:
            if not self.adaint_fixed:
                self.adaint_fixed = True
                self.adaint.requires_grad_(False)
                get_root_logger().info(
                    'Fix AdaInt for {} iters.'.format(self.n_fix_iters))
        elif self.en_adaint and self.cnt_iters == self.n_fix_iters:
            self.adaint.requires_grad_(True)
            if self.adaint_fixed:
                self.adaint_fixed = False
                get_root_logger().info(
                    'Unfix AdaInt after {} iters.'.format(self.n_fix_iters))

        outputs = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(outputs.pop('losses'))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        outputs.update({'log_vars': log_vars})
        self.cnt_iters += 1
        return outputs

    def val_step(self, data_batch, **kwargs):
        return self.forward_test(**data_batch, **kwargs)

    def evaluate(self, output, gt, meta=None):
        crop_border = self.test_cfg.crop_border
        output = tensor2img(output)
        gt = tensor2img(gt)
        mask = None
        if any(metric in ('PSNR_hc', 'deltaE_hc') for metric in self.test_cfg.metrics):
            if not meta or len(meta) != 1 or not meta[0].get('mask_path'):
                raise ValueError('HC metrics require mask_path metadata for each image')
            mask = mmcv.imread(meta[0]['mask_path'], flag='grayscale', backend='cv2')
            if mask is None:
                raise FileNotFoundError(meta[0]['mask_path'])
        eval_result = dict()
        for metric in self.test_cfg.metrics:
            if metric in ('PSNR_hc', 'deltaE_hc'):
                eval_result[metric] = self.allowed_metrics[metric](
                    output, gt, mask, crop_border)
            else:
                eval_result[metric] = self.allowed_metrics[metric](
                    output, gt, crop_border)
        return eval_result
