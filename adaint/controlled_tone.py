"""Constrained global mappings and a small, achromatic spatial correction."""
import math

import torch
from torch import nn

from .modules import ConditionEncoder, PredictionHead, identity_lut
from .semantic_modules import RegionMapping, DualLocalAffineDecoder
from .safe_affine import decode_p3, encode_p3


class BoundedExpertLUT(nn.Module):
    """Convex mixing of bounded experts, anchored to the actual AdaInt nodes.

    No extra residual bank is added after the convex mixture. Distinct small
    initial experts break symmetry without the double-zero product problem.
    """
    def __init__(self, n_vertices, n_ranks, condition_dim, hidden_dim,
                 expert_limit=.4):
        super().__init__()
        if not math.isfinite(expert_limit) or not 0 < expert_limit <= 1:
            raise ValueError('expert_limit must be in (0,1]')
        self.expert_limit = float(expert_limit)
        self.head = PredictionHead(condition_dim, hidden_dim, n_ranks)
        nn.init.zeros_(self.head.output.weight)
        nn.init.zeros_(self.head.output.bias)
        self.expert_raw = nn.Parameter(torch.randn(n_ranks, 3, n_vertices,
                                                   n_vertices, n_vertices) * .02)

    def forward(self, condition, vertices):
        weights = self.head(condition).softmax(-1)
        experts = (identity_lut(vertices)[:, None] +
                   self.expert_limit * self.expert_raw.tanh()[None]).clamp(0, 1)
        lut = (weights[:, :, None, None, None, None] * experts).sum(1)
        return weights, lut


class SharedRegionMapping(nn.Module):
    """One whole-image mapping plus a low-rank, bounded foreground difference."""
    def __init__(self, cnn_features, stats_features, condition_dim, hidden_dim,
                 n_vertices, n_ranks, curve_points, min_interval, lut_mode,
                 residual_rank=3, residual_limit=.08):
        super().__init__()
        if lut_mode != 'bounded':
            raise ValueError('shared mapping requires bounded experts')
        if residual_rank < 1 or not 0 < residual_limit <= 1:
            raise ValueError('invalid shared foreground residual settings')
        self.base = RegionMapping(cnn_features, stats_features, condition_dim,
            hidden_dim, n_vertices, n_ranks, curve_points, min_interval, 'bounded')
        self.foreground_encoder = ConditionEncoder(cnn_features,
                                                    stats_features * 2 + 2, condition_dim)
        self.residual_head = PredictionHead(condition_dim, hidden_dim, residual_rank)
        nn.init.normal_(self.residual_head.output.weight, std=1e-4)
        nn.init.zeros_(self.residual_head.output.bias)
        self.residual_basis = nn.Parameter(torch.randn(residual_rank, 3,
            n_vertices, n_vertices, n_vertices) * .02)
        self.residual_rank = int(residual_rank)
        self.residual_limit = float(residual_limit)

    @property
    def adaint(self):
        return self.base.adaint

    def forward(self, cnn, global_stats, foreground_stats, area, reliability):
        # The base condition is whole-image information, not background-only
        # information. Foreground and background share exactly the same nodes.
        base_condition, weights, vertices, base = self.base(cnn, global_stats,
            global_stats, torch.zeros_like(area), torch.ones_like(reliability))
        foreground_condition = self.foreground_encoder(cnn, torch.cat(
            (global_stats, foreground_stats, area, reliability), 1))
        coefficients = self.residual_head(foreground_condition).tanh()
        difference = self.residual_limit * torch.einsum('bm,mcijk->bcijk',
            coefficients, self.residual_basis.tanh()) / self.residual_rank
        # Clamp the actual foreground table, not only individual coefficients.
        # The difference before clipping is bounded by residual_limit.
        foreground = (base + difference).clamp(0, 1)
        return ([foreground_condition, base_condition], [weights, weights],
                [vertices, vertices], [foreground, base])


class ExposureLocalDecoder(DualLocalAffineDecoder):
    """Single scale, one scalar per branch; no RGB offsets or refinement heads."""
    def __init__(self, *args):
        super().__init__(*args)
        channels = self.outputs[0].in_channels
        self.outputs = nn.ModuleList([nn.Conv2d(channels, 1, 1, bias=False)
                                      for _ in range(2)])
        for output in self.outputs:
            nn.init.zeros_(output.weight)


def exposure_correction(global_rgb, field, mean_tanh, active, limit):
    """Bound log exposure AFTER guidance, and cap positive gain by headroom.

    Centering log exposure does not imply zero-mean RGB correction. Every
    channel shares the same linear gain. A zero field is an exact identity.
    """
    log_gain = .5 * limit * (field.tanh() - mean_tanh) * active
    linear = decode_p3(global_rgb)
    headroom = -linear.amax(1, keepdim=True).clamp_min(1e-12).log()
    log_gain = torch.minimum(log_gain, headroom.clamp_min(0))
    corrected = encode_p3(linear * log_gain.exp())
    delta = corrected - global_rgb
    # Remove floating round-trip noise at exact identity without killing the
    # gradient of the zero-initialized exposure head.
    return delta - torch.where(log_gain == 0, delta.detach(), torch.zeros_like(delta))
