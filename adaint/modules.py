"""Components of the single global-LUT plus local-affine architecture."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def stratified_pixels(imgs, size):
    """One deterministic centre sample per stratum, without averaging pixels."""
    height, width = imgs.shape[-2:]
    ys = ((torch.arange(size, device=imgs.device).float() + 0.5)
          * height / size).long().clamp(max=height - 1)
    xs = ((torch.arange(size, device=imgs.device).float() + 0.5)
          * width / size).long().clamp(max=width - 1)
    return imgs.index_select(-2, ys).index_select(-1, xs)


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

    def forward(self, imgs, samples=None):
        small = (stratified_pixels(imgs, self.pool_size)
                 if samples is None else samples).clamp(0, 1)
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
    """Monotone RGB prior with fixed endpoints; sampled at AdaInt vertices."""
    def __init__(self, curve_points, condition_dim, hidden_dim):
        super().__init__()
        if curve_points < 2:
            raise ValueError('curve_points must be at least 2')
        self.curve_points = int(curve_points)
        self.head = PredictionHead(condition_dim, hidden_dim, 3*curve_points)
        nn.init.zeros_(self.head.output.weight)
        nn.init.zeros_(self.head.output.bias)

    def predict_curves(self, condition):
        increments = self.head(condition).view(-1, 3, self.curve_points).softmax(-1)
        return F.pad(increments.cumsum(-1), (1, 0), value=0.)

    def apply_curves(self, values, curves):
        positions = values.clamp(0, 1)*self.curve_points
        left = positions.floor().long().clamp(max=self.curve_points-1)
        fraction = positions-left.to(positions.dtype)
        flat = left.flatten(2)
        low = curves.gather(2, flat).view_as(values)
        high = curves.gather(2, flat+1).view_as(values)
        return low*(1-fraction)+high*fraction


class AdaptiveIntervals(nn.Module):
    """Independent RGB coordinates with a strictly positive interval floor."""
    def __init__(self, n_vertices, condition_dim, hidden_dim, min_interval=0.001):
        super().__init__()
        if n_vertices < 2 or not 0 < min_interval < 1.0 / (n_vertices-1):
            raise ValueError('min_interval must be positive and leave interval mass')
        self.n_vertices = int(n_vertices)
        self.min_interval = float(min_interval)
        self.head = PredictionHead(condition_dim, hidden_dim, 3*(n_vertices-1))
        nn.init.zeros_(self.head.output.weight)
        nn.init.zeros_(self.head.output.bias)

    def forward(self, condition):
        intervals = self.head(condition).view(-1, 3, self.n_vertices-1).softmax(-1)
        intervals = self.min_interval+(1-self.min_interval*(self.n_vertices-1))*intervals
        return F.pad(intervals.cumsum(-1), (1, 0), value=0.)


class NormalizedResidualLUT(nn.Module):
    """Full-resolution bases, unit RMS per basis, unconstrained signed weights."""
    def __init__(self, n_vertices, n_ranks, condition_dim, hidden_dim):
        super().__init__()
        if n_vertices < 2 or n_ranks < 1:
            raise ValueError('n_vertices >= 2 and n_ranks >= 1 required')
        self.head = PredictionHead(condition_dim, hidden_dim, n_ranks)
        # Nonzero, distinct bases and small coefficients avoid a double-zero product.
        nn.init.normal_(self.head.output.weight, std=1e-4)
        nn.init.zeros_(self.head.output.bias)
        axis = torch.linspace(-1, 1, n_vertices)
        b, g, r = torch.meshgrid(axis, axis, axis)
        patterns = torch.stack((r, g, b, r*g, r*b, g*b, r*g*b,
                                r.square(), g.square(), b.square()))
        mixing = torch.randn(n_ranks, 3, len(patterns))
        bases = torch.einsum('mcp,pijk->mcijk', mixing, patterns)
        bases = bases / bases.square().mean((1, 2, 3, 4), keepdim=True).sqrt()
        self.basis_luts = nn.Parameter(bases)

    def normalized_bases(self):
        rms = self.basis_luts.square().mean((1, 2, 3, 4), keepdim=True)
        return self.basis_luts / rms.clamp_min(1e-12).sqrt()

    def forward(self, condition):
        weights = self.head(condition)
        residual = torch.einsum('bm,mcijk->bcijk', weights, self.normalized_bases())
        return weights, residual


def vertex_volume_weights(vertices):
    """Trapezoidal integration weights on each nonuniform RGB coordinate axis."""
    intervals = torch.diff(vertices, dim=-1)
    return (F.pad(intervals, (1, 0)) + F.pad(intervals, (0, 1))) * 0.5


def identity_lut(vertices):
    batch, _, count = vertices.shape
    return torch.stack((
        vertices[:, 0, None, None, :].expand(batch, count, count, count),
        vertices[:, 1, None, :, None].expand(batch, count, count, count),
        vertices[:, 2, :, None, None].expand(batch, count, count, count)), dim=1)


def lut_regularizations(lut, vertices):
    """Integrate displacement energy/slopes and diagonal negative slopes in RGB space.

    LUT storage axes are B,G,R; vertices are R,G,B. Width-weighted quadrature
    avoids weighting dense adaptive regions merely by their node count.
    """
    node_weights = vertex_volume_weights(vertices)
    volume = (node_weights[:, 2, :, None, None] *
              node_weights[:, 1, None, :, None] *
              node_weights[:, 0, None, None, :])[:, None]
    displacement = lut - identity_lut(vertices)
    magnitude = (displacement.square() * volume).sum((2, 3, 4)).mean()
    smooth, mono = lut.new_zeros(()), lut.new_zeros(())
    for color, axis in ((0, 4), (1, 3), (2, 2)):
        shape = [vertices.shape[0], 1, 1, 1, 1]
        shape[axis] = vertices.shape[-1] - 1
        widths = torch.diff(vertices[:, color], dim=-1).view(shape)
        safe_widths = widths.clamp_min(1e-8)
        edge_weights = []
        for coordinate in range(3):
            edge_weights.append(torch.diff(vertices[:, coordinate], dim=-1)
                                if coordinate == color else node_weights[:, coordinate])
        measure = (edge_weights[2][:, :, None, None] *
                   edge_weights[1][:, None, :, None] *
                   edge_weights[0][:, None, None, :])[:, None]
        slope = torch.diff(displacement, dim=axis) / safe_widths
        smooth = smooth + (slope.square() * measure).sum((2, 3, 4)).mean()
        diagonal = torch.diff(lut[:, color:color+1], dim=axis) / safe_widths
        mono = mono + (F.relu(-diagonal).square() * measure).sum((2, 3, 4)).mean() / 3
    return smooth, mono, magnitude


class LocalAffineDecoder(nn.Module):
    """Multiscale FiLM-conditioned four-coefficient local correction.

    Filter a full-context field before cropping; high-resolution guidance is
    per-pixel luminance, so crops use the same coordinate system as inference.
    """
    def __init__(self, condition_dim, resolution=128, channels=16,
                 radius=2, epsilon=0.001):
        super().__init__()
        if resolution < 32 or channels < 4 or radius < 1 or epsilon <= 0:
            raise ValueError('invalid local decoder or guided-filter settings')
        self.resolution, self.radius, self.epsilon = int(resolution), int(radius), float(epsilon)
        self.shallow = nn.Sequential(nn.Conv2d(3, channels, 3, 2, 1), nn.LeakyReLU(.2))
        self.deep = nn.Sequential(
            nn.Conv2d(channels, channels*2, 3, 2, 1), nn.LeakyReLU(.2),
            nn.Conv2d(channels*2, channels*2, 3, 1, 1), nn.LeakyReLU(.2))
        self.fusion = nn.Sequential(nn.Conv2d(channels*3, channels*2, 3, 1, 1),
                                    nn.LeakyReLU(.2))
        self.film = nn.Linear(condition_dim, channels*4)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.output = nn.Conv2d(channels*2, 4, 1, bias=False)
        nn.init.zeros_(self.output.weight)

    def box_mean(self, value):
        return F.avg_pool2d(value, 2*self.radius+1, stride=1,
                            padding=self.radius, count_include_pad=False)

    def forward(self, source, condition):
        small = F.interpolate(source, (self.resolution,)*2, mode='bilinear', align_corners=False)
        shallow = self.shallow(small)
        deep = F.interpolate(self.deep(shallow), shallow.shape[-2:], mode='bilinear', align_corners=False)
        features = self.fusion(torch.cat((shallow, deep), dim=1))
        gamma, beta = self.film(condition).chunk(2, dim=1)
        features = F.leaky_relu((1+gamma[:, :, None, None])*features+
                               beta[:, :, None, None], negative_slope=.2)
        low = self.output(features)
        low_source = F.interpolate(source, low.shape[-2:], mode='bilinear', align_corners=False)
        guidance = luminance(low_source)
        mean_g, mean_p = self.box_mean(guidance), self.box_mean(low)
        variance = self.box_mean(guidance.square())-mean_g.square()
        covariance = self.box_mean(guidance*low)-mean_g*mean_p
        slope = covariance/(variance.clamp_min(0)+self.epsilon)
        offset = mean_p-slope*mean_g
        return low, self.box_mean(slope), self.box_mean(offset), low_source

    @staticmethod
    def upsample(slope, offset, target, crop_box=None):
        return (crop_map(slope, crop_box, target.shape[-2:])*luminance(target)+
                crop_map(offset, crop_box, target.shape[-2:]))
