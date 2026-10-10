"""Display P3/D65 color supervision; no learned parameters or inference layers."""
import math
import numbers

import torch
import torch.nn as nn


def display_p3_to_lab(rgb):
    """Encoded RGB NCHW -> CIE Lab with a D65 reference white.

    Display P3 uses the sRGB transfer curve, but different primaries. Matrix
    and transfer definitions: https://www.w3.org/TR/css-color-4/#color-conversion-code
    Both XYZ and Lab use D65 here; this is not CSS's D50-adapted Lab.
    Extend the transfer curve by reflection outside the gamut, and the Lab
    linear branch below its threshold, so predictions retain useful gradients.
    """
    if rgb.ndim != 4 or rgb.size(1) != 3 or not rgb.is_floating_point():
        raise ValueError('Display P3 conversion requires floating-point NCHW RGB')
    # Keep double precision for numeric checks; use float32 for half inputs.
    if rgb.dtype in (torch.float16, torch.bfloat16):
        rgb = rgb.float()
    magnitude = rgb.abs()
    nonlinear = rgb.sign() * ((magnitude + 0.055) / 1.055).pow(2.4)
    linear = torch.where(magnitude <= 0.04045, rgb / 12.92, nonlinear)
    matrix = rgb.new_tensor([
        [608311 / 1250200, 189793 / 714400, 198249 / 1000160],
        [35783 / 156275, 247089 / 357200, 198249 / 2500400],
        [0., 32229 / 714400, 5220557 / 5000800]])
    xyz = torch.einsum('ij,bjhw->bihw', matrix, linear)
    # D65 chromaticity x=0.3127, y=0.3290, consistent with the P3 matrix.
    white = rgb.new_tensor([0.3127 / 0.3290, 1.,
                            (1. - 0.3127 - 0.3290) / 0.3290])
    relative = xyz / white.view(1, 3, 1, 1)
    threshold = (6. / 29.) ** 3
    # Evaluate a safe inactive power branch too: torch.where evaluates both.
    f = torch.where(relative > threshold,
                    relative.clamp_min(threshold).pow(1. / 3.),
                    relative * (841. / 108.) + 4. / 29.)
    x, y, z = f.unbind(1)
    return torch.stack((116. * y - 16., 500. * (x - y),
                        200. * (y - z)), dim=1)


def p3_delta_e_map(pred, target, epsilon=0.):
    """Per-pixel Delta E 76 in Lab units, optionally smoothed at equality."""
    _, error, _ = p3_error_maps(pred, target, epsilon)
    return error


def p3_error_maps(pred, target, epsilon=0.):
    """Compute distances per pixel BEFORE pooling, to avoid color cancellation."""
    if pred.shape != target.shape:
        raise ValueError('color supervision requires aligned prediction and GT')
    difference = display_p3_to_lab(pred) - display_p3_to_lab(target)
    lab = (difference.square().sum(1) + epsilon ** 2).sqrt() - epsilon
    ab = (difference[:, 1:].square().sum(1) + epsilon ** 2).sqrt() - epsilon
    return difference, lab, ab


def grid_cell_means(error_map, grid_size=16):
    """Per-image means of non-overlapping rectangular grid cells.

    Partition the actual paired crop, not the full-image context. Uneven
    dimensions are split at floor(i * size / count); every pixel belongs to
    exactly one cell. Small images use fewer cells rather than duplicate them.
    """
    height, width = error_map.shape[-2:]
    rows, columns = min(grid_size, height), min(grid_size, width)
    row_sizes = [(i + 1) * height // rows - i * height // rows
                 for i in range(rows)]
    col_sizes = [(i + 1) * width // columns - i * width // columns
                 for i in range(columns)]
    row_means = torch.stack([part.mean(-2) for part in
                             torch.split(error_map, row_sizes, dim=-2)], dim=-2)
    cells = torch.stack([part.mean(-1) for part in
                         torch.split(row_means, col_sizes, dim=-1)], dim=-1)
    return cells.flatten(1)


def grid_top_mean(error_map, grid_size=16, top_fraction=0.1):
    cells = grid_cell_means(error_map, grid_size)
    count = max(1, math.ceil(cells.size(1) * top_fraction))
    return cells.topk(count, dim=1).values.mean(1)


class P3LabColorLoss(nn.Module):
    """Global color distance plus a modest worst-grid term, normalized by 100."""
    def __init__(self, grid_size=16, top_fraction=0.1, hard_weight=0.25,
                 hard_mode='lab'):
        super().__init__()
        if (isinstance(grid_size, bool) or
                not isinstance(grid_size, numbers.Integral) or grid_size < 1):
            raise ValueError('color_grid_size must be a positive integer')
        if not 0 < top_fraction <= 1 or not 0 <= hard_weight <= 1:
            raise ValueError('invalid color top fraction or hard-region weight')
        if hard_mode not in ('lab', 'ab'):
            raise ValueError('color_hard_mode must be lab or ab')
        self.grid_size = int(grid_size)
        self.top_fraction = float(top_fraction)
        self.hard_weight = float(hard_weight)
        self.hard_mode = hard_mode

    def statistics(self, pred, target, epsilon=0.):
        # Preserve the meaning of the existing deltaE_P3_top metric in all modes.
        error = p3_delta_e_map(pred, target, epsilon)
        mean = error.mean((-1, -2))
        hard = grid_top_mean(error, self.grid_size, self.top_fraction)
        return mean, hard

    def evaluation_statistics(self, pred, target):
        difference, lab, ab = p3_error_maps(pred, target)
        return dict(deltaE_P3=lab.mean((-1, -2)),
                    deltaE_P3_top=grid_top_mean(lab, self.grid_size, self.top_fraction),
                    chroma_P3=ab.mean((-1, -2)),
                    chroma_P3_top=grid_top_mean(ab, self.grid_size, self.top_fraction),
                    chroma_P3_p90=torch.quantile(ab.flatten(1), .9, dim=1),
                    L_error_P3=difference[:, 0].abs().mean((-1, -2)))

    def forward(self, pred, target):
        difference, lab, ab = p3_error_maps(pred, target, epsilon=1e-3)
        mean = lab.mean((-1, -2))
        error = ab if self.hard_mode == 'ab' else lab
        cells = grid_cell_means(error, self.grid_size)
        count = max(1, math.ceil(cells.size(1) * self.top_fraction))
        selected = cells.topk(count, dim=1)
        hard = selected.values.mean(1)
        loss = ((1. - self.hard_weight) * mean + self.hard_weight * hard).mean() / 100.
        # Inspect the selected training grids without adding another objective.
        with torch.no_grad():
            light = grid_cell_means(difference[:, 0].square(), self.grid_size)
            chroma = grid_cell_means(difference[:, 1:].square().sum(1), self.grid_size)
            light = light.gather(1, selected.indices).mean(1)
            chroma = chroma.gather(1, selected.indices).mean(1)
            light_fraction = light / (light + chroma).clamp_min(1e-12)
        return loss, dict(color_delta_e=mean.detach().mean(),
                          color_delta_e_top=grid_top_mean(lab.detach(), self.grid_size,
                                                         self.top_fraction).mean(),
                          color_chroma=ab.detach().mean(),
                          color_chroma_top=grid_top_mean(ab.detach(), self.grid_size,
                                                        self.top_fraction).mean(),
                          color_hard_L_fraction=light_fraction.mean())
