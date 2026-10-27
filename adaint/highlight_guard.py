"""Optional training-only constraints for highlight-safe semantic enhancement.

These functions contain no learned parameters or buffers. RGB and the luminance
proxy are in the existing encoded [0, 1] convention; this module does not change
the Display P3 conversion or the enhancer's forward output. Losses retain raw
predictions so that values outside the display range receive corrective gradients.
"""
import math
import numbers

import torch
import torch.nn.functional as F


DEFAULTS = dict(
    over_weight=0., final_range_weight=0., lut_range_weight=0., keep_weight=0.,
    top_fraction=.1, over_margin=2. / 255., keep_error_threshold=.03,
    reference_config=None, reference_checkpoint=None,
    trainable_scope='all', detach_result_condition=False)


def _real(value, name, minimum=0., maximum=None, minimum_open=False):
    if (isinstance(value, bool) or not isinstance(value, numbers.Real) or
            not math.isfinite(value)):
        raise ValueError(name + ' must be a finite real number')
    if value < minimum or (minimum_open and value == minimum):
        raise ValueError(name + ' is below its allowed range')
    if maximum is not None and value > maximum:
        raise ValueError(name + ' is above its allowed range')
    return float(value)


def guard_options(value=None):
    """Validate opt-in settings without allocating state or changing RNG state.

    ``keep_error_threshold`` is a per-pixel RGB *RMSE* threshold. For example,
    .03 means approximately 7.65/255 RGB RMSE, not an MSE tolerance of .03.
    """
    options = dict(DEFAULTS)
    if value is not None:
        if not isinstance(value, dict):
            raise ValueError('model.highlight_guard must be a mapping')
        unknown = sorted(set(value) - set(options))
        if unknown:
            raise ValueError('unknown highlight_guard options: ' + ', '.join(unknown))
        options.update(value)
    for name in ('over_weight', 'final_range_weight', 'lut_range_weight', 'keep_weight'):
        options[name] = _real(options[name], 'highlight_guard.' + name)
    options['top_fraction'] = _real(options['top_fraction'], 'highlight_guard.top_fraction',
                                    maximum=1., minimum_open=True)
    for name in ('over_margin', 'keep_error_threshold'):
        options[name] = _real(options[name], 'highlight_guard.' + name, maximum=1.)
    if options['trainable_scope'] not in ('all', 'heads'):
        raise ValueError("highlight_guard.trainable_scope must be 'all' or 'heads'")
    if not isinstance(options['detach_result_condition'], bool):
        raise ValueError('highlight_guard.detach_result_condition must be boolean')
    for name in ('reference_config', 'reference_checkpoint'):
        path = options[name]
        if path is not None and (not isinstance(path, str) or not path.strip()):
            raise ValueError('highlight_guard.' + name + ' must be a nonempty path or null')
    if options['keep_weight'] and not all(options[name] for name in
                                           ('reference_config', 'reference_checkpoint')):
        raise ValueError('keep_weight > 0 requires reference_config and reference_checkpoint')
    return options


def _image(value, name):
    if (not torch.is_tensor(value) or not value.is_floating_point() or
            value.ndim != 4 or value.size(1) != 3):
        raise ValueError(name + ' must be a floating-point B,3,H,W tensor')
    if not value.numel():
        raise ValueError(name + ' must not be empty')
    # Quantiles and squared RGB errors should not be accumulated in float16.
    return value.float() if value.dtype in (torch.float16, torch.bfloat16) else value


def _mask(mask, image):
    if mask is None:
        return torch.ones_like(image[:, :1])
    if not torch.is_tensor(mask) or mask.shape != image[:, :1].shape:
        raise ValueError('region mask must align with B,1,H,W image coordinates')
    mask = mask.detach().to(image)
    if not torch.isfinite(mask).all() or (mask < 0).any() or (mask > 1).any():
        raise ValueError('region mask must contain finite weights in [0,1]')
    return mask


def encoded_luminance(image):
    """The project's encoded RGB luma proxy, not physical linear luminance."""
    coefficients = image.new_tensor([.2126, .7152, .0722]).view(1, 3, 1, 1)
    return (image * coefficients).sum(1, keepdim=True)


def _regional_reduce(error, weights, min_area=0.):
    """Equal image weight after within-image pixel normalization.

    Empty regions do not dilute the mean. A fully empty batch still returns a
    differentiable zero tied to ``error``, with finite zero backward gradients.
    """
    mass = weights.sum((1, 2, 3))
    valid = (mass > 0) & (weights.mean((1, 2, 3)) >= min_area)
    per_image = (error * weights).sum((1, 2, 3)) / mass.clamp_min(1e-12)
    indicator = valid.to(error)
    result = (per_image * indicator).sum() / indicator.sum().clamp_min(1.)
    return result, indicator


def gt_highlight_mask(gt, mask, top_fraction=.1, min_area=.0001):
    """Detached per-image, per-region GT quantile selection.

    The threshold is the (1 - top_fraction) quantile of GT encoded luma among
    pixels with positive regional weight. All ties at the threshold are kept,
    so a flat region is treated uniformly instead of selecting arbitrary pixels.
    Returned weights retain the supplied mask's soft weights. Tiny/empty regions
    return zero weights. Selection uses GT only, never prediction brightness.
    """
    top_fraction = _real(top_fraction, 'top_fraction', maximum=1., minimum_open=True)
    min_area = _real(min_area, 'min_area', maximum=1.)
    target = _image(gt, 'gt').detach()
    if not torch.isfinite(target).all():
        raise ValueError('gt must contain finite values')
    weights = _mask(mask, target)
    luma = encoded_luminance(target)
    selected = torch.zeros_like(weights)
    # Batch sizes are small in native-size enhancement; each image may have a
    # different number of region pixels and therefore its own quantile.
    for index in range(target.size(0)):
        region = weights[index, 0]
        if not (region.sum() > 0 and region.mean() >= min_area):
            continue
        values = luma[index, 0][region > 0]
        threshold = torch.quantile(values, 1. - top_fraction)
        selected[index, 0] = region * (luma[index, 0] >= threshold).to(region)
    return selected.detach()


def highlight_over_statistics(output, gt, mask, top_fraction=.1,
                              margin=2. / 255., min_area=.0001):
    """Return unweighted, per-image-normalized bright-region diagnostics.

    ``over_mean`` is mean positive luma excess above ``margin`` (RGB units);
    ``over_mse`` is its square loss; ``signed_mean`` retains darkening as negative.
    Statistics use raw output without clamping. Callers can detach log values.
    """
    margin = _real(margin, 'margin', maximum=1.)
    prediction = _image(output, 'output')
    target = _image(gt, 'gt').detach().to(prediction)
    if prediction.shape != target.shape:
        raise ValueError('output and gt must have identical shapes')
    weights = gt_highlight_mask(target, mask, top_fraction, min_area).to(prediction)
    difference = encoded_luminance(prediction) - encoded_luminance(target)
    excess = F.relu(difference - margin)
    over_mean, valid = _regional_reduce(excess, weights)
    over_mse, _ = _regional_reduce(excess.square(), weights)
    signed_mean, _ = _regional_reduce(difference, weights)
    over_fraction, _ = _regional_reduce((difference > margin).to(prediction), weights)
    return dict(over_mean=over_mean, over_mse=over_mse,
                signed_mean=signed_mean, over_fraction=over_fraction,
                valid_count=valid.sum(), valid_fraction=valid.mean(),
                selected_fraction=weights.mean())


def highlight_over_loss(output, gt, mask, top_fraction=.1,
                        margin=2. / 255., min_area=.0001):
    """Squared one-sided excess in GT-relative highlight regions, no weight."""
    return highlight_over_statistics(output, gt, mask, top_fraction,
                                     margin, min_area)['over_mse']


def range_loss(value):
    """Mean raw-channel/LUT-node squared distance outside [0,1], no weight.

    Apply this to the actual fused LUT or final RGB, not to rescalable basis
    parameters. The mean is over all scalar values; two branches should be
    averaged by the caller to avoid doubling the loss merely by using two LUTs.
    """
    if not torch.is_tensor(value) or not value.is_floating_point() or not value.numel():
        raise ValueError('range_loss requires a nonempty floating-point tensor')
    work = value.float() if value.dtype in (torch.float16, torch.bfloat16) else value
    return (F.relu(work - 1.).square() + F.relu(-work).square()).mean()


def good_region_keep_loss(output, reference, gt, mask, error_threshold=.03,
                          margin=2. / 255., min_area=.0001, top_fraction=.1):
    """Anchor only accurate, in-range teacher pixels, excluding bad highlights.

    ``error_threshold`` is per-pixel RGB RMSE. Teacher and GT are detached.
    A teacher pixel is retained only if it lies in [0,1], its RGB MSE is at most
    error_threshold**2, and it is not excessively bright in the configured GT
    top-fraction region. The default remains top 10% for legacy callers.
    Student outputs stay raw. This scalar is unweighted and macro-averaged over
    images containing at least one accepted pixel in a valid original region.
    """
    error_threshold = _real(error_threshold, 'error_threshold', maximum=1.)
    margin = _real(margin, 'margin', maximum=1.)
    min_area = _real(min_area, 'min_area', maximum=1.)
    prediction = _image(output, 'output')
    teacher = _image(reference, 'reference').detach().to(prediction)
    target = _image(gt, 'gt').detach().to(prediction)
    if prediction.shape != target.shape or teacher.shape != target.shape:
        raise ValueError('output, reference and gt must have identical shapes')
    if not torch.isfinite(teacher).all():
        raise ValueError('reference must contain finite values')
    weights = _mask(mask, prediction)
    region_valid = ((weights.mean((1, 2, 3)) >= min_area) &
                    (weights.sum((1, 2, 3)) > 0)).to(prediction)[:, None, None, None]
    teacher_mse = (teacher - target).square().mean(1, keepdim=True)
    accurate = teacher_mse <= error_threshold ** 2
    in_range = ((teacher >= 0) & (teacher <= 1)).all(1, keepdim=True)
    highlights = gt_highlight_mask(target, weights, top_fraction, min_area) > 0
    too_bright = encoded_luminance(teacher) - encoded_luminance(target) > margin
    accepted = accurate & in_range & ~(highlights & too_bright)
    weights = weights * region_valid * accepted.to(prediction)
    error = (prediction - teacher).square().mean(1, keepdim=True)
    return _regional_reduce(error, weights)[0]
