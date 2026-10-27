"""Behavioral checks for raw highlight constraints and selective anchoring."""
import copy

import pytest
import torch

from adaint.highlight_guard import (
    guard_options, gt_highlight_mask, highlight_over_loss,
    highlight_over_statistics, range_loss, good_region_keep_loss)


def gray(values):
    return torch.tensor(values, dtype=torch.float64).view(1, 1, 1, -1).repeat(1, 3, 1, 1)


def test_default_configuration_is_inactive_and_does_not_mutate_or_consume_rng():
    source = dict(over_weight=1., trainable_scope='heads')
    original = copy.deepcopy(source)
    state = torch.random.get_rng_state()
    options = guard_options(source)
    assert source == original and options is not source
    assert torch.equal(torch.random.get_rng_state(), state)
    assert options['keep_error_threshold'] == .03
    defaults = guard_options()
    assert all(defaults[k] == 0 for k in
               ('over_weight', 'lut_range_weight', 'final_range_weight', 'keep_weight'))
    assert defaults['trainable_scope'] == 'all'
    assert not defaults['detach_result_condition']


@pytest.mark.parametrize('value', [
    dict(over_weight=-1), dict(final_range_weight=float('nan')),
    dict(lut_range_weight=True), dict(top_fraction=0), dict(top_fraction=1.1),
    dict(over_margin=float('inf')), dict(keep_error_threshold='0.03'),
    dict(trainable_scope='almost_all'), dict(detach_result_condition=1),
    dict(typo=True), dict(keep_weight=1.), dict(reference_config='')])
def test_invalid_options_fail_early(value):
    with pytest.raises(ValueError):
        guard_options(value)


def test_relative_gt_highlights_detect_dark_skin_and_ignore_bright_prediction_elsewhere():
    target = gray([.1, .2, .3, .4])
    prediction = gray([.95, .2, .3, .6]).requires_grad_()
    mask = torch.ones_like(target[:, :1])
    selected = gt_highlight_mask(target, mask, top_fraction=.25)
    assert torch.equal(selected, selected.new_tensor([[[[0., 0., 0., 1.]]]]))
    loss = highlight_over_loss(prediction, target, mask, top_fraction=.25, margin=0.)
    assert loss.item() == pytest.approx(.04)
    loss.backward()
    assert torch.count_nonzero(prediction.grad[..., :3]) == 0
    assert (prediction.grad[..., 3] > 0).all()
    assert torch.isfinite(prediction.grad).all()


def test_each_image_is_normalized_before_average_and_empty_regions_do_not_dilute():
    target = gray([.4, .4, .4, .4]).repeat(3, 1, 1, 1)
    prediction = target.clone()
    prediction[0] += .1
    prediction[1] += .3
    prediction[2] += .5
    mask = target.new_tensor([[[[1., 0., 0., 0.]]],
                              [[[1., 1., 1., 1.]]],
                              [[[0., 0., 0., 0.]]]])
    result = highlight_over_statistics(prediction, target, mask, margin=0.)
    assert result['over_mse'].item() == pytest.approx((.01 + .09) / 2)
    assert result['over_mean'].item() == pytest.approx(.2)
    assert result['valid_count'].item() == 2
    assert result['valid_fraction'].item() == pytest.approx(2. / 3.)


def test_flat_gt_keeps_ties_uniformly_and_tiny_region_is_excluded():
    target = gray([.25, .25, .25, .25])
    mask = torch.ones_like(target[:, :1])
    assert torch.equal(gt_highlight_mask(target, mask, .1), mask)
    tiny = mask * .00001
    assert torch.count_nonzero(gt_highlight_mask(target, tiny, .1, min_area=.0001)) == 0


def test_margin_darkening_and_exact_zero_have_finite_zero_gradients():
    target = gray([.4, .6])
    prediction = gray([.3, .605]).requires_grad_()
    mask = torch.ones_like(target[:, :1])
    loss = highlight_over_loss(prediction, target, mask, top_fraction=1., margin=.01)
    loss.backward()
    assert loss.item() == 0
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))
    assert highlight_over_statistics(prediction, target, mask,
                                     top_fraction=1., margin=.01)['signed_mean'] < 0


def test_no_gt_or_teacher_gradient_and_all_empty_loss_remains_differentiable():
    target = gray([.2, .8]).requires_grad_()
    teacher = target.detach().clone().requires_grad_()
    prediction = (target.detach() + .1).requires_grad_()
    empty = torch.zeros_like(target[:, :1])
    loss = (highlight_over_loss(prediction, target, empty) +
            good_region_keep_loss(prediction, teacher, target, empty))
    loss.backward()
    assert loss.item() == 0
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))
    assert target.grad is None and teacher.grad is None


def test_range_penalty_uses_raw_values_and_points_gradients_back_into_range():
    prediction = torch.tensor([-.2, .2, .8, 1.3], dtype=torch.float64, requires_grad=True)
    loss = range_loss(prediction)
    assert loss.item() == pytest.approx((.04 + .09) / 4)
    loss.backward()
    assert prediction.grad[0] < 0 and prediction.grad[-1] > 0
    assert torch.equal(prediction.grad[1:3], torch.zeros_like(prediction.grad[1:3]))
    assert torch.isfinite(prediction.grad).all()
    moved = prediction.detach() - .1 * prediction.grad
    assert range_loss(moved) < loss
    assert torch.equal(prediction.detach(), prediction.new_tensor([-.2, .2, .8, 1.3]))


def test_keep_threshold_is_rmse_and_does_not_preserve_teacher_highlight_errors():
    target = gray([.2, .2, .2, .8, .8])
    # Pixel0: accurate. Pixel1: RMSE .05 > .03 despite MSE .0025 < .03.
    # Pixel2: out of range. Pixel3: accurate RMSE but a bad bright-region excess.
    # Pixel4: accurate and inside the allowed bright-region margin.
    teacher = gray([.21, .25, 1.01, .82, .805]).requires_grad_()
    prediction = gray([.31, .95, .9, .95, .905]).requires_grad_()
    mask = torch.ones_like(target[:, :1])
    loss = good_region_keep_loss(prediction, teacher, target, mask,
                                 error_threshold=.03, margin=.01)
    assert loss.item() == pytest.approx(.01)
    loss.backward()
    assert (prediction.grad[..., 0] > 0).all() and (prediction.grad[..., 4] > 0).all()
    assert torch.count_nonzero(prediction.grad[..., 1:4]) == 0
    assert teacher.grad is None


def test_soft_weights_are_respected_and_half_precision_loss_accumulates_in_float32():
    target = gray([.2, .2])
    prediction = gray([.3, .5]).requires_grad_()
    mask = target.new_tensor([[[[1., .5]]]])
    loss = highlight_over_loss(prediction, target, mask, top_fraction=1., margin=0.)
    assert loss.item() == pytest.approx((.01 + .5 * .09) / 1.5)
    half_pred = prediction.detach().half().requires_grad_()
    half_loss = highlight_over_loss(half_pred, target.half(), mask.half(),
                                    top_fraction=1., margin=0.)
    assert half_loss.dtype == torch.float32
    half_loss.backward()
    assert torch.isfinite(half_pred.grad).all()


def test_keep_highlight_exclusion_uses_configured_fraction():
    target = gray([.2, .3, .4, .7, .8])
    teacher = gray([.2, .3, .4, .72, .8])
    prediction = gray([.2, .3, .4, .92, .8]).requires_grad_()
    mask = torch.ones_like(target[:, :1])
    # The .7 target pixel is below the top-10% threshold, but belongs to the
    # top half. It must not be anchored to a known overbright teacher there.
    default = good_region_keep_loss(prediction, teacher, target, mask,
                                    error_threshold=.03, margin=.01)
    configured = good_region_keep_loss(prediction, teacher, target, mask,
                                       error_threshold=.03, margin=.01, top_fraction=.5)
    assert default.item() == pytest.approx(.04 / 5.)
    assert configured.item() == 0
    configured.backward()
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))
