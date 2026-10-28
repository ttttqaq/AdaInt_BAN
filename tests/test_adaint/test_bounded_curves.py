"""Operator regressions for bounded LUTs, spatial curves and bright supervision."""
import pytest
import torch

from adaint.bounded_curves import (NUMERICAL_MARGIN, BoundedCurveDecoder,
                                   bounded_lut, symmetric_highlight_losses)
from adaint.modules import identity_lut
from adaint.postprocess import ResultAwareLocalDecoder


def decoder(**kwargs):
    return BoundedCurveDecoder(16, 32, 4, 1, .001, **kwargs)


def vertices(batch=1, count=9):
    return torch.linspace(0, 1, count).view(1, 1, count).expand(batch, 3, count).clone()


def constant_fields(model, values, size=(7, 9)):
    low = torch.as_tensor(values).float().view(1, -1, 1, 1).expand(1, -1, *size).clone()
    return [(low, torch.zeros_like(low), low)]


def test_lut_initialization_tracks_current_nonuniform_vertices():
    vertex = vertices()
    vertex[:, 0] = vertex[:, 0].square()
    vertex.requires_grad_()
    residual = torch.zeros(1, 3, 9, 9, 9, requires_grad=True)
    result = bounded_lut(residual, vertex)
    expected = NUMERICAL_MARGIN + (1 - 2 * NUMERICAL_MARGIN) * (
        .001 + .998 * identity_lut(vertex))
    assert torch.allclose(result, expected, atol=2e-7, rtol=0)
    result.mean().backward()
    assert residual.grad.abs().min() > 0
    assert vertex.grad is not None and torch.isfinite(vertex.grad).all()
    assert result.min() > 0 and result.max() < 1


@pytest.mark.parametrize('value', [-1.e30, 1.e30])
def test_extreme_lut_logits_remain_strictly_inside_range_without_clamp(value):
    result = bounded_lut(torch.full((1, 3, 9, 9, 9), value), vertices())
    assert torch.isfinite(result).all()
    assert (result > 0).all() and (result < 1).all()
    assert result.min() >= NUMERICAL_MARGIN * .99
    assert result.max() <= 1 - NUMERICAL_MARGIN * .99


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_invalid_lut_inputs_are_rejected(bad):
    value = torch.zeros(1, 3, 9, 9, 9)
    value[0, 0, 0, 0, 0] = bad
    with pytest.raises(ValueError, match='finite'):
        bounded_lut(value, vertices())
    bad_vertices = vertices()
    bad_vertices[..., 1] = bad_vertices[..., 0]
    with pytest.raises(ValueError, match='increase strictly'):
        bounded_lut(torch.zeros(1, 3, 9, 9, 9), bad_vertices)


@pytest.mark.parametrize('guidance', ['luminance', 'rgb'])
def test_zero_heads_give_near_identity_for_black_white_and_random_rgb(guidance):
    model = decoder(result_condition=True, multiscale=True, guidance=guidance).eval()
    source = torch.rand(2, 3, 36, 48)
    source[:, :, 0] = 0
    source[:, :, -1] = 1
    conditions = [torch.randn(2, 16), torch.randn(2, 16)]
    fields, _ = model(source, conditions, [source, source], torch.ones_like(source[:, :1]))
    output = model.apply_fields(fields[0], source, source)
    expected = NUMERICAL_MARGIN + (1 - 2 * NUMERICAL_MARGIN) * (.001 + .998 * source)
    assert torch.allclose(output, expected, atol=3e-7, rtol=0)
    assert (output > 0).all() and (output < 1).all()
    assert fields[0][0][0].size(1) == 18
    assert len(fields[0]) == 2
    (output - .4).square().mean().backward()
    for head in (model.outputs[0], model.refinement_outputs[0]):
        assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0
        assert torch.isfinite(head.weight.grad).all()


def test_curve_endpoint_is_learnable_and_white_can_be_compressed():
    model = decoder()
    source = torch.ones(1, 3, 16, 24)
    logits = torch.zeros(model.curve_channels)
    # Allocate large white reserve independently to all RGB curves.
    logits.view(3, -1)[:, -1] = 8.
    fields = constant_fields(model, logits)
    fields[0][2].requires_grad_()
    output = model.apply_fields(fields, source, source)
    assert output.max() < .3
    (output - .5).square().mean().backward()
    gradient = fields[0][2].grad.view(1, 3, 6, 7, 9)
    assert gradient[:, :, -1].abs().sum() > 0
    assert torch.isfinite(gradient).all()


def test_random_extreme_curve_knots_and_outputs_are_monotone_and_bounded():
    model = decoder(segments=8)
    raw = torch.randn(1, model.segments + 2, 1, 1) * 1.e5
    knots = model.curve_knots(raw)
    assert (torch.diff(knots, dim=1) >= 0).all()
    assert (knots > 0).all() and (knots < 1).all()
    values = raw.flatten().repeat(3)
    fields = constant_fields(model, values, size=(1, 1))
    ramp = torch.linspace(0, 1, 257).view(1, 1, 1, -1).expand(1, 3, 1, -1)
    output = model.apply_fields(fields, ramp, ramp)
    assert (torch.diff(output, dim=-1) >= -1e-7).all()
    assert (output > 0).all() and (output < 1).all()


@pytest.mark.parametrize('guidance', ['luminance', 'rgb'])
def test_nonzero_multiscale_curve_crop_consistency_and_feature_gradients(guidance):
    torch.manual_seed(29)
    model = decoder(result_condition=True, multiscale=True, guidance=guidance)
    for head in list(model.outputs) + list(model.refinement_outputs):
        torch.nn.init.normal_(head.weight, std=.1)
    source = torch.rand(1, 3, 48, 64)
    global_output = torch.rand_like(source)
    conditions = [torch.randn(1, 16), torch.randn(1, 16)]
    fields, _ = model(source, conditions, [global_output, global_output], source[:, :1])
    full = model.apply_fields(fields[0], source, global_output)
    box = source.new_tensor([[.25, .25, .75, .75]])
    crop = model.apply_fields(fields[0], source[..., 12:36, 16:48],
                              global_output[..., 12:36, 16:48], box)
    assert torch.allclose(crop, full[..., 12:36, 16:48], atol=3e-6, rtol=0)
    full.square().mean().backward()
    assert model.shallow[0].weight.grad.abs().sum() > 0
    assert model.result_adapter.weight.grad.abs().sum() > 0
    assert torch.isfinite(model.result_adapter.weight.grad).all()


def test_curve_construction_preserves_shared_feature_initialization_and_rng():
    settings = dict(result_condition=True, multiscale=True, guidance='luminance')
    torch.manual_seed(2026)
    old = ResultAwareLocalDecoder(16, 32, 4, 1, .001, **settings)
    old_state = torch.random.get_rng_state()
    torch.manual_seed(2026)
    new = decoder(**settings)
    assert torch.equal(old_state, torch.random.get_rng_state())
    for name, value in old.state_dict().items():
        if not name.startswith(('outputs.', 'refinement_outputs.')):
            assert torch.equal(value, new.state_dict()[name])


def test_curve_rejects_out_of_range_rgb_and_nonfinite_latents():
    model = decoder()
    fields = constant_fields(model, torch.zeros(model.curve_channels))
    guide = torch.zeros(1, 3, 8, 12)
    with pytest.raises(ValueError, match=r'\[0,1\]'):
        model.apply_fields(fields, guide, guide - .0001)
    with pytest.raises(ValueError, match='finite'):
        model.curve_knots(torch.full((1, 6, 2, 2), float('inf')))


def test_symmetric_highlights_penalize_darkening_and_brightening_equally():
    gt = torch.full((2, 3, 8, 12), .5, requires_grad=True)
    mask = torch.ones_like(gt[:, :1])
    low = symmetric_highlight_losses(gt - .1, gt, (mask,), top_fraction=.1)
    high = symmetric_highlight_losses(gt + .1, gt, (mask,), top_fraction=.1)
    assert low[0].item() == pytest.approx(.01)
    assert torch.allclose(low[0], high[0], atol=2e-8, rtol=0)
    assert low[1] == 0 and high[1] == 0
    output = torch.full_like(gt, .6, requires_grad=True)
    sum(symmetric_highlight_losses(output, gt, (mask,))).backward()
    assert output.grad.abs().sum() > 0 and gt.grad is None


def test_highlight_gradient_only_uses_edges_with_both_selected_endpoints():
    gt = torch.tensor([.1, .2, .8, .9]).view(1, 1, 1, 4).expand(1, 3, 1, 4)
    mask = torch.ones_like(gt[:, :1])
    output = gt.clone()
    output[..., 1] += .4  # The dark-to-bright boundary must not contribute.
    rgb, gradient = symmetric_highlight_losses(output, gt, (mask,), top_fraction=.5)
    assert rgb == 0 and gradient == 0
    output = gt.clone()
    output[..., -1] -= .2
    rgb, gradient = symmetric_highlight_losses(output, gt, (mask,), top_fraction=.5)
    assert rgb.item() == pytest.approx(.02)
    assert gradient.item() == pytest.approx(.04)


def test_empty_regions_do_not_dilute_symmetric_losses_and_all_empty_has_zero_gradient():
    gt = torch.full((1, 3, 5, 7), .5)
    output = torch.full_like(gt, .6, requires_grad=True)
    empty, full = torch.zeros_like(gt[:, :1]), torch.ones_like(gt[:, :1])
    expected = symmetric_highlight_losses(output, gt, (full,))
    actual = symmetric_highlight_losses(output, gt, (empty, full))
    assert all(torch.equal(x, y) for x, y in zip(expected, actual))
    losses = symmetric_highlight_losses(output, gt, (empty, empty))
    assert all(value == 0 and torch.isfinite(value) for value in losses)
    sum(losses).backward()
    assert torch.count_nonzero(output.grad) == 0
