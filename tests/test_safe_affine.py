import copy
import pytest
import torch

from adaint.safe_affine import (protect_rgb, feasible_rgb, decode_p3,
                                highlight_losses, safe_options)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.float16])
def test_range_and_protected_identity(dtype):
    torch.manual_seed(2026)
    source = torch.rand(2, 3, 27, 31, dtype=dtype)
    raw = (torch.rand_like(source) * 5 - 2).requires_grad_()
    output, aux = protect_rgb(raw, source)
    assert output.min() >= 0 and output.max() <= 1
    safe = ~aux['risk'].expand_as(raw)
    assert torch.equal(output[safe], raw.float()[safe] if dtype == torch.float16 else raw[safe])
    output.square().mean().backward()
    assert torch.isfinite(raw.grad).all()


def test_black_and_white_anchors_and_outward_steps():
    source = torch.tensor([0., 1., .5]).view(1, 3, 1, 1)
    raw = torch.tensor([-.1, 1.1, .7]).view_as(source).requires_grad_()
    output, active = feasible_rgb(raw, source)
    assert active.all() and torch.equal(output, source)
    output.sum().backward()
    assert torch.isfinite(raw.grad).all()


def test_linear_rgb_ratio_and_quantized_highlight_separation():
    raw = torch.tensor([.96, 1.08, 1.2, 1.4]).view(1, 1, 1, 4).expand(1, 3, 1, 4)
    output, _ = protect_rgb(raw, torch.full_like(raw, .8))
    codes = (output[0, 0, 0] * 255).round()
    assert (torch.diff(codes) > 0).all() and codes[-1] < 255
    color = torch.tensor([1.15, .8, .6]).view(1, 3, 1, 1)
    mapped, _ = protect_rgb(color, torch.full_like(color, .5))
    ratio = decode_p3(mapped) / decode_p3(color)
    assert torch.allclose(ratio, ratio[:, :1].expand_as(ratio), atol=1e-6)


def test_knee_continuity_and_nonzero_overshoot_gradient():
    raw = torch.tensor([.92 - 1e-5, .92, .92 + 1e-5, 1.3]).view(1, 1, 1, 4).expand(1, 3, 1, 4).clone().requires_grad_()
    output, _ = protect_rgb(raw, torch.full_like(raw, .5))
    assert (output[..., 2] - output[..., 1]).abs().max() < 2e-5
    output.sum().backward()
    assert torch.isfinite(raw.grad).all() and raw.grad[..., -1].sum() > 0


def test_symmetric_highlight_and_gradient_supervision():
    target = torch.full((1, 3, 7, 9), .7)
    masks = (torch.ones(1, 1, 7, 9), torch.zeros(1, 1, 7, 9))
    bright = highlight_losses(target + .1, target, masks)[0]
    dark = highlight_losses(target - .1, target, masks)[0]
    assert torch.allclose(bright, dark, atol=1e-7)
    empty = highlight_losses(target.requires_grad_(), target.detach(), (masks[1],))
    sum(empty).backward()
    assert torch.isfinite(target.grad).all()


@pytest.mark.parametrize('bad', [{'knee': 1.}, {'raw_weight': -1}, {'enabled': 1}, {'extra': 2}])
def test_invalid_options(bad):
    with pytest.raises(ValueError):
        safe_options(bad)


def test_full_model_backward_and_legacy_state_compatibility():
    from mmcv import Config
    from mmedit.models import build_model
    from adaint.config_utils import load_config
    cfg = load_config('adaint/configs/face_outdoor_semantic_postprocess.yaml')
    model_cfg = copy.deepcopy(cfg.model)
    model_cfg.safe_affine = dict(enabled=True)
    model = build_model(model_cfg, train_cfg=dict(n_fix_iters=0), test_cfg=cfg.test_cfg)
    assert all(p.requires_grad for p in model.parameters())
    assert not any('safe_affine' in key for key in model.state_dict())
    legacy_cfg = copy.deepcopy(cfg.model)
    legacy = build_model(legacy_cfg, train_cfg=dict(n_fix_iters=0), test_cfg=cfg.test_cfg)
    legacy.load_state_dict(model.state_dict(), strict=True)
    image = torch.rand(1, 3, 32, 32)
    mask = torch.ones(1, 1, 32, 32)
    out = model.forward_train(image, image * .9, lq_context=image,
        context_samples=image, crop_box=torch.tensor([[0., 0., 1., 1.]]),
        foreground_mask=mask, supervision_mask=mask, foreground_context=mask,
        foreground_samples=mask, foreground_region_samples=image,
        background_region_samples=image, region_area=torch.tensor([[1., 0.]]))
    losses = out['losses']
    assert losses['safe_final_clip_fraction'] == 0
    assert losses['safe_protected_max_change'] == 0
    total = sum(value for name, value in losses.items() if name.startswith('loss'))
    total.backward()
    assert torch.isfinite(total) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
