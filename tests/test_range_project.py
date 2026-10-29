import copy

import pytest
import torch

from adaint.range_project import (project_rgb, range_error, difficult_weights,
                                  difficult_losses, range_options)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.float16])
def test_range_and_all_legal_channel_identity(dtype):
    torch.manual_seed(2026)
    raw = (torch.rand(2, 3, 19, 23, dtype=dtype) * 4 - 1).requires_grad_()
    output = project_rgb(raw)
    assert output.min() >= 0 and output.max() <= 1
    legal = (raw >= 0) & (raw <= 1)
    assert torch.equal(output[legal], raw.to(output)[legal])
    (output.square().mean() + raw.float().square().mean()).backward()
    assert torch.isfinite(raw.grad).all()


def test_normal_highlights_and_boundary_anchor_counterexample():
    raw = torch.tensor([.96, .98, 1., -1e-6, .6, .4]).reshape(1, 3, 1, 2)
    expected = raw.clone().clamp(0, 1)
    assert torch.equal(project_rgb(raw), expected)
    assert torch.equal(project_rgb(raw)[raw >= 0], raw[raw >= 0])


def test_gt_is_common_zero_and_raw_gradient_survives_projection():
    gt = torch.tensor([.96, .98, 1.]).view(1, 3, 1, 1)
    assert range_error(gt).sum() == 0
    assert torch.equal(project_rgb(gt), gt)
    raw = torch.tensor([1.2, -.2, 1.1]).view_as(gt).requires_grad_()
    target = torch.tensor([.8, .2, .9]).view_as(gt)
    loss = (raw - target).square().mean() + range_error(raw).mean()
    loss.backward()
    assert (raw.grad[:, [0, 2]] > 0).all() and (raw.grad[:, 1] < 0).all()


def test_difficult_background_includes_gt_midtone_and_undershoot():
    gt = torch.tensor([.5, .95, .5, .95]).view(1, 1, 1, 4).expand(1, 3, 1, 4)
    raw = torch.tensor([1.2, .95, -.2, .95]).view(1, 1, 1, 4).expand_as(gt).clone().requires_grad_()
    mask = torch.ones(1, 1, 1, 4)
    chosen = difficult_weights(raw, gt, mask, .5)
    assert torch.equal(chosen.flatten(), torch.tensor([1., 0., 1., 0.]))
    assert not chosen.requires_grad
    rgb, _ = difficult_losses(raw, project_rgb(raw), gt, (mask,), .5)
    rgb.backward()
    assert raw.grad[..., 0].abs().sum() > 0 and raw.grad[..., 2].abs().sum() > 0


def test_selected_to_unselected_edge_is_supervised_and_empty_region_safe():
    gt = torch.tensor([.5, .2]).view(1, 1, 1, 2).expand(1, 3, 1, 2)
    raw = torch.tensor([.8, .2]).view(1, 1, 1, 2).expand_as(gt).clone().requires_grad_()
    mask = torch.ones(1, 1, 1, 2)
    rgb, gradient = difficult_losses(raw, project_rgb(raw), gt, (mask,), .5)
    assert gradient > 0
    empty = difficult_losses(raw, project_rgb(raw), gt, (mask * 0,), .5)
    assert all(value == 0 for value in empty)
    (rgb + gradient + sum(empty)).backward()
    assert torch.isfinite(raw.grad).all()


@pytest.mark.parametrize('bad', [{'enabled': 1}, {'hard_fraction': 0},
                                {'range_weight': -1}, {'gradient_weight': float('nan')},
                                {'unknown': 2}])
def test_invalid_configuration(bad):
    with pytest.raises(ValueError):
        range_options(bad)


def test_nonfinite_proposal_fails():
    with pytest.raises(FloatingPointError):
        project_rgb(torch.full((1, 3, 1, 1), float('nan')))


@pytest.mark.parametrize('scene,count', [('indoor', 67), ('outdoor', 85), ('club', 142)])
def test_scratch_configs_schedule_and_saved_protection(scene, count, tmp_path):
    from adaint.config_utils import load_config
    from adaint.experiment_utils import resolve_training_schedule, save_generation_configs
    cfg = load_config('adaint/configs/face_' + scene + '_semantic_range_project.yaml')
    assert cfg.infer_after_generation is True and cfg.model.range_project.enabled
    assert not cfg.model.postprocess.freeze_global and cfg.model.pretrained is False
    assert cfg.adaint_fix_epochs == 0 and cfg.gen == 1 and not cfg.enable_kd
    assert all(cfg.get(key) is None for key in ('postprocess_init', 'load_from', 'resume_from'))
    cfg.gpus = 1
    cfg.work_dir = str(tmp_path)
    resolve_training_schedule(cfg, count)
    assert cfg.total_iters == count * 100 and cfg.train_cfg.n_fix_iters == 0
    save_generation_configs(cfg)
    saved = load_config(str(tmp_path / 'inference_config.py'))
    assert saved.model.range_project.enabled and saved.infer_after_generation is True


def test_full_model_raw_training_and_legacy_parameter_compatibility():
    from adaint.config_utils import load_config
    from mmedit.models import build_model
    cfg = load_config('adaint/configs/face_outdoor_semantic_postprocess.yaml')
    legacy_cfg = copy.deepcopy(cfg.model)
    model_cfg = copy.deepcopy(legacy_cfg)
    model_cfg.range_project = dict(enabled=True)
    model = build_model(model_cfg, train_cfg=dict(n_fix_iters=0), test_cfg=cfg.test_cfg)
    legacy = build_model(legacy_cfg, train_cfg=dict(n_fix_iters=0), test_cfg=cfg.test_cfg)
    legacy.load_state_dict(model.state_dict(), strict=True)
    assert all(p.requires_grad for p in model.parameters())
    assert not any('range_project' in key for key in model.state_dict())
    image = torch.rand(1, 3, 32, 32)
    mask = torch.ones(1, 1, 32, 32)
    result = model.forward_train(image, image * .9, lq_context=image,
        context_samples=image, crop_box=torch.tensor([[0., 0., 1., 1.]]),
        foreground_mask=mask, supervision_mask=mask, foreground_context=mask,
        foreground_samples=mask, foreground_region_samples=image,
        background_region_samples=image, region_area=torch.tensor([[1., 0.]]))
    losses = result['losses']
    assert losses['range_final_clip_fraction'] == 0
    assert losses['range_legal_max_change'] == 0
    total = sum(value for key, value in losses.items() if key.startswith('loss'))
    total.backward()
    assert torch.isfinite(total)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
