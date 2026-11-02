import copy

import pytest
import torch

from adaint.controlled_tone import (BoundedExpertLUT, SharedRegionMapping,
                                    exposure_correction)
from adaint.modules import identity_lut
from adaint.safe_affine import decode_p3


def test_bounded_experts_use_actual_nonuniform_coordinates_and_gradients():
    torch.manual_seed(2026)
    module = BoundedExpertLUT(5, 3, 8, 8)
    vertices = torch.tensor([0., .01, .2, .8, 1.]).view(1, 1, 5).expand(2, 3, 5)
    condition = torch.randn(2, 8)
    weights, lut = module(condition, vertices)
    assert lut.min() >= 0 and lut.max() <= 1
    assert torch.allclose(weights.sum(1), torch.ones(2))
    (lut - .4).square().mean().backward()
    assert module.expert_raw.grad.abs().sum() > 0
    assert torch.isfinite(module.head.output.weight.grad).all()
    with torch.no_grad():
        module.expert_raw.zero_()
    assert torch.allclose(module(condition, vertices)[1], identity_lut(vertices), atol=1e-7)


def test_shared_background_is_independent_of_foreground_statistics():
    torch.manual_seed(2026)
    module = SharedRegionMapping(8, 6, 8, 8, 5, 3, 4, .001, 'bounded')
    cnn, global_stats = torch.randn(2, 8), torch.randn(2, 6)
    a = module(cnn, global_stats, torch.randn(2, 6), torch.ones(2, 1) * .3, torch.ones(2, 1))
    b = module(cnn, global_stats, torch.randn(2, 6), torch.ones(2, 1) * .8, torch.ones(2, 1))
    assert torch.equal(a[3][1], b[3][1])
    assert a[2][0] is a[2][1]
    assert (a[3][0] - a[3][1]).abs().max() <= .080001
    assert all(lut.min() >= 0 and lut.max() <= 1 for lut in a[3])
    (a[3][0] - .5).square().mean().backward()
    assert module.residual_head.output.weight.grad.abs().sum() > 0


def test_exposure_identity_keeps_trainable_gradient_and_positive_headroom():
    rgb = torch.tensor([.8, .5, .3]).view(1, 3, 1, 1).expand(1, 3, 3, 4)
    field = torch.zeros(1, 1, 3, 4, requires_grad=True)
    mean = torch.zeros(1, 1, 1, 1)
    delta = exposure_correction(rgb, field, mean, torch.ones_like(mean), .15)
    assert torch.equal(delta, torch.zeros_like(delta))
    delta.sum().backward()
    assert torch.isfinite(field.grad).all() and field.grad.abs().sum() > 0
    bright = torch.tensor([.9999, .8, .5]).view(1, 3, 1, 1)
    output = bright + exposure_correction(bright, torch.ones_like(mean) * 100,
                                           mean, torch.ones_like(mean), .15)
    assert output.max() <= 1.000001 and output.min() >= 0
    ratio = decode_p3(output) / decode_p3(bright)
    assert torch.allclose(ratio, ratio[:, :1].expand_as(ratio), atol=1e-5)


@pytest.mark.parametrize('scene,count', [('indoor', 67), ('outdoor', 85)])
@pytest.mark.parametrize('index', range(1, 7))
def test_ablation_protocol_and_five_epoch_checkpoints(scene, count, index):
    from adaint.config_utils import load_config
    from adaint.experiment_utils import resolve_training_schedule
    cfg = load_config('adaint/configs/face_' + scene + '_shared_tone_' + str(index) + '.py')
    cfg.gpus = 1
    resolve_training_schedule(cfg, count)
    assert cfg.checkpoint_config.interval == count * 5
    assert cfg.evaluation.interval == count * 5
    assert cfg.total_iters == count * 100
    assert cfg.infer_after_generation and cfg.model.safe_affine.enabled
    assert not cfg.enable_kd and cfg.gen == 1
    assert all(cfg.get(key) is None for key in ('postprocess_init', 'load_from', 'resume_from'))


@pytest.mark.parametrize('index', range(1, 7))
def test_full_model_backward_empty_regions_and_safety(index):
    from adaint.config_utils import load_config
    from mmedit.models import build_model
    cfg = load_config('adaint/configs/face_indoor_shared_tone_' + str(index) + '.py')
    model = build_model(copy.deepcopy(cfg.model), train_cfg=dict(n_fix_iters=0), test_cfg=cfg.test_cfg)
    image = torch.rand(1, 3, 32, 32)
    mask = torch.ones(1, 1, 32, 32)
    result = model.forward_train(image, image * .9, lq_context=image,
        context_samples=image, crop_box=torch.tensor([[0., 0., 1., 1.]]),
        foreground_mask=mask, supervision_mask=mask, foreground_context=mask,
        foreground_samples=mask, foreground_region_samples=image,
        background_region_samples=image, region_area=torch.tensor([[1., 0.]]))
    losses = result['losses']
    assert losses['safe_final_clip_fraction'] == 0
    assert losses['safe_protected_max_change'] == 0
    total = sum(v for k, v in losses.items() if k.startswith('loss'))
    total.backward()
    assert torch.isfinite(total)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
