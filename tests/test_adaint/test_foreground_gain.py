"""Identity, task gradients, permanent freezing and crop/checkpoint isolation."""
import copy

import pytest
import torch
import torch.nn.functional as F
from mmcv.runner import build_optimizer

from adaint.config_utils import load_config
from adaint.experiment_utils import (initialize_foreground_gain_from_baseline,
                                     load_enhancer_checkpoint, resolve_training_schedule)
from adaint.foreground_gain import ForegroundSpatialGain
from adaint.model_semantic import AiLUTDual, region_samples
from adaint.modules import stratified_pixels
from adaint.spatial_tone import decode_p3


def small_model(**kwargs):
    options = dict(lut_mode='basis', n_vertices=9, n_ranks=3, curve_points=4,
        stats_bins=8, stats_pool_size=16, condition_dim=32, head_hidden_dim=16,
        context_size=32, local_resolution=32, local_channels=4, region_samples_size=8,
        foreground_gain_enabled=True, gain_channels=4, gain_max=2.,
        region_lightness_weight=.1, foreground_lightness_weight=1.,
        background_lightness_weight=0., shadow_weight=0.,
        train_cfg=dict(total_iters=10, n_fix_iters=0))
    options.update(kwargs)
    return AiLUTDual(**options)


def batch():
    image = torch.rand(2, 3, 48, 64, device='cuda') * .6 + .05
    mask = torch.zeros_like(image[:, :1])
    mask[:, :, 8:36, 8:44] = 1
    target = image.clone()
    target[0] *= .9
    target[1] *= 1.1
    return dict(lq=image, gt=target, foreground_mask=mask, supervision_mask=mask,
        lq_context=image, context_samples=stratified_pixels(image, 16),
        foreground_context=mask, foreground_samples=stratified_pixels(mask, 16),
        foreground_region_samples=region_samples(image, mask, 8),
        background_region_samples=region_samples(image, 1 - mask, 8),
        region_area=torch.cat((mask.mean((-1, -2)), 1 - mask.mean((-1, -2))), 1),
        crop_box=image.new_tensor([[0., 0., 1., 1.]]).repeat(2, 1))


def module_fixture():
    torch.manual_seed(2026)
    module = ForegroundSpatialGain(8, 12, channels=4)
    features = torch.randn(2, 8, 16, 16)
    source = torch.rand(2, 3, 48, 64) * .6 + .02
    low = F.interpolate(source, (16, 16), mode='bilinear', align_corners=False)
    base = low * .9
    mask = torch.ones(2, 1, 16, 16)
    condition = torch.randn(2, 12)
    active = torch.ones(2, 1, 1, 1)
    field = module.predict(features, low, base, mask, condition, active)
    return module, source, field, (features, low, base, mask, condition, active)


def test_zero_head_is_exact_identity_including_black_signed_and_extended_rgb():
    module, image, field, _ = module_fixture()
    image[:, :, 0, :5] = torch.tensor([0., 1e-8, -.01, 1., 1.2])[None, None]
    image.requires_grad_()
    output, gain, log_gain = module.apply_gain(image, field, image)
    assert torch.equal(output, image)
    assert torch.equal(gain, torch.ones_like(gain))
    assert torch.equal(log_gain, torch.zeros_like(log_gain))
    output.sum().backward()
    assert torch.isfinite(image.grad).all()


@pytest.mark.parametrize('sign', [-1., 1.])
def test_gain_is_bidirectional_bounded_and_preserves_common_channel_ratios(sign):
    module, image, _, args = module_fixture()
    with torch.no_grad():
        module.output.bias.fill_(sign * 100)
    field = module.predict(*args)
    image[:] = image.new_tensor([.3, .2, .1])[None, :, None, None]
    output, gain, log_gain = module.apply_gain(image, field, image)
    assert gain.min() >= .5 and gain.max() <= 2.
    assert (gain > 1).all() if sign > 0 else (gain < 1).all()
    before, after = decode_p3(image), decode_p3(output)
    assert torch.allclose(after[:, 1:] / after[:, :1], before[:, 1:] / before[:, :1], atol=2e-6)
    assert torch.allclose(log_gain.exp(), gain)


def test_common_headroom_and_empty_region_are_stable():
    module, image, _, args = module_fixture()
    with torch.no_grad():
        module.output.bias.fill_(100)
    field = module.predict(*args)
    image[:] = image.new_tensor([.99, .3, .1])[None, :, None, None]
    output, gain, _ = module.apply_gain(image, field, image)
    assert output.max() <= 1 + 1e-6 and gain.min() >= 1
    field['active'].zero_()
    empty, gain, log_gain = module.apply_gain(image, field, image)
    assert torch.equal(empty, image) and (gain == 1).all() and (log_gain == 0).all()


def test_identity_has_immediate_task_gradient_and_adapter_sees_post_lut_result():
    module, image, field, args = module_fixture()
    output = module.apply_gain(image, field, image)[0]
    (output - image * .8).square().mean().backward()
    assert torch.isfinite(module.output.weight.grad).all()
    assert module.output.weight.grad.abs().sum() > 0
    with torch.no_grad():
        module.output.weight.normal_(std=.1)
    a = module.predict(*args)
    other = list(args)
    other[2] = args[2] * .4
    b = module.predict(*other)
    ga = module.apply_gain(image, a, image)[1]
    gb = module.apply_gain(image, b, image)[1]
    assert not torch.allclose(ga, gb)


def test_full_context_gain_matches_nontrivial_crop_and_regularizers_use_applied_gain():
    module, image, _, args = module_fixture()
    with torch.no_grad():
        module.output.weight.normal_(std=.1)
        module.output.bias.fill_(.2)
    field = module.predict(*args)
    full, gain, log_gain = module.apply_gain(image, field, image)
    crop = image[..., 12:36, 16:48]
    box = image.new_tensor([[.25, .25, .75, .75]]).repeat(2, 1)
    output, _, _ = module.apply_gain(crop, field, crop, box)
    assert torch.allclose(output, full[..., 12:36, 16:48], atol=2e-6)
    _, _, context_log = module.apply_gain(args[2], field, args[1])
    mag, smooth = module.regularizers(log_gain, torch.ones_like(gain), context_log,
                                      args[1], args[3], .0001)
    assert mag > 0 and smooth > 0 and torch.isfinite(mag + smooth)
    empty = module.regularizers(log_gain, torch.zeros_like(gain), context_log,
                                 args[1], torch.zeros_like(args[3]), .0001)
    assert all(x.item() == 0 for x in empty)


def test_only_adapter_is_trainable_and_train_mode_keeps_all_shared_modules_in_eval():
    model = small_model()
    for repeat in range(2):
        model.train()
        assert model.foreground_gain.training and model.training
        assert all(not module.training for name, module in model.named_children()
                   if name != 'foreground_gain')
        assert all(parameter.requires_grad == name.startswith('foreground_gain.')
                   for name, parameter in model.named_parameters())
    assert not any(name.startswith('spatial_tone.') for name in model.state_dict())
    assert not hasattr(model.foreground_gain, 'curve_heads')
    model.eval()
    assert not any(module.training for module in model.modules())


def test_strict_baseline_initialization_and_gain_checkpoint_round_trip(tmp_path):
    baseline = small_model(foreground_gain_enabled=False)
    baseline.cnt_iters.fill_(456)
    path = str(tmp_path / 'baseline.pth')
    torch.save(dict(state_dict=baseline.state_dict()), path)
    model = small_model()
    initialize_foreground_gain_from_baseline(model, path)
    assert model.cnt_iters.item() == 0
    for key, value in baseline.state_dict().items():
        if key != 'cnt_iters': assert torch.equal(value, model.state_dict()[key])
    with pytest.raises(RuntimeError, match='incompatible'):
        load_enhancer_checkpoint(model, path)
    state = copy.deepcopy(baseline.state_dict())
    state.pop(next(key for key in state if key.startswith('branches.background.')))
    torch.save(dict(state_dict=state), path)
    with pytest.raises(RuntimeError, match='incompatible'):
        initialize_foreground_gain_from_baseline(model, path)
    legacy = small_model(foreground_gain_enabled=False, spatial_tone_enabled=True)
    torch.save(dict(state_dict=legacy.state_dict()), path)
    with pytest.raises(RuntimeError, match='incompatible'):
        initialize_foreground_gain_from_baseline(model, path)
    model.cnt_iters.fill_(500)
    torch.save(dict(state_dict=model.state_dict()), path)
    resumed = small_model()
    load_enhancer_checkpoint(resumed, path)
    assert resumed.cnt_iters.item() == 500
    assert all(not p.requires_grad for name, p in resumed.named_parameters()
               if not name.startswith('foreground_gain.'))


def test_config_keeps_original_data_and_uses_frozen_gain_training():
    cfg = load_config('adaint/configs/face_niguang_foreground_gain.yaml')
    baseline = load_config('adaint/configs/face_niguang_semantic_basis.yaml')
    assert cfg.data == baseline.data and cfg.semantic_masks == baseline.semantic_masks
    assert cfg.seed == 2026 and cfg.model.foreground_gain_enabled
    assert not cfg.model.spatial_tone_enabled and cfg.model.shadow_weight == 0
    assert cfg.model.background_lightness_weight == 0 and not cfg.get('tone_init_from')
    assert cfg.work_dir != baseline.work_dir and cfg.model.gain_max == 2
    cfg.gpus = 1
    resolve_training_schedule(cfg, 76)
    assert cfg.total_iters == 7600 and cfg.train_cfg.n_fix_iters == 0
    assert cfg.train_cfg.get('tone_warmup_iters', 0) == 0


@pytest.mark.parametrize('options', [dict(spatial_tone_enabled=True),dict(local_enabled=False)])
def test_incompatible_modes_are_rejected(options):
    with pytest.raises(ValueError, match='foreground gain requires'):
        small_model(**options)


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='real AiLUT CUDA kernel')


@CUDA
def test_task_gradients_and_base_background_remain_fixed_across_training_and_resume(tmp_path):
    torch.manual_seed(15)
    model = small_model(train_cfg=dict(total_iters=10, n_fix_iters=1)).cuda().train()
    data = batch()
    before = {name: p.detach().clone() for name, p in model.named_parameters()
              if not name.startswith('foreground_gain.')}
    args = {k:v for k,v in data.items() if k not in ('lq','gt','supervision_mask')}
    with torch.no_grad():
        initial = model.forward_dummy(data['lq'], True, **args)[3]
        background = initial['branch_outputs'][1].clone()
    optimizer = build_optimizer(model, dict(type='Adam', lr=.001, weight_decay=.01))
    for step in range(3):
        model.train()
        result = model.train_step(data, optimizer)
        assert all(torch.isfinite(torch.tensor(x)) for x in result['log_vars'].values())
        assert not model.backbone.training
        assert all(not p.requires_grad and p.grad is None and torch.equal(p, before[n])
                   for n,p in model.named_parameters() if n in before)
        assert model.foreground_gain.output.weight.grad.abs().sum() > 0
    assert model.foreground_gain.adapter[0].weight.grad.abs().sum() > 0
    logs = result['log_vars']
    assert 'loss_background_lightness' not in logs
    assert 'loss_foreground_lightness' in logs and 'loss_gain_smooth' in logs
    assert not any(k.startswith('loss_tone_') for k in logs)
    with torch.no_grad():
        output, _, _, aux = model.forward_dummy(data['lq'], True, **args)
        assert torch.equal(aux['branch_outputs'][1], background)
        zero = data['foreground_mask'].expand_as(output) == 0
        assert torch.equal(output[zero], background[zero])
    path = str(tmp_path / 'resume.pth')
    torch.save(dict(state_dict=model.state_dict()), path)
    resumed = small_model().cuda().train()
    load_enhancer_checkpoint(resumed, path)
    resumed.train_step(data, build_optimizer(resumed, dict(type='Adam',lr=.001)))
    assert resumed.cnt_iters.item() == 4
    assert all(not p.requires_grad for n,p in resumed.named_parameters() if n in before)


@CUDA
def test_baseline_identity_full_model_crop_and_absent_foreground(tmp_path):
    torch.manual_seed(21)
    baseline = small_model(foreground_gain_enabled=False).cuda().eval()
    for head in baseline.local_decoder.outputs:
        torch.nn.init.normal_(head.weight, std=.01)
    path = str(tmp_path / 'base.pth')
    torch.save(dict(state_dict=baseline.state_dict()), path)
    model = small_model().cuda().eval()
    initialize_foreground_gain_from_baseline(model, path)
    data = batch()
    args = {k:v for k,v in data.items() if k not in ('lq','gt','supervision_mask')}
    with torch.no_grad():
        expected = baseline.forward_dummy(data['lq'], **args)[0]
        full = model.forward_dummy(data['lq'], **args)[0]
        assert torch.equal(expected, full)
        model.foreground_gain.output.weight.normal_(std=.1)
        model.foreground_gain.output.bias.fill_(.2)
        full, _, _, aux = model.forward_dummy(data['lq'], True, **args)
        for delta,mask in zip(aux['context_deltas'],aux['local_masks']):
            assert ((delta * mask).sum((-1,-2))/mask.sum((-1,-2))).abs().max() < 1e-6
        crop = data['lq'][...,12:36,16:48]
        args.update(foreground_mask=data['foreground_mask'][...,12:36,16:48],
                    crop_box=crop.new_tensor([[.25,.25,.75,.75]]).repeat(2,1))
        cropped = model.forward_dummy(crop, **args)[0]
        assert torch.allclose(cropped, full[...,12:36,16:48],atol=3e-6)
        empty=torch.zeros_like(data['foreground_mask'])
        output, _, _, aux=model.forward_dummy(data['lq'],True,foreground_mask=empty)
        original=baseline.forward_dummy(data['lq'],foreground_mask=empty)[0]
        assert torch.equal(output,original) and (aux['foreground_gain']==1).all()
