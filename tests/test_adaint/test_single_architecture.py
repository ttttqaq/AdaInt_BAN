"""Functional and gradient regressions for the branch's sole enhancement model."""
import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from mmcv import Config

from adaint.config_utils import load_config
from adaint.context_transform import ContextAwareAugment
from adaint.experiment_utils import (inference_config, load_enhancer_checkpoint,
    resolve_training_schedule, save_generation_configs, LUTResumeCounterHook)
from adaint.model import AiLUT
from adaint.modules import (NormalizedResidualLUT, ColorStatistics, crop_map,
    identity_lut, lut_regularizations, stratified_pixels)
from adaint.loss import KD_Loss
from tools.prepare_lut_experiment import prepare


def small_model(**extra):
    options = dict(n_vertices=9, curve_points=4, n_ranks=3, stats_bins=8,
                   stats_pool_size=16, condition_dim=32, head_hidden_dim=16,
                   local_resolution=32, local_channels=4, context_size=32,
                   train_cfg=dict(total_iters=101, n_fix_iters=0))
    options.update(extra)
    return AiLUT(**options)


def test_signed_router_has_unit_rms_bases_scale_invariance_and_large_logit_gradients():
    lut = NormalizedResidualLUT(9, 3, 8, 4)
    with torch.no_grad():
        lut.head.output.weight.zero_()
        lut.head.output.bias.fill_(30.)
    condition = torch.randn(2, 8)
    weights, residual = lut(condition)
    assert torch.equal(weights, torch.full_like(weights, 30.))
    weights.sum().backward()
    assert torch.equal(lut.head.output.bias.grad, torch.full_like(lut.head.output.bias, 2.))
    bases = lut.normalized_bases()
    assert torch.allclose(bases.square().mean((1, 2, 3, 4)), torch.ones(3), atol=1e-6)
    with torch.no_grad():
        lut.basis_luts.mul_(10.)
    assert torch.allclose(residual, lut(condition)[1], atol=1e-4)


def test_nonuniform_lut_regularization_uses_real_slopes_and_diagonal_only():
    uniform = torch.linspace(0, 1, 9)
    for axis in (uniform, uniform.pow(3)):
        vertices = axis.view(1, 1, 9).repeat(1, 3, 1)
        identity = identity_lut(vertices)
        assert all(value.abs() < 1e-7 for value in lut_regularizations(identity, vertices))
        lut = identity.clone()
        lut[:, 0] = -identity[:, 0]
        smooth, mono, _ = lut_regularizations(lut, vertices)
        assert smooth.item() == pytest.approx(4./3., abs=1e-5)
        assert mono.item() == pytest.approx(1./3., abs=1e-5)
        lut = identity.clone()
        lut[:, 0] -= .2*identity[:, 1]
        assert lut_regularizations(lut, vertices)[1] == 0


def test_stratified_statistics_keep_extreme_colors_and_training_samples_match():
    torch.manual_seed(2)
    binary = torch.randint(0, 2, (1, 1, 128, 128)).float().repeat(1, 3, 1, 1)
    stats = ColorStatistics(8, 16)
    samples = stratified_pixels(binary, 16)
    assert torch.equal(stats(binary), stats(binary, samples))
    assert stats(binary)[:, 4*8+8:4*8+16].sum() > 3.9


def test_context_augmentation_alignment_float_resize_and_samples():
    random.seed(4)
    image = np.random.RandomState(7).randint(0, 256, (48, 64, 3)).astype(np.uint8)
    result = ContextAwareAugment(context_size=32, samples_size=16,
                                 brightness=0, saturation=0, flip_ratio=0)(
                                     dict(lq=image.copy(), gt=image.copy()))
    box = result['crop_box'].numpy()
    x0, y0, x1, y1 = np.rint(box*[64, 48, 64, 48]).astype(int)
    assert np.array_equal(result['lq'], image[y0:y1, x0:x1])
    assert np.array_equal(result['gt'], result['lq'])
    expected = torch.nn.functional.interpolate(
        torch.from_numpy(image).permute(2, 0, 1)[None].float()/255., (32, 32),
        mode='bilinear', align_corners=False)[0].permute(1, 2, 0).numpy()
    assert np.allclose(result['lq_context'], expected, atol=1e-7)
    assert result['context_samples'].shape == (16, 16, 3)


def test_crop_map_pixel_edges_and_gradient():
    ramp = torch.arange(80.).view(1, 1, 8, 10).requires_grad_()
    result = crop_map(ramp, torch.tensor([[.2, .25, .8, .75]]), (4, 6))
    assert torch.allclose(result, ramp[:, :, 2:6, 2:8], atol=1e-5)
    result.sum().backward()
    assert ramp.grad.sum() == 24


def test_kd_decay_and_gt_has_strict_preference():
    model = small_model()
    for iteration, expected in ((0, 1.), (30, 1.), (65, .5), (100, 0.)):
        model.cnt_iters.fill_(iteration)
        assert model.kd_multiplier() == pytest.approx(expected)
    gt, teacher = torch.tensor(0.), torch.tensor(1.)
    midpoint = (torch.tensor(.5)-gt).square()+.1*(torch.tensor(.5)-teacher).square()
    near_gt = (torch.tensor(.1)-gt).square()+.1*(torch.tensor(.1)-teacher).square()
    assert near_gt < midpoint


def test_frozen_teacher_receives_context_and_no_gradients():
    class Teacher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(.2))
            self.context = None
        def forward_dummy(self, x, **context):
            self.context = context
            return x+self.bias, None, None
    kd = KD_Loss.__new__(KD_Loss)
    torch.nn.Module.__init__(kd)
    kd.teacher, kd.loss_weight = Teacher(), .1
    x = torch.zeros(1, 3, 4, 4)
    pred = x.clone().requires_grad_()
    kd(pred, x, crop_box='context').backward()
    assert kd.teacher.context == dict(crop_box='context')
    assert kd.teacher.bias.grad is None and pred.grad.abs().sum() > 0


def test_single_model_configs_and_strict_checkpoint_loading(tmp_path):
    import yaml
    with open('adaint/configs/ppr10k.yaml') as handle:
        base = yaml.safe_load(handle)
    train, test = prepare(base, 'unit_single', generations=4)
    assert train['model'] == test['model'] and train['model']['type'] == 'AiLUT'
    assert train['enable_kd'] and train['kdloss_type'] == 'MSELoss' and train['kdloss_weight'] < 1
    cfg = load_config('adaint/configs/ppr10k.yaml')
    cfg.gpus, cfg.work_dir = 1, str(tmp_path)
    resolve_training_schedule(cfg, 8875)
    assert cfg.train_cfg.total_iters == 10500
    save_generation_configs(cfg)
    assert load_config(str(tmp_path/'inference_config.py')).train_cfg is None
    model = small_model()
    state = copy.deepcopy(model.state_dict())
    state['kd_loss.teacher.weight'] = torch.ones(1)
    path = tmp_path/'model.pth'
    torch.save(dict(state_dict=state), str(path))
    load_enhancer_checkpoint(model, str(path))
    model.kd_loss = torch.nn.Linear(1, 1)
    teacher_state = copy.deepcopy(model.kd_loss.state_dict())
    load_enhancer_checkpoint(model, str(path))
    assert all(torch.equal(value, model.kd_loss.state_dict()[key])
               for key, value in teacher_state.items())
    state.pop('local_decoder.output.weight')
    torch.save(dict(state_dict=state), str(path))
    with pytest.raises(RuntimeError):
        load_enhancer_checkpoint(model, str(path))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='real AiLUT CUDA kernel')
def test_original_coordinate_identity_and_local_crop_agreement():
    torch.manual_seed(4)
    model = small_model().cuda().eval()
    with torch.no_grad():
        model.lut_generator.head.output.weight.zero_()
        model.lut_generator.head.output.bias.zero_()
        # Deliberately nonuniform coordinates still reproduce identity.
        model.adaint.head.output.bias.copy_(torch.randn_like(model.adaint.head.output.bias))
    full = torch.rand(1, 3, 48, 64, device='cuda')
    with torch.no_grad():
        y, _, _, aux = model.forward_dummy(full, True)
        assert (y-full).abs().max() < 2e-6
        assert aux['local_delta'].abs().max() == 0
        model.local_decoder.output.weight.normal_(std=.02)
        y, _, _, full_aux = model.forward_dummy(full, True)
        crop = full[:, :, 12:36, 16:48]
        box = full.new_tensor([[.25, .25, .75, .75]])
        samples = stratified_pixels(full, 16)
        crop_y, _, _, crop_aux = model.forward_dummy(crop, True, full, samples, box)
        assert torch.allclose(crop_aux['delta_mean'], full_aux['delta_mean'], atol=1e-7)
        assert torch.allclose(crop_y, y[:, :, 12:36, 16:48], atol=3e-6)
        assert full_aux['context_delta'].mean((-1, -2)).abs().max() < 1e-7
        assert crop_aux['local_delta'].abs().mean() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='real AiLUT CUDA kernel')
def test_zero_local_head_receives_gradient_and_all_losses_train():
    torch.manual_seed(5)
    model = small_model().cuda().eval()
    x = torch.rand(2, 3, 32, 48, device='cuda')
    context = torch.rand(2, 3, 48, 64, device='cuda')
    samples = stratified_pixels(context, 16)
    boxes = x.new_tensor([[.25, .25, .75, .75], [.1, .1, .9, .9]])
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    for step in range(3):
        result = model.forward_train(x, x*.8, context, samples, boxes)
        loss, logs = model.parse_losses(result['losses'])
        assert all(np.isfinite(value) for value in logs.values())
        optimizer.zero_grad()
        loss.backward()
        assert model.local_decoder.output.weight.grad.abs().sum() > 0
        for module in (model.tone_curve.head.output, model.adaint.head.output,
                       model.lut_generator.head.output):
            assert module.weight.grad is not None and torch.isfinite(module.weight.grad).all()
        assert model.lut_generator.basis_luts.grad.abs().sum() > 0
        optimizer.step()
    assert model.local_decoder.film.weight.grad.abs().sum() > 0
    assert model.forward_dummy(x)[1].std(0).mean() > 0
