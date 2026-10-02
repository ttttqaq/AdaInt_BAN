import copy
import random

import numpy as np
import pytest
import torch
from mmcv import Config

from adaint.config_utils import load_config
from adaint.context_transform import ContextAwareAugment
from adaint.experiment_utils import (
    inference_config, load_enhancer_checkpoint, resolve_training_schedule,
    save_generation_configs, LUTResumeCounterHook)
from adaint.model_architecture_v2 import AiLUTArchitectureV2
from adaint.model_refinement import (
    AiLUTArchitectureV3, AiLUTGlobalLocalV3, AiLUTRefinementBase,
    RefinementStatistics, crop_map, normalized_lab)
from adaint.loss import KD_Loss
from mmedit.models.builder import build_loss
from tools.prepare_lut_experiment import prepare


def small_model(cls=AiLUTRefinementBase, **extra):
    options = dict(n_vertices=9, coarse_vertices=5, curve_points=4,
                   stats_bins=8, stats_pool_size=16, condition_dim=32,
                   head_hidden_dim=16, vertex_embedding_dim=8,
                   guide_resolution=32, guide_channels=4,
                   use_interval_floor=False, use_flexible_tone_endpoints=False,
                   use_diagonal_monotonicity=False, monotonicity_factor=2,
                   train_cfg=dict(total_iters=101, n_fix_iters=0))
    options.update(extra)
    return cls(**options)


def test_lab_reference_colors_and_finite_out_of_range_gradient():
    rgb = torch.tensor([[[[0., 1., 1.]], [[0., 1., 0.]], [[0., 1., 0.]]]])
    lab = normalized_lab(rgb)
    assert torch.allclose(lab[0, :, 0, 0], torch.zeros(3), atol=1e-5)
    assert torch.allclose(lab[0, :, 0, 1], torch.tensor([1., 0., 0.]), atol=1e-4)
    assert torch.allclose(lab[0, :, 0, 2],
                          torch.tensor([.5324, 80.092 / 128, 67.203 / 128]), atol=1e-3)
    outside = torch.tensor([[[[-.1]], [[1.2]], [[0.]]]], requires_grad=True)
    normalized_lab(outside).square().sum().backward()
    assert torch.isfinite(outside.grad).all() and outside.grad.abs().sum() > 0


def test_stratified_statistics_preserve_extremes_removed_by_pooling():
    torch.manual_seed(2)
    binary = torch.randint(0, 2, (1, 1, 128, 128)).float().repeat(1, 3, 1, 1)
    pooled = RefinementStatistics(8, 16, False)(binary)
    sampled = RefinementStatistics(8, 16, True)(binary)
    dark_start = 4 * 8 + 8
    bright_start = dark_start + 4
    assert pooled[:, dark_start:bright_start + 4].sum() == 0
    assert sampled[:, dark_start:bright_start + 4].sum() > 3.9


def test_crop_map_pixel_edge_coordinates_and_gradient():
    ramp = torch.arange(80.).view(1, 1, 8, 10).requires_grad_()
    result = crop_map(ramp, torch.tensor([[.2, .25, .8, .75]]), (4, 6))
    assert torch.allclose(result, ramp[:, :, 2:6, 2:8], atol=1e-5)
    result.sum().backward()
    assert torch.allclose(ramp.grad[:, :, 2:6, 2:8], torch.ones(1, 1, 4, 6))


def test_context_augmentation_alignment_and_samples():
    random.seed(4)
    image = np.random.RandomState(7).randint(0, 256, (48, 64, 3)).astype(np.uint8)
    result = ContextAwareAugment(context_size=32, samples_size=16,
                                 brightness=0, saturation=0, flip_ratio=0)(
                                     dict(lq=image.copy(), gt=image.copy()))
    box = result['crop_box'].numpy()
    x0, y0, x1, y1 = np.rint(box * [64, 48, 64, 48]).astype(int)
    assert np.array_equal(result['lq'], image[y0:y1, x0:x1])
    assert np.array_equal(result['gt'], result['lq'])
    assert result['lq_context'].shape == (32, 32, 3)
    assert result['context_samples'].shape == (16, 16, 3)


def test_diagonal_monotonicity_scale_and_cross_channel_freedom():
    model = small_model()
    vertices = torch.linspace(0, 1, 9).view(1, 1, 9).repeat(1, 3, 1)
    residual = torch.zeros(1, 2, 3, 9, 9, 9)
    residual[:, :, 0] = -2 * vertices[:, 0].view(1, 1, 1, 1, 9)
    _, all_penalty = model._regularizations(vertices, residual)
    model.use_diagonal_monotonicity = True
    _, diagonal_penalty = model._regularizations(vertices, residual)
    assert diagonal_penalty > 0
    assert torch.allclose(all_penalty, diagonal_penalty)
    residual.zero_()
    residual[:, :, 0] = -.2 * vertices[:, 1].view(1, 1, 1, 9, 1)
    assert model._regularizations(vertices, residual)[1] == 0
    model.use_diagonal_monotonicity = False
    assert model._regularizations(vertices, residual)[1] > 0


def test_kd_decay_endpoints_and_resume_counter():
    model = small_model(use_kd_decay=True)
    for iteration, expected in ((0, 1.), (30, 1.), (65, .55), (100, .1)):
        model.cnt_iters.fill_(iteration)
        assert model.kd_multiplier() == pytest.approx(expected)
    model.use_kd_decay = False
    assert model.kd_multiplier() == 1


def test_confidence_kd_reduces_unreliable_teacher_and_freezes_gradients():
    class Teacher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(.2))
        def forward_dummy(self, x):
            return x + self.bias, None, None
    kd = KD_Loss.__new__(KD_Loss)
    torch.nn.Module.__init__(kd)
    kd.teacher = Teacher()
    kd.loss = build_loss(dict(type='L1Loss', loss_weight=1.))
    kd.elementwise_loss = build_loss(dict(type='L1Loss', reduction='none'))
    x = torch.zeros(1, 3, 4, 4)
    pred = x.clone().requires_grad_()
    reliable = kd(pred, x, gt=x + .2, use_confidence=True)
    unreliable = kd(pred, x, gt=x, use_confidence=True)
    assert unreliable < reliable * .05
    unreliable.backward()
    assert kd.teacher.bias.grad is None and pred.grad is not None


def test_epoch_schedule_follows_dataset_and_batch_size():
    cfg = load_config('adaint/configs/face_indoor_refinement.yaml')
    cfg.gpus = 1
    resolve_training_schedule(cfg, 67)
    assert cfg.total_iters == 6700 and cfg.train_cfg.n_fix_iters == 335
    cfg.data.train_dataloader.samples_per_gpu = 2
    resolve_training_schedule(cfg, 1001)
    assert cfg.total_iters == 50100 and cfg.train_cfg.n_fix_iters == 2505
    assert cfg.train_cfg.total_iters == cfg.total_iters


def test_generated_configs_match_and_context_toggle_is_wired(tmp_path):
    import yaml
    with open('adaint/configs/face_indoor_refinement.yaml') as handle:
        base = yaml.safe_load(handle)
    train, test = prepare(base, 'unit_context', 'all')
    assert train['model'] == test['model']
    assert train['model']['type'] == 'AiLUTArchitectureV3'
    assert not any(key in train['model'] for key in ('use_post_tone', 'use_guided_decoder'))
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(train))
    cfg = load_config(str(path))
    steps = cfg.data.train.pipeline
    assert sum(step['type'] == 'ContextAwareAugment' for step in steps) == 1
    assert 'crop_box' in steps[-1]['keys']
    derived = inference_config(cfg)
    assert derived.model == cfg.model and derived.train_cfg is None
    cfg.model.gen = 2
    cfg.model.kd_loss = dict(type='KD_Loss')
    derived = inference_config(cfg)
    assert 'gen' not in derived.model and 'kd_loss' not in derived.model


def test_strict_checkpoint_drops_teacher_but_rejects_shape_mismatch(tmp_path):
    model = small_model()
    state = dict(model.state_dict())
    state['kd_loss.teacher.fake'] = torch.ones(1)
    path = tmp_path / 'teacher.pth'
    torch.save(dict(state_dict=state), str(path))
    load_enhancer_checkpoint(model, str(path))
    state['tone_curve.head.output.bias'] = torch.zeros(99)
    torch.save(dict(state_dict=state), str(path))
    with pytest.raises(RuntimeError):
        load_enhancer_checkpoint(model, str(path))


def test_yaml_source_saves_reloadable_generation_configs(tmp_path):
    cfg = load_config('adaint/configs/face_indoor_refinement.yaml')
    cfg.work_dir = str(tmp_path)
    cfg.model.gen = 1
    cfg.model.kd_loss = dict(type='KD_Loss')
    save_generation_configs(cfg)
    resolved = load_config(str(tmp_path / 'resolved_config.py'))
    derived = load_config(str(tmp_path / 'inference_config.py'))
    assert resolved.model.kd_loss.type == 'KD_Loss'
    assert derived.model.type == cfg.model.type
    assert derived.train_cfg is None and 'kd_loss' not in derived.model


def test_resume_counter_corrects_legacy_zero_based_metadata():
    import logging
    from types import SimpleNamespace
    class Runner:
        def __init__(self):
            self.model = SimpleNamespace(module=SimpleNamespace(cnt_iters=torch.tensor([5.])))
            self._iter = 4
            self.logger = logging.getLogger('resume_test')
        @property
        def iter(self):
            return self._iter
    runner = Runner()
    LUTResumeCounterHook().before_run(runner)
    assert runner.iter == 5


@pytest.mark.skipif(not torch.cuda.is_available(), reason='real AiLUT CUDA extension')
def test_v2_control_forward_equivalence():
    control = small_model().cuda().eval()
    v2 = AiLUTArchitectureV2(
        n_vertices=9, coarse_vertices=5, curve_points=4,
        stats_bins=8, stats_pool_size=16, condition_dim=32,
        head_hidden_dim=16, vertex_embedding_dim=8,
        guide_resolution=32, guide_channels=4,
        use_interval_floor=False, use_flexible_tone_endpoints=False).cuda().eval()
    v2.load_state_dict(control.state_dict(), strict=True)
    x = torch.rand(1, 3, 24, 36, device='cuda')
    with torch.no_grad():
        assert torch.allclose(control.forward_dummy(x)[0], v2.forward_dummy(x)[0], atol=1e-6)


@pytest.mark.parametrize('cls', [AiLUTArchitectureV3, AiLUTGlobalLocalV3])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='real AiLUT CUDA extension')
def test_new_architectures_full_context_losses_and_backward(cls):
    model = small_model(
        cls, use_full_image_context=True, use_lab_loss=True,
        use_rgb_mse_loss=True, use_low_frequency_loss=True,
        use_edge_aware_smoothness=True, use_stratified_statistics=True).cuda().eval()
    x = torch.rand(1, 3, 24, 36, device='cuda')
    context = torch.rand(1, 3, 48, 64, device='cuda')
    samples = torch.rand(1, 3, 16, 16, device='cuda')
    box = torch.tensor([[.2, .1, .8, .9]], device='cuda')
    y, _, _, aux = model.forward_dummy(
        x, return_aux=True, lq_context=context, context_samples=samples, crop_box=box)
    assert y.shape == x.shape and torch.isfinite(y).all()
    guide = aux['guide']
    if cls is AiLUTArchitectureV3:
        assert torch.allclose(guide.sum(1), torch.ones_like(guide[:, 0]), atol=1e-5)
        assert (guide >= 0).all()
    else:
        assert torch.equal(guide[:, 0], torch.ones_like(guide[:, 0]))
        assert guide[:, 1:].abs().max() <= model.local_gate_scale
    outputs = model.forward_train(
        x, x * .8, lq_context=context, context_samples=samples, crop_box=box)
    loss, logs = model.parse_losses(outputs['losses'])
    loss.backward()
    assert all(np.isfinite(value) for value in logs.values())
    for module in (model.tone_curve.head.output,
                   model.post_tone_encoder.stats_projection,
                   model.spatial_guide.output, model.adaint.head.output):
        assert module.weight.grad is not None
        assert torch.isfinite(module.weight.grad).all()
        assert module.weight.grad.abs().sum() > 0
