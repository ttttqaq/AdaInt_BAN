"""Tone invariants, checkpoint isolation, crop alignment and staged gradients."""
import copy

import pytest
import torch
from mmcv import ConfigDict

from adaint.color import display_p3_to_lab
from adaint.config_utils import load_config
from adaint.experiment_utils import (initialize_tone_from_baseline, load_enhancer_checkpoint,
                                     public_training_config, resolve_training_schedule)
from adaint.model_semantic import AiLUTDual, region_samples
from adaint.modules import stratified_pixels
from adaint.semantic_dataset import SemanticFace
from adaint.spatial_tone import (SpatialLuminanceTone, decode_p3, encode_p3,
                                 p3_luminance, region_lightness_loss)


def model_options(**kwargs):
    options = dict(lut_mode='basis', n_vertices=9, n_ranks=3, curve_points=4,
        stats_bins=8, stats_pool_size=16, condition_dim=32, head_hidden_dim=16,
        context_size=32, local_resolution=32, local_channels=4, region_samples_size=8,
        spatial_tone_enabled=True, tone_resolution=32, tone_channels=4,
        region_lightness_weight=.1, train_cfg=dict(total_iters=10, n_fix_iters=0))
    options.update(kwargs)
    return options


def sample_batch():
    image = torch.rand(2, 3, 48, 64, device='cuda') * .7
    mask = torch.zeros_like(image[:, :1])
    mask[:, :, 8:35, 8:43] = 1
    return dict(lq=image, gt=image * .85, foreground_mask=mask, supervision_mask=mask,
        lq_context=image, context_samples=stratified_pixels(image, 16),
        foreground_context=mask, foreground_samples=stratified_pixels(mask, 16),
        foreground_region_samples=region_samples(image, mask, 8),
        background_region_samples=region_samples(image, 1 - mask, 8),
        region_area=torch.cat((mask.mean((-1, -2)), 1 - mask.mean((-1, -2))), 1),
        crop_box=image.new_tensor([[0., 0., 1., 1.]]).repeat(2, 1))


def tone_fixture(curves=3):
    torch.manual_seed(2026)
    tone = SpatialLuminanceTone(8, 12, curves=curves, resolution=32, channels=4)
    image = torch.rand(2, 3, 48, 64) * .8
    mask = torch.rand_like(image[:, :1])
    conditions = [torch.randn(2, 8), torch.randn(2, 8)]
    return tone, image, mask, conditions


@pytest.mark.parametrize('curves', [1, 3])
def test_identity_linear_p3_black_signed_values_and_finite_gradients(curves):
    tone, image, mask, conditions = tone_fixture(curves)
    image[:, :, 0, :4] = torch.tensor([0., 1e-8, -.01, 1.1])[None, None]
    image.requires_grad_()
    fields, _ = tone.predict(image, mask, conditions)
    output, gain = tone.apply_tone(image, fields[0], image)
    assert torch.allclose(output, image, atol=2e-6)
    assert torch.allclose(gain, torch.ones_like(gain), atol=2e-6)
    output.sum().backward()
    assert torch.isfinite(image.grad).all()
    assert torch.allclose(encode_p3(decode_p3(image)), image, atol=2e-6)
    white = torch.ones(1, 3, 1, 1)
    assert p3_luminance(white).item() == pytest.approx(1.)
    # Neutral linear Y and Lab L* agree with the existing supervision convention.
    gray = torch.full((1, 3, 1, 1), .2)
    y = p3_luminance(decode_p3(gray))
    assert display_p3_to_lab(gray)[0, 0, 0, 0].item() == pytest.approx((116 * y.pow(1/3) - 16).item())


def test_curves_stay_monotone_bounded_and_common_gain_avoids_channel_clipping():
    tone, image, mask, conditions = tone_fixture()
    with torch.no_grad():
        for head in tone.curve_heads:
            head.output.bias.view(3, 16)[:, 0] = 100
    fields, reference = tone.predict(image, mask, conditions)
    for field in fields:
        assert (torch.diff(field['knots'], dim=-1) >= 0).all()
        assert (field['slopes'] <= 4 + 1e-6).all()
        assert torch.allclose(field['knots'][..., 0], torch.zeros(2, 3))
        assert torch.allclose(field['knots'][..., -1], torch.ones(2, 3))
    image[:] = image.new_tensor([.99, .5, .1])[None, :, None, None]
    output, gain = tone.apply_tone(image, fields[0], image)
    assert gain.max() <= 4 and output.max() <= 1 + 1e-6
    before, after = decode_p3(image), decode_p3(output)
    assert torch.allclose(after[:, 1:] / after[:, :1], before[:, 1:] / before[:, :1], atol=1e-5)
    curve, spatial = tone.regularizers(fields, reference, mask, .0001)
    assert curve > 0 and spatial >= 0
    assert all(torch.isfinite(x) for x in tone.regularizers(fields, reference, torch.zeros_like(mask), .0001))


def test_full_context_filtered_logits_apply_to_crop_in_same_coordinates():
    tone, image, mask, conditions = tone_fixture()
    with torch.no_grad():
        for head in tone.curve_heads:
            head.output.bias.normal_(std=.5)
    fields, _ = tone.predict(image, mask, conditions)
    full, _ = tone.apply_tone(image, fields[0], image)
    crop = image[..., 12:36, 16:48]
    box = image.new_tensor([[.25, .25, .75, .75]]).repeat(2, 1)
    result, _ = tone.apply_tone(crop, fields[0], crop, box)
    assert torch.allclose(result, full[..., 12:36, 16:48], atol=3e-6)


def test_mask_and_conditions_affect_spatial_weights_and_break_curve_symmetry():
    tone, image, mask, conditions = tone_fixture()
    fields, _ = tone.predict(image, mask, conditions)
    alternate, _ = tone.predict(image, 1 - mask, conditions)
    assert not torch.equal(fields[0]['low_logits'], alternate[0]['low_logits'])
    output, _ = tone.apply_tone(image, fields[0], image)
    (output - image * .8).square().mean().backward()
    gradient = tone.curve_heads[0].output.bias.grad.view(3, 16)
    assert not torch.allclose(gradient[0], gradient[1], atol=1e-10, rtol=1e-5)
    assert tone.curve_heads[0] is not tone.curve_heads[1]
    assert tone.films[0] is not tone.films[1]


def test_lightness_loss_uses_valid_unweighted_area_and_gt_error():
    difference = torch.full((3, 1, 10, 10), .1, requires_grad=True)
    mask = torch.ones_like(difference)
    mask[1:] = 0
    mask[1, :, 0, 0] = 1
    difference.data[1:] = .9
    value = region_lightness_loss(difference, torch.zeros_like(mask), mask, .02, 100.)
    assert value.item() == pytest.approx((.1 ** 2 + 1e-6) ** .5 - .001)
    value.backward()
    assert difference.grad[1:].sum() == 0
    assert region_lightness_loss(difference, mask, torch.zeros_like(mask), .02).item() == 0
    assert region_lightness_loss(torch.zeros_like(difference), mask, mask, .02).item() == 0


def test_baseline_initialization_only_allows_new_tone_keys_and_resets_counter(tmp_path):
    baseline = AiLUTDual(**model_options(spatial_tone_enabled=False))
    baseline.cnt_iters.fill_(999)
    path = str(tmp_path / 'baseline.pth')
    torch.save(dict(state_dict=baseline.state_dict()), path)
    model = AiLUTDual(**model_options())
    initialize_tone_from_baseline(model, path)
    assert model.cnt_iters.item() == 0
    for key, value in baseline.state_dict().items():
        if key != 'cnt_iters':
            assert torch.equal(model.state_dict()[key], value)
    with pytest.raises(RuntimeError, match='incompatible'):
        load_enhancer_checkpoint(model, path)
    state = copy.deepcopy(baseline.state_dict())
    state.pop(next(key for key in state if key.startswith('branches.background.')))
    torch.save(dict(state_dict=state), path)
    with pytest.raises(RuntimeError, match='incompatible tone baseline'):
        initialize_tone_from_baseline(model, path)
    torch.save(dict(state_dict=model.state_dict()), path)
    with pytest.raises(RuntimeError, match='incompatible tone baseline'):
        initialize_tone_from_baseline(model, path)
    load_enhancer_checkpoint(AiLUTDual(**model_options()), path)


def test_config_resolves_stages_and_preserves_data_split():
    cfg = load_config('adaint/configs/face_niguang_semantic_spatial_tone.yaml')
    baseline = load_config('adaint/configs/face_niguang_semantic_basis.yaml')
    assert cfg.data == baseline.data and cfg.semantic_masks == baseline.semantic_masks
    cfg.gpus = 1
    resolve_training_schedule(cfg, 76)
    assert cfg.train_cfg.tone_warmup_iters == 228
    assert cfg.total_iters == 7600 and cfg.train_cfg.n_fix_iters == 0
    assert 'tone_warmup_iters' not in public_training_config(cfg).get('train_cfg', {})
    assert cfg.optimizers.paramwise_cfg.custom_keys.spatial_tone.lr_mult == 10


def test_highlights_metrics_exclude_absent_samples():
    model = AiLUTDual(**model_options(test_cfg=ConfigDict(
        metrics=['deltaL_P3_highlights', 'deltaL_P3_background'], crop_border=0)))
    image = torch.full((1, 3, 8, 8), .9)
    mask = torch.zeros_like(image[:, :1])
    metrics = model.evaluate(image * .9, image, foreground_mask=mask)
    assert metrics['deltaL_P3_highlights'] > 0 and metrics['_valid_highlights'] == 1
    empty = model.evaluate(image * .1, image * .1, foreground_mask=mask)
    assert empty['_valid_highlights'] == 0
    dataset = object.__new__(SemanticFace)
    dataset.data_infos = [dict(), dict()]
    result = dataset.evaluate([dict(eval_result=metrics), dict(eval_result=empty)])
    assert result['deltaL_P3_highlights'] == metrics['deltaL_P3_highlights']
    assert result['highlights_valid_count'] == 1


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='AiLUT CUDA kernel')


@CUDA
def test_warmup_freezes_baseline_then_joint_gradients_and_resume_phase(tmp_path):
    torch.manual_seed(33)
    model = AiLUTDual(**model_options(train_cfg=dict(total_iters=10, n_fix_iters=0,
                                                     tone_warmup_iters=1))).cuda().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    data = sample_batch()
    before = {key: value.detach().clone() for key, value in model.named_parameters()}
    logs = model.train_step(data, optimizer)['log_vars']
    assert all(torch.equal(before[key], value) for key, value in model.named_parameters()
               if not key.startswith('spatial_tone.'))
    assert any(not torch.equal(before[key], value) for key, value in model.named_parameters()
               if key.startswith('spatial_tone.curve_heads.'))
    assert not model.backbone.training and model.spatial_tone.training
    assert all(torch.isfinite(torch.tensor(value)) for value in logs.values())
    model.train()
    model.train_step(data, optimizer)
    assert model.backbone.training and not model._tone_warmup
    for head in model.spatial_tone.curve_heads:
        assert torch.isfinite(head.output.weight.grad).all() and head.output.weight.grad.abs().sum() > 0
    for head in model.spatial_tone.spatial_heads:
        assert torch.isfinite(head.weight.grad).all() and head.weight.grad.abs().sum() > 0
    for branch in model.branches.values():
        assert branch.lut_generator.basis_luts.grad.abs().sum() > 0
    path = str(tmp_path / 'joint.pth')
    torch.save(dict(state_dict=model.state_dict()), path)
    resumed = AiLUTDual(**model_options(train_cfg=dict(total_iters=10, n_fix_iters=0,
                                                       tone_warmup_iters=1))).cuda().train()
    load_enhancer_checkpoint(resumed, path)
    resumed.train_step(data, torch.optim.Adam(resumed.parameters(), lr=1e-5))
    assert not resumed._tone_warmup and resumed.cnt_iters.item() == 3


@CUDA
def test_new_tone_preserves_baseline_and_full_model_crop_consistency(tmp_path):
    baseline = AiLUTDual(**model_options(spatial_tone_enabled=False)).cuda().eval()
    path = str(tmp_path / 'baseline.pth')
    torch.save(dict(state_dict=baseline.state_dict()), path)
    model = AiLUTDual(**model_options()).cuda().eval()
    initialize_tone_from_baseline(model, path)
    data = sample_batch()
    args = {key: value for key, value in data.items() if key not in ('lq', 'gt', 'supervision_mask')}
    with torch.no_grad():
        expected = baseline.forward_dummy(data['lq'], **args)[0]
        output = model.forward_dummy(data['lq'], **args)[0]
        assert torch.allclose(output, expected, atol=2e-6)
        for head in model.spatial_tone.curve_heads:
            head.output.bias.normal_(std=.5)
        full = model.forward_dummy(data['lq'], **args)[0]
        crop = data['lq'][..., 12:36, 16:48]
        args.update(foreground_mask=data['foreground_mask'][..., 12:36, 16:48],
                    crop_box=crop.new_tensor([[.25, .25, .75, .75]]).repeat(2, 1))
        result = model.forward_dummy(crop, **args)[0]
        assert torch.allclose(result, full[..., 12:36, 16:48], atol=3e-6)
