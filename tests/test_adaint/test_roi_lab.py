"""ROI geometry, color/gamut math, all-parameter training and strict initialization."""
import copy
import random

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from mmcv import ConfigDict
from PIL import Image

from adaint.color import display_p3_to_lab
from adaint.config_utils import load_config
from adaint.experiment_utils import (initialize_roi_lab_from_basis, load_enhancer_checkpoint,
                                    resolve_training_schedule, inference_config)
from adaint.model_roi_lab import AiLUTROILab
from adaint.roi_lab import (ForegroundROILab, extract_roi, sample_roi, paste_roi,
    lab_to_linear_p3, encode_p3, decode_p3, compress_gamut, apply_lab_residual)
from adaint.semantic_pipeline import SemanticContextAugment, LoadSemanticMasks
from tests.test_adaint.test_semantic_dual import batch, small_model


def roi_model(**kwargs):
    options = dict(n_vertices=9, curve_points=4, n_ranks=3, stats_bins=8,
        stats_pool_size=16, condition_dim=32, head_hidden_dim=16,
        local_resolution=32, local_channels=4, context_size=32, region_samples_size=8,
        roi_size=32, roi_channels=4, color_loss_weight=.1,
        train_cfg=dict(total_iters=101, n_fix_iters=0),
        test_cfg=ConfigDict(metrics=['PSNR', 'deltaL_P3_foreground', 'deltaAB_P3_foreground',
                                   'deltaL_P3_background', 'deltaL_P3_highlights'], crop_border=0))
    options.update(kwargs)
    return AiLUTROILab(**options)


def roi_batch(device='cuda'):
    data = batch(device)
    roi, mask, boxes = extract_roi(data['lq'], data['foreground_mask'], 32, .15)
    data.update(foreground_roi=roi, foreground_roi_mask=mask, foreground_roi_box=boxes,
                foreground_roi_gt=sample_roi(data['gt'], boxes, 32),
                foreground_roi_supervision=sample_roi(data['supervision_mask'], boxes, 32, True))
    return data


def test_p3_d65_roundtrip_and_finite_black_gradients():
    torch.manual_seed(4)
    rgb = (torch.rand(2, 3, 7, 11, dtype=torch.double) * 1.3 - .1).requires_grad_()
    recovered = encode_p3(lab_to_linear_p3(display_p3_to_lab(rgb)))
    assert torch.allclose(recovered, rgb, atol=2e-7, rtol=1e-6)
    values = torch.tensor([0., 1e-7, .0031308, .04045, 1.], dtype=torch.double)
    rgb = values[None, None, None].expand(1, 3, 2, -1).clone().requires_grad_()
    delta = torch.zeros_like(rgb, requires_grad=True)
    result, actual = apply_lab_residual(rgb, delta)
    assert torch.allclose(result, rgb, atol=2e-6)
    (result.square().mean() + actual.square().mean()).backward()
    assert torch.isfinite(rgb.grad).all() and torch.isfinite(delta.grad).all()


def test_gamut_compression_preserves_luminance_and_has_finite_gradients():
    linear = torch.tensor([[[[-.15, 1.3, .2]], [[.5, .2, .2]], [[.3, -.1, .2]]]], requires_grad=True)
    result = compress_gamut(linear)
    w = linear.new_tensor([35783 / 156275, 247089 / 357200, 198249 / 2500400])[None, :, None, None]
    assert result.min() >= -1e-6 and result.max() <= 1 + 1e-6
    assert torch.allclose((result * w).sum(1), (linear * w).sum(1), atol=1e-6)
    assert torch.allclose(result[..., 2], linear[..., 2], atol=1e-6)
    result.square().mean().backward()
    assert torch.isfinite(linear.grad).all()


def test_native_roi_retains_detail_lost_by_whole_image_downsampling():
    image = torch.zeros(1, 3, 1024, 1024)
    image[:, 0, 100:108, 100:104] = 1
    image[:, 2, 100:108, 104:108] = 1
    mask = torch.zeros(1, 1, 1024, 1024)
    mask[..., 100:108, 100:108] = 1
    native, roi_mask, box = extract_roi(image, mask, 32, 0)
    coarse = F.interpolate(image, (32, 32), mode='bilinear', align_corners=False)
    fake_detail = sample_roi(coarse, box, 32)
    assert native.std() > .3 and fake_detail.abs().max() == 0
    assert roi_mask.mean() > .9


def test_square_roi_padding_empty_mask_and_crop_coordinate_equivalence():
    image = torch.rand(2, 3, 40, 80)
    mask = torch.zeros(2, 1, 40, 80)
    mask[0, :, :20, :10] = 1
    roi, m, box = extract_roi(image, mask, 32, .25)
    side = box[0, 2:] - box[0, :2]
    assert side[0] * 80 == pytest.approx(float(side[1] * 40), abs=1e-5)
    assert box[0, 0] < 0 and m[0].min() == 0
    assert torch.isfinite(roi).all() and m[1].sum() == 0
    field = torch.randn(2, 3, 32, 32)
    full = paste_roi(field, box, (40, 80))
    crop = field.new_tensor([[.125, .25, .75, .9]]).repeat(2, 1)
    cropped = paste_roi(field, box, (26, 50), crop)
    assert torch.allclose(full[..., 10:36, 10:60], cropped, atol=2e-5)


def test_centered_lab_fields_are_bounded_and_empty_foreground_is_inactive():
    torch.manual_seed(6)
    module = ForegroundROILab(16, 4, 8., 4.)
    roi = torch.rand(2, 3, 32, 32)
    mask = torch.zeros(2, 1, 32, 32)
    mask[0, :, 3:27, 5:25] = 1
    active = torch.tensor([1., 0.]).view(2, 1, 1, 1)
    features, addition = module.encode(roi, mask, active)
    assert addition.abs().max() == 0
    with torch.no_grad():
        module.output.weight.normal_(0, .1)
    field = module.decode(features, roi, torch.rand(2, 16), mask, active)
    mean = (field[0] * mask[0]).sum((-1, -2)) / mask[0].sum()
    assert mean.abs().max() < 1e-6 and (field.abs() <= module.limits + 1e-6).all()
    assert field[1].abs().max() == 0
    field.square().mean().backward()
    assert torch.isfinite(module.output.weight.grad).all() and module.output.weight.grad.abs().sum() > 0


def test_spatial_head_can_learn_opposite_lightness_corrections_in_one_roi():
    # A mechanism check beyond nonzero gradients: fit a known, zero-mean
    # correction that lifts the dark half and dims the bright half.
    torch.manual_seed(14)
    module = ForegroundROILab(16, 4, 8., 4.)
    source = torch.full((1, 3, 32, 32), .25)
    source[..., 16:] = .65
    mask = torch.ones(1, 1, 32, 32)
    active = torch.ones(1, 1, 1, 1)
    condition = torch.zeros(1, 16)
    target = torch.full((1, 1, 32, 32), 1.)
    target[..., 16:] = -1.
    optimizer = torch.optim.Adam(module.parameters(), lr=.003)
    initial = target.square().mean()
    for step in range(80):
        features, _ = module.encode(source, mask, active)
        prediction = module.decode(features, source, condition, mask, active)[:, :1]
        loss = (prediction - target).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss < initial * .3
    assert prediction[..., :16].mean() > .5 and prediction[..., 16:].mean() < -.5
    assert prediction.mean().abs() < 1e-5


def test_full_roi_and_gt_are_captured_before_aligned_crop_and_flip():
    random.seed(4)
    image = np.random.RandomState(9).randint(0, 256, (48, 64, 3)).astype(np.uint8)
    target = (image.astype(np.float32) * .8).astype(np.uint8)
    mask = np.zeros((48, 64, 1), np.float32)
    mask[5:34, 10:35] = 1
    result = SemanticContextAugment(context_size=32, samples_size=16, region_samples_size=8,
        roi_size=32, roi_margin=.15, flip_ratio=1., brightness=0, saturation=0,
        crop_ratio=(.6, .7), require_foreground=True)(
        dict(lq=image.copy(), gt=target.copy(), foreground_mask=mask.copy(), supervision_mask=mask.copy()))
    source = torch.from_numpy(np.flip(image, 1).copy()).permute(2, 0, 1)[None].float() / 255
    gt = torch.from_numpy(np.flip(target, 1).copy()).permute(2, 0, 1)[None].float() / 255
    expected = sample_roi(source, result['foreground_roi_box'][None], 32)
    expected_gt = sample_roi(gt, result['foreground_roi_box'][None], 32)
    assert np.allclose(result['foreground_roi'], expected[0].permute(1, 2, 0).numpy())
    assert np.allclose(result['foreground_roi_gt'], expected_gt[0].permute(1, 2, 0).numpy())
    assert result['supervision_mask'].sum() >= .5 * mask.sum()


def test_explicit_mask_exclusion_is_shared_by_fusion_supervision_without_editing_files(tmp_path):
    mask = np.zeros((100, 80), np.uint8)
    mask[20:75, 25:55] = 255
    path = tmp_path / '82_face_mask.png'
    Image.fromarray(mask).save(str(path))
    loader = LoadSemanticMasks(str(tmp_path), foreground_ignore_boxes={'82': [[0, 0, 1, .42]]})
    result = loader(dict(lq_path='82_input.jpg', lq=np.zeros((100, 80, 3), np.uint8)))
    assert result['supervision_mask'][:42].sum() == 0
    assert result['foreground_mask'][:41].sum() == 0
    assert np.array_equal(np.asarray(Image.open(path)), mask)


def test_config_joint_schedule_roi_collection_and_original_configs_unchanged():
    cfg = load_config('adaint/configs/face_niguang_roi_lab.yaml')
    cfg.gpus = 1
    resolve_training_schedule(cfg, 76)
    assert cfg.train_cfg.n_fix_iters == 0 and cfg.total_iters == 76 * 30
    augment = next(s for s in cfg.data.train.pipeline if s['type'] == 'SemanticContextAugment')
    assert augment['roi_size'] == 128 and augment['brightness'] == augment['saturation'] == 0
    keys = cfg.data.train.pipeline[-1]['keys']
    assert all(k in keys for k in ('foreground_roi', 'foreground_roi_gt', 'foreground_roi_box'))
    assert cfg.optimizers.paramwise_cfg.custom_keys.backbone.lr_mult == .1
    assert cfg.optimizers.paramwise_cfg.custom_keys.branches.lr_mult == .2
    assert cfg.semantic_masks.foreground_ignore_boxes['82'][0][-1] == .42
    old = load_config('adaint/configs/face_niguang_semantic_basis.yaml')
    assert old.adaint_fix_epochs == 5 and 'foreground_ignore_boxes' not in old.semantic_masks
    assert not any('roi_size' in s for s in old.data.train.pipeline)
    assert inference_config(cfg).model.type == 'AiLUTROILab'


def test_basis_initialization_only_allows_replaced_fg_parameters_and_new_roi_keys(tmp_path):
    torch.manual_seed(9)
    baseline = small_model(lut_mode='basis')
    baseline.cnt_iters.fill_(300)
    state = copy.deepcopy(baseline.state_dict())
    path = str(tmp_path / 'basis.pth')
    torch.save(dict(state_dict=state), path)
    model = roi_model()
    initialize_roi_lab_from_basis(model, path)
    assert model.cnt_iters.item() == 0
    for name, value in model.state_dict().items():
        if not name.startswith('foreground_roi.') and name != 'cnt_iters':
            assert torch.equal(value, state[name]), name
    assert not any(name.startswith(('local_decoder.outputs.0.', 'local_decoder.films.0.'))
                   for name in model.state_dict())
    assert all(p.requires_grad for p in model.parameters())
    # Full inference/resume loading remains strict; migration is NOT a blanket non-strict load.
    with pytest.raises(RuntimeError, match='incompatible'):
        load_enhancer_checkpoint(model, path)
    for broken in ('missing', 'unexpected', 'shape'):
        data = copy.deepcopy(state)
        if broken == 'missing':
            data.pop('branches.background.adaint.head.output.weight')
        elif broken == 'unexpected':
            data['spatial_tone.weight'] = torch.ones(1)
        else:
            data['backbone.0.0.weight'] = torch.ones(1)
        torch.save(dict(state_dict=data), path)
        with pytest.raises(RuntimeError, match='incompatible'):
            initialize_roi_lab_from_basis(roi_model(), path)
    torch.save(dict(state_dict=model.state_dict()), path)
    restored = roi_model()
    load_enhancer_checkpoint(restored, path)
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())


def test_freeze_configuration_is_rejected_and_all_modules_follow_train_mode():
    with pytest.raises(ValueError, match='joint training'):
        roi_model(train_cfg=dict(n_fix_iters=10))
    model = roi_model().train()
    assert all(m.training for m in model.modules())
    assert all(p.requires_grad for p in model.parameters())
    model.eval()
    assert all(not m.training for m in model.modules())


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='native AiLUT CUDA kernel')


@CUDA
def test_all_retained_parameter_groups_get_gradients_and_update_with_no_freeze():
    torch.manual_seed(22)
    model = roi_model().cuda().train()
    data = roi_batch()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    groups = ('backbone.', 'branches.foreground.', 'branches.background.', 'local_decoder.',
              'branches.foreground.adaint.', 'branches.background.adaint.',
              'foreground_roi.shallow.', 'foreground_roi.condition.', 'foreground_roi.output.')
    before = {n:p.detach().clone() for n,p in model.named_parameters()}
    seen = {group:0. for group in groups}
    for step in range(3):
        result = model.train_step(data, optimizer)
        assert all(np.isfinite(v) for v in result['log_vars'].values())
        assert all(p.requires_grad for p in model.parameters()) and not model.adaint_fixed
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        for group in groups:
            seen[group] += sum(float(p.grad.abs().sum()) for n,p in model.named_parameters() if n.startswith(group))
    for group in groups:
        assert seen[group] > 0, group
        assert any(not torch.equal(before[n], p) for n,p in model.named_parameters() if n.startswith(group)), group


@CUDA
def test_roi_gt_changes_losses_but_never_prediction_or_condition():
    torch.manual_seed(23)
    model = roi_model().cuda().eval()
    data = roi_batch()
    with torch.no_grad():
        original = model.forward_train(**data)
        changed = dict(data)
        changed['foreground_roi_gt'] = torch.zeros_like(data['foreground_roi_gt'])
        alternate = model.forward_train(**changed)
    assert torch.allclose(original['results']['output'], alternate['results']['output'], atol=2e-5)
    assert original['losses']['loss_roi_mean_lightness'] != alternate['losses']['loss_roi_mean_lightness']


@CUDA
@pytest.mark.parametrize('value', [0., 1.])
def test_empty_or_full_foreground_is_finite_at_inference(value):
    model = roi_model().cuda().eval()
    image = torch.rand(1, 3, 48, 64, device='cuda')
    mask = torch.full((1, 1, 48, 64), value, device='cuda')
    with torch.no_grad():
        output, _, _, aux = model.forward_dummy(image, True, foreground_mask=mask)
    assert torch.isfinite(output).all()
    if value == 0:
        assert torch.equal(output, aux['branch_outputs'][1])
        assert aux['roi_requested_delta'].abs().max() == 0


@CUDA
def test_native_inference_roi_matches_full_training_roi_context():
    torch.manual_seed(24)
    model = roi_model().cuda().eval()
    data = roi_batch()
    keys = ('lq_context','context_samples','crop_box','foreground_mask','foreground_context',
            'foreground_samples','foreground_region_samples','background_region_samples','region_area',
            'foreground_roi','foreground_roi_mask','foreground_roi_box')
    with torch.no_grad():
        native = model.forward_dummy(data['lq'], foreground_mask=data['foreground_mask'])[0]
        contextual = model.forward_dummy(data['lq'], **{k:data[k] for k in keys})[0]
    assert torch.allclose(native, contextual, atol=2e-4)
