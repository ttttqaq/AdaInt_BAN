"""Mechanism, data alignment, CUDA gradients, checkpoint and evaluation regressions."""
import copy
import random

import numpy as np
import pytest
import torch
from mmcv import ConfigDict
from PIL import Image

from adaint.config_utils import load_config
from adaint.experiment_utils import load_enhancer_checkpoint, resolve_training_schedule
from adaint.model_semantic import AiLUTDual, region_samples
from adaint.semantic_modules import RegionColorStatistics, masked_per_image
from adaint.semantic_pipeline import LoadSemanticMasks, SemanticContextAugment, region_pixels
from adaint.semantic_dataset import SemanticFace
from adaint.modules import stratified_pixels
from mmedit.datasets import build_dataset


def small_model(**kwargs):
    options = dict(n_vertices=9, curve_points=4, n_ranks=3, stats_bins=8,
        stats_pool_size=16, condition_dim=32, head_hidden_dim=16,
        local_resolution=32, local_channels=4, context_size=32, region_samples_size=8,
        color_loss_weight=.1, train_cfg=dict(total_iters=101, n_fix_iters=0))
    options.update(kwargs)
    return AiLUTDual(**options)


def batch(device='cuda'):
    image = torch.rand(2, 3, 48, 64, device=device)
    mask = torch.zeros(2, 1, 48, 64, device=device)
    mask[0, :, 4:35, 7:43] = 1
    mask[1, :, 11:26, 17:33] = 1
    return dict(lq=image, gt=image * .85, foreground_mask=mask, supervision_mask=mask,
        lq_context=image, context_samples=stratified_pixels(image, 16),
        foreground_context=mask, foreground_samples=stratified_pixels(mask, 16),
        foreground_region_samples=region_samples(image, mask, 8),
        background_region_samples=region_samples(image, 1 - mask, 8),
        region_area=torch.cat((mask.mean((-1, -2)), 1 - mask.mean((-1, -2))), 1),
        crop_box=image.new_tensor([[0., 0., 1., 1.]]).repeat(2, 1))


def test_region_statistics_exclude_masked_black_and_normalize_weights():
    image = torch.zeros(1, 3, 16, 16)
    image[:, :, :2, :3] = .8
    mask = torch.zeros(1, 1, 16, 16)
    mask[:, :, :2, :3] = 1
    criterion = RegionColorStatistics(8, 16)
    features = criterion(image, image, mask)
    assert torch.allclose(features[:, 32:36], torch.full((1, 4), .8), atol=1e-6)
    assert torch.allclose(features, criterion(image, image, mask * .2), atol=1e-5)
    assert torch.isfinite(criterion(image, image, torch.zeros_like(mask))).all()


def test_region_loss_is_per_image_and_empty_regions_do_not_dilute_it():
    error = torch.stack((torch.full((10, 10), 2.), torch.full((10, 10), 6.),
                         torch.full((10, 10), 99.))).requires_grad_()
    mask = torch.zeros(3, 1, 10, 10)
    mask[0, :, :1, :1], mask[1] = 1, 1
    value, _, valid = masked_per_image(error, mask)
    assert value.item() == 4 and valid.tolist() == [1, 1, 0]
    value.backward()
    assert error.grad[2].sum() == 0
    zero = masked_per_image(error, torch.zeros_like(mask))[0]
    assert zero.item() == 0 and torch.isfinite(zero)


def test_original_region_sampling_preserves_single_pixel_foreground():
    image = np.zeros((128, 128, 3), np.uint8)
    image[1, 1] = [190, 70, 30]
    mask = np.zeros((128, 128, 1), np.float32)
    mask[1, 1] = 1
    assert np.all(region_pixels(image, mask, 8) == [190, 70, 30])
    tensor = torch.from_numpy(image).permute(2, 0, 1)[None].float() / 255
    weights = torch.from_numpy(mask).permute(2, 0, 1)[None]
    assert torch.allclose(region_samples(tensor, weights, 8),
                          tensor[:, :, 1:2, 1:2].expand(1, 3, 8, 8))
    assert region_pixels(image, np.zeros_like(mask), 8).sum() == 0


def test_mask_loader_uses_input_id_separate_supervision_and_missing_is_error(tmp_path):
    mask = np.zeros((16, 20), np.uint8)
    mask[3:9, 4:11] = 255
    Image.fromarray(mask).save(str(tmp_path / '12_face_mask.png'))
    loader = LoadSemanticMasks(str(tmp_path), feather_radius=1)
    result = loader(dict(lq_path='/input/12_input.jpg', lq=np.zeros((16, 20, 3), np.uint8)))
    assert np.array_equal(result['supervision_mask'][..., 0], mask / 255)
    assert ((result['foreground_mask'] > 0) & (result['foreground_mask'] < 1)).any()
    assert 'face_mask' not in result
    with pytest.raises(FileNotFoundError):
        loader(dict(lq_path='/input/13_input.jpg', lq=np.zeros((16, 20, 3), np.uint8)))


def test_joint_flip_crop_samples_and_context_use_same_mask_coordinates():
    random.seed(7)
    image = np.random.RandomState(3).randint(0, 256, (48, 64, 3)).astype(np.uint8)
    mask = np.zeros((48, 64, 1), np.float32)
    mask[3:22, 7:31] = 1
    result = SemanticContextAugment(context_size=32, samples_size=16, region_samples_size=8,
        flip_ratio=1., brightness=0., saturation=0.)(
        dict(lq=image.copy(), gt=image.copy(), foreground_mask=mask.copy(),
             supervision_mask=mask.copy(), face_mask=mask.copy()))
    flipped_image, flipped_mask = np.flip(image, 1).copy(), np.flip(mask, 1).copy()
    x0, y0, x1, y1 = np.rint(result['crop_box'].numpy() * [64, 48, 64, 48]).astype(int)
    assert np.array_equal(result['lq'], flipped_image[y0:y1, x0:x1])
    assert np.array_equal(result['foreground_mask'], flipped_mask[y0:y1, x0:x1])
    assert np.array_equal(result['face_mask'], result['supervision_mask'])
    expected = stratified_pixels(torch.from_numpy(flipped_mask).permute(2, 0, 1)[None], 16)
    assert np.array_equal(result['foreground_samples'], expected[0].permute(1, 2, 0).numpy())
    assert result['region_area'][0] == pytest.approx(mask.mean())
    assert result['foreground_region_samples'].shape == (8, 8, 3)


@pytest.mark.parametrize('scene,count', [('indoor', 67), ('outdoor', 85)])
def test_config_wiring_independent_heads_and_epoch_budget(scene, count):
    cfg = load_config('adaint/configs/face_' + scene + '_semantic_dual.yaml')
    assert cfg.model.type == 'AiLUTDual' and not cfg.enable_kd and cfg.gen == 1
    pipeline = cfg.data.train.pipeline
    assert [step['type'] for step in pipeline].count('LoadSemanticMasks') == 1
    assert 'SemanticContextAugment' in [step['type'] for step in pipeline]
    cfg.gpus = 1
    resolve_training_schedule(cfg, count)
    assert cfg.total_iters == count * cfg.train_epochs
    assert cfg.train_cfg.n_fix_iters == count * cfg.adaint_fix_epochs
    model = small_model()
    a, b = model.branches.values()
    for name in ('adaint', 'tone_curve', 'lut_generator', 'condition_encoder'):
        assert getattr(a, name) is not getattr(b, name)
    assert not any(key.startswith(('condition_encoder.', 'lut_generator.', 'adaint.', 'tone_curve.'))
                   for key in model.state_dict())


def test_checkpoint_round_trip_rejects_single_mapping_schema(tmp_path):
    model = small_model()
    path = str(tmp_path / 'dual.pth')
    torch.save(dict(state_dict=model.state_dict()), path)
    load_enhancer_checkpoint(small_model(), path)
    state = copy.deepcopy(model.state_dict())
    state.pop(next(key for key in state if key.startswith('branches.background.')))
    torch.save(dict(state_dict=state), path)
    with pytest.raises(RuntimeError, match='incompatible'):
        load_enhancer_checkpoint(model, path)


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='real AiLUT CUDA kernel')


@CUDA
def test_same_rgb_can_have_different_outputs_and_mask_selects_after_query():
    model = small_model(local_enabled=False).cuda().eval()
    with torch.no_grad():
        for name, sign in (('foreground', 1.), ('background', -1.)):
            mapping = model.branches[name]
            mapping.lut_generator.basis_luts.fill_(1.)
            mapping.lut_generator.head.output.weight.zero_()
            mapping.lut_generator.head.output.bias.fill_(sign * .1 / 3)
        model.branches['background'].adaint.head.output.bias.copy_(torch.linspace(-1, 1, 24))
        image = torch.full((1, 3, 48, 64), .5, device='cuda')
        mask = torch.zeros(1, 1, 48, 64, device='cuda')
        mask[:, :, :, :32] = 1.
        output, _, vertices = model.forward_dummy(image, foreground_mask=mask)
        assert torch.allclose(output[:, :, :, :32], image[:, :, :, :32] + .1, atol=5e-5)  # baseline einsum may use TF32 on Ampere
        assert torch.allclose(output[:, :, :, 32:], image[:, :, :, 32:] - .1, atol=5e-5)
        assert not torch.allclose(vertices[:, 0], vertices[:, 1])
        soft, _, _ = model.forward_dummy(image, foreground_mask=torch.full_like(mask, .25))
        assert torch.allclose(soft, torch.full_like(image, .45), atol=5e-5)


@CUDA
def test_p3_and_regional_losses_reach_both_independent_heads_and_local_outputs():
    torch.manual_seed(17)
    model = small_model().cuda().eval()
    result = model.forward_train(**batch())
    losses = result['losses']
    loss = sum(value for key, value in losses.items() if key.startswith('loss_'))
    assert torch.isfinite(loss)
    loss.backward()
    for mapping in model.branches.values():
        for parameter in (mapping.tone_curve.head.output.weight, mapping.adaint.head.output.weight,
                          mapping.lut_generator.head.output.weight, mapping.lut_generator.basis_luts,
                          mapping.condition_encoder.stats_projection.weight):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0
    for output in model.local_decoder.outputs:
        assert torch.isfinite(output.weight.grad).all() and output.weight.grad.abs().sum() > 0
    assert losses['foreground_valid_fraction'].item() == 1
    assert losses['background_valid_fraction'].item() == 1


@CUDA
def test_region_centering_crop_consistency_and_empty_masks_stay_finite():
    torch.manual_seed(22)
    model = small_model().cuda().eval()
    for output in model.local_decoder.outputs:
        torch.nn.init.normal_(output.weight, std=.02)
    data = batch()
    arguments = {key: value for key, value in data.items() if key not in ('lq', 'gt', 'supervision_mask')}
    full, _, _, aux = model.forward_dummy(data['lq'], True, **arguments)
    for delta, mask in zip(aux['context_deltas'], aux['local_masks']):
        mean = (delta * mask).sum((-1, -2)) / mask.sum((-1, -2))
        assert mean.abs().max() < 1e-6
    cropped = data['lq'][..., 12:36, 16:48]
    arguments.update(crop_box=cropped.new_tensor([[.25, .25, .75, .75]]).repeat(2, 1),
                     foreground_mask=data['foreground_mask'][..., 12:36, 16:48])
    crop, _, _ = model.forward_dummy(cropped, **arguments)
    assert torch.allclose(crop, full[..., 12:36, 16:48], atol=3e-6)
    for fill in (0., 1.):
        output, _, _, aux = model.forward_dummy(data['lq'], True,
                                              foreground_mask=torch.full_like(data['foreground_mask'], fill))
        assert torch.isfinite(output).all()
        assert aux['local_deltas'][0 if fill == 0 else 1].abs().max() == 0


@CUDA
def test_both_adaint_heads_freeze_and_unfreeze_with_existing_train_step():
    model = small_model(train_cfg=dict(total_iters=101, n_fix_iters=1)).cuda().eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train_step(batch(), optimizer)
    assert model.cnt_iters.item() == 1
    assert all(not p.requires_grad for b in model.branches.values() for p in b.adaint.parameters())
    model.train_step(batch(), optimizer)
    assert all(p.requires_grad for b in model.branches.values() for p in b.adaint.parameters())


@CUDA
def test_mask_aware_validation_saves_output_and_regions(tmp_path):
    metrics = ['PSNR', 'deltaE_P3', 'deltaE_P3_top', 'deltaE_P3_foreground',
               'deltaE_P3_background', 'deltaL_P3_foreground', 'deltaAB_P3_foreground']
    model = small_model(test_cfg=ConfigDict(metrics=metrics, crop_border=1)).cuda().eval()
    data = batch()
    result = model.forward_test(data['lq'], gt=data['gt'], foreground_mask=data['foreground_mask'],
        supervision_mask=data['supervision_mask'],
        meta=[dict(lq_path='a_input.jpg'), dict(lq_path='b_input.jpg')],
        save_image=True, save_path=str(tmp_path))
    assert len(result) == 2 and (tmp_path / 'a_input.png').exists()
    assert all(set(metrics).issubset(item['eval_result']) for item in result)
    result = model.forward_test([data['lq'][0], data['lq'][1, :, :32, :40]],
        gt=[data['gt'][0], data['gt'][1, :, :32, :40]],
        foreground_mask=[data['foreground_mask'][0], data['foreground_mask'][1, :, :32, :40]],
        supervision_mask=[data['supervision_mask'][0], data['supervision_mask'][1, :, :32, :40]])
    assert len(result) == 2


def test_absent_regions_are_excluded_from_dataset_metric_average():
    dataset = object.__new__(SemanticFace)
    dataset.data_infos = [dict(), dict()]
    output = dataset.evaluate([
        dict(eval_result=dict(PSNR=20., deltaE_P3_foreground=9., _valid_foreground=1.)),
        dict(eval_result=dict(PSNR=30., deltaE_P3_foreground=0., _valid_foreground=0.))])
    assert output['PSNR'] == 25 and output['deltaE_P3_foreground'] == 9
    assert output['foreground_valid_count'] == 1

def test_configurable_checkpoint_selection_minimizes_region_error():
    from adaint.experiment_utils import select_best_checkpoint
    records = [dict(mode='val', epoch=1, PSNR=30., deltaE_P3_foreground=9.),
               dict(mode='val', epoch=2, PSNR=29., deltaE_P3_foreground=4.),
               dict(mode='val', epoch=3, PSNR=28., deltaE_P3_foreground=float('nan'))]
    assert select_best_checkpoint(records, '/tmp', 2)[2].endswith('epoch_1.pth')
    assert select_best_checkpoint(records, '/tmp', 2, 'deltaE_P3_foreground', 'less')[2].endswith('epoch_2.pth')
    with pytest.raises(ValueError):
        select_best_checkpoint(records, '/tmp', 2, rule='invalid')


@CUDA
def test_optional_face_and_boundary_losses_use_separate_masks():
    model = small_model(face_color_weight=.1, boundary_gradient_weight=.01,
        test_cfg=ConfigDict(metrics=['deltaE_P3_face'], crop_border=1)).cuda().eval()
    data = batch()
    data['face_mask'] = data['foreground_mask'].clone()
    data['face_mask'][..., :12, :] = 0
    losses = model.forward_train(**data)['losses']
    assert torch.isfinite(losses['loss_face_color'])
    assert torch.isfinite(losses['loss_boundary_gradient'])
    with torch.no_grad():
        outputs = model.forward_test(data['lq'], data['gt'], foreground_mask=data['foreground_mask'],
                                     face_mask=data['face_mask'])
    assert all(item['eval_result']['deltaE_P3_face'] > 0 for item in outputs)

def test_region_checkpoint_nan_overwrite_does_not_reuse_stale_measurement():
    from adaint.experiment_utils import select_best_checkpoint
    records = [dict(mode='val', epoch=1, deltaE_P3_foreground=4.),
               dict(mode='val', epoch=2, deltaE_P3_foreground=3.),
               dict(mode='val', epoch=2, deltaE_P3_foreground=float('nan'))]
    assert select_best_checkpoint(records, '/tmp', 2, 'deltaE_P3_foreground', 'less')[2].endswith('epoch_1.pth')
