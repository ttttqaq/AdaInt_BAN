"""Color references, local-error selection and end-to-end supervision checks."""
import copy

import pytest
import torch
from mmcv import ConfigDict

from adaint.color import (display_p3_to_lab, grid_top_mean,
                          P3LabColorLoss)
from adaint.config_utils import load_config
from adaint.experiment_utils import load_enhancer_checkpoint, resolve_training_schedule
from adaint.model import AiLUT
from adaint.modules import stratified_pixels


def small_model(**extra):
    options = dict(n_vertices=9, curve_points=4, n_ranks=3, stats_bins=8,
                   stats_pool_size=16, condition_dim=32, head_hidden_dim=16,
                   local_resolution=32, local_channels=4, context_size=32,
                   train_cfg=dict(total_iters=101, n_fix_iters=0))
    options.update(extra)
    return AiLUT(**options)


def test_display_p3_d65_black_white_gray_and_red_reference():
    rgb = torch.tensor([[0., 0., 0.], [1., 1., 1.], [.5, .5, .5],
                        [1., 0., 0.]], dtype=torch.float64).view(4, 3, 1, 1)
    lab = display_p3_to_lab(rgb).flatten(1)
    assert torch.allclose(lab[0], lab.new_tensor([0., 0., 0.]), atol=1e-10)
    assert torch.allclose(lab[1], lab.new_tensor([100., 0., 0.]), atol=1e-10)
    assert torch.allclose(lab[2], lab.new_tensor([53.3889647, 0., 0.]), atol=1e-6)
    # Display P3's red is substantially more saturated than sRGB red.
    assert torch.allclose(lab[3], lab.new_tensor([54.9665571, 94.0925463,
                                                94.7699260]), atol=1e-5)


def test_zero_loss_has_zero_finite_gradients_and_half_uses_float32():
    pred = torch.rand(2, 3, 19, 23, requires_grad=True)
    loss, stats = P3LabColorLoss()(pred, pred.detach())
    assert loss.item() == 0 and all(value.item() == 0 for value in stats.values())
    loss.backward()
    assert torch.equal(pred.grad, torch.zeros_like(pred))
    assert display_p3_to_lab(pred.detach().half()).dtype == torch.float32


def test_out_of_gamut_and_black_have_finite_corrective_gradients():
    pred = torch.tensor([-.1, 0., 1.2], dtype=torch.float64).view(1, 3, 1, 1)
    pred.requires_grad_()
    target = pred.detach().clamp(.1, .9)
    criterion = P3LabColorLoss()
    assert torch.autograd.gradcheck(lambda x: criterion(x, target)[0], (pred,),
                                    eps=1e-6, atol=1e-5)
    loss, _ = criterion(pred, target)
    loss.backward()
    assert torch.isfinite(pred.grad).all() and (pred.grad.abs() > 0).all()
    assert criterion(pred.detach() - .0001 * pred.grad, target)[0] < loss


def test_uneven_disjoint_grid_and_top_selection_are_per_image():
    error = torch.ones(2, 5, 7)
    error[0, :2, :3], error[0, :2, 3:] = 9., 4.
    error[0, 2:, :3], error[0, 2:, 3:] = 2., 1.
    error[1] = 6.
    assert torch.equal(grid_top_mean(error, 2, .25), torch.tensor([9., 6.]))
    assert torch.equal(grid_top_mean(error, 2, .5), torch.tensor([6.5, 6.]))
    assert grid_top_mean(torch.tensor([[[3.]]]), 16, .1).item() == 3.


def test_opposite_color_errors_do_not_cancel_and_difficult_cells_get_more_gradient():
    target = torch.full((1, 3, 32, 32), .5)
    pred = target.clone()
    pred[:, 0, :, :16] += .1
    pred[:, 0, :, 16:] -= .1
    assert torch.allclose(pred.mean((-1, -2)), target.mean((-1, -2)))
    assert P3LabColorLoss()(pred, target)[0] > .01
    error = torch.ones(1, 32, 32)
    error[:, :2, :2] = 10.
    error.requires_grad_()
    (.75 * error.mean() + .25 * grid_top_mean(error).mean()).backward()
    assert error.grad[0, 0, 0] > error.grad[0, 31, 31]


@pytest.mark.parametrize('scene,count', [('indoor', 67), ('outdoor', 85)])
def test_single_generation_configs_and_optional_kd_reaches_zero(scene, count):
    cfg = load_config('adaint/configs/face_{}.yaml'.format(scene))
    assert cfg.gen == 1 and not cfg.enable_kd
    assert cfg.model.color_loss_weight == .01
    assert cfg.test_cfg.metrics == ['PSNR', 'deltaE_P3', 'deltaE_P3_top']
    assert scene + '_train.txt' in cfg.data.train.ann_file
    cfg.gpus = 1
    resolve_training_schedule(cfg, count)
    assert cfg.total_iters == count * 100
    model = small_model(kd_decay_end_ratio=cfg.model.kd_decay_end_ratio,
                        train_cfg=dict(total_iters=101))
    for iteration, expected in ((0, 1.), (30, 1.), (55, .5), (80, 0.), (100, 0.)):
        model.cnt_iters.fill_(iteration)
        assert model.kd_multiplier() == pytest.approx(expected)


def test_existing_checkpoint_schema_and_p3_metrics(tmp_path):
    model = small_model(color_loss_weight=.01,
                        test_cfg=ConfigDict(metrics=['deltaE_P3', 'deltaE_P3_top'],
                                            crop_border=1))
    # The color objective adds no parameters/buffers to an existing checkpoint.
    assert not any(key.startswith('color_loss.') for key in model.state_dict())
    path = str(tmp_path / 'checkpoint.pth')
    torch.save(dict(state_dict=copy.deepcopy(model.state_dict())), path)
    load_enhancer_checkpoint(model, path)
    gt = torch.full((1, 3, 19, 23), .5)
    pred = gt.clone()
    pred[:, 0, 1:5, 1:5] += .15
    metrics = model.evaluate(pred, gt)
    assert metrics['deltaE_P3_top'] > metrics['deltaE_P3'] > 0
    assert model.evaluate(gt, gt) == {'deltaE_P3': 0., 'deltaE_P3_top': 0.}


@pytest.mark.skipif(not torch.cuda.is_available(), reason='real AiLUT CUDA kernel')
def test_p3_loss_backpropagates_through_global_lut_and_local_affine():
    torch.manual_seed(17)
    model = small_model(color_loss_weight=.01).cuda().eval()
    image = torch.rand(1, 3, 48, 64, device='cuda')
    box = image.new_tensor([[0., 0., 1., 1.]])
    target = image * .8
    result = model.forward_train(image, target, image, stratified_pixels(image, 16), box)
    losses = result['losses']
    assert losses['loss_color'] > 0 and losses['color_delta_e_top'] >= losses['color_delta_e']
    losses['loss_color'].backward()
    for parameter in (model.local_decoder.output.weight,
                      model.lut_generator.head.output.weight,
                      model.tone_curve.head.output.weight):
        assert parameter.grad.abs().sum() > 0 and torch.isfinite(parameter.grad).all()
