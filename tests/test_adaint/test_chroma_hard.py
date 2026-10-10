"""Focused regressions for color hard mining, gain migration and selection."""
import copy
import json

import pytest
import torch
from mmcv import ConfigDict

from adaint.color import P3LabColorLoss, grid_top_mean, p3_error_maps
from adaint.config_utils import load_config
from adaint.experiment_utils import load_enhancer_checkpoint, select_best_checkpoint
from adaint.modules import LocalAffineDecoder, stratified_pixels
from adaint.model import AiLUT
from tools.prepare_lut_experiment import prepare
from tools.train_kd_unified import export_best_checkpoints


def small_model(**extra):
    options = dict(n_vertices=9, curve_points=4, n_ranks=3, stats_bins=8,
                   stats_pool_size=16, condition_dim=32, head_hidden_dim=16,
                   local_resolution=32, local_channels=4, context_size=32,
                   train_cfg=dict(total_iters=101, n_fix_iters=0))
    options.update(extra)
    return AiLUT(**options)


def test_ab_mining_ignores_luminance_and_keeps_global_lab(monkeypatch):
    # Left cell: brightness error 30; right cell: alternating +/- a error 6.
    # Pixel-first distances must retain 6 even though the cell's mean a is 0.
    lab = torch.zeros(1, 3, 2, 4)
    lab[:, 0, :, :2] = 30
    lab[:, 1, :, 2] = 6
    lab[:, 1, :, 3] = -6
    monkeypatch.setattr('adaint.color.display_p3_to_lab', lambda x: x)
    target = torch.zeros_like(lab)
    diff, e_lab, e_ab = p3_error_maps(lab, target, epsilon=1e-3)
    assert diff[:, 1, :, 2:].mean() == 0
    ab_loss, stats = P3LabColorLoss(2, .25, .25, 'ab')(lab, target)
    old_loss, _ = P3LabColorLoss(2, .25, .25, 'lab')(lab, target)
    expected = (.75 * e_lab.mean() + .25 * grid_top_mean(e_ab, 2, .25).mean()) / 100
    assert ab_loss == pytest.approx(expected.item())
    assert old_loss > ab_loss
    assert stats['color_hard_L_fraction'] == 0
    assert stats['color_chroma_top'] == pytest.approx(5.999, abs=1e-6)
    assert stats['color_delta_e_top'] == pytest.approx(29.999, abs=1e-5)


@pytest.mark.parametrize('mode', ['lab', 'ab'])
def test_zero_and_out_of_gamut_gradients(mode):
    pred = torch.rand(2, 3, 5, 7, requires_grad=True)
    loss, stats = P3LabColorLoss(hard_mode=mode)(pred, pred.detach())
    loss.backward()
    assert loss == 0 and torch.equal(pred.grad, torch.zeros_like(pred))
    assert all(value == 0 for value in stats.values())
    extreme = torch.tensor([-.1, 0., 1.2], dtype=torch.float64).view(1, 3, 1, 1)
    extreme.requires_grad_()
    target = extreme.detach().clamp(.1, .9)
    assert torch.autograd.gradcheck(lambda x: P3LabColorLoss(hard_mode=mode)(x, target)[0],
                                    (extreme,), eps=1e-6, atol=1e-5)


def test_metrics_keep_lab_top_definition_and_track_chroma_and_light():
    metrics = ['deltaE_P3', 'deltaE_P3_top', 'chroma_P3', 'chroma_P3_top',
               'chroma_P3_p90', 'L_error_P3']
    model = small_model(color_hard_mode='ab', test_cfg=ConfigDict(metrics=metrics, crop_border=0))
    gt = torch.full((1, 3, 8, 8), .5)
    neutral = gt + .1
    scores = model.evaluate(neutral, gt)
    assert scores['L_error_P3'] > 9 and scores['deltaE_P3_top'] > 9
    assert scores['chroma_P3_top'] < 1e-4
    assert model.evaluate(gt, gt) == dict.fromkeys(metrics, 0.)


def test_rgb_gain_adds_only_64_parameters_and_preserves_regularization():
    shared = LocalAffineDecoder(32, channels=16)
    rgb = LocalAffineDecoder(32, channels=16, gain_mode='rgb')
    assert sum(p.numel() for p in rgb.parameters()) - sum(p.numel() for p in shared.parameters()) == 64
    torch.manual_seed(7)
    coefficients = torch.randn(2, 4, 7, 9)
    expanded = torch.cat((coefficients[:, :1].repeat(1, 3, 1, 1), coefficients[:, 1:]), 1)
    image = torch.rand(2, 3, 7, 9)
    assert torch.equal(shared.correction(coefficients, image), rgb.correction(expanded, image))
    assert torch.allclose(shared.smoothness(coefficients, image, 10),
                          rgb.smoothness(expanded, image, 10), atol=1e-6)
    assert rgb.output.weight.count_nonzero() == 0


def test_nonzero_legacy_checkpoint_migration_preserves_full_and_crop_outputs(tmp_path):
    torch.manual_seed(7)
    shared = small_model().eval()
    with torch.no_grad():
        shared.local_decoder.output.weight.normal_(std=.02)
    rgb = small_model(local_gain_mode='rgb').eval()
    state = copy.deepcopy(shared.state_dict())
    checkpoint = tmp_path / 'shared.pth'
    torch.save(dict(state_dict=state), str(checkpoint))
    load_enhancer_checkpoint(rgb, str(checkpoint))
    with pytest.raises(RuntimeError, match='load_from'):
        load_enhancer_checkpoint(rgb, str(checkpoint), allow_gain_migration=False)
    # Exercise the standard PyTorch/MMCV module loading path too.
    rgb.load_state_dict(state, strict=True)
    image = torch.rand(1, 3, 48, 64)
    samples = stratified_pixels(image, 16)
    box = torch.tensor([[.25, .25, .75, .75]])
    with torch.no_grad():
        before = shared.forward_dummy(image, True)
        after = rgb.forward_dummy(image, True)
        crop = rgb.forward_dummy(image[:, :, 12:36, 16:48], True, image, samples, box)
    assert torch.allclose(before[0], after[0], atol=2e-6)
    assert torch.allclose(before[3]['delta_mean'], after[3]['delta_mean'], atol=1e-7)
    assert torch.allclose(crop[0], after[0][:, :, 12:36, 16:48], atol=3e-6)
    assert after[3]['context_delta'].mean((-1, -2)).abs().max() < 1e-7
    assert torch.allclose(before[3]['context_delta'], after[3]['context_delta'], atol=1e-7)


@pytest.mark.parametrize('hard,gain', [('ab', 'shared'), ('lab', 'rgb')])
def test_real_cpu_kernel_backward_and_optimizer_step(hard, gain):
    torch.manual_seed(17)
    model = small_model(color_loss_weight=.3, color_hard_mode=hard, local_gain_mode=gain).eval()
    image = torch.rand(1, 3, 32, 40)
    box = torch.tensor([[0., 0., 1., 1.]])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    result = model.forward_train(image, image * .8, image, stratified_pixels(image, 16), box)
    loss, logs = model.parse_losses(result['losses'])
    assert all(torch.isfinite(torch.tensor(value)) for value in logs.values())
    loss.backward()
    for parameter in (model.local_decoder.output.weight, model.lut_generator.head.output.weight,
                      model.tone_curve.head.output.weight):
        assert parameter.grad.abs().sum() > 0 and torch.isfinite(parameter.grad).all()
    optimizer.step()
    assert torch.isfinite(model.forward_dummy(image)[0]).all()


def test_color_selection_uses_final_psnr_bound_and_latest_resume_scores(tmp_path):
    records = [dict(mode='val', epoch=1, PSNR=30., chroma_P3_top=2.),
               dict(mode='val', epoch=2, PSNR=29.98, chroma_P3_top=3.),
               dict(mode='val', epoch=3, PSNR=30.12, chroma_P3_top=4.),
               dict(mode='val', epoch=4, PSNR=30.05, chroma_P3_top=3.5),
               dict(mode='train', epoch=5, PSNR=99., chroma_P3_top=0.)]
    assert select_best_checkpoint(records, str(tmp_path), 2)[1] == 3
    assert select_best_checkpoint(records, str(tmp_path), 2, 'chroma_P3_top')[1] == 1
    assert select_best_checkpoint(records, str(tmp_path), 2, 'chroma_P3_top', .1)[1] == 4
    records.append(dict(mode='val', epoch=4, PSNR=30.05, chroma_P3_top=5.))
    assert select_best_checkpoint(records, str(tmp_path), 2, 'chroma_P3_top', .1)[1] == 3
    with pytest.raises(ValueError):
        select_best_checkpoint(records, str(tmp_path), 2, 'chroma_P3_top', -.1)
    for epoch in (1, 2, 3, 4):
        (tmp_path / ('epoch_%s.pth' % epoch)).write_bytes(str(epoch).encode())
    (tmp_path / 'unit.log.json').write_text('\n'.join(json.dumps(row) for row in records))
    cfg = ConfigDict(work_dir=str(tmp_path), iters_per_epoch=2, log_interval_epochs=1,
                     checkpoint_selection=dict(metric='chroma_P3_top', psnr_tolerance=.1))
    summary = export_best_checkpoints(cfg)
    assert (tmp_path / 'best_color.pth').read_bytes() == b'1'
    assert (tmp_path / 'best.pth').read_bytes() == b'3'
    assert (tmp_path / 'best_psnr.pth').read_bytes() == b'3'
    assert summary['checkpoints']['best.pth']['source'] == 'epoch_3.pth'


def test_ablation_configs_change_only_one_method_at_a_time():
    import yaml
    cfg = load_config('adaint/configs/face_indoor_chroma.yaml')
    assert cfg.gen == 1 and not cfg.enable_kd and cfg.train_epochs == 100
    assert cfg.model.color_loss_weight == .3 and cfg.model.color_hard_weight == .25
    with open('adaint/configs/face_indoor_chroma.yaml') as handle:
        base = yaml.safe_load(handle)
    a, _ = prepare(base, 'unit_a', ['model.color_hard_mode=lab'])
    c, _ = prepare(base, 'unit_c', ['model.color_hard_mode=lab', 'model.local_gain_mode=rgb'])
    assert a['model']['local_gain_mode'] == 'shared'
    assert c['model']['color_hard_mode'] == 'lab'
    assert a['data'] == c['data'] == base['data']
    assert a['checkpoint_selection'] == c['checkpoint_selection'] == base['checkpoint_selection']
    baseline = load_config('adaint/configs/face_indoor.yaml')
    for key, value in baseline.model.items():
        assert cfg.model[key] == value
    assert cfg.data.train.ann_file == baseline.data.train.ann_file
