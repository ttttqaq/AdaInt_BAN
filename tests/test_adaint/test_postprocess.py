"""CPU regressions with the real AiLUT extension, including legacy replay."""
import copy
import subprocess
import types

import pytest
import torch

from adaint.config_utils import load_config
from adaint.experiment_utils import (initialize_postprocess, load_enhancer_checkpoint,
                                    save_generation_configs,
                                    resolve_training_schedule)
from adaint.postprocess import postprocess_options
from tests.test_adaint.test_semantic_dual import small_model, batch


@pytest.fixture(scope='module')
def legacy_class():
    source = subprocess.check_output(['git', 'show', '1d3898a:adaint/model_semantic.py']).decode()
    source = source.replace('@MODELS.register_module()\n', '')
    module = types.ModuleType('adaint._postprocess_reference')
    module.__package__ = 'adaint'
    exec(compile(source, '<basis-lut 1d3898a>', 'exec'), module.__dict__)
    return module.AiLUTDual


def legacy_small(cls):
    return cls(n_vertices=9, curve_points=4, n_ranks=3, stats_bins=8,
               stats_pool_size=16, condition_dim=32, head_hidden_dim=16,
               local_resolution=32, local_channels=4, context_size=32, region_samples_size=8,
               color_loss_weight=.1, lut_mode='basis',
               train_cfg=dict(total_iters=101, n_fix_iters=0))


def forward_args(data):
    return {key: value for key, value in data.items() if key not in ('lq', 'gt', 'supervision_mask')}


def test_all_off_replays_original_checkpoint_and_losses_exactly(legacy_class):
    torch.manual_seed(12)
    old = legacy_small(legacy_class).eval()
    for output in old.local_decoder.outputs:
        torch.nn.init.normal_(output.weight, std=.02)
    current = small_model(lut_mode='basis', postprocess={}).eval()
    current.load_state_dict(old.state_dict(), strict=True)
    data = batch('cpu')
    with torch.no_grad():
        expected = old.forward_dummy(data['lq'], **forward_args(data))[0]
        actual = current.forward_dummy(data['lq'], **forward_args(data))[0]
        old_losses = old.forward_train(**data)['losses']
        new_losses = current.forward_train(**data)['losses']
    assert torch.equal(expected, actual)
    assert set(old_losses) == set(new_losses)
    assert all(torch.equal(value, new_losses[name]) for name, value in old_losses.items())


def test_migration_preserves_nonzero_local_output_and_resets_counter(tmp_path):
    old = small_model(lut_mode='basis').eval()
    for output in old.local_decoder.outputs:
        torch.nn.init.normal_(output.weight, std=.02)
    old.cnt_iters.fill_(444)
    path = str(tmp_path / 'base.pth')
    torch.save(dict(state_dict=old.state_dict(), meta=dict(code_revision='baseline', epoch=6)), path)
    model = small_model(lut_mode='basis', postprocess=dict(result_condition=True,
        global_dc=True, regional_dc=True, multiscale=True, freeze_global=True)).eval()
    info = initialize_postprocess(model, path)
    assert info['counter_reset'] and info['source_epoch'] == 6 and info['new_keys']
    assert model.cnt_iters.item() == 0
    data = batch('cpu')
    with torch.no_grad():
        before = old.forward_dummy(data['lq'], **forward_args(data))[0]
        after = model.forward_dummy(data['lq'], **forward_args(data))[0]
    assert torch.allclose(before, after, atol=2e-7, rtol=0)
    state = copy.deepcopy(old.state_dict())
    state.pop(next(k for k in state if k.startswith('branches.background.')))
    torch.save(dict(state_dict=state), path)
    with pytest.raises(RuntimeError, match='incompatible postprocess'):
        initialize_postprocess(model, path)


@pytest.mark.parametrize('options', [dict(result_condition=True), dict(global_dc=True),
    dict(regional_dc=True), dict(multiscale=True), dict(guidance='rgb'),
    dict(result_condition=True, global_dc=True, regional_dc=True, multiscale=True, guidance='rgb')])
def test_switches_forward_backward_and_frozen_global_stays_exact(options):
    model = small_model(lut_mode='basis', postprocess=dict(options, freeze_global=True)).train()
    assert not model.backbone.training and not model.branches.training
    assert all(not p.requires_grad for p in model.backbone.parameters())
    frozen = {name: p.detach().clone() for name, p in model.named_parameters()
              if name.startswith(('backbone.', 'branches.'))}
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    data = batch('cpu')
    # Two steps let initially zero output heads expose their upstream parameters.
    for _ in range(2):
        result = model.train_step(data, optimizer)
        assert all(torch.isfinite(torch.tensor(x)) for x in result['log_vars'].values())
    for name, p in model.named_parameters():
        if name in frozen:
            assert torch.equal(p, frozen[name]) and p.grad is None
    active = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert active and all(torch.isfinite(p.grad).all() for p in active)
    if options.get('global_dc'):
        assert model.postprocess_calibration.global_head[-1].weight.grad.abs().sum() > 0
    if options.get('regional_dc'):
        assert model.postprocess_calibration.regional_head[-1].weight.grad.abs().sum() > 0
    if options.get('multiscale'):
        assert all(p.weight.grad.abs().sum() > 0 for p in model.local_decoder.refinement_outputs)
    if options.get('result_condition'):
        assert model.local_decoder.result_adapter.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('guidance', ['luminance', 'rgb'])
def test_nonzero_all_features_crop_consistency_centering_and_empty_masks(guidance):
    model = small_model(lut_mode='basis', postprocess=dict(result_condition=True,
        global_dc=True, regional_dc=True, multiscale=True, guidance=guidance)).eval()
    for output in list(model.local_decoder.outputs) + list(model.local_decoder.refinement_outputs):
        torch.nn.init.normal_(output.weight, std=.02)
    for head in (model.postprocess_calibration.global_head, model.postprocess_calibration.regional_head):
        torch.nn.init.normal_(head[-1].weight, std=.02)
    data = batch('cpu')
    args = forward_args(data)
    with torch.no_grad():
        full, _, _, aux = model.forward_dummy(data['lq'], True, **args)
        for delta, mask in zip(aux['context_deltas'], aux['local_masks']):
            assert ((delta * mask).sum((-1, -2)) / mask.sum((-1, -2))).abs().max() < 1e-6
        crop = data['lq'][..., 12:36, 16:48]
        args.update(crop_box=crop.new_tensor([[.25, .25, .75, .75]]).repeat(2, 1),
                    foreground_mask=data['foreground_mask'][..., 12:36, 16:48])
        actual = model.forward_dummy(crop, **args)[0]
        assert torch.allclose(actual, full[..., 12:36, 16:48], atol=4e-6)
        for fill in (0., 1.):
            y, _, _, a = model.forward_dummy(data['lq'], True,
                foreground_mask=torch.full_like(data['foreground_mask'], fill))
            assert torch.isfinite(y).all() and torch.count_nonzero(a['regional_dc']) == 0


def test_bounded_signed_dc_and_calibration_only():
    model = small_model(lut_mode='basis', local_enabled=False,
        postprocess=dict(global_dc=True, regional_dc=True, freeze_global=True,
                         global_dc_limit=.07, regional_dc_limit=.03)).eval()
    with torch.no_grad():
        model.postprocess_calibration.global_head[-1].bias.copy_(torch.tensor([-100., 100., .5]))
        model.postprocess_calibration.regional_head[-1].bias.fill_(100.)
        data = batch('cpu')
        output, _, _, aux = model.forward_dummy(data['lq'], True, **forward_args(data))
    assert torch.isfinite(output).all()
    assert aux['global_dc'].min() == pytest.approx(-.07)
    assert aux['global_dc'].max() == pytest.approx(.07)
    assert aux['regional_dc'].max() == pytest.approx(.03)


def test_joint_training_reaches_global_heads_and_rgb_constant_guide_is_finite():
    model = small_model(lut_mode='basis', postprocess=dict(result_condition=True,
        global_dc=True, regional_dc=True, multiscale=True, guidance='rgb')).eval()
    data = batch('cpu')
    result = model.forward_train(**data)
    loss = sum(value for name, value in result['losses'].items() if name.startswith('loss_'))
    loss.backward()
    for branch in model.branches.values():
        for parameter in (branch.lut_generator.basis_luts, branch.adaint.head.output.weight):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0
    model.zero_grad()
    for output in model.local_decoder.outputs:
        torch.nn.init.normal_(output.weight, std=.02)
    image = torch.full((1, 3, 24, 32), .5)
    output = model.forward_dummy(image, foreground_mask=torch.full((1, 1, 24, 32), .5))[0]
    output.square().mean().backward()
    assert torch.isfinite(output).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_new_checkpoint_roundtrip_and_strict_loading_rejects_old_schema(tmp_path):
    opts = dict(global_dc=True, result_condition=True, multiscale=True, guidance='rgb')
    model = small_model(lut_mode='basis', postprocess=opts).eval()
    path = str(tmp_path / 'new.pth')
    torch.save(dict(state_dict=model.state_dict()), path)
    other = small_model(lut_mode='basis', postprocess=opts).eval()
    load_enhancer_checkpoint(other, path)
    assert all(torch.equal(x, other.state_dict()[n]) for n, x in model.state_dict().items())
    torch.save(dict(state_dict=small_model(lut_mode='basis').state_dict()), path)
    with pytest.raises(RuntimeError, match='incompatible AiLUT'):
        load_enhancer_checkpoint(other, path)


def test_optional_features_preserve_seeded_initialization_of_common_parameters():
    torch.manual_seed(2026)
    simple = small_model(lut_mode='basis', postprocess=dict(global_dc=True))
    torch.manual_seed(2026)
    extended = small_model(lut_mode='basis', postprocess=dict(global_dc=True,
        regional_dc=True, result_condition=True, multiscale=True))
    expected = extended.state_dict()
    for name, value in simple.state_dict().items():
        assert torch.equal(value, expected[name]), name


@pytest.mark.parametrize('scene,count', [('indoor', 67), ('niguang', 76), ('outdoor', 85)])
def test_config_schedule_and_inference_serialization(scene, count, tmp_path):
    cfg = load_config('adaint/configs/face_' + scene + '_semantic_postprocess.yaml')
    assert cfg.model.postprocess.freeze_global and cfg.model.postprocess.global_dc
    assert cfg.train_epochs == 30 and cfg.adaint_fix_epochs == 0
    cfg.gpus = 1
    cfg.work_dir = str(tmp_path)
    resolve_training_schedule(cfg, count)
    assert cfg.total_iters == count * 30 and cfg.train_cfg.n_fix_iters == 0
    save_generation_configs(cfg)
    saved = load_config(str(tmp_path / 'inference_config.py'))
    assert saved.model.postprocess == cfg.model.postprocess
    assert saved.test_cfg.metrics is None


@pytest.mark.parametrize('options', [dict(typo=True), dict(guidance='learned'),
    dict(global_dc_limit=0), dict(result_condition='yes'),
    dict(global_dc=True, fixed_rgb_bias=[0., 0., 0.])])
def test_invalid_options_are_rejected(options):
    with pytest.raises(ValueError):
        postprocess_options(options)
