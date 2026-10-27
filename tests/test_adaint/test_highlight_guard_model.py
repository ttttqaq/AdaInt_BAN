"""Guard integration: legacy replay, frozen scopes and raw diagnostics."""
import copy
import subprocess
import types

import pytest
import torch
from mmcv import Config, ConfigDict

from adaint.experiment_utils import inference_config, load_enhancer_checkpoint
from tests.test_adaint.test_semantic_dual import small_model, batch


POST = dict(result_condition=True, global_dc=True, regional_dc=True,
            multiscale=True, guidance='luminance')


def arguments(data):
    return {key: value for key, value in data.items()
            if key not in ('lq', 'gt', 'supervision_mask')}


@pytest.fixture(scope='module')
def baseline_class():
    source = subprocess.check_output(
        ['git', 'show', '0757473:adaint/model_semantic.py']).decode()
    source = source.replace('@MODELS.register_module()\n', '')
    module = types.ModuleType('adaint._guard_baseline')
    module.__package__ = 'adaint'
    exec(compile(source, '<0757473 semantic model>', 'exec'), module.__dict__)
    return module.AiLUTDual


def tiny_arguments():
    return dict(n_vertices=9, curve_points=4, n_ranks=3, stats_bins=8,
        stats_pool_size=16, condition_dim=32, head_hidden_dim=16,
        local_resolution=32, local_channels=4, context_size=32, region_samples_size=8,
        color_loss_weight=.1, lut_mode='basis', postprocess=copy.deepcopy(POST),
        train_cfg=dict(total_iters=101, n_fix_iters=0))


def test_guard_off_replays_current_baseline_outputs_state_and_all_losses(baseline_class):
    torch.manual_seed(2026)
    old = baseline_class(**tiny_arguments()).eval()
    for head in list(old.local_decoder.outputs) + list(old.local_decoder.refinement_outputs):
        torch.nn.init.normal_(head.weight, std=.02)
    current = small_model(lut_mode='basis', postprocess=POST, highlight_guard={}).eval()
    assert set(current.state_dict()) == set(old.state_dict())
    current.load_state_dict(old.state_dict(), strict=True)
    data = batch('cpu')
    with torch.no_grad():
        expected = old.forward_dummy(data['lq'], **arguments(data))[0]
        actual = current.forward_dummy(data['lq'], **arguments(data))[0]
        expected_losses = old.forward_train(**data)['losses']
        actual_losses = current.forward_train(**data)['losses']
    assert torch.equal(expected, actual)
    assert set(expected_losses) == set(actual_losses)
    for name, value in expected_losses.items():
        assert torch.equal(value, actual_losses[name]), name


def test_heads_scope_only_updates_declared_heads_and_never_unfreezes_adaint():
    model = small_model(lut_mode='basis', postprocess=POST,
        highlight_guard=dict(trainable_scope='heads', over_weight=1.)).train()
    assert model.training and not model.backbone.training and not model.branches.training
    assert not model.local_decoder.training
    trainable = {name for name, value in model.named_parameters() if value.requires_grad}
    assert trainable
    assert all('.lut_generator.head.output.' in name or
               name.startswith(('local_decoder.outputs.', 'local_decoder.refinement_outputs.'))
               for name in trainable)
    frozen = {name: value.detach().clone() for name, value in model.named_parameters()
              if name not in trainable}
    before = {name: value.detach().clone() for name, value in model.named_parameters()
              if name in trainable}
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    for _ in range(2):
        result = model.train_step(batch('cpu'), optimizer)
        assert all(torch.isfinite(torch.tensor(value)) for value in result['log_vars'].values())
    for name, value in model.named_parameters():
        if name in frozen:
            assert torch.equal(value, frozen[name]) and value.grad is None, name
    assert any(not torch.equal(value, before[name]) for name, value in model.named_parameters()
               if name in before)
    assert all(not parameter.requires_grad for branch in model.branches.values()
               for parameter in branch.adaint.parameters())


def test_detach_only_changes_result_condition_gradient_path_not_forward():
    standard = small_model(lut_mode='basis', postprocess=POST).eval()
    detached = small_model(lut_mode='basis', postprocess=POST,
        highlight_guard=dict(detach_result_condition=True)).eval()
    detached.load_state_dict(standard.state_dict(), strict=True)
    seen = []
    original_forward = detached.local_decoder.forward

    def capture(source, conditions, global_context=None, foreground_context=None):
        seen.extend(value.requires_grad for value in global_context)
        return original_forward(source, conditions, global_context, foreground_context)

    detached.local_decoder.forward = capture
    data = batch('cpu')
    expected = standard.forward_dummy(data['lq'], **arguments(data))[0]
    actual = detached.forward_dummy(data['lq'], **arguments(data))[0]
    assert torch.equal(expected, actual) and seen == [False, False]
    actual.square().mean().backward()
    for branch in detached.branches.values():
        gradient = branch.lut_generator.basis_luts.grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_reference_is_rng_neutral_frozen_and_absent_from_checkpoint_and_inference(tmp_path):
    from mmedit.models import build_model
    reference = small_model(lut_mode='basis', postprocess=POST).eval()
    checkpoint = str(tmp_path / 'reference.pth')
    torch.save(dict(state_dict=reference.state_dict()), checkpoint)
    options = tiny_arguments()
    options.pop('train_cfg')
    options['type'] = 'AiLUTDual'
    config = str(tmp_path / 'reference.py')
    (tmp_path / 'reference.py').write_text(
        'model = ' + repr(options) + '\ntest_cfg = dict(metrics=None, crop_border=0)\n')
    guard = dict(keep_weight=1., reference_config=config, reference_checkpoint=checkpoint,
                 keep_error_threshold=.03, trainable_scope='heads')
    model = small_model(lut_mode='basis', postprocess=POST, highlight_guard=guard).train()
    model.load_state_dict(reference.state_dict(), strict=True)
    before = torch.get_rng_state().clone()
    loaded = model._guard_reference_model(torch.device('cpu'))
    assert torch.equal(before, torch.get_rng_state())
    assert not loaded.training and all(not p.requires_grad for p in loaded.parameters())
    assert set(model.state_dict()) == set(reference.state_dict())
    assert all('reference' not in name for name, _ in model.named_parameters())
    data = batch('cpu')
    with torch.no_grad():
        data['gt'] = reference.forward_dummy(data['lq'], **arguments(data))[0]
        model.branches['foreground'].lut_generator.head.output.bias.add_(.01)
    losses = model.forward_train(**data)['losses']
    assert losses['loss_guard_keep'].item() > 0
    losses['loss_guard_keep'].backward()
    assert all(p.grad is None for p in loaded.parameters())
    cfg = Config(dict(model=dict(options, highlight_guard=guard),
                      test_cfg=dict(metrics=['PSNR'], crop_border=0)))
    exported = inference_config(cfg)
    assert 'reference_checkpoint' not in exported.model.highlight_guard
    assert 'keep_weight' not in exported.model.highlight_guard
    assert 'trainable_scope' not in exported.model.highlight_guard
    clean = build_model(exported.model, train_cfg=None, test_cfg=exported.test_cfg)
    saved = str(tmp_path / 'enhancer.pth')
    torch.save(dict(state_dict=model.state_dict()), saved)
    load_enhancer_checkpoint(clean, saved)
    assert clean._guard_reference is None


def test_raw_diagnostics_are_measured_before_inherited_cpu_image_clamp():
    metrics = ['PSNR', 'raw_high_clip', 'raw_high_clip_foreground',
               'raw_high_clip_background', 'highlight_over_foreground',
               'highlight_over_background']
    model = small_model(test_cfg=ConfigDict(metrics=metrics, crop_border=0)).eval()
    output = torch.full((1, 3, 4, 4), .6)
    output[:, 0, 0, 0] = 1.2
    gt = torch.full_like(output, .5)
    mask = torch.zeros((1, 1, 4, 4))
    mask[..., :2] = 1
    result = model.evaluate(output, gt, foreground_mask=mask, supervision_mask=mask)
    assert result['raw_high_clip'] == pytest.approx(1 / 16)
    assert result['raw_high_clip_foreground'] == pytest.approx(1 / 8)
    assert result['raw_high_clip_background'] == 0
    assert result['highlight_over_foreground'] > result['highlight_over_background'] > 0
    # The inherited helper really did clamp its CPU alias; our metrics survived it.
    assert output.max().item() == 1


def test_enabled_guard_losses_reach_fused_lut_and_local_heads():
    model = small_model(lut_mode='basis', postprocess=POST,
        highlight_guard=dict(over_weight=1., lut_range_weight=.1,
                             final_range_weight=.1)).eval()
    with torch.no_grad():
        for branch in model.branches.values():
            branch.lut_generator.basis_luts[0].add_(.3)
    losses = model.forward_train(**batch('cpu'))['losses']
    expected = {'loss_guard_over', 'loss_guard_lut_range', 'loss_guard_final_range'}
    assert expected.issubset(losses)
    assert all(torch.isfinite(losses[name]) and losses[name] > 0 for name in expected)
    sum(losses[name] for name in expected).backward()
    assert all(branch.lut_generator.basis_luts.grad.abs().sum() > 0
               for branch in model.branches.values())
    assert any(head.weight.grad is not None and head.weight.grad.abs().sum() > 0
               for head in model.local_decoder.outputs)
