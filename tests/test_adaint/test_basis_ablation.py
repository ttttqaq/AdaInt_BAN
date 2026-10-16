"""Isolate basis-LUT generation while keeping the semantic enhancer unchanged."""
import copy
import subprocess

import pytest
import torch
import torch.nn.functional as F

from adaint.config_utils import load_config
from adaint.experiment_utils import load_enhancer_checkpoint, resolve_training_schedule
from adaint.modules import identity_lut
from adaint.semantic_modules import BasisLUT, RegionMapping
from tests.test_adaint.test_semantic_dual import small_model, batch


def mapping(mode='basis'):
    return RegionMapping(16, 10, 8, 4, 5, 3, 4, .001, lut_mode=mode)


def conditions():
    return (torch.randn(2, 16), torch.rand(2, 10), torch.rand(2, 10),
            torch.tensor([[.3], [.7]]), torch.ones(2, 1))


def test_raw_basis_fusion_matches_main_linear_bank_and_has_no_extra_tone_or_identity():
    branch = mapping()
    assert not hasattr(branch, 'tone_curve')
    generator = branch.lut_generator
    with torch.no_grad():
        generator.basis_luts.normal_(0, .2)
        generator.head.output.weight.zero_()
        generator.head.output.bias.copy_(torch.tensor([.4, -.2, .7]))
    arguments = conditions()
    condition, weights, vertices, lut = branch(*arguments)
    # main/AdaInt stores the same bank as a bias-free Linear's weight matrix.
    expected = F.linear(weights, generator.basis_luts.flatten(1).t()).view_as(lut)
    assert torch.allclose(lut, expected, atol=1e-7)
    assert torch.equal(weights[0], torch.tensor([.4, -.2, .7]))
    assert vertices.shape == (2, 3, 5)
    assert condition.shape == (2, 8)
    # Doubling raw basis amplitude must double output: no RMS normalization.
    with torch.no_grad():
        generator.basis_luts.mul_(2)
    assert torch.allclose(branch(*arguments)[3], 2 * lut, atol=1e-7)
    with torch.no_grad():
        generator.basis_luts.zero_()
    assert torch.count_nonzero(branch(*arguments)[3]) == 0


def test_basis_bank_initializes_as_identity_plus_zero_bases_and_all_get_gradient():
    generator = BasisLUT(5, 3, 8, 4)
    uniform = torch.linspace(0, 1, 5).view(1, 1, 5).expand(1, 3, 5)
    assert torch.equal(generator.basis_luts[0], identity_lut(uniform)[0])
    assert torch.count_nonzero(generator.basis_luts[1:]) == 0
    assert torch.equal(generator.head.output.bias, torch.ones(3))
    _, lut = generator(torch.rand(2, 8))
    lut.square().mean().backward()
    assert torch.all(generator.basis_luts.grad.flatten(1).abs().sum(1) > 0)


def test_seeded_initialization_of_all_common_model_parameters_is_identical():
    torch.manual_seed(41)
    tone = small_model()
    torch.manual_seed(41)
    basis = small_model(lut_mode='basis')
    expected = tone.state_dict()
    checked = 0
    for name, value in basis.state_dict().items():
        if '.lut_generator.basis_luts' in name or '.lut_generator.head.output.bias' in name:
            continue
        assert torch.equal(value, expected[name]), name
        checked += 1
    assert checked > 50
    assert not any('tone_curve' in name for name in basis.state_dict())
    assert sum(p.numel() for p in basis.parameters()) < sum(p.numel() for p in tone.parameters())


def test_default_and_explicit_tone_mode_retain_the_original_checkpoint_schema():
    torch.manual_seed(5)
    original = small_model()
    torch.manual_seed(5)
    explicit = small_model(lut_mode='tone_residual')
    assert set(original.state_dict()) == set(explicit.state_dict())
    for name, value in original.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[name]), name
    assert all(hasattr(branch, 'tone_curve') for branch in explicit.branches.values())


@pytest.mark.parametrize('mode', ['invalid', '', None])
def test_unknown_mode_is_rejected(mode):
    with pytest.raises(ValueError, match='lut_mode'):
        small_model(lut_mode=mode)


@pytest.mark.parametrize('scene', ['indoor', 'outdoor'])
def test_ablation_config_only_changes_lut_mode_and_output_names(scene):
    tone = load_config('adaint/configs/face_' + scene + '_semantic_dual.yaml')
    basis = load_config('adaint/configs/face_' + scene + '_semantic_basis.yaml')
    control, experiment = copy.deepcopy(tone._cfg_dict), copy.deepcopy(basis._cfg_dict)
    assert experiment['model']['lut_mode'] == 'basis'
    assert control['model'].get('lut_mode', 'tone_residual') == 'tone_residual'
    assert experiment['model']['color_loss_weight'] == (.3 if scene == 'indoor' else .1)
    assert 'semantic_basis' in experiment['work_dir']
    for config in (control, experiment):
        config.pop('work_dir')
        config.pop('exp_name')
        config['model'].pop('lut_mode', None)
    assert control == experiment
    tone.gpus = basis.gpus = 1
    resolve_training_schedule(tone, 67)
    resolve_training_schedule(basis, 67)
    assert tone.total_iters == basis.total_iters
    assert tone.train_cfg.n_fix_iters == basis.train_cfg.n_fix_iters


def test_only_lut_generation_changes_in_existing_production_files():
    # Guard the user's request: forward_train/dummy/test and every loss stay intact.
    old_model = subprocess.check_output(['git', 'show', 'fcaf8bc:adaint/model_semantic.py']).decode()
    with open('adaint/model_semantic.py') as handle:
        new_model = handle.read()
    start = '    def forward_dummy('
    assert old_model[old_model.index(start):] == new_model[new_model.index(start):]
    old_modules = subprocess.check_output(['git', 'show', 'fcaf8bc:adaint/semantic_modules.py']).decode()
    with open('adaint/semantic_modules.py') as handle:
        new_modules = handle.read()
    start = 'class DualLocalAffineDecoder'
    assert old_modules[old_modules.index(start):] == new_modules[new_modules.index(start):]


def test_basis_checkpoint_roundtrip_and_tone_checkpoint_is_rejected(tmp_path):
    path = str(tmp_path / 'basis.pth')
    model = small_model(lut_mode='basis')
    torch.save(dict(state_dict=model.state_dict()), path)
    restored = small_model(lut_mode='basis')
    load_enhancer_checkpoint(restored, path)
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in model.state_dict().items())
    torch.save(dict(state_dict=small_model().state_dict()), path)
    with pytest.raises(RuntimeError, match='incompatible'):
        load_enhancer_checkpoint(restored, path)


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='real AiLUT CUDA kernel')


@CUDA
def test_basis_mode_reuses_losses_and_has_gradients_through_both_heads_nodes_and_local():
    torch.manual_seed(17)
    model = small_model(lut_mode='basis').cuda().eval()
    data = batch()
    result = model.forward_train(**data)
    losses = result['losses']
    torch.manual_seed(17)
    control = small_model().cuda().eval()
    assert set(losses) == set(control.forward_train(**data)['losses'])
    loss = sum(value for name, value in losses.items() if name.startswith('loss_'))
    assert torch.isfinite(loss)
    loss.backward()
    for branch in model.branches.values():
        assert not hasattr(branch, 'tone_curve')
        for parameter in (branch.lut_generator.head.output.weight,
                          branch.lut_generator.basis_luts, branch.adaint.head.output.weight,
                          branch.condition_encoder.stats_projection.weight):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0
        assert torch.all(branch.lut_generator.basis_luts.grad.flatten(1).abs().sum(1) > 0)
    for output in model.local_decoder.outputs:
        assert torch.isfinite(output.weight.grad).all() and output.weight.grad.abs().sum() > 0


@CUDA
def test_basis_mode_preserves_spatial_fusion_centering_crop_consistency_and_empty_regions():
    model = small_model(lut_mode='basis').cuda().eval()
    for output in model.local_decoder.outputs:
        torch.nn.init.normal_(output.weight, std=.02)
    data = batch()
    arguments = {name: value for name, value in data.items() if name not in ('lq', 'gt', 'supervision_mask')}
    with torch.no_grad():
        full, _, vertices, aux = model.forward_dummy(data['lq'], True, **arguments)
        expected = data['foreground_mask'] * aux['branch_outputs'][0] + (1 - data['foreground_mask']) * aux['branch_outputs'][1]
        assert torch.equal(full, expected)
        for delta, mask in zip(aux['context_deltas'], aux['local_masks']):
            mean = (delta * mask).sum((-1, -2)) / mask.sum((-1, -2))
            assert mean.abs().max() < 1e-6
        cropped = data['lq'][..., 12:36, 16:48]
        arguments.update(crop_box=cropped.new_tensor([[.25, .25, .75, .75]]).repeat(2, 1),
                         foreground_mask=data['foreground_mask'][..., 12:36, 16:48])
        crop, _, _ = model.forward_dummy(cropped, **arguments)
        assert torch.allclose(crop, full[..., 12:36, 16:48], atol=3e-6)
        for fill in (0., 1.):
            output, _, _ = model.forward_dummy(data['lq'],
                foreground_mask=torch.full_like(data['foreground_mask'], fill))
            assert torch.isfinite(output).all()


@CUDA
def test_training_freezes_and_unfreezes_both_basis_mode_adaint_heads():
    model = small_model(lut_mode='basis', train_cfg=dict(total_iters=3, n_fix_iters=1)).cuda().eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train_step(batch(), optimizer)
    assert all(not p.requires_grad for branch in model.branches.values() for p in branch.adaint.parameters())
    model.train_step(batch(), optimizer)
    assert all(p.requires_grad for branch in model.branches.values() for p in branch.adaint.parameters())
