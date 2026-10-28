"""End-to-end bounded mapping, all-parameter learning and old-path regression."""
import copy

import pytest
import torch

from tests.test_adaint.test_semantic_dual import small_model, batch


POST = dict(result_condition=True, global_dc=False, regional_dc=False,
            multiscale=True, guidance='luminance', freeze_global=False)


def model_options():
    return dict(lut_mode='basis', postprocess=copy.deepcopy(POST),
        bounded_curves=dict(enabled=True, segments=4, epsilon=.001,
                            highlight_weight=1., gradient_weight=.1))


def args(data):
    return {key: value for key, value in data.items()
            if key not in ('lq', 'gt', 'supervision_mask')}


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_bounded_mapping_and_full_trainability(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.manual_seed(2026)
    model = small_model(**model_options()).to(device).train()
    assert all(p.requires_grad for p in model.parameters())
    assert not hasattr(model, 'postprocess_calibration')
    data = batch(device)
    output, _, _, aux = model.forward_dummy(data['lq'], True, **args(data))
    assert 0 <= output.min() <= output.max() <= 1
    for lut in aux['luts']:
        assert 0 <= lut.min() <= lut.max() <= 1
    for vertices in aux['vertices']:
        assert torch.equal(vertices[..., 0], torch.zeros_like(vertices[..., 0]))
        assert torch.equal(vertices[..., -1], torch.ones_like(vertices[..., -1]))
    losses = model.forward_train(**data)['losses']
    assert 'loss_curve_highlight' in losses and 'loss_curve_gradient' in losses
    sum(v for k, v in losses.items() if k.startswith('loss')).backward()
    for prefix in ('backbone.', 'branches.foreground.adaint.',
                   'branches.foreground.lut_generator.basis_luts',
                   'local_decoder.outputs.', 'local_decoder.refinement_outputs.'):
        gradients = [p.grad for n, p in model.named_parameters() if n.startswith(prefix)]
        assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients), prefix
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train_step(data, optimizer)
    assert all(p.requires_grad for p in model.parameters())
    model.eval()
    with torch.no_grad():
        for value in (0., 1.):
            image = torch.full_like(data['lq'], value)
            for mask_value in (0., .5, 1.):
                mask = torch.full_like(data['foreground_mask'], mask_value)
                result = model.forward_dummy(image, foreground_mask=mask)[0]
                assert torch.isfinite(result).all()
                assert 0 <= result.min() <= result.max() <= 1


def test_bounded_configuration_rejects_unbounded_or_frozen_additions():
    for key, value in [('global_dc', True), ('regional_dc', True), ('freeze_global', True)]:
        options = model_options()
        options['postprocess'][key] = value
        with pytest.raises(ValueError):
            small_model(**options)
    options = model_options()
    options['train_cfg'] = dict(n_fix_iters=1)
    with pytest.raises(ValueError):
        small_model(**options)


def test_old_model_schema_is_unchanged_when_bounded_disabled():
    torch.manual_seed(14)
    old = small_model(lut_mode='basis', postprocess=POST).eval()
    torch.manual_seed(14)
    explicit = small_model(lut_mode='basis', postprocess=POST,
                           bounded_curves=dict(enabled=False)).eval()
    assert set(old.state_dict()) == set(explicit.state_dict())
    for name, value in old.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[name]), name
    data = batch('cpu')
    with torch.no_grad():
        assert torch.equal(old.forward_dummy(data['lq'], **args(data))[0],
                           explicit.forward_dummy(data['lq'], **args(data))[0])
