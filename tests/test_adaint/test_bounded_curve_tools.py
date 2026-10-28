"""Protocol invariants for matched from-scratch bounded-curve experiments."""
import copy
import importlib.util
import json
from pathlib import Path
import runpy
import sys
import types

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / 'tools' / 'run_bounded_curves.py'


@pytest.fixture
def generator():
    spec = importlib.util.spec_from_file_location('bounded_curve_tools_under_test', str(SCRIPT))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_config():
    """A saved inference config carrying stale fine-tuning settings on purpose."""
    return dict(
        model=dict(type='AiLUTDual', lut_mode='basis', local_enabled=True,
                   pretrained=True, gen=2, kd_loss=dict(model_path='old_teacher.pth'),
                   color_loss_weight=.1,
                   postprocess=dict(result_condition=True, global_dc=True, regional_dc=True,
                                    multiscale=True, guidance='luminance', freeze_global=True,
                                    fixed_rgb_bias=None, global_dc_limit=.08,
                                    regional_dc_limit=.04),
                   highlight_guard=dict(over_weight=1., keep_weight=.1,
                                        reference_checkpoint='old_teacher.pth',
                                        trainable_scope='heads', detach_result_condition=True)),
        data=dict(
            workers_per_gpu=4,
            train=dict(ann_file='ann/train.txt', dir_lq='/data/input', dir_gt='/data/gt',
                       pipeline=[dict(type='LoadSemanticMasks', foreground_root='/data/masks'),
                                 dict(type='SemanticContextAugment', context_size=256,
                                      samples_size=64, region_samples_size=32,
                                      brightness=.2, saturation=.2, crop_ratio=(.6, 1.),
                                      flip_ratio=.5),
                                 dict(type='ImageToTensor', keys=['lq', 'gt', 'foreground_mask'])]),
            val=dict(ann_file='ann/val.txt', dir_lq='/data/input', dir_gt='/data/gt',
                     pipeline=[dict(type='LoadSemanticMasks', foreground_root='/data/masks')]),
            test=dict(ann_file='ann/test.txt', dir_lq='/data/test',
                      pipeline=[dict(type='LoadSemanticMasks', foreground_root='/data/test_masks')]),
            train_dataloader=dict(samples_per_gpu=3, workers_per_gpu=4),
            val_dataloader=dict(samples_per_gpu=2), test_dataloader=dict(samples_per_gpu=2)),
        validation_test_cfg=dict(metrics=['PSNR', 'deltaE_P3_foreground'], crop_border=0),
        test_cfg=dict(metrics=None), train_cfg=dict(n_fix_iters=500),
        optimizers=dict(type='Adam', lr=2e-5, weight_decay=0.), lr_config={},
        postprocess_init='old.pth', load_from='old.pth', resume_from='old.pth',
        gen=3, enable_kd=True, train_epochs=30,
        semantic_masks=dict(region_label='face_hair_neck', train_foreground_root='/data/masks'),
        custom_hooks=[dict(type='LUTResumeCounterHook'), dict(type='UnrelatedHook')])


@pytest.mark.parametrize('variant', ['legacy', 'bounded', 'detail', 'no_result', 'no_fine'])
def test_every_variant_removes_pretraining_freezing_and_teacher_state(generator, variant):
    cfg = generator.prepare_config(source_config(), 'unit', 'work_dirs/unit', variant)
    assert cfg['postprocess_init'] is cfg['load_from'] is cfg['resume_from'] is None
    assert cfg['gen'] == 1 and not cfg['enable_kd']
    assert cfg['train_cfg'] == {} and cfg['adaint_fix_epochs'] == 0
    assert cfg['train_epochs'] == 100 and cfg['seed'] == 2026
    assert cfg['optimizers']['lr'] == 1e-4
    assert cfg['lr_config'] == dict(policy='CosineAnnealing', min_lr=1e-6, by_epoch=False)
    assert not cfg['model']['pretrained']
    assert 'gen' not in cfg['model'] and 'kd_loss' not in cfg['model']
    assert not cfg['model']['postprocess']['freeze_global']
    guard = cfg['model']['highlight_guard']
    assert guard['trainable_scope'] == 'all' and not guard['detach_result_condition']
    assert all(guard[key] == 0 for key in
               ('over_weight', 'lut_range_weight', 'final_range_weight', 'keep_weight'))
    assert guard['reference_config'] is guard['reference_checkpoint'] is None
    assert cfg['custom_hooks'] == [dict(type='UnrelatedHook')]


@pytest.mark.parametrize('variant', ['legacy', 'bounded', 'detail', 'no_result', 'no_fine'])
def test_variants_keep_data_splits_masks_and_geometry_but_disable_jitter(generator, variant):
    source = source_config()
    before = copy.deepcopy(source)
    cfg = generator.prepare_config(source, 'unit', 'work_dirs/unit', variant)
    expected_train = copy.deepcopy(source['data']['train'])
    expected_train['pipeline'][1].update(brightness=0., saturation=0.)
    assert cfg['data']['train'] == expected_train
    for split in ('val', 'test'):
        assert cfg['data'][split] == source['data'][split]
    assert cfg['semantic_masks'] == source['semantic_masks']
    for split in ('train', 'val', 'test'):
        assert cfg['data'][split + '_dataloader'] == dict(
            samples_per_gpu=1, workers_per_gpu=2, persistent_workers=True)
    assert source == before


def test_variants_isolate_structure_loss_and_decoder_ablations(generator):
    configs = {name: generator.prepare_config(source_config(), 'unit', 'work_dirs/unit', name)
               for name in generator.VARIANTS}
    legacy = configs['legacy']['model']
    assert not legacy['bounded_curves']['enabled']
    assert legacy['postprocess']['global_dc'] and legacy['postprocess']['regional_dc']
    assert legacy['postprocess']['result_condition'] and legacy['postprocess']['multiscale']
    for name in generator.VARIANTS[1:]:
        model = configs[name]['model']
        assert model['bounded_curves']['enabled']
        assert model['bounded_curves']['segments'] == 4
        assert model['bounded_curves']['epsilon'] == .001
        assert not model['postprocess']['global_dc'] and not model['postprocess']['regional_dc']
        assert model['postprocess']['fixed_rgb_bias'] is None
        assert model['postprocess']['guidance'] == 'luminance'
    assert configs['bounded']['model']['bounded_curves']['highlight_weight'] == 0
    assert configs['bounded']['model']['bounded_curves']['gradient_weight'] == 0
    for name in ('detail', 'no_result', 'no_fine'):
        assert configs[name]['model']['bounded_curves']['highlight_weight'] == 1.
        assert configs[name]['model']['bounded_curves']['gradient_weight'] == .1
        assert configs[name]['model']['bounded_curves']['top_fraction'] == .1
    for name, changed_key in (('no_result', 'result_condition'), ('no_fine', 'multiscale')):
        expected = copy.deepcopy(configs['detail'])
        expected['model']['postprocess'][changed_key] = False
        assert configs[name] == expected


def test_bounded_variant_clears_inherited_fixed_bias_and_zero_workers_are_valid(generator):
    source = source_config()
    source['model']['postprocess'].update(global_dc=False, fixed_rgb_bias=[.01, -.02, .03])
    cfg = generator.prepare_config(source, 'unit', 'work_dirs/unit', 'detail', workers=0,
                                   highlight_weight=.5, gradient_weight=.02)
    assert cfg['model']['postprocess']['fixed_rgb_bias'] is None
    assert cfg['model']['bounded_curves']['highlight_weight'] == .5
    assert cfg['model']['bounded_curves']['gradient_weight'] == .02
    for split in ('train', 'val', 'test'):
        assert cfg['data'][split + '_dataloader']['workers_per_gpu'] == 0
        assert not cfg['data'][split + '_dataloader']['persistent_workers']


def test_config_serialization_is_executable_and_cannot_overwrite_existing_file(generator, tmp_path):
    cfg = generator.prepare_config(source_config(), 'unit', 'work_dirs/unit', 'detail')
    path = tmp_path / 'config.py'
    generator.write_python_config(path, cfg)
    loaded = runpy.run_path(str(path))
    for key in cfg:
        assert loaded[key] == cfg[key]
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        generator.write_python_config(path, dict(unwanted=True))
    assert path.read_bytes() == original


@pytest.fixture
def cli_context(generator, monkeypatch, tmp_path):
    root = tmp_path / 'repo'
    (root / 'ann').mkdir(parents=True)
    for split, ids in (('train', '1\n2\n'), ('val', '3\n'), ('test', '4\n5\n')):
        (root / 'ann' / (split + '.txt')).write_text(ids)
    source_path = root / 'saved_inference.py'
    source_path.write_text('# A saved config; model weights are intentionally absent.\n')
    source = source_config()

    class Config(dict):
        @property
        def _cfg_dict(self):
            return self

    adaint = types.ModuleType('adaint')
    adaint.__path__ = []
    config_utils = types.ModuleType('adaint.config_utils')
    config_utils.load_config = lambda _: Config(copy.deepcopy(source))
    monkeypatch.setitem(sys.modules, 'adaint', adaint)
    monkeypatch.setitem(sys.modules, 'adaint.config_utils', config_utils)
    monkeypatch.setattr(generator, 'ROOT', root)
    monkeypatch.setattr(generator, 'code_revision', lambda: 'unit_code_revision')
    launched = []
    monkeypatch.setattr(generator.subprocess, 'check_call',
                        lambda command, **kwargs: launched.append((command, kwargs)))
    suite = tmp_path / 'suite'
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), '--scene', 'outdoor', '--source-config',
                                    str(source_path), '--suite-dir', str(suite), '--prefix', 'unit'])
    return dict(root=root, suite=suite, source=source, source_path=source_path, launched=launched)


def test_cli_only_prepares_reproducible_text_artifacts_and_no_icc_export(
        generator, cli_context, capsys):
    generator.main()
    manifest = json.loads((cli_context['suite'] / 'manifest_outdoor.json').read_text())
    assert manifest['initialization'] == 'from_scratch'
    assert manifest['source_checkpoint'] is manifest['teacher_checkpoint'] is None
    assert manifest['source_config_sha256'] == generator.sha256(cli_context['source_path'])
    assert manifest['parent_revision'] == generator.PARENT_REVISION
    assert manifest['code_revision'] == 'unit_code_revision'
    assert manifest['selection']['split'] == 'val'
    assert not cli_context['launched']
    assert len(manifest['runs']) == 5
    for split in ('train', 'val', 'test'):
        annotation = cli_context['root'] / 'ann' / (split + '.txt')
        assert manifest['data_annotations'][split]['sha256'] == generator.sha256(annotation)
    for run in manifest['runs']:
        cfg = runpy.run_path(run['config'])
        assert not cfg['infer_after_generation']
        assert cfg['data']['val'] == cli_context['source']['data']['val']
        assert cfg['data']['test'] == cli_context['source']['data']['test']
        # Preparation must not launch training, inference, or p3.py implicitly.
        assert run['command'][1].endswith('tools/train_kd_unified.py')
        assert '--p3' not in run['command'] and '--resume-from' not in run['command']
        assert run['command'][-2:] == ['--gpus', '1']
        assert not Path(run['work_dir']).exists()
    files = [path for path in cli_context['suite'].rglob('*') if path.is_file()]
    assert {path.suffix for path in files} == {'.py', '.json'}
    capsys.readouterr()


@pytest.mark.parametrize('collision', ['manifest', 'config', 'experiment'])
def test_cli_refuses_existing_paths_before_writing_any_configs(
        generator, cli_context, collision):
    suite, root = cli_context['suite'], cli_context['root']
    # Collide with the final variant to catch partial writes of earlier groups.
    name = 'unit_outdoor_no_fine_s2026'
    if collision == 'manifest':
        path = suite / 'manifest_outdoor.json'
    elif collision == 'config':
        path = suite / 'configs' / (name + '.py')
    else:
        path = root / 'work_dirs' / name / 'keep.txt'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'preserve this existing experiment')
    before = {str(file): file.read_bytes() for directory in (root, suite)
              if directory.exists() for file in directory.rglob('*') if file.is_file()}
    with pytest.raises(FileExistsError):
        generator.main()
    after = {str(file): file.read_bytes() for directory in (root, suite)
             if directory.exists() for file in directory.rglob('*') if file.is_file()}
    assert after == before
    assert not cli_context['launched']
