"""Prepare matched from-scratch bounded-curve experiments; launch only on request.

The source is an old experiment's saved inference config, used only for its
architecture and data pipeline. No source model or teacher checkpoint is loaded.
Generated manifests are accepted by tools/launch_highlight_guard.py.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from pprint import pformat
import re
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_VARIANTS = ('legacy', 'bounded', 'detail', 'no_result', 'no_fine')
VARIANTS = DEFAULT_VARIANTS + ('legacy_jitter',)
PARENT_REVISION = 'af1616db97fa9e2a8c457901c22ac42a765064e3'
GUARD_METRICS = ('raw_high_clip', 'raw_high_clip_foreground',
                 'raw_high_clip_background', 'highlight_over_foreground',
                 'highlight_over_background')


def plain(value):
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(plain(item) for item in value)
    return value


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_config(source, name, work_dir, variant, epochs=100,
                   learning_rate=1e-4, workers=2, segments=4, epsilon=.001,
                   highlight_weight=1., gradient_weight=.1, top_fraction=.1,
                   seed=2026):
    """Copy a resolved source pipeline into a fresh all-parameter experiment.

    Highlight/gradient weights are experiment starting values, not tuned optima.
    The ``bounded`` group isolates the structural replacement with the old loss;
    ``detail`` then adds symmetric highlight and gradient supervision.
    """
    if variant not in VARIANTS:
        raise ValueError('unknown variant: ' + variant)
    cfg = plain(copy.deepcopy(source))
    model = cfg.get('model', {})
    if model.get('type') != 'AiLUTDual':
        raise ValueError('bounded-curve experiments require AiLUTDual')
    if model.get('lut_mode', 'tone_residual') != 'basis':
        raise ValueError('source config must describe the existing basis-LUT model')
    if not model.get('local_enabled', True):
        raise ValueError('bounded-curve experiments require the local decoder')
    if not model.get('postprocess'):
        raise ValueError('source must include its saved postprocess settings')
    cfg.update(exp_name=name, work_dir=str(work_dir), seed=int(seed),
               gen=1, enable_kd=False, train_epochs=int(epochs),
               adaint_fix_epochs=0, checkpoint_interval_epochs=1,
               evaluation_interval_epochs=1, log_interval_epochs=1,
               infer_after_generation=False, postprocess_init=None,
               load_from=None, resume_from=None, train_cfg={})
    # These values must never be inherited from a resolved fine-tuning run.
    model.pop('kd_loss', None)
    model.pop('gen', None)
    model['pretrained'] = False
    model['highlight_guard'] = dict(
        over_weight=0., lut_range_weight=0., final_range_weight=0.,
        keep_weight=0., top_fraction=float(top_fraction), over_margin=2. / 255.,
        keep_error_threshold=.03, reference_config=None,
        reference_checkpoint=None, trainable_scope='all',
        detach_result_condition=False)
    legacy = variant in ('legacy', 'legacy_jitter')
    detail = variant in ('detail', 'no_result', 'no_fine')
    model['bounded_curves'] = dict(
        enabled=not legacy, segments=int(segments), epsilon=float(epsilon),
        highlight_weight=float(highlight_weight) if detail else 0.,
        gradient_weight=float(gradient_weight) if detail else 0.,
        top_fraction=float(top_fraction))
    postprocess = model['postprocess']
    postprocess['freeze_global'] = False
    if not legacy:
        postprocess.update(global_dc=False, regional_dc=False, fixed_rgb_bias=None)
    if variant == 'no_result':
        postprocess['result_condition'] = False
    if variant == 'no_fine':
        postprocess['multiscale'] = False

    augmentations = 0
    for step in cfg['data']['train']['pipeline']:
        if step['type'] == 'SemanticContextAugment':
            if variant != 'legacy_jitter':
                step.update(brightness=0., saturation=0.)
            augmentations += 1
        elif step['type'] == 'RandomColorJitter':
            raise ValueError('source pipeline must first be resolved by load_config')
    if augmentations != 1:
        raise ValueError('source needs exactly one resolved SemanticContextAugment')

    validation = copy.deepcopy(cfg.get('validation_test_cfg') or cfg.get('test_cfg') or {})
    metrics = list(validation.get('metrics') or [
        'PSNR', 'deltaE_P3', 'deltaE_P3_top', 'deltaE_P3_foreground',
        'deltaE_P3_background'])
    for metric in GUARD_METRICS:
        if metric not in metrics:
            metrics.append(metric)
    validation.update(metrics=metrics, crop_border=validation.get('crop_border', 0))
    cfg['test_cfg'] = validation
    cfg['validation_test_cfg'] = copy.deepcopy(validation)
    cfg['data']['workers_per_gpu'] = int(workers)
    for split in ('train', 'val', 'test'):
        if split not in cfg['data']:
            continue
        loader = cfg['data'].setdefault(split + '_dataloader', {})
        loader.update(samples_per_gpu=1, workers_per_gpu=int(workers),
                      persistent_workers=workers > 0)
    cfg['optimizers']['lr'] = float(learning_rate)
    cfg['lr_config'] = dict(policy='CosineAnnealing', min_lr=1e-6, by_epoch=False)
    cfg['best_checkpoint_metric'], cfg['best_checkpoint_rule'] = 'PSNR', 'greater'
    if cfg.get('custom_hooks'):
        cfg['custom_hooks'] = [hook for hook in cfg['custom_hooks']
                               if hook.get('type') != 'LUTResumeCounterHook']
    return cfg


def write_python_config(path, cfg):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against a concurrent preparation process.
    with path.open('x') as handle:
        handle.write('\n\n'.join('{} = {}'.format(key, pformat(plain(value), width=100))
                                 for key, value in cfg.items()) + '\n')


def annotation_records(source):
    records = {}
    for split in ('train', 'val', 'test'):
        dataset = source.get('data', {}).get(split)
        if not dataset or not dataset.get('ann_file'):
            continue
        path = Path(dataset['ann_file']).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        path = path.resolve()
        records[split] = dict(path=str(path), sha256=sha256(path),
                              count=len(path.read_text().split()),
                              dir_lq=dataset.get('dir_lq'), dir_gt=dataset.get('dir_gt'))
    if not all(split in records for split in ('train', 'val')):
        raise ValueError('source config needs train and val annotation files')
    return records


def code_revision():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unavailable'


def augmentation_record(cfg):
    step = next(step for step in cfg['data']['train']['pipeline']
                if step['type'] == 'SemanticContextAugment')
    return dict(brightness=step.get('brightness', .2),
                saturation=step.get('saturation', .2),
                geometry='unchanged from saved source config')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', choices=('indoor', 'outdoor'), required=True)
    parser.add_argument('--source-config', type=Path, required=True,
                        help='saved inference config; architecture/data only, no weights')
    parser.add_argument('--suite-dir', type=Path,
                        default=Path('work_dirs/bc_20261028_suite'))
    parser.add_argument('--prefix', default='bc_20261028')
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(DEFAULT_VARIANTS))
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--segments', type=int, default=4)
    parser.add_argument('--epsilon', type=float, default=.001)
    parser.add_argument('--highlight-weight', type=float, default=1.,
                        help='symmetric highlight loss starting weight, not a tuned optimum')
    parser.add_argument('--gradient-weight', type=float, default=.1,
                        help='gradient loss starting weight, not a tuned optimum')
    parser.add_argument('--top-fraction', type=float, default=.1)
    parser.add_argument('--parent-revision', default=PARENT_REVISION)
    parser.add_argument('--execute', choices=VARIANTS,
                        help='explicitly train one prepared variant; otherwise just prepare')
    args = parser.parse_args()
    finite = all(math.isfinite(value) for value in
                 (args.lr, args.epsilon, args.highlight_weight,
                  args.gradient_weight, args.top_fraction))
    if (not finite or args.epochs < 1 or args.lr < 1e-6 or args.workers < 0 or
            args.segments < 2 or not 0 < args.epsilon < .5 or
            args.highlight_weight < 0 or args.gradient_weight < 0 or
            not 0 < args.top_fraction <= 1):
        parser.error('invalid budget, learning rate, worker count or curve/loss settings')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', args.prefix):
        parser.error('--prefix must contain only letters, numbers, underscores and hyphens')
    if len(set(args.variants)) != len(args.variants):
        parser.error('--variants must not contain duplicates')
    if args.execute and args.execute not in args.variants:
        parser.error('--execute must be included in --variants')

    from adaint.config_utils import load_config
    source_config = args.source_config.expanduser().resolve()
    source = load_config(str(source_config))
    annotations = annotation_records(source)
    suite = args.suite_dir.expanduser().resolve()
    manifest_path = suite / ('manifest_' + args.scene + '.json')
    if manifest_path.exists():
        raise FileExistsError('refusing to overwrite manifest: ' + str(manifest_path))
    runs, configs = [], []
    for variant in args.variants:
        name = '{}_{}_{}_s{}'.format(args.prefix, args.scene, variant, args.seed)
        destination = ROOT / 'work_dirs' / name
        cfg_path = suite / 'configs' / (name + '.py')
        # Refuse any existing experiment, even if it has no latest checkpoint.
        if destination.exists() or destination.is_symlink():
            raise FileExistsError('experiment path already exists: ' + str(destination))
        if cfg_path.exists() or cfg_path.is_symlink():
            raise FileExistsError('config path already exists: ' + str(cfg_path))
        cfg = prepare_config(source._cfg_dict, name, destination, variant,
                             epochs=args.epochs, learning_rate=args.lr,
                             workers=args.workers, segments=args.segments,
                             epsilon=args.epsilon, highlight_weight=args.highlight_weight,
                             gradient_weight=args.gradient_weight,
                             top_fraction=args.top_fraction, seed=args.seed)
        command = [sys.executable, str(ROOT / 'tools/train_kd_unified.py'), str(cfg_path),
                   '--seed', str(args.seed), '--gpus', '1']
        runs.append(dict(variant=variant, name=name, config=str(cfg_path),
                         work_dir=str(destination), command=command,
                         shell_command=' '.join(shlex.quote(part) for part in command),
                         bounded_curves=cfg['model']['bounded_curves'],
                         postprocess=cfg['model']['postprocess'],
                         augmentation=augmentation_record(cfg),
                         initialization='from_scratch'))
        configs.append((cfg_path, cfg))

    manifest = dict(
        scene=args.scene, source_config=str(source_config),
        source_config_sha256=sha256(source_config), initialization='from_scratch',
        source_checkpoint=None, teacher_checkpoint=None, parent_revision=args.parent_revision,
        code_revision=code_revision(), epochs=args.epochs, learning_rate=args.lr,
        seed=args.seed, segments=args.segments, epsilon=args.epsilon,
        highlight_weight=args.highlight_weight, gradient_weight=args.gradient_weight,
        top_fraction=args.top_fraction, data_annotations=annotations,
        data_annotation_hashes={split: row['sha256'] for split, row in annotations.items()},
        semantic_masks=plain(source.get('semantic_masks', {})),
        augmentation=dict(
            by_variant={run['variant']: run['augmentation'] for run in runs},
            geometry='unchanged from saved source config'),
        selection=dict(metric='PSNR', rule='greater', split='val'),
        note='All runs train from scratch with all model parameters enabled, identical data, '
             'seed and budget. No checkpoint or frozen teacher is used. '
             'Optional legacy_jitter retains the source brightness/saturation jitter; '
             'all other variants disable it. Actual per-run jitter is recorded. '
             'Detail loss weights are declared experimental starting values. '
             'Inference exports raw PNG; P3 ICC is a separate p3.py step.', runs=runs)
    # All paths, annotation hashes and configs were checked before the first write.
    for cfg_path, cfg in configs:
        write_python_config(cfg_path, cfg)
    suite.mkdir(parents=True, exist_ok=True)
    with manifest_path.open('x') as handle:
        json.dump(manifest, handle, indent=2)
        handle.write('\n')
    print(json.dumps(manifest, indent=2))
    if args.execute:
        selected = next(run for run in runs if run['variant'] == args.execute)
        subprocess.check_call(selected['command'], cwd=str(ROOT))


if __name__ == '__main__':
    main()
