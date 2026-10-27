"""Prepare isolated, equal-budget highlight-guard experiments (no launch by default)."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from pprint import pformat
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VARIANTS = ('legacy_control', 'heads_control', 'range', 'over', 'range_over', 'full', 'detach')
GUARD_METRICS = ('raw_high_clip', 'raw_high_clip_foreground', 'raw_high_clip_background',
                 'highlight_over_foreground', 'highlight_over_background')


def plain(value):
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(plain(item) for item in value)
    return value


def sha256(path):
    digest = hashlib.sha256()
    with open(str(path), 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def guard_variant(name, source_config, source_checkpoint, args):
    if name not in VARIANTS:
        raise ValueError('unknown variant: ' + name)
    options = dict(over_weight=0., lut_range_weight=0., final_range_weight=0.,
                   keep_weight=0., top_fraction=args.top_fraction,
                   over_margin=args.over_margin, keep_error_threshold=args.keep_error_threshold,
                   reference_config=None, reference_checkpoint=None,
                   trainable_scope='all' if name == 'legacy_control' else 'heads',
                   detach_result_condition=name == 'detach')
    if name in ('range', 'range_over', 'full', 'detach'):
        options.update(lut_range_weight=args.lut_range_weight,
                       final_range_weight=args.final_range_weight)
    if name in ('over', 'range_over', 'full', 'detach'):
        options['over_weight'] = args.over_weight
    if name in ('full', 'detach'):
        options.update(keep_weight=args.keep_weight, reference_config=str(source_config),
                       reference_checkpoint=str(source_checkpoint))
    return options


def prepare_config(source, name, work_dir, source_checkpoint, guard, epochs,
                   learning_rate, workers=2):
    """Restore a saved inference config to a fresh fine-tuning config."""
    cfg = plain(copy.deepcopy(source))
    if cfg.get('model', {}).get('type') != 'AiLUTDual':
        raise ValueError('highlight experiments require AiLUTDual')
    cfg.update(exp_name=name, work_dir=str(work_dir), gen=1, enable_kd=False,
               train_epochs=int(epochs), adaint_fix_epochs=0,
               checkpoint_interval_epochs=1, evaluation_interval_epochs=1,
               log_interval_epochs=1, infer_after_generation=False,
               postprocess_init=str(source_checkpoint), load_from=None, resume_from=None,
               train_cfg={})
    cfg['model'].pop('kd_loss', None)
    cfg['model'].pop('gen', None)
    cfg['model']['highlight_guard'] = guard
    cfg['model']['postprocess']['freeze_global'] = False
    validation = copy.deepcopy(cfg.get('validation_test_cfg') or cfg.get('test_cfg') or {})
    metrics = list(validation.get('metrics') or ['PSNR', 'deltaE_P3',
        'deltaE_P3_foreground', 'deltaE_P3_background'])
    for metric in GUARD_METRICS:
        if metric not in metrics:
            metrics.append(metric)
    validation.update(metrics=metrics, crop_border=validation.get('crop_border', 0))
    cfg['test_cfg'] = validation
    cfg['validation_test_cfg'] = copy.deepcopy(validation)
    cfg['data']['workers_per_gpu'] = int(workers)
    for split in ('train', 'val', 'test'):
        loader = cfg['data'].setdefault(split + '_dataloader', {})
        loader['workers_per_gpu'] = int(workers)
        loader['persistent_workers'] = workers > 0
    cfg['optimizers']['lr'] = float(learning_rate)
    cfg['lr_config'] = dict(policy='CosineAnnealing', min_lr=1e-6, by_epoch=False)
    cfg['best_checkpoint_metric'], cfg['best_checkpoint_rule'] = 'PSNR', 'greater'
    # Resolved pipeline is retained; load_config detects LoadSemanticMasks and
    # does not insert a second semantic augmentation pipeline.
    return cfg


def write_python_config(path, cfg):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n\n'.join('{} = {}'.format(k, pformat(plain(v), width=100))
                              for k, v in cfg.items()) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', choices=('indoor', 'outdoor'), required=True)
    parser.add_argument('--source-config', type=Path, required=True)
    parser.add_argument('--source-checkpoint', type=Path, required=True)
    parser.add_argument('--suite-dir', type=Path, default=Path('work_dirs/hg_20261027_suite'))
    parser.add_argument('--prefix', default='hg_20261027')
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--over-weight', type=float, default=1.)
    parser.add_argument('--lut-range-weight', type=float, default=.01)
    parser.add_argument('--final-range-weight', type=float, default=.1)
    parser.add_argument('--keep-weight', type=float, default=.1)
    parser.add_argument('--top-fraction', type=float, default=.1)
    parser.add_argument('--over-margin', type=float, default=2./255.)
    parser.add_argument('--keep-error-threshold', type=float, default=.03)
    parser.add_argument('--execute', choices=VARIANTS,
                        help='explicitly train one prepared variant; otherwise only write configs')
    args = parser.parse_args()
    if args.epochs < 1 or args.lr <= 0 or args.workers < 0:
        parser.error('invalid budget, learning rate or worker count')
    from adaint.config_utils import load_config
    source_config, checkpoint = args.source_config.resolve(), args.source_checkpoint.resolve()
    source = load_config(str(source_config))
    suite = args.suite_dir.resolve()
    suite.mkdir(parents=True, exist_ok=True)
    runs = []
    for variant in args.variants:
        name = '{}_{}_{}_s{}'.format(args.prefix, args.scene, variant, args.seed)
        destination = ROOT/'work_dirs'/name
        cfg_path = suite/'configs'/(name+'.py')
        guard = guard_variant(variant, source_config, checkpoint, args)
        cfg = prepare_config(source._cfg_dict, name, destination, checkpoint, guard,
                             args.epochs, args.lr, args.workers)
        if cfg_path.exists():
            raise FileExistsError('refusing to overwrite experiment config: ' + str(cfg_path))
        if (destination/'gen_0/latest.pth').exists():
            raise FileExistsError('experiment already has training outputs: ' + str(destination))
        write_python_config(cfg_path, cfg)
        command = [sys.executable, str(ROOT/'tools/train_kd_unified.py'), str(cfg_path),
                   '--seed', str(args.seed)]
        runs.append(dict(variant=variant, name=name, config=str(cfg_path),
                         work_dir=str(destination), command=command,
                         shell_command=' '.join(shlex.quote(c) for c in command), guard=guard))
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT)).decode().strip()
    except subprocess.CalledProcessError:
        revision = 'unavailable'
    manifests = dict(scene=args.scene, source_config=str(source_config),
        source_config_sha256=sha256(source_config), source_checkpoint=str(checkpoint),
        source_checkpoint_sha256=sha256(checkpoint), code_revision=revision,
        epochs=args.epochs, learning_rate=args.lr, seed=args.seed,
        data_annotation_hashes={split:sha256(source.data[split].ann_file)
                               for split in ('train','val')},
        note='All variants use identical data, initialization, seed and budget. '
             'No test images or expert screenshots are used for training/selection.', runs=runs)
    manifest_path = suite/('manifest_'+args.scene+'.json')
    if manifest_path.exists():
        raise FileExistsError(str(manifest_path))
    manifest_path.write_text(json.dumps(manifests, indent=2))
    print(json.dumps(manifests, indent=2))
    if args.execute:
        selected = [run for run in runs if run['variant'] == args.execute]
        if not selected:
            parser.error('--execute must be included in --variants')
        subprocess.check_call(selected[0]['command'], cwd=str(ROOT))


if __name__ == '__main__':
    main()
