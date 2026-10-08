"""Generate isolated, matching training/inference configs for LUT experiments."""
import argparse
import copy
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml


MODEL_TYPES = dict(v2='AiLUTRefinementBase', v3='AiLUTArchitectureV3',
                   global_local='AiLUTGlobalLocalV3')
METHODS = dict(
    diagonal='use_diagonal_monotonicity', kd_decay='use_kd_decay',
    confidence_kd='use_confidence_kd', stratified='use_stratified_statistics',
    edge_smoothness='use_edge_aware_smoothness', lab='use_lab_loss',
    mse='use_rgb_mse_loss', low_frequency='use_low_frequency_loss',
    full_context='use_full_image_context')
PRESETS = ['baseline', 'diagonal', 'kd_decay', 'confidence_kd', 'stratified',
           'edge_smoothness', 'lab', 'mse', 'low_frequency', 'full_context',
           'v3', 'global_local', 'all']


def configure_ppr10k_data(cfg, data_root=None, target=None):
    """Use the supplied train/validation folders verbatim, for expert a/b/c."""
    cfg = copy.deepcopy(cfg)
    root = data_root or cfg['data_root']
    target = target or cfg.get('target', 'a')
    if target not in ('a', 'b', 'c'):
        raise ValueError('PPR10K target must be a, b or c')
    cfg['data_root'], cfg['target'] = root, target
    for split in ('train', 'val', 'test'):
        if split not in cfg['data']:
            continue
        dataset = cfg['data'][split]
        folder = 'prr_train' if split == 'train' else 'prr_value'
        dataset.update(type='PPR10KFolder',
                       dir_lq=os.path.join(root, folder, 'source'),
                       dir_gt=os.path.join(root, folder, 'target_' + target))
        if split != 'train':
            dataset['dir_mask'] = os.path.join(root, 'masks_360p')
    return cfg


def prepare(base, name, preset='baseline', architecture=None, enable=(),
            overrides=(), output_dir='adaint/configs/generated', generations=None,
            epochs=None):
    cfg = copy.deepcopy(base)
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]*', name):
        raise ValueError('experiment name must be a simple alphanumeric identifier')
    if preset not in PRESETS:
        raise ValueError('unknown preset')
    chosen = architecture or ('global_local' if preset == 'global_local' else
                              'v3' if preset in ('v3', 'all') else 'v2')
    cfg['model']['type'] = MODEL_TYPES[chosen]
    cfg['model']['n_spatial_luts'] = 3 if chosen == 'global_local' else 2
    # New architectures are selected by model type, never by boolean switches.
    for key in ('guided_filter_radius', 'guided_filter_epsilon', 'local_gate_scale'):
        cfg['model'].pop(key, None)
    if chosen != 'v2':
        cfg['model'].update(guided_filter_radius=2, guided_filter_epsilon=0.001)
    if chosen == 'global_local':
        cfg['model']['local_gate_scale'] = 0.5
    selected = set(enable)
    if preset in METHODS:
        selected.add(preset)
    if preset == 'all':
        selected.update(('diagonal', 'kd_decay', 'stratified', 'edge_smoothness',
                         'lab', 'full_context'))
    for method, key in METHODS.items():
        cfg['model'][key] = method in selected
    cfg['exp_name'] = name
    cfg['work_dir'] = './work_dirs/' + name
    cfg['teacher_config_path'] = os.path.join(output_dir, name + '.yaml')
    if generations is not None:
        cfg['gen'] = generations
    if epochs is not None:
        cfg['train_epochs'] = epochs
    for assignment in overrides:
        path, value = assignment.split('=', 1)
        cursor = cfg
        parts = path.split('.')
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = yaml.safe_load(value)
    if cfg['gen'] < 1 or cfg['train_epochs'] < 1:
        raise ValueError('gen and train_epochs must be positive')
    test = copy.deepcopy(cfg)
    test['model'].pop('kd_loss', None)
    test['model'].pop('gen', None)
    test['train_cfg'] = None
    test['validation_test_cfg'] = copy.deepcopy(cfg['test_cfg'])
    test['test_cfg'] = dict(metrics=None, crop_border=0)
    test['data'] = dict(workers_per_gpu=1,
                        test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),
                        test=copy.deepcopy(cfg['data']['test']))
    if 'val' in cfg['data']:
        test['data']['val'] = copy.deepcopy(cfg['data']['val'])
    test['work_dir'] = cfg['work_dir'] + '/test'
    return cfg, test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--preset', choices=PRESETS, default='baseline')
    parser.add_argument('--architecture', choices=list(MODEL_TYPES))
    parser.add_argument('--enable', nargs='*', choices=list(METHODS), default=[])
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE')
    parser.add_argument('--gen', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--dataset', choices=['face', 'ppr10k'], default='face')
    parser.add_argument('--target', choices=['a', 'b', 'c'],
                        help='PPR10K groundtruth expert (default: a)')
    parser.add_argument('--data-root', help='override the PPR10K dataset directory')
    parser.add_argument('--base', help='override the selected dataset template')
    parser.add_argument('--output-dir', default='adaint/configs/generated')
    args = parser.parse_args()
    base_path = args.base or ('adaint/configs/ppr10k_refinement.yaml'
                             if args.dataset == 'ppr10k' else
                             'adaint/configs/face_indoor_refinement.yaml')
    with open(base_path) as handle:
        base = yaml.safe_load(handle)
    if args.dataset == 'ppr10k' or base.get('dataset') == 'ppr10k':
        base = configure_ppr10k_data(base, args.data_root, args.target)
    elif args.target or args.data_root:
        parser.error('--target and --data-root require a PPR10K template')
    cfg, test = prepare(base, args.name, args.preset, args.architecture,
                        args.enable, args.set, args.output_dir, args.gen, args.epochs)
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, args.name + '.yaml')
    test_path = os.path.join(args.output_dir, args.name + '_test.yaml')
    # Avoid accidentally changing an experiment's recorded config.
    if os.path.exists(train_path) or os.path.exists(test_path):
        raise FileExistsError('choose a new experiment name; output configs already exist')
    for path, contents in ((train_path, cfg), (test_path, test)):
        with open(path, 'w') as handle:
            yaml.safe_dump(contents, handle, sort_keys=False)
        print(path)


if __name__ == '__main__':
    main()
