"""Create named train/test configurations for this branch's single AiLUT model."""
import argparse
import copy
import os
import re
import yaml


def configure_ppr10k_data(cfg, data_root=None, target=None):
    root, target = data_root or cfg['data_root'], target or cfg.get('target', 'a')
    if target not in ('a', 'b', 'c'):
        raise ValueError('PPR10K target must be a, b or c')
    cfg['data_root'], cfg['target'] = root, target
    for split in ('train', 'val', 'test'):
        if split not in cfg['data']:
            continue
        dataset = cfg['data'][split]
        folder = 'prr_train' if split == 'train' else 'prr_value'
        dataset.update(type='PPR10KFolder', dir_lq=os.path.join(root, folder, 'source'),
                       dir_gt=os.path.join(root, folder, 'target_'+target))
        if split != 'train':
            dataset['dir_mask'] = os.path.join(root, 'masks_360p')
    return cfg


def prepare(base, name, overrides=(), generations=None, epochs=None):
    cfg = copy.deepcopy(base)
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]*', name):
        raise ValueError('experiment name must be a simple alphanumeric identifier')
    cfg.update(exp_name=name, work_dir='./work_dirs/'+name)
    if generations is not None:
        cfg['gen'], cfg['enable_kd'] = generations, generations > 1
    if epochs is not None:
        cfg['train_epochs'] = epochs
    for assignment in overrides:
        path, value = assignment.split('=', 1)
        cursor, parts = cfg, path.split('.')
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = yaml.safe_load(value)
    if cfg['model']['type'] != 'AiLUT':
        raise ValueError('this branch supports only model.type=AiLUT')
    if cfg['gen'] < 1 or cfg['train_epochs'] < 1:
        raise ValueError('gen and train_epochs must be positive')
    if cfg['gen'] > 1 and not cfg.get('enable_kd'):
        raise ValueError('multiple generations require enable_kd; use gen=1 for GT only')
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
    test['work_dir'] = cfg['work_dir']+'/test'
    return cfg, test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--dataset', choices=['face', 'ppr10k'], default='ppr10k')
    parser.add_argument('--target', choices=['a', 'b', 'c'])
    parser.add_argument('--data-root')
    parser.add_argument('--base')
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE')
    parser.add_argument('--gen', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--output-dir', default='adaint/configs/generated')
    args = parser.parse_args()
    path = args.base or ('adaint/configs/ppr10k.yaml' if args.dataset == 'ppr10k'
                         else 'adaint/configs/face_indoor.yaml')
    with open(path) as handle:
        base = yaml.safe_load(handle)
    if base.get('dataset') == 'ppr10k':
        base = configure_ppr10k_data(base, args.data_root, args.target)
    elif args.target or args.data_root:
        parser.error('--target and --data-root require a PPR10K template')
    cfg, test = prepare(base, args.name, args.set, args.gen, args.epochs)
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, args.name+'.yaml')
    test_path = os.path.join(args.output_dir, args.name+'_test.yaml')
    if os.path.exists(train_path) or os.path.exists(test_path):
        raise FileExistsError('choose a new name; experiment configs already exist')
    for path, contents in ((train_path, cfg), (test_path, test)):
        with open(path, 'w') as handle:
            yaml.safe_dump(contents, handle, sort_keys=False)
        print(path)


if __name__ == '__main__':
    main()
