"""Training budgets, inference configs and strict enhancer checkpoint loading."""
import copy
import math
import numbers
import os
from pprint import pformat

import torch
from mmcv import Config
from mmcv.runner import HOOKS, Hook, get_dist_info
from .epoch_hooks import configure_epoch_display

ENHANCER_TYPES = ('AiLUT', 'AiLUTDual')


@HOOKS.register_module()
class LUTResumeCounterHook(Hook):
    """Align old MMCV's zero-based resume metadata with completed model steps."""
    def before_run(self, runner):
        model = getattr(runner.model, 'module', runner.model)
        completed = int(model.cnt_iters.item())
        if completed != runner.iter:
            runner.logger.info('Align resumed iteration %s to model count %s',
                               runner.iter, completed)
            runner._iter = completed


def resolve_training_schedule(cfg, dataset_size, distributed=False):
    if cfg.get('train_epochs', None) is not None:
        loader = cfg.data.get('train_dataloader', {})
        samples = loader.get('samples_per_gpu', cfg.data.get('samples_per_gpu', 1))
        world = get_dist_info()[1] if distributed else cfg.get('gpus', 1)
        drop_last = loader.get('drop_last', cfg.data.get('drop_last', False))
        if distributed:
            per_rank = int(math.ceil(dataset_size / world))
            steps = (per_rank // samples if drop_last else
                     int(math.ceil(per_rank / samples)))
        else:
            batch = samples * world
            steps = (dataset_size // batch if drop_last else
                     int(math.ceil(dataset_size / batch)))
        if steps < 1 or cfg.train_epochs < 1:
            raise ValueError('training schedule has no batches or epochs')
        cfg.train_cfg = cfg.get('train_cfg') or {}
        cfg.total_iters = int(steps * cfg.train_epochs)
        cfg.train_cfg.n_fix_iters = int(steps * cfg.get('adaint_fix_epochs', 5))
        if cfg.get('tone_warmup_epochs') is not None:
            warmup = cfg.tone_warmup_epochs
            if (isinstance(warmup, bool) or not isinstance(warmup, numbers.Integral) or
                    not 0 <= warmup < cfg.train_epochs):
                raise ValueError('tone_warmup_epochs must be an integer in [0, train_epochs)')
            if warmup and not cfg.model.get('spatial_tone_enabled', False):
                raise ValueError('tone warmup requires spatial_tone_enabled')
            cfg.train_cfg.tone_warmup_iters = int(steps * warmup)
        if cfg.get('checkpoint_config'):
            cfg.checkpoint_config.interval = int(steps * cfg.get('checkpoint_interval_epochs', 1))
        if cfg.get('evaluation'):
            cfg.evaluation.interval = int(steps * cfg.get('evaluation_interval_epochs', 1))
        cfg.iters_per_epoch = steps
        if cfg.get('log_interval_epochs') is not None:
            for key in ('train_epochs', 'adaint_fix_epochs', 'checkpoint_interval_epochs',
                        'evaluation_interval_epochs', 'log_interval_epochs'):
                value = cfg.get(key, 5 if key == 'adaint_fix_epochs' else 1)
                minimum = 0 if key == 'adaint_fix_epochs' else 1
                if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
                    raise ValueError('{} must be an integer >= {}'.format(key, minimum))
            if cfg.evaluation_interval_epochs % cfg.checkpoint_interval_epochs:
                raise ValueError('evaluation_interval_epochs must be a positive multiple '
                                 'of checkpoint_interval_epochs')
            if cfg.train_epochs < cfg.evaluation_interval_epochs:
                raise ValueError('train_epochs must reach evaluation_interval_epochs')
            cfg.workflow = cfg.get('workflow') or [('train', 1)]
            if cfg.get('lr_config'):
                cfg.lr_config.by_epoch = False
            cfg.checkpoint_config.update(type='LUTEpochCheckpointHook', by_epoch=False,
                                         steps_per_epoch=steps)
            cfg.log_config.interval = steps * cfg.log_interval_epochs
            for hook in cfg.log_config.hooks:
                if hook['type'] in ('TextLoggerHook', 'LUTEpochTextLoggerHook'):
                    hook.update(type='LUTEpochTextLoggerHook', by_epoch=False,
                                steps_per_epoch=steps, total_epochs=cfg.train_epochs)
    if cfg.model.type in ENHANCER_TYPES:
        cfg.train_cfg.total_iters = cfg.total_iters
    return cfg


def public_training_config(cfg):
    """Export epoch parameters without duplicating internal iteration values."""
    result = Config(copy.deepcopy(cfg._cfg_dict))
    if result.get('log_interval_epochs') is None:
        return result
    for key in ('total_iters', 'iters_per_epoch', 'workflow'):
        result.pop(key, None)
    if result.get('train_cfg') is not None:
        for key in ('n_fix_iters', 'total_iters', 'tone_warmup_iters'):
            result.train_cfg.pop(key, None)
        if not result.train_cfg:
            result.pop('train_cfg', None)
    for key in ('checkpoint_config', 'log_config', 'evaluation'):
        if result.get(key):
            result[key].pop('interval', None)
            result[key].pop('by_epoch', None)
    if result.get('checkpoint_config'):
        result.checkpoint_config.pop('steps_per_epoch', None)
        if result.checkpoint_config.get('type') == 'LUTEpochCheckpointHook':
            result.checkpoint_config.pop('type')
    if result.get('lr_config'):
        result.lr_config.pop('by_epoch', None)
    for hook in result.get('log_config', {}).get('hooks', []):
        for key in ('by_epoch', 'steps_per_epoch', 'total_epochs', 'interval'):
            hook.pop(key, None)
    return result


def select_best_checkpoint(records, work_dir, steps_per_epoch=None, metric='PSNR', rule='greater'):
    """Read new epoch logs and existing iteration logs when resuming."""
    if rule not in ('greater', 'less'):
        raise ValueError("checkpoint rule must be 'greater' or 'less'")
    candidates = {}
    for entry in records:
        if entry.get('mode') != 'val' or metric not in entry:
            continue
        if 'iter' in entry:
            step = entry['iter']
            progress = step / steps_per_epoch if steps_per_epoch else step
            filename = 'iter_{}.pth'.format(step)
        elif steps_per_epoch and 'epoch' in entry:
            progress = entry['epoch']
            filename = 'epoch_{}.pth'.format(entry['epoch'])
        else:
            continue
        # A resumed run can overwrite the same epoch/iteration file. Retain
        # its latest validation, otherwise an older score can describe new weights.
        if not math.isfinite(entry[metric]):
            candidates.pop(filename, None)
            continue
        candidates[filename] = (entry[metric], progress, os.path.join(work_dir, filename))
    if not candidates:
        raise ValueError('no finite validation {} recorded; training must reach its evaluation period'.format(metric))
    return max(candidates.values(), key=lambda item: item[0] if rule == 'greater' else -item[0])


def inference_config(cfg):
    result = Config(copy.deepcopy(cfg._cfg_dict))
    result.validation_test_cfg = copy.deepcopy(
        cfg.get('validation_test_cfg', cfg.get('test_cfg', {})))
    result.model.pop('kd_loss', None)
    result.model.pop('gen', None)
    result.train_cfg = None
    result.test_cfg = dict(metrics=None, crop_border=0)
    for key in ('load_from', 'resume_from'):
        result[key] = None
    return result


def save_generation_configs(cfg):
    """Write Python configs even when the source Config came from YAML.

    MMCV 1.3 cannot dump YAML inputs to .py, and its pretty_text depends on an
    old YAPF API. Use standard Python serialization without changing packages.
    """
    def plain(value):
        if isinstance(value, dict):
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, list):
            return [plain(item) for item in value]
        if isinstance(value, tuple):
            return tuple(plain(item) for item in value)
        return value

    os.makedirs(cfg.work_dir, exist_ok=True)
    for name, contents in (('resolved_config.py', public_training_config(cfg)),
                           ('inference_config.py', public_training_config(inference_config(cfg)))):
        with open(os.path.join(cfg.work_dir, name), 'w') as handle:
            handle.write('\n\n'.join(
                '{} = {}'.format(key, pformat(plain(value), width=100))
                for key, value in contents._cfg_dict.items()) + '\n')


def load_enhancer_checkpoint(model, filename):
    """Discard frozen teacher weights, but reject every enhancer shape/key mismatch."""
    checkpoint = torch.load(filename, map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint)
    state = {key[7:] if key.startswith('module.') else key: value
             for key, value in state.items()}
    state = {key: value for key, value in state.items()
             if not key.startswith('kd_loss.')}
    current = model.state_dict()
    expected = {key: value for key, value in current.items()
                if not key.startswith('kd_loss.')}
    missing = sorted(set(expected)-set(state))
    unexpected = sorted(set(state)-set(expected))
    shapes = sorted(key for key in set(state) & set(expected)
                    if state[key].shape != expected[key].shape)
    if missing or unexpected or shapes:
        raise RuntimeError('incompatible AiLUT checkpoint: missing={}, unexpected={}, '
                           'shape_mismatches={}'.format(missing, unexpected, shapes))
    # A training model may already contain its frozen, separately loaded teacher.
    # Validate every enhancer key, then preserve that teacher during preflight.
    current.update(state)
    model.load_state_dict(current, strict=True)
    return checkpoint


def initialize_tone_from_baseline(model, filename):
    """Copy ONLY an exactly matching dual baseline; start a fresh tone schedule.

    All old keys/shapes must match. Only the newly initialized spatial_tone
    namespace may be absent. Resuming a tone checkpoint uses strict loading.
    Also supports the supervision-only ablation with tone disabled.
    """
    if type(model).__name__ != 'AiLUTDual':
        raise ValueError('tone_init_from requires AiLUTDual')
    checkpoint = torch.load(filename, map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint)
    state = {key[7:] if key.startswith('module.') else key: value for key, value in state.items()}
    current = model.state_dict()
    expected = {key: value for key, value in current.items() if not key.startswith('spatial_tone.')}
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    shapes = sorted(key for key in set(state) & set(expected) if state[key].shape != expected[key].shape)
    if missing or unexpected or shapes:
        raise RuntimeError('incompatible tone baseline: missing={}, unexpected={}, shape_mismatches={}'.format(
            missing, unexpected, shapes))
    current.update({key: value for key, value in state.items() if key != 'cnt_iters'})
    current['cnt_iters'] = torch.zeros_like(current['cnt_iters'])
    model.load_state_dict(current, strict=True)
    return checkpoint
