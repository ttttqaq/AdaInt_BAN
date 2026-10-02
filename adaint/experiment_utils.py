"""Training budgets, inference configs and strict enhancer checkpoint loading."""
import copy
import math
import os
from pprint import pformat

import torch
from mmcv import Config
from mmcv.runner import HOOKS, Hook, get_dist_info

REFINEMENT_TYPES = ('AiLUTRefinementBase', 'AiLUTArchitectureV3', 'AiLUTGlobalLocalV3')


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
        cfg.total_iters = int(steps * cfg.train_epochs)
        cfg.train_cfg.n_fix_iters = int(steps * cfg.get('adaint_fix_epochs', 5))
        if cfg.get('checkpoint_config'):
            cfg.checkpoint_config.interval = int(steps * cfg.get('checkpoint_interval_epochs', 1))
        if cfg.get('evaluation'):
            cfg.evaluation.interval = int(steps * cfg.get('evaluation_interval_epochs', 1))
        cfg.iters_per_epoch = steps
    if cfg.model.type in REFINEMENT_TYPES:
        cfg.train_cfg.total_iters = cfg.total_iters
    return cfg


def inference_config(cfg):
    result = Config(copy.deepcopy(cfg._cfg_dict))
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
    for name, contents in (('resolved_config.py', cfg),
                           ('inference_config.py', inference_config(cfg))):
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
    model.load_state_dict(state, strict=True)
    return checkpoint
