import json
import logging
from types import SimpleNamespace

import pytest
from mmcv import Config
from mmcv.runner import LogBuffer

from adaint.config_utils import load_config
from adaint.epoch_hooks import LUTEpochTextLoggerHook, LUTEpochCheckpointHook
from adaint.experiment_utils import (
    public_training_config, resolve_training_schedule, save_generation_configs,
    select_best_checkpoint)


def test_epoch_only_config_resolves_runtime_and_exports_without_iteration_knobs(tmp_path):
    cfg = load_config('adaint/configs/ppr10k.yaml')
    assert 'total_iters' not in cfg and 'train_cfg' not in cfg
    assert 'interval' not in cfg.log_config and 'interval' not in cfg.evaluation
    cfg.gpus = 1
    resolve_training_schedule(cfg, 8875)
    assert cfg.total_iters == 10500 and cfg.train_cfg.n_fix_iters == 175
    assert cfg.checkpoint_config.interval == 35 and cfg.evaluation.interval == 525
    assert cfg.log_config.interval == 35
    assert cfg.checkpoint_config.type == 'LUTEpochCheckpointHook'
    cfg.work_dir = str(tmp_path)
    save_generation_configs(cfg)
    saved = load_config(str(tmp_path/'resolved_config.py'))
    assert 'total_iters' not in saved and 'iters_per_epoch' not in saved
    assert 'interval' not in saved.log_config and 'interval' not in saved.checkpoint_config
    resolve_training_schedule(saved, 8875)
    assert saved.total_iters == 10500 and saved.log_config.interval == 35


def test_epoch_period_validation_and_batch_change_recompute_all_internal_steps():
    cfg = load_config('adaint/configs/ppr10k.yaml')
    cfg.gpus = 1
    cfg.data.train_dataloader.samples_per_gpu = 32
    cfg.log_interval_epochs = 2
    resolve_training_schedule(cfg, 8875)
    assert cfg.iters_per_epoch == 278
    assert cfg.log_config.interval == 556
    assert cfg.total_iters == 83400 and cfg.evaluation.interval == 4170
    cfg.evaluation_interval_epochs = 3
    cfg.checkpoint_interval_epochs = 2
    with pytest.raises(ValueError, match='evaluation_interval_epochs'):
        resolve_training_schedule(cfg, 8875)


def fake_runner(tmp_path, hook):
    runner = SimpleNamespace(iter=0, epoch=999, max_iters=12, _max_iters=12, mode='train', rank=0,
                             world_size=1, work_dir=str(tmp_path), timestamp='unit',
                             logger=logging.getLogger('epoch_unit'), log_buffer=LogBuffer(),
                             hooks=[hook], meta=None, current_lr=lambda: [1e-4])
    hook._get_max_memory = lambda runner: 42
    hook.before_run(runner)
    return runner


def test_training_and_validation_print_at_epoch_boundaries_without_losing_losses(tmp_path):
    hook = LUTEpochTextLoggerHook(steps_per_epoch=2, total_epochs=6, interval=2)
    runner = fake_runner(tmp_path, hook)
    runner.log_buffer.update(dict(loss=2., time=.1), 1)
    runner.log_buffer.update(dict(loss=4., time=.1), 1)
    runner.iter = 1
    hook.log_before_evaluation(runner)
    runner.log_buffer.clear()
    runner.log_buffer.output['PSNR'] = 20.
    runner.log_buffer.ready = True
    hook.after_train_iter(runner)
    rows = [json.loads(line) for line in (tmp_path/'unit.log.json').read_text().splitlines()]
    assert [row['mode'] for row in rows] == ['train', 'val']
    assert [row['epoch'] for row in rows] == [1, 1]
    assert rows[0]['loss'] == 3. and rows[1]['PSNR'] == 20.
    assert all('iter' not in row for row in rows)


def test_logging_period_accumulates_training_across_more_frequent_validation(tmp_path):
    hook = LUTEpochTextLoggerHook(steps_per_epoch=2, total_epochs=6, interval=4)
    runner = fake_runner(tmp_path, hook)
    for epoch, loss in ((1, 2.), (2, 4.)):
        runner.log_buffer.update(dict(loss=loss, time=.1), 2)
        runner.iter = epoch * 2 - 1
        hook.log_before_evaluation(runner)
        runner.log_buffer.clear()
        runner.log_buffer.output['PSNR'] = 20.
        runner.log_buffer.ready = True
        hook.after_train_iter(runner)
    rows = [json.loads(line) for line in (tmp_path/'unit.log.json').read_text().splitlines()]
    training = [row for row in rows if row['mode'] == 'train']
    assert len(training) == 1 and training[0]['epoch'] == 2 and training[0]['loss'] == 3.
    assert len([row for row in rows if row['mode'] == 'val']) == 2


def test_epoch_checkpoint_name_and_existing_log_compatibility(tmp_path):
    hook = LUTEpochCheckpointHook(steps_per_epoch=2, interval=2)
    calls = []
    runner = SimpleNamespace(iter=3, max_iters=12, work_dir=str(tmp_path), rank=0,
                             logger=logging.getLogger('checkpoint_unit'), meta={},
                             save_checkpoint=lambda out, **kwargs: calls.append(kwargs))
    hook.before_run(runner)
    hook.after_train_iter(runner)
    assert calls[0]['filename_tmpl'] == 'epoch_2.pth'
    assert runner.meta['hook_msgs']['last_ckpt'].endswith('epoch_2.pth')
    value, epoch, filename = select_best_checkpoint([
        dict(mode='val', epoch=2, PSNR=20.), dict(mode='val', iter=6, PSNR=21.)],
        str(tmp_path), steps_per_epoch=2)
    assert value == 21. and epoch == 3. and filename.endswith('iter_6.pth')


def test_resume_uses_latest_validation_for_overwritten_checkpoint(tmp_path):
    value, epoch, filename = select_best_checkpoint([
        dict(mode='val', epoch=1, PSNR=20.),
        dict(mode='val', epoch=2, PSNR=25.),
        dict(mode='val', epoch=2, PSNR=19.)], str(tmp_path), steps_per_epoch=2)
    assert value == 20. and epoch == 1 and filename.endswith('epoch_1.pth')
