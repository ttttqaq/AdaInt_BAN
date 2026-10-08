"""Epoch-facing logging/checkpoints over the existing iteration-based runner."""
import logging
import re
from collections import OrderedDict

import torch
from mmcv.runner import HOOKS, LogBuffer, master_only
from mmcv.runner.dist_utils import allreduce_params
from mmcv.runner.hooks.checkpoint import CheckpointHook
from mmcv.runner.hooks.logger.text import TextLoggerHook


@HOOKS.register_module()
class LUTEpochTextLoggerHook(TextLoggerHook):
    def __init__(self, steps_per_epoch, total_epochs, **kwargs):
        kwargs['by_epoch'] = False
        super().__init__(**kwargs)
        self.steps_per_epoch = int(steps_per_epoch)
        self.total_epochs = int(total_epochs)
        self._pending = LogBuffer()
        self._seen = {}

    def _collect_train_history(self, runner):
        # Validation clears the runner buffer. Keep unprinted training values
        # across evaluations, and never copy a previously collected value twice.
        for key, values in runner.log_buffer.val_history.items():
            previous, start = self._seen.get(key, (None, 0))
            if previous is not values:
                start = 0
            self._pending.val_history.setdefault(key, []).extend(values[start:])
            self._pending.n_history.setdefault(key, []).extend(
                runner.log_buffer.n_history[key][start:])
            self._seen[key] = (values, len(values))

    def _write(self, runner, mode, values):
        epoch = (runner.iter + self.steps_per_epoch) // self.steps_per_epoch
        record = OrderedDict(mode=mode, epoch=epoch)
        rates = runner.current_lr()
        record['lr'] = rates[0] if isinstance(rates, list) else {
            name: rate[0] for name, rate in rates.items()}
        values = dict(values)
        values.pop('eval_iter_num', None)
        for name in ('time', 'data_time'):
            if name in values:
                record['epoch_' + name] = values.pop(name) * self.steps_per_epoch
        if mode == 'train' and torch.cuda.is_available():
            record['memory'] = self._get_max_memory(runner)
        record.update(values)
        prefix = 'Epoch{} [{}/{}]'.format(
            '(val)' if mode == 'val' else '', epoch, self.total_epochs)
        fields = []
        for name, value in record.items():
            if name in ('mode', 'epoch'):
                continue
            if isinstance(value, float):
                value = '{:.3e}'.format(value) if name == 'lr' else '{:.4f}'.format(value)
            fields.append('{}: {}'.format(name, value))
        runner.logger.info(prefix + '\t' + ', '.join(fields))
        self._dump_log(record, runner)
        return record

    def _write_train(self, runner):
        self._collect_train_history(runner)
        if self._pending.val_history:
            self._pending.average()
            self._write(runner, 'train', self._pending.output)
            self._pending.clear()

    def log_before_evaluation(self, runner):
        self._collect_train_history(runner)
        if self.every_n_iters(runner, self.interval) or self.is_last_iter(runner):
            self._write_train(runner)

    def after_train_iter(self, runner):
        if runner.log_buffer.ready and self.get_mode(runner) == 'val':
            self._write(runner, 'val', runner.log_buffer.output)
        elif self.every_n_iters(runner, self.interval) or self.is_last_iter(runner):
            self._write_train(runner)
        if self.reset_flag:
            runner.log_buffer.clear_output()


@HOOKS.register_module()
class LUTEpochCheckpointHook(CheckpointHook):
    def __init__(self, steps_per_epoch, **kwargs):
        kwargs['by_epoch'] = False
        super().__init__(**kwargs)
        self.steps_per_epoch = int(steps_per_epoch)

    def after_train_iter(self, runner):
        if self.every_n_iters(runner, self.interval) or (
                self.save_last and self.is_last_iter(runner)):
            epoch = (runner.iter + 1) // self.steps_per_epoch
            runner.logger.info('Saving checkpoint at epoch %s', epoch)
            if self.sync_buffer:
                allreduce_params(runner.model.buffers())
            self._save_checkpoint(runner)

    @master_only
    def _save_checkpoint(self, runner):
        epoch = (runner.iter + 1) // self.steps_per_epoch
        filename = 'epoch_{}.pth'.format(epoch)
        arguments = dict(self.args, filename_tmpl=filename)
        runner.save_checkpoint(self.out_dir, save_optimizer=self.save_optimizer, **arguments)
        if runner.meta is not None:
            runner.meta.setdefault('hook_msgs', {})['last_ckpt'] = self.file_client.join_path(
                self.out_dir, filename)
        if self.max_keep_ckpts > 0:
            interval_epochs = self.interval // self.steps_per_epoch
            for old_epoch in range(epoch - self.max_keep_ckpts * interval_epochs,
                                   0, -interval_epochs):
                path = self.file_client.join_path(self.out_dir, 'epoch_{}.pth'.format(old_epoch))
                if not self.file_client.isfile(path):
                    break
                self.file_client.remove(path)


class _EpochDisplayFilter(logging.Filter):
    def __init__(self, steps_per_epoch):
        super().__init__()
        self.steps_per_epoch = steps_per_epoch

    def filter(self, record):
        message = record.getMessage()
        message = re.sub(r'max: (\d+) iters', lambda match: 'max: {:g} epochs'.format(
            int(match.group(1)) / self.steps_per_epoch), message)
        message = re.sub(r'(Fix AdaInt for|Unfix AdaInt after) (\d+) iters',
                         lambda match: '{} {:g} epochs'.format(
                             match.group(1), int(match.group(2)) / self.steps_per_epoch), message)
        if message.startswith('resumed from epoch:'):
            message = 'Loaded training checkpoint'
        if message.startswith('Align resumed iteration'):
            count = int(message.rsplit(' ', 1)[-1])
            message = 'Restored training progress: epoch {:g}'.format(count / self.steps_per_epoch)
        record.msg, record.args = message, ()
        return True


def configure_epoch_display(logger, cfg):
    for existing in list(logger.filters):
        if isinstance(existing, _EpochDisplayFilter):
            logger.removeFilter(existing)
    if cfg.get('log_interval_epochs') is not None:
        logger.addFilter(_EpochDisplayFilter(cfg.iters_per_epoch))
