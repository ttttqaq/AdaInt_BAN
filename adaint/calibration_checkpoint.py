"""Select trained scratch checkpoints without an external reference or fallback."""
import json
from pathlib import Path
import shutil

from .experiment_utils import select_best_checkpoint


def validate_scratch_start(cfg):
    if cfg.get('load_from'):
        raise ValueError('scratch training does not accept load_from; use --resume-from for interrupted runs')
    work_dir = Path(cfg.work_dir)
    if not cfg.get('resume_from') and any(
            list(work_dir.glob(pattern)) for pattern in ('*.pth', '*.log.json')):
        raise ValueError('existing training outputs: choose a new --work-dir or resume the same run')


def save_calibration_selection(cfg, records, steps_per_epoch):
    selected = dict(training_initialization='scratch')
    for metric, rule, name in [('PSNR', 'greater', 'best_psnr.pth'),
                               ('deltaE_P3_foreground', 'less', 'best_foreground.pth')]:
        score, progress, source = select_best_checkpoint(
            records, cfg.work_dir, steps_per_epoch, metric, rule)
        shutil.copy(source, str(Path(cfg.work_dir) / name))
        selected[name] = dict(score=score, progress=progress, source=source)
    metric = cfg.get('best_checkpoint_metric', 'PSNR')
    rule = cfg.get('best_checkpoint_rule', 'greater')
    score, progress, source = select_best_checkpoint(
        records, cfg.work_dir, steps_per_epoch, metric, rule)
    shutil.copy(source, str(Path(cfg.work_dir) / 'best.pth'))
    selected.update(metric=metric, rule=rule, score=score, progress=progress, source=source)
    (Path(cfg.work_dir) / 'best_selection.json').write_text(json.dumps(selected, indent=2))
    return selected
