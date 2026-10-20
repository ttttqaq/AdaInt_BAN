"""Joint foreground/background checkpoint policy with an explicit baseline fallback."""
import json
import math
from pathlib import Path
import shutil

import torch

from .calibration_reference import validate_reference_cache
from .experiment_utils import initialize_bilateral_from_basis, select_best_checkpoint


def initialize_calibration_training(model, cfg):
    manifest, metrics = validate_reference_cache(cfg)
    initialize_bilateral_from_basis(model, cfg.calibration_init_from)
    if model.n_fix_iters or not all(p.requires_grad for p in model.parameters()):
        raise ValueError('all student parameters must train, including both AdaInt/local heads')
    fallback = Path(cfg.work_dir) / 'initial_basis.pth'
    if fallback.exists():
        saved = torch.load(str(fallback), map_location='cpu')
        if saved.get('meta', {}).get('reference_sha256') != manifest['checkpoint_sha256']:
            raise ValueError('work directory belongs to another reference checkpoint')
        expected, existing = model.state_dict(), saved.get('state_dict', {})
        if set(expected) != set(existing) or any(expected[k].shape != existing[k].shape for k in expected):
            raise ValueError('fallback architecture changed; use a new work directory')
        for key in expected:
            if not key.startswith('foreground_calibration.') and not torch.equal(expected[key].cpu(), existing[key]):
                raise ValueError('baseline fallback does not match current basis source')
        if any(existing['foreground_calibration.output.' + name].count_nonzero() for name in ('weight', 'bias')):
            raise ValueError('fallback is not the zero-calibration baseline')
    else:
        torch.save(dict(state_dict={key:value.cpu() for key,value in model.state_dict().items()},
            meta=dict(epoch=0, iter=0, reference_sha256=manifest['checkpoint_sha256'],
                      baseline_fallback=True)), str(fallback))
    # Resume/load happens afterwards through strict full-student checkpoint loading.
    return metrics


def choose_guarded_checkpoint(records, work_dir, baseline, tolerances, steps_per_epoch):
    limits = {
        'PSNR': ('greater', baseline['PSNR'] - tolerances['psnr_drop']),
        'deltaL_P3_background': ('less', baseline['deltaL_P3_background'] + tolerances['background_L_increase']),
        'deltaL_P3_highlights': ('less', baseline['deltaL_P3_highlights'] + tolerances['highlight_L_increase']),
        'deltaL_P3_foreground': ('less', baseline['deltaL_P3_foreground'] + tolerances['foreground_L_increase'])}
    if any(not math.isfinite(value) or value < 0 for value in tolerances.values()):
        raise ValueError('guard tolerances must be finite and nonnegative')
    latest = {}
    for record in records:
        if record.get('mode') != 'val':
            continue
        if 'iter' in record:
            filename, progress = 'iter_{}.pth'.format(record['iter']), record['iter'] / steps_per_epoch
        elif 'epoch' in record:
            filename, progress = 'epoch_{}.pth'.format(record['epoch']), record['epoch']
        else:
            continue
        latest[filename] = (record, progress)
    eligible = []
    for filename, (record, progress) in latest.items():
        score = record.get('deltaE_P3_foreground', float('nan'))
        if not math.isfinite(score) or score >= baseline['deltaE_P3_foreground']:
            continue
        if any(not math.isfinite(record.get(key, float('nan'))) or
               (record[key] < limit if rule == 'greater' else record[key] > limit)
               for key,(rule,limit) in limits.items()):
            continue
        if not (Path(work_dir) / filename).is_file():
            continue
        eligible.append((score, -record['PSNR'], progress, str(Path(work_dir) / filename)))
    if eligible:
        selected = min(eligible)
        return dict(source=selected[3], progress=selected[2], score=selected[0],
                    using_baseline_fallback=False, eligible_progress=[x[2] for x in eligible], limits=limits)
    return dict(source=str(Path(work_dir) / 'initial_basis.pth'), progress=0,
                score=baseline['deltaE_P3_foreground'], using_baseline_fallback=True,
                eligible_progress=[], limits=limits)


def save_calibration_selection(cfg, records, steps_per_epoch):
    manifest, baseline = validate_reference_cache(cfg)
    selected = choose_guarded_checkpoint(records, cfg.work_dir, baseline,
                                         cfg.calibration_selection_tolerances, steps_per_epoch)
    for metric, rule, name in [('PSNR','greater','best_psnr.pth'),
                                ('deltaE_P3_foreground','less','best_foreground.pth')]:
        score, progress, source = select_best_checkpoint(records, cfg.work_dir, steps_per_epoch, metric, rule)
        shutil.copy(source, str(Path(cfg.work_dir) / name))
        selected[name] = dict(score=score, progress=progress, source=source)
    shutil.copy(selected['source'], str(Path(cfg.work_dir) / 'best.pth'))
    selected.update(metric='deltaE_P3_foreground', baseline=baseline,
        reference_sha256=manifest['checkpoint_sha256'], tolerances=dict(cfg.calibration_selection_tolerances))
    (Path(cfg.work_dir) / 'best_selection.json').write_text(json.dumps(selected, indent=2))
    return selected
