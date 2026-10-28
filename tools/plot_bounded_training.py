"""Refresh bounded-curve training summaries and figures without touching logs.

Example (safe to rerun while training):
    python tools/plot_bounded_training.py --suite work_dirs/bc_20261028_suite

Use --historical to display the old checkpoint PSNR as a labelled reference.
It is NOT a matched from-scratch control and is never used to select a winner.
Only the standard library, NumPy and Matplotlib are required; no model is loaded.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import tempfile


HISTORICAL_PSNR = {'outdoor': 25.858, 'indoor': 29.917}
COLORS = {'legacy': '#6b7280', 'bounded': '#2563eb', 'detail': '#059669',
          'no_result': '#dc2626', 'no_fine': '#9333ea'}
LABELS = {'legacy': 'Legacy, from scratch', 'bounded': 'Bounded structure',
          'detail': 'Bounded + detail losses', 'no_result': 'Without result condition',
          'no_fine': 'Without fine scale'}


def finite(value):
    """Booleans, null and non-finite log entries are not numeric metrics."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def read_json(path, warnings):
    try:
        with path.open() as handle:
            return json.load(handle)
    except (OSError, ValueError) as error:
        warnings.append({'path': str(path), 'reason': str(error)})
        return None


def load_logs(directory):
    """Last valid occurrence wins, following the trainer's file/line ordering."""
    latest, warnings, paths, overwritten = {}, [], [], 0
    for path in sorted(directory.glob('*.log.json')):
        paths.append(str(path.resolve()))
        try:
            handle = path.open()
        except OSError as error:
            warnings.append({'path': str(path), 'reason': str(error)})
            continue
        with handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError as error:
                    # A writer may be in the middle of its final JSON line.
                    warnings.append({'path': str(path), 'line': line_number,
                                     'reason': 'Skipped unreadable JSON: ' + str(error)})
                    continue
                if not isinstance(row, dict) or row.get('mode') not in ('train', 'val'):
                    continue
                epoch = finite(row.get('epoch'))
                if epoch is None or epoch < 1 or not epoch.is_integer():
                    warnings.append({'path': str(path), 'line': line_number,
                                     'reason': 'Skipped non-positive/non-integer epoch'})
                    continue
                key = (row['mode'], int(epoch))
                overwritten += key in latest
                latest[key] = dict(row, epoch=int(epoch), _log_path=str(path.resolve()),
                                  _log_line=line_number)
    records = {mode: [latest[key] for key in sorted(latest)
                      if key[0] == mode] for mode in ('train', 'val')}
    return records, paths, warnings, overwritten


def metric_rows(records, name):
    return [(row['epoch'], finite(row.get(name))) for row in records
            if finite(row.get(name)) is not None]


def train_loss(row):
    """Report the logged optimization objective; never infer a train PSNR."""
    value = finite(row.get('loss'))
    if value is not None:
        return value, 'logged loss'
    terms = [finite(value) for key, value in row.items() if key.startswith('loss_')]
    terms = [value for value in terms if value is not None]
    return (sum(terms), 'sum of logged loss_* terms') if terms else (None, None)


def resolve_work_dir(run, suite):
    path = Path(run['work_dir']).expanduser()
    if not path.is_absolute():
        # Generated manifests normally store absolute paths. Relative work_dirs
        # are repository-relative; suite-local copied fixtures are also supported.
        project = Path(__file__).resolve().parents[1]
        candidates = [project / path, suite / path, suite.parent / path]
        path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    return path.resolve()


def summarize_run(run, manifest, suite):
    work_dir = resolve_work_dir(run, suite)
    generation = work_dir if work_dir.name == 'gen_0' else work_dir / 'gen_0'
    records, log_paths, warnings, overwritten = load_logs(generation)
    expected = int(manifest.get('epochs', 100))
    train, validation = records['train'], records['val']
    candidates = [row for row in validation if finite(row.get('PSNR')) is not None]
    # Same-metric ties select the earlier epoch, matching chronological max().
    best = max(candidates, key=lambda row: (float(row['PSNR']), -row['epoch'])) if candidates else None
    resources_path = generation / 'runtime_resources.json'
    resources = read_json(resources_path, warnings) if resources_path.is_file() else None
    if resources is not None and not isinstance(resources, dict):
        warnings.append({'path': str(resources_path), 'reason': 'Runtime resources is not a mapping'})
        resources = None
    last_train = max((row['epoch'] for row in train), default=0)
    last_val = max((row['epoch'] for row in validation), default=0)
    reached_budget = last_train >= expected and last_val >= expected
    completed = reached_budget and resources is not None
    if completed:
        status = 'training_completed'
    elif reached_budget:
        status = 'budget_logged_awaiting_runtime_resources'
    elif train or validation:
        status = 'in_progress_or_interrupted'
    else:
        status = 'no_training_logs'
    epoch_times = [value for _, value in metric_rows(train, 'epoch_time')]
    memories = [value for _, value in metric_rows(train, 'memory')]
    foreground = [row for row in validation
                  if finite(row.get('deltaE_P3_foreground')) is not None]
    minimum_fg = min(foreground, key=lambda row: float(row['deltaE_P3_foreground'])) if foreground else None
    last = validation[-1] if validation else {}
    required_epochs = set(range(1, expected + 1))
    item = dict(
        scene=manifest.get('scene'), variant=run.get('variant', run.get('name')),
        name=run.get('name'), initialization=run.get('initialization', manifest.get('initialization')),
        work_dir=str(work_dir), generation_dir=str(generation),
        manifest_epochs=expected, seed=manifest.get('seed'), learning_rate=manifest.get('learning_rate'),
        status=status, training_completed=completed,
        train_budget_logged=last_train >= expected, validation_budget_logged=last_val >= expected,
        last_train_epoch=last_train, last_val_epoch=last_val,
        train_epochs_recorded=len(train), val_epochs_recorded=len(validation),
        all_train_epochs_recorded=required_epochs.issubset({row['epoch'] for row in train}),
        all_val_epochs_recorded=required_epochs.issubset({row['epoch'] for row in validation}),
        best_val_epoch=best['epoch'] if best else None,
        best_val_PSNR=finite(best.get('PSNR')) if best else None,
        foreground_deltaE_at_best_PSNR=finite(best.get('deltaE_P3_foreground')) if best else None,
        raw_high_clip_at_best_PSNR=finite(best.get('raw_high_clip')) if best else None,
        raw_high_clip_percent_at_best_PSNR=(100. * finite(best.get('raw_high_clip'))
            if best and finite(best.get('raw_high_clip')) is not None else None),
        best_epoch_checkpoint=str(generation / ('epoch_{}.pth'.format(best['epoch']))) if best else None,
        minimum_foreground_deltaE=finite(minimum_fg.get('deltaE_P3_foreground')) if minimum_fg else None,
        minimum_foreground_deltaE_epoch=minimum_fg['epoch'] if minimum_fg else None,
        last_val_PSNR=finite(last.get('PSNR')),
        last_val_foreground_deltaE=finite(last.get('deltaE_P3_foreground')),
        last_val_raw_high_clip=finite(last.get('raw_high_clip')),
        last_train_loss=train_loss(train[-1])[0] if train else None,
        train_loss_sources=sorted({train_loss(row)[1] for row in train if train_loss(row)[1]}),
        logged_peak_memory_mib=max(memories) if memories else None,
        mean_logged_train_epoch_seconds=(sum(epoch_times) / len(epoch_times)) if epoch_times else None,
        total_logged_train_epoch_seconds=sum(epoch_times) if epoch_times else None,
        runtime_resources=resources, runtime_resources_path=str(resources_path) if resources else None,
        parameters=resources.get('parameters') if resources else None,
        trainable_parameters=resources.get('trainable_parameters') if resources else None,
        elapsed_training_seconds=resources.get('elapsed_training_seconds') if resources else None,
        peak_allocated_bytes=resources.get('peak_allocated_bytes') if resources else None,
        peak_reserved_bytes=resources.get('peak_reserved_bytes') if resources else None,
        gpu=resources.get('gpu') if resources else None,
        code_revision=resources.get('code_revision') if resources else manifest.get('code_revision'),
        source_logs=log_paths, overwritten_mode_epoch_records=overwritten, warnings=warnings)
    return item, records


def atomic_json(path, value):
    with tempfile.NamedTemporaryFile(mode='w', dir=str(path.parent), suffix='.tmp',
                                     delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
    temporary.replace(path)


def csv_rows(path, items):
    names = [name for name in items[0] if name not in
             ('runtime_resources', 'source_logs', 'warnings', 'train_loss_sources')] if items else [
                 'scene', 'variant', 'name', 'status']
    with tempfile.NamedTemporaryFile(mode='w', dir=str(path.parent), suffix='.tmp',
                                     newline='', delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(items)
    temporary.replace(path)


def save_figure(figure, stem):
    outputs = []
    for extension in ('png', 'pdf'):
        path = stem.with_suffix('.' + extension)
        with tempfile.NamedTemporaryFile(dir=str(path.parent), suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
        figure.savefig(str(temporary), format=extension, dpi=180, bbox_inches='tight')
        temporary.replace(path)
        outputs.append(str(path.resolve()))
    return outputs


def plot_scene(scene, items, series, output, historical=False):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42})
    palette = plt.get_cmap('tab10')
    figures = []
    note = ('Solid curves: matched from-scratch runs. Historical checkpoint: different training/budget.'
            if historical and scene in HISTORICAL_PSNR else
            'All curves: from-scratch runs. Best markers use validation PSNR; no train PSNR is inferred.')
    specifications = [('PSNR', 'Validation PSNR (dB)', 1.),
                      ('deltaE_P3_foreground', 'Validation foreground P3 Delta E', 1.),
                      ('raw_high_clip', 'Validation raw upper-bound exceedance (%)', 100.)]
    figure, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for index, item in enumerate(items):
        records = series[item['name']]['val']
        variant = item['variant']
        label = LABELS.get(variant, variant)
        color = COLORS.get(variant, palette(index % 10))
        for axis, (key, title, scale) in zip(axes, specifications):
            points = metric_rows(records, key)
            if points:
                values = np.asarray(points, dtype=float)
                axis.plot(values[:, 0], values[:, 1] * scale, color=color,
                          linewidth=1.7, label=label)
            best = next((row for row in records if row['epoch'] == item['best_val_epoch']), None)
            value = finite(best.get(key)) if best else None
            if value is not None:
                axis.scatter([best['epoch']], [value * scale], marker='*', s=85,
                             color=color, edgecolors='white', linewidths=.5, zorder=4)
                axis.annotate('e{}'.format(best['epoch']), (best['epoch'], value * scale),
                              xytext=(4, 5 + 2 * (index % 2)), textcoords='offset points',
                              fontsize=8, color=color)
    for axis, (_, title, _) in zip(axes, specifications):
        axis.set_title(title)
        axis.set_xlabel('Epoch')
        axis.grid(alpha=.2)
        if not axis.lines:
            axis.text(.5, .5, 'No validation records yet', ha='center', va='center',
                      transform=axis.transAxes)
    if historical and scene in HISTORICAL_PSNR:
        value = HISTORICAL_PSNR[scene]
        axes[0].axhline(value, color='#111827', linestyle='--', linewidth=1.2,
                       label='Historical checkpoint: {:.3f} dB (different budget)'.format(value))
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc='lower center', ncol=min(3, len(handles)),
                      bbox_to_anchor=(.5, 0), frameon=False, fontsize=9)
    figure.suptitle('{} - validation history'.format(scene.capitalize()), fontsize=14)
    figure.text(.5, .11, note, ha='center', fontsize=8)
    figure.tight_layout(rect=(0, .15, 1, .94))
    figures.extend(save_figure(figure, output / (scene + '_validation')))
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for index, item in enumerate(items):
        records = series[item['name']]['train']
        color = COLORS.get(item['variant'], palette(index % 10))
        label = LABELS.get(item['variant'], item['variant'])
        points = [(row['epoch'], train_loss(row)[0]) for row in records
                  if train_loss(row)[0] is not None]
        for axis, values in zip(axes, (points, metric_rows(records, 'memory'),
                                       metric_rows(records, 'epoch_time'))):
            if values:
                values = np.asarray(values, dtype=float)
                axis.plot(values[:, 0], values[:, 1], color=color, linewidth=1.7, label=label)
    titles = ('Training optimization loss (logged)', 'Logged GPU peak memory (MiB)',
              'Logged train epoch time (s)')
    for axis, title in zip(axes, titles):
        axis.set_title(title)
        axis.set_xlabel('Epoch')
        axis.grid(alpha=.2)
        if not axis.lines:
            axis.text(.5, .5, 'No training values yet', ha='center', va='center',
                      transform=axis.transAxes)
    if axes[0].lines and all((line.get_ydata() > 0).all() for line in axes[0].lines):
        axes[0].set_yscale('log')
        axes[0].set_ylabel('Logarithmic scale')
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc='lower center', ncol=min(3, len(handles)),
                      bbox_to_anchor=(.5, 0), frameon=False, fontsize=9)
    figure.suptitle('{} - training and resources'.format(scene.capitalize()), fontsize=14)
    figure.text(.5, .11,
                'Training loss is an optimization objective, not train PSNR. Epoch time excludes unlogged overhead.',
                ha='center', fontsize=8)
    figure.tight_layout(rect=(0, .15, 1, .94))
    figures.extend(save_figure(figure, output / (scene + '_training')))
    plt.close(figure)
    return figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='default: SUITE/training_curves')
    parser.add_argument('--historical', action='store_true',
                        help='show clearly labelled old PSNR reference, not a matched control')
    parser.add_argument('--no-plots', action='store_true',
                        help='refresh JSON/CSV only; useful before installing NumPy/Matplotlib')
    args = parser.parse_args()
    suite = args.suite.expanduser().resolve()
    output = args.output.expanduser().resolve() if args.output else suite / 'training_curves'
    paths = sorted(suite.glob('manifest_*.json'))
    if not paths:
        parser.error('no manifest_*.json found under ' + str(suite))
    output.mkdir(parents=True, exist_ok=True)
    items, series, warnings, provenance = [], {}, [], []
    for path in paths:
        manifest = read_json(path, warnings)
        if not isinstance(manifest, dict) or manifest.get('scene') not in ('indoor', 'outdoor'):
            warnings.append({'path': str(path), 'reason': 'Not a valid indoor/outdoor manifest'})
            continue
        provenance.append(dict(path=str(path.resolve()), scene=manifest['scene'],
                               initialization=manifest.get('initialization'),
                               code_revision=manifest.get('code_revision')))
        for run in manifest.get('runs', []):
            if not isinstance(run, dict) or not run.get('work_dir') or not run.get('name'):
                warnings.append({'path': str(path), 'reason': 'Skipped malformed run entry'})
                continue
            if run['name'] in series:
                raise ValueError('duplicate experiment name across manifests: ' + run['name'])
            item, records = summarize_run(run, manifest, suite)
            item['manifest'] = str(path.resolve())
            items.append(item)
            series[item['name']] = records
    artifacts = []
    if not args.no_plots:
        for scene in ('outdoor', 'indoor'):
            scene_items = [item for item in items if item['scene'] == scene]
            if scene_items:
                artifacts.extend(plot_scene(scene, scene_items, series, output, args.historical))
    summary = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(), suite=str(suite),
        manifests=provenance, experiments=items,
        experiments_count=len(items), completed_count=sum(item['training_completed'] for item in items),
        all_training_completed=bool(items) and all(item['training_completed'] for item in items),
        historical_PSNR_reference=HISTORICAL_PSNR if args.historical else None,
        historical_reference_note='Historical checkpoints are not matched from-scratch controls; '
                                  'their original training/budget differs. They are never used for selection.',
        metric_note='Best epoch maximizes finite validation PSNR; other best-row metrics refer to '
                    'that same epoch. raw_high_clip is a fraction, its percent field is 100x. '
                    'Training losses are logged objectives; no training PSNR is inferred. '
                    'Runtime elapsed time belongs to the recorded training process and may not '
                    'include earlier processes after a resume.',
        ordering_note='Sorted *.log.json files, then source line order; the last readable '
                      '(mode, epoch) record replaces earlier resumed records.',
        figures=artifacts, warnings=warnings)
    atomic_json(output / 'summary.json', summary)
    csv_rows(output / 'summary.csv', items)
    print(json.dumps(dict(output=str(output), experiments=len(items),
                          completed=summary['completed_count'], figures=artifacts,
                          warnings=len(warnings) + sum(len(item['warnings']) for item in items)), indent=2))


if __name__ == '__main__':
    main()
