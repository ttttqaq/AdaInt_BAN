"""Wait for the declared training suite, evaluate idle GPUs, then build reports.

This command owns only evaluation_state.json, eval/, reports/ and summary files.
It never changes the launcher's training state or signals training processes.
Run once per suite; an advisory file lock rejects concurrent orchestrators.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ('train', 'val', 'test')
TEST_ROOT = Path('/DB/data/yujietang/IQ-LUT/face_dataset/test_final')
MASK_ROOT = Path('/DB/data/yujietang/IQ-LUT/face_dataset/test_final_masks/masks')


def read_json(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else default


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def devices():
    output = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,memory.free',
        '--format=csv,noheader,nounits']).decode()
    return {row.split(',')[1].strip(): dict(index=int(row.split(',')[0]),
                free_mb=int(row.split(',')[2])) for row in output.strip().splitlines()}


def pid_alive(pid, command=None):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    proc = Path('/proc') / str(pid)
    try:
        if proc.exists():
            if (proc / 'stat').read_text().split(') ', 1)[1].startswith('Z '):
                return False
            if command and (proc / 'cmdline').read_bytes().split(b'\0')[:-1] != [
                    str(part).encode() for part in command]:
                return False
    except (OSError, IndexError):
        return False
    return True


def verified_evaluation(directory, config=None, checkpoint=None):
    """Accept only a finished canonical evaluation and successful separate P3 step."""
    directory = Path(directory)
    try:
        report = read_json(directory / 'evaluation.json')
        verification = read_json(directory / 'p3_verification.json')
        if not report or not isinstance(verification, list):
            return False
        count = report.get('count', len(report.get('records', [])))
        if count <= 0 or len(verification) != count:
            return False
        if any(not row.get('pixels_equal') or not Path(row['raw']).is_file() or
               not Path(row['p3']).is_file() for row in verification):
            return False
        for path, key in ((config, 'config_sha256'), (checkpoint, 'checkpoint_sha256')):
            if path is not None and report.get(key) != file_hash(path):
                return False
        return isinstance(report.get('summary'), dict)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def load_runs(suite):
    runs, sources = [], []
    for scene in ('outdoor', 'indoor'):
        path = suite / ('manifest_' + scene + '.json')
        if not path.is_file():
            continue
        manifest = read_json(path)
        sources.append(dict(path=str(path), sha256=file_hash(path)))
        if manifest.get('scene') != scene:
            raise ValueError('scene does not match manifest filename: ' + str(path))
        for run in manifest['runs']:
            item = dict(run, scene=scene)
            work_dir = Path(item['work_dir'])
            if not work_dir.is_absolute():
                work_dir = ROOT / work_dir
            item.update(work_dir=str(work_dir.resolve()),
                        config=str(work_dir.resolve() / 'gen_0' / 'inference_config.py'),
                        checkpoint=str(work_dir.resolve() / 'gen_0' / 'best.pth'))
            runs.append(item)
    if not runs or len({run['name'] for run in runs}) != len(runs):
        raise ValueError('suite needs nonempty manifests with unique run names')
    return runs, sources


def evaluation_command(run, split, output, memory_fraction):
    command = [sys.executable, str(ROOT / 'tools/evaluate_highlight_guard.py'),
               'test' if split == 'test' else 'paired', '--config', run['config'],
               '--checkpoint', run['checkpoint'], '--output', str(output),
               '--device', 'cuda:0', '--memory-fraction', str(memory_fraction),
               '--decompose', '--p3']
    if split == 'test':
        command.extend(['--input-root', str(TEST_ROOT), '--mask-root', str(MASK_ROOT),
                        '--ids-file', str(ROOT / 'work_dirs/test_final' /
                                          (run['scene'] + '_ids.txt'))])
    else:
        command.extend(['--split', split])
    return command


def mark_failed(job, reason, split=None):
    job.update(status='failed', error=reason, finished_unix=time.time())
    if split:
        job['phases'][split].update(status='failed', error=reason)
    for phase in job['phases'].values():
        if phase['status'] in ('pending', 'waiting'):
            phase.update(status='skipped', error=reason)


def finish_phase(active, state, exit_code):
    item = state['jobs'][active['name']]
    phase = item['phases'][active['split']]
    if active.get('handle'):
        active['handle'].close()
    valid = verified_evaluation(phase['output'], item['config'], item['checkpoint'])
    if valid and exit_code in (None, 0):
        phase.update(status='completed', finished_unix=time.time(), exit_code=exit_code)
        item.update(status='ready', error=None)
        print('Evaluated {} {}'.format(active['name'], active['split']), flush=True)
    else:
        message = 'evaluator exit {} or incomplete/mismatched RGB/P3 artifacts'.format(exit_code)
        mark_failed(item, message, active['split'])
        phase.update(exit_code=exit_code, finished_unix=time.time())
        print('Failed {} {}: {}'.format(active['name'], active['split'], message), flush=True)
    phase.pop('pid', None)


def summarize(suite, runs, state):
    rows, selections = [], {}
    for run in runs:
        job = state['jobs'][run['name']]
        for split in SPLITS:
            phase = job['phases'][split]
            row = dict(scene=run['scene'], variant=run['variant'], name=run['name'], split=split,
                       training_status=job.get('training_status'), status=phase['status'],
                       evaluation=str(Path(phase['output']) / 'evaluation.json'),
                       checkpoint=run['checkpoint'], error=phase.get('error'))
            if phase['status'] == 'completed':
                report = read_json(row['evaluation'], {})
                row.update(count=report.get('count'), **report.get('summary', {}))
            rows.append(row)
    for scene in sorted({run['scene'] for run in runs}):
        candidates, rejected = [], []
        for row in rows:
            if row['scene'] != scene or row['split'] != 'val' or row['variant'].startswith('legacy'):
                continue
            reasons = []
            if row['status'] != 'completed' or not finite(row.get('PSNR')):
                reasons.append('no completed finite validation PSNR')
            for metric in ('raw_high_clip', 'raw_low_clip'):
                if not finite(row.get(metric)) or row[metric] != 0.:
                    reasons.append('validation {} is missing or nonzero'.format(metric))
            if reasons:
                rejected.append(dict(variant=row['variant'], reasons=reasons))
            else:
                candidates.append(row)
        winner = max(candidates, key=lambda row: (row['PSNR'],
            -row['deltaE_P3'] if finite(row.get('deltaE_P3')) else -float('inf'), row['variant'])) if candidates else None
        selections[scene] = dict(
            selected=winner['variant'] if winner else None,
            selected_name=winner['name'] if winner else None,
            checkpoint=winner['checkpoint'] if winner else None,
            validation_metrics={key: winner.get(key) for key in
                                ('PSNR', 'deltaE_P3', 'deltaE_P3_foreground',
                                 'raw_high_clip', 'raw_low_clip')} if winner else None,
            rejected=rejected, policy='Validation only: highest PSNR among non-legacy variants '
                'with exactly zero validation raw upper/lower overflow; no train/test selection.')
    report = dict(collected_unix=time.time(), protocol='Best checkpoint per run is validation-PSNR selected; '
                  'suite variant selection uses validation only. Raw diagnostics precede clamp; '
                  'P3 export is separately verified. Paired output is GT-sized; test is native.',
                  rows=rows, selection=selections)
    save_json(suite / 'experiment_summary.json', report)
    keys = ['scene', 'variant', 'name', 'split', 'status', 'training_status', 'PSNR',
            'deltaE_P3', 'deltaE_P3_foreground', 'raw_high_clip', 'raw_low_clip']
    keys += sorted({key for row in rows for key in row} - set(keys))
    with (suite / 'experiment_summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return report


def build_reports(suite, runs, state, summary):
    for scene, selection in summary['selection'].items():
        scene_runs = [run for run in runs if run['scene'] == scene]
        models = [dict(name=run['variant'], dir=str(suite / 'eval' / scene / run['variant']))
                  for run in scene_runs]
        baseline_root = ROOT / 'work_dirs/hg_20261027_suite/baselines'
        original = {split: str(baseline_root / (scene + '_' +
                    {'train': 'train_nchw', 'val': 'val_nchw', 'test': 'test_nchw'}[split]))
                    for split in SPLITS}
        models.insert(0, dict(name='original', splits=original))
        if scene == 'outdoor':
            models.insert(1, dict(name='highlight_guard',
                                 dir=str(ROOT / 'work_dirs/hg_20261027_suite/candidates/range_over')))
        missing = []
        for model in models:
            for split in SPLITS:
                path = Path(model.get('splits', {}).get(split, str(Path(model.get('dir', '.')) / split)))
                if not (path / 'evaluation.json').is_file():
                    missing.append(dict(model=model['name'], split=split, directory=str(path)))
        available = [run['variant'] for run in scene_runs if any(
            state['jobs'][run['name']]['phases'][split]['status'] == 'completed' for split in SPLITS)]
        if not available:
            state['reports'][scene] = dict(status='skipped', error='no completed current evaluations',
                                          missing=missing)
            continue
        candidate = selection['selected'] or ('legacy' if 'legacy' in available else available[0])
        report_suite = suite / 'reports' / scene
        report_manifest = dict(scene=scene, splits=list(SPLITS), models=models,
                               reference='original', candidate=candidate, selection=selection,
                               missing_evaluations=missing,
                               note='If no eligible bounded candidate exists, comparison candidate '
                                    'is a display fallback and is not a validation winner.')
        manifest_path = report_suite / 'report_manifest.json'
        save_json(manifest_path, report_manifest)
        manifest_hash = file_hash(manifest_path)
        previous = state['reports'].get(scene, {})
        if (previous.get('status') == 'completed' and
                previous.get('manifest_sha256') == manifest_hash and
                (report_suite / 'report/report.html').is_file()):
            continue
        command = [sys.executable, str(ROOT / 'tools/analyze_bounded_curves.py'),
                   '--suite', str(report_suite), '--output', str(report_suite / 'report'),
                   '--reference', 'original', '--candidate', candidate]
        log = report_suite / ('report_{}.console.log'.format(time.time_ns()))
        item = dict(status='running', command=command, console_log=str(log),
                    manifest_sha256=manifest_hash, missing=missing, started_unix=time.time())
        state['reports'][scene] = item
        save_json(suite / 'evaluation_state.json', state)
        environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with log.open('x') as handle:
            code = subprocess.call(command, cwd=str(ROOT), env=environment,
                                   stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT)
        item.update(status='completed' if code == 0 else 'failed', exit_code=code,
                    finished_unix=time.time(), report=str(report_suite / 'report/report.html'))
        save_json(suite / 'evaluation_state.json', state)
        print('Report {} {}'.format(scene, item['status']), flush=True)


def orchestrate(args):
    suite = args.suite.expanduser().resolve()
    runs, sources = load_runs(suite)
    path = suite / 'evaluation_state.json'
    state = read_json(path, {})
    old_sources = state.get('manifest_sources')
    if old_sources and old_sources != sources:
        raise ValueError('suite manifests changed since this evaluation state was created')
    state.update(schema_version=1, suite=str(suite), manifest_sources=sources,
                 memory_fraction=args.memory_fraction, min_free_mb=args.min_free_mb,
                 orchestrator_pid=os.getpid())
    state.setdefault('started_unix', time.time())
    state.setdefault('jobs', {})
    state.setdefault('reports', {})
    active = {}
    for run in runs:
        job = state['jobs'].setdefault(run['name'], dict(
            scene=run['scene'], variant=run['variant'], config=run['config'],
            checkpoint=run['checkpoint'], work_dir=run['work_dir'], status='waiting', phases={}))
        for split in SPLITS:
            phase = job['phases'].setdefault(split, dict(status='pending',
                output=str(suite / 'eval' / run['scene'] / run['variant'] / split)))
            if phase['status'] == 'running' and pid_alive(phase.get('pid'), phase.get('command')):
                gpu = job.get('gpu_uuid')
                if not gpu or gpu in active:
                    raise RuntimeError('cannot safely recover active GPU evaluation: ' + run['name'])
                active[gpu] = dict(name=run['name'], split=split, pid=phase['pid'],
                                   command=phase['command'], process=None)
            elif verified_evaluation(phase['output'], run['config'], run['checkpoint']):
                phase.update(status='completed', recovered=True)
            elif phase['status'] in ('running', 'completed') or args.retry_failed:
                phase.update(status='pending', error=None)
        if args.retry_failed and job['status'] == 'failed':
            job.update(status='waiting', error=None)
    save_json(path, state)

    try:
        while True:
            for gpu, item in list(active.items()):
                process = item.get('process')
                code = process.poll() if process is not None else None
                finished = code is not None if process is not None else not pid_alive(item['pid'], item['command'])
                if finished:
                    finish_phase(item, state, code)
                    del active[gpu]
            training = read_json(suite / 'launcher/run_state.json', {})
            training_jobs = training.get('jobs', [])
            by_name = {job['name']: job for job in training_jobs}
            busy = {job.get('gpu_uuid') for job in training_jobs if job.get('status') == 'training'}
            queued = any(job.get('status') == 'queued' for job in training_jobs)
            pending = []
            for run in runs:
                job = state['jobs'][run['name']]
                train = by_name.get(run['name'], {})
                job['training_status'] = train.get('status', 'not_yet_in_launcher_state')
                if job['status'] == 'failed':
                    continue
                if all(phase['status'] == 'completed' for phase in job['phases'].values()):
                    job.update(status='completed', finished_unix=job.get('finished_unix', time.time()))
                    continue
                if any(phase['status'] == 'running' for phase in job['phases'].values()):
                    continue
                if train.get('status') == 'failed':
                    mark_failed(job, 'training failed with exit {}'.format(train.get('exit_code')))
                    continue
                if train.get('status') != 'completed':
                    job['status'] = 'waiting_training'
                    continue
                if not all(Path(run[key]).is_file() for key in ('config', 'checkpoint')):
                    mark_failed(job, 'completed training did not produce inference_config.py and best.pth')
                    continue
                gpu = train.get('gpu_uuid')
                if not gpu:
                    mark_failed(job, 'completed launcher job has no GPU UUID')
                    continue
                job.update(gpu_uuid=gpu, gpu=train.get('gpu'), status='waiting_gpu')
                if not queued and gpu not in busy and gpu not in active:
                    pending.append((run, gpu))
            if pending:
                inventory = devices()
                for run, gpu in pending:
                    if gpu in active or gpu not in inventory or inventory[gpu]['free_mb'] < args.min_free_mb:
                        continue
                    job = state['jobs'][run['name']]
                    split = next((part for part in SPLITS if job['phases'][part]['status'] == 'pending'), None)
                    if split is None:
                        mark_failed(job, 'no runnable evaluation split remains')
                        continue
                    phase = job['phases'][split]
                    output = Path(phase['output'])
                    output.mkdir(parents=True, exist_ok=True)
                    command = evaluation_command(run, split, output, args.memory_fraction)
                    log = output / ('evaluation_{}.console.log'.format(time.time_ns()))
                    handle = log.open('x')
                    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
                        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1',
                        PYTHONDONTWRITEBYTECODE='1')
                    process = subprocess.Popen(command, cwd=str(ROOT), env=environment,
                        stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                        start_new_session=True)
                    phase.update(status='running', pid=process.pid, command=command,
                                 console_log=str(log), started_unix=time.time(),
                                 free_mb_at_launch=inventory[gpu]['free_mb'])
                    job['status'] = 'evaluating'
                    active[gpu] = dict(name=run['name'], split=split, pid=process.pid,
                        process=process, handle=handle, command=command)
                    save_json(path, state)
                    print('Started {} {} on {}'.format(run['name'], split, gpu), flush=True)
            state['updated_unix'] = time.time()
            save_json(path, state)
            if not active and all(job['status'] in ('completed', 'failed') for job in state['jobs'].values()):
                break
            time.sleep(args.wait_poll)
    except KeyboardInterrupt:
        # Stop only evaluator children started by this invocation, never training.
        for item in active.values():
            process = item.get('process')
            if process is not None:
                process.terminate()
                process.wait()
                if item.get('handle'):
                    item['handle'].close()
                state['jobs'][item['name']]['phases'][item['split']]['status'] = 'pending'
        state['interrupted_unix'] = time.time()
        save_json(path, state)
        raise
    summary = summarize(suite, runs, state)
    if not getattr(args, 'no_reports', False):
        build_reports(suite, runs, state, summary)
    else:
        state['reports_skipped'] = True
    state.update(finished_unix=time.time(), status='completed_with_failures' if any(
        job['status'] == 'failed' for job in state['jobs'].values()) or any(
        item['status'] == 'failed' for item in state['reports'].values()) else 'completed')
    save_json(path, state)
    print('Suite {}: {}'.format(state['status'], suite / 'experiment_summary.json'), flush=True)
    return 1 if state['status'] == 'completed_with_failures' else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--memory-fraction', type=float, default=.22)
    parser.add_argument('--min-free-mb', type=int, default=6000)
    parser.add_argument('--wait-poll', type=float, default=10.)
    parser.add_argument('--retry-failed', action='store_true',
                        help='retry failed/interrupted evaluator phases on this invocation')
    parser.add_argument('--no-reports', action='store_true',
                        help='evaluate and summarize a supplementary control suite only')
    args = parser.parse_args()
    if (not math.isfinite(args.memory_fraction) or not 0 < args.memory_fraction <= 1 or
            args.min_free_mb < 0 or not math.isfinite(args.wait_poll) or args.wait_poll <= 0):
        parser.error('invalid memory fraction, minimum free memory or poll interval')
    suite = args.suite.expanduser().resolve()
    if not suite.is_dir():
        parser.error('suite directory does not exist')
    with (suite / '.evaluation.lock').open('a+') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('another evaluation orchestrator is already running for this suite')
        lock.seek(0)
        lock.truncate()
        lock.write(str(os.getpid()) + '\n')
        lock.flush()
        raise SystemExit(orchestrate(args))


if __name__ == '__main__':
    main()
