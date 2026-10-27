"""Run prepared experiments with one task per GPU and an allocator budget.

Only launches commands in explicitly supplied suite manifests. Existing GPU
processes are observed, never stopped or changed.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def devices():
    output = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,memory.free',
        '--format=csv,noheader,nounits']).decode()
    return {int(row.split(',')[0]): dict(uuid=row.split(',')[1].strip(),
                free_mb=int(row.split(',')[2])) for row in output.strip().splitlines()}


def write_state(path, jobs):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(updated_unix=time.time(), jobs=jobs), indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifests', nargs='+', type=Path)
    parser.add_argument('--gpus', nargs='+', type=int, default=list(range(8)))
    parser.add_argument('--min-free-mb', type=int, default=7000)
    parser.add_argument('--memory-fraction', type=float, default=.22)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.memory_fraction <= 1:
        parser.error('memory fraction must be in (0, 1]')
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output/'run_state.json'
    if state_path.exists():
        raise FileExistsError('refusing to replace existing scheduler state: ' + str(state_path))
    jobs = []
    for path in args.manifests:
        manifest = json.loads(path.read_text())
        for run in manifest['runs']:
            jobs.append(dict(name=run['name'], command=run['command'],
                             config=run['config'], work_dir=run['work_dir'],
                             status='queued', manifest=str(path.resolve())))
    active = {}
    root = Path(__file__).resolve().parents[1]
    while any(job['status'] in ('queued', 'training') for job in jobs):
        for gpu, (process, handle, job) in list(active.items()):
            code = process.poll()
            if code is not None:
                handle.close()
                job.update(status='completed' if code == 0 else 'failed',
                           exit_code=code, finished_unix=time.time())
                print('{} {} exit={}'.format(job['name'], job['status'], code), flush=True)
                del active[gpu]
        inventory = devices()
        for gpu in args.gpus:
            if gpu in active or gpu not in inventory or inventory[gpu]['free_mb'] < args.min_free_mb:
                continue
            queued = next((job for job in jobs if job['status'] == 'queued'), None)
            if queued is None:
                break
            log = args.output/(queued['name']+'.console.log')
            handle = log.open('x')
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=inventory[gpu]['uuid'],
                OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1',
                PYTHONDONTWRITEBYTECODE='1',
                ADAINT_CUDA_MEMORY_FRACTION=str(args.memory_fraction))
            process = subprocess.Popen(queued['command'], cwd=str(root), env=environment,
                stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                start_new_session=True)
            queued.update(status='training', gpu=gpu, gpu_uuid=inventory[gpu]['uuid'],
                free_mb_at_launch=inventory[gpu]['free_mb'], pid=process.pid,
                started_unix=time.time(), console_log=str(log),
                memory_fraction=args.memory_fraction)
            active[gpu] = (process, handle, queued)
            print('Started {} on GPU {} pid={}'.format(queued['name'],gpu,process.pid), flush=True)
        write_state(state_path, jobs)
        if any(job['status'] in ('queued', 'training') for job in jobs):
            time.sleep(10)
    write_state(state_path, jobs)
    raise SystemExit(1 if any(job['status'] == 'failed' for job in jobs) else 0)


if __name__ == '__main__':
    main()
