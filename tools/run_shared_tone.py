"""DB28-only controlled ablations, native exports, P3 tagging and evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path('/DB/data/yujietang/IQ-LUT/image_enhance/AdaInt_BAN')
IQ = PROJECT.parents[1]
SUITE = PROJECT / 'work_dirs/shared_tone_suite_20261102'
PYTHON = '/DB/rhome/yujietang/miniconda3/envs/AdaInt_BAN/bin/python'
EVAL_PYTHON = '/DB/rhome/yujietang/miniconda3/bin/python'
LABELS = ['safe_full', 'global_only', 'bounded_dual', 'shared_global',
          'global_aux', 'scalar_exposure']


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def revision():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT)).decode().strip()


def configuration(scene, index):
    return ROOT / ('adaint/configs/face_' + scene + '_shared_tone_' + str(index) + '.py')


def tag_folder(source, target):
    import p3
    source, target = Path(source), Path(target)
    files = sorted(source.glob('*_result.png'))
    if not files:
        raise RuntimeError('No exported PNGs: ' + str(source))
    for path in files:
        message = p3.file_process(str(path), str(target / path.name))
        if '处理成功' not in message:
            raise RuntimeError(message)
    if len(list(target.glob('*_result.png'))) != len(files):
        raise RuntimeError('P3 export count mismatch')


def infer_test(cfg, generation):
    import numpy as np
    import torch
    from PIL import Image
    from mmedit.models import build_model
    from adaint.experiment_utils import inference_config, load_enhancer_checkpoint
    from adaint.semantic_pipeline import LoadSemanticMasks

    scene = cfg.ablation.scene if 'scene' in cfg.ablation else ('indoor' if 'indoor' in cfg.exp_name else 'outdoor')
    ids = (PROJECT / 'work_dirs/test_final' / (scene + '_ids.txt')).read_text().split()
    mask_loader = LoadSemanticMasks(
        foreground_root=str(IQ / 'face_dataset/test_final_masks/masks'),
        feather_radius=cfg.semantic_masks.get('feather_radius', 1))
    test_cfg = inference_config(cfg)
    model = build_model(test_cfg.model, train_cfg=None, test_cfg=test_cfg.test_cfg)
    checkpoint = load_enhancer_checkpoint(model, str(generation / 'best.pth'))
    model.cuda().eval()
    folder = generation / 'test_infer'
    raw_folder = folder / 'raw'
    raw_folder.mkdir(parents=True, exist_ok=True)
    records = []
    with torch.no_grad():
        for sid in ids:
            path = IQ / 'face_dataset/test_final' / (sid + '_input.jpg')
            image = np.asarray(Image.open(path).convert('RGB')).copy()
            masks = mask_loader(dict(lq=image, lq_path=str(path)))
            tensor = torch.from_numpy(image).permute(2, 0, 1)[None].float().cuda() / 255.
            mask = torch.from_numpy(masks['foreground_mask']).permute(2, 0, 1)[None].cuda()
            output, _, _, aux = model.forward_dummy(tensor, return_aux=True, foreground_mask=mask)
            raw = aux['raw_output']
            if not torch.isfinite(output).all() or output.min() < 0 or output.max() > 1:
                raise RuntimeError('Unsafe final output for test ID ' + sid)
            codes = output[0].permute(1, 2, 0).mul(255).round().byte().cpu().numpy()
            Image.fromarray(codes).save(raw_folder / (sid + '_result.png'))
            foreground = masks['supervision_mask'][..., 0] > 0
            near_white = (codes >= 250).all(-1)
            records.append(dict(id=sid, input=str(path), size=list(image.shape[:2]),
                proposal_min=raw.min().item(), proposal_max=raw.max().item(),
                proposal_high_fraction=(raw > 1).any(1).float().mean().item(),
                final_min=output.min().item(), final_max=output.max().item(),
                final_clip_fraction=((output < 0) | (output > 1)).float().mean().item(),
                near_white_fraction=float(near_white.mean()),
                face_near_white_fraction=float(near_white[foreground].mean()) if foreground.any() else None,
                background_near_white_fraction=float(near_white[~foreground].mean()) if (~foreground).any() else None,
                any_255_fraction=float((codes == 255).any(-1).mean())))
    manifest = dict(scene=scene, checkpoint=str(generation / 'best.pth'),
        checkpoint_sha256=sha(generation / 'best.pth'), selected_epoch=checkpoint.get('meta', {}).get('epoch'),
        code_revision=revision(), host=socket.gethostname(), records=records,
        regression_ids=[sid for sid in ['1', '16', '17'] if sid in ids],
        note='No test GT; highlight fractions are diagnostics, not exposure-correctness scores.')
    save(folder / 'manifest.json', manifest)
    del model, tensor, output, raw, aux
    torch.cuda.empty_cache()
    tag_folder(raw_folder, folder / 'raw_p3')
    return manifest


def evaluate(cfg, generation):
    command = [EVAL_PYTHON, str(IQ / 'eval.py'), '--data-dir', cfg.data.train.dir_gt,
        '--result-dir', str(generation / 'train_infer_p3'),
        '--mask-dir', cfg.semantic_masks.train_foreground_root,
        '--output-dir', str(generation / 'evaluation'), '--color-space', 'display-p3',
        '--deltae', 'both']
    subprocess.run(command, cwd=str(ROOT), check=True)
    # Native inference intentionally includes train+val. Preserve both the
    # original evaluator output and explicit split means to avoid mixing them.
    import csv
    with (generation / 'evaluation/metrics_per_image.csv').open(encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    summaries = {}
    for split in ['train', 'val']:
        ids = set(Path(cfg.data[split].ann_file).read_text().split())
        chosen = [r for r in rows if r['sample_id'] in ids]
        values = {}
        for key in rows[0]:
            if key.startswith('result_') and any(key.endswith(s) for s in ['_psnr', '_delta_e00', '_delta_eab']):
                numbers = [float(r[key]) for r in chosen if r[key] not in ['', 'None']]
                values[key] = sum(numbers) / len(numbers) if numbers else None
        summaries[split] = dict(count=len(chosen), metrics=values)
    save(generation / 'evaluation/split_summary.json', summaries)
    return summaries


def run(scene, index, gpu, smoke=False):
    if socket.gethostname().split('.')[0].upper() != 'DB28':
        raise RuntimeError('Training for this suite is authorized on DB28 only')
    os.chdir(str(ROOT))
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    sys.path.insert(0, str(ROOT))
    from adaint.config_utils import load_config
    from adaint.experiment_utils import save_generation_configs
    cfg = load_config(str(configuration(scene, index)))
    config_path = configuration(scene, index)
    if smoke:
        cfg.exp_name = 'face_' + scene + '_shared_tone_smoke_1'
        cfg.work_dir = str(PROJECT / 'work_dirs' / cfg.exp_name)
        cfg.train_epochs = 5
        for split in ['train', 'val']:
            ids = Path(cfg.data[split].ann_file).read_text().split()[:2]
            file = Path(cfg.work_dir) / (split + '_ids.txt')
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text('\n'.join(ids) + '\n')
            cfg.data[split].ann_file = str(file)
        cfg.gpus = 1
        save_generation_configs(cfg)
        config_path = Path(cfg.work_dir) / 'resolved_config.py'
    destination = Path(cfg.work_dir)
    generation = destination / 'gen_0'
    state_path = destination / 'status.json'
    if generation.exists():
        raise RuntimeError('Refusing to overwrite existing generation: ' + str(generation))
    state = dict(scene=scene, index=index, label=LABELS[index - 1], gpu=gpu,
        host=socket.gethostname(), pid=os.getpid(), config=str(config_path),
        config_sha256=sha(config_path), code_revision=revision(), seed=2026,
        started=time.time(), status='training', smoke=smoke)
    save(state_path, state)
    try:
        subprocess.run([PYTHON, 'tools/train_kd_unified.py', str(config_path), '--seed', '2026'],
                       cwd=str(ROOT), env=os.environ.copy(), check=True)
        state.update(status='p3_export', training_finished=time.time()); save(state_path, state)
        tag_folder(generation / 'train_infer', generation / 'train_infer_p3')
        state['status'] = 'test_inference'; save(state_path, state)
        test = infer_test(cfg, generation)
        state['status'] = 'evaluation'; save(state_path, state)
        summaries = evaluate(cfg, generation)
        state.update(status='completed', finished=time.time(), split_metrics=summaries,
                     test_count=len(test['records']), regression_ids=test['regression_ids'],
                     checkpoint_sha256=sha(generation / 'best.pth'))
        save(state_path, state)
    except Exception:
        state.update(status='failed', finished=time.time(), error=traceback.format_exc())
        save(state_path, state)
        raise


def launch(gpus):
    if socket.gethostname().split('.')[0].upper() != 'DB28':
        raise RuntimeError('Launch on DB28 only')
    if len(gpus) < 4 or len(set(gpus)) != len(gpus):
        raise ValueError('12 jobs require four distinct empty GPUs, three jobs per GPU')
    import fcntl
    SUITE.mkdir(parents=True, exist_ok=True)
    with (SUITE / 'launch.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = SUITE / 'manifest.json'
        if manifest_path.exists():
            raise RuntimeError('Suite already launched; inspect its manifest/status first')
        inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used',
                                            '--format=csv,noheader,nounits']).decode().splitlines()
        cards = {int(line.split(',')[0]): line.split(',') for line in inventory}
        running = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid',
                                          '--format=csv,noheader']).decode()
        for gpu in gpus:
            if int(cards[gpu][2]) > 5 or cards[gpu][1].strip() in running:
                raise RuntimeError('GPU no longer completely empty: ' + str(gpu))
        jobs = []
        for index in range(1, 7):
            for scene in ['indoor', 'outdoor']:
                name = 'face_' + scene + '_shared_tone_' + str(index)
                destination = PROJECT / 'work_dirs' / name
                if destination.exists():
                    raise RuntimeError('Existing experiment directory: ' + str(destination))
                jobs.append(dict(scene=scene, index=index, exp_name=name,
                                 gpu=gpus[len(jobs) // 3], work_dir=str(destination)))
        manifest = dict(code_revision=revision(), host=socket.gethostname(),
            started=time.time(), epochs=100, seed=2026, jobs_per_gpu=3, jobs=jobs,
            checkpoint_interval_epochs=5, evaluation_interval_epochs=5,
            regression_ids=['1', '16', '17'], labels=LABELS)
        save(manifest_path, manifest)
        for job in jobs:
            destination = Path(job['work_dir']); destination.mkdir(parents=True)
            log = (destination / 'console.log').open('w')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(job['gpu']),
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT))
            command = [PYTHON, str(Path(__file__).resolve()), 'run', '--scene', job['scene'],
                       '--index', str(job['index']), '--gpu', str(job['gpu'])]
            process = subprocess.Popen(command, cwd=str(ROOT), env=env,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            log.close(); job['pid'] = process.pid
            save(manifest_path, manifest)
        print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'smoke', 'launch', 'status'])
    parser.add_argument('--scene', choices=['indoor', 'outdoor'], default='outdoor')
    parser.add_argument('--index', type=int, choices=range(1, 7), default=6)
    parser.add_argument('--gpu', type=int, default=4)
    parser.add_argument('--gpus', default='4,5,6,7')
    args = parser.parse_args()
    if args.action in ['run', 'smoke']:
        run(args.scene, args.index, args.gpu, smoke=args.action == 'smoke')
    elif args.action == 'launch':
        launch([int(x) for x in args.gpus.split(',')])
    else:
        manifest = json.loads((SUITE / 'manifest.json').read_text())
        for job in manifest['jobs']:
            path = Path(job['work_dir']) / 'status.json'
            state = json.loads(path.read_text()) if path.exists() else {'status': 'starting'}
            print(job['exp_name'], job['gpu'], job.get('pid'), state['status'])


if __name__ == '__main__':
    main()
