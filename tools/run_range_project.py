"""One scratch run per scene; built-in train/val export, then one blind test pass."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCENES = ('indoor', 'outdoor', 'club')
SUITE = ROOT / 'work_dirs/rp_20261029_suite'
TEST = ROOT / 'work_dirs/test_final_range_project_20261029'


def configuration(scene):
    return ROOT / ('adaint/configs/face_' + scene + '_semantic_range_project.yaml')


def generation(scene):
    return ROOT / ('work_dirs/rp_20261029_' + scene + '_s2026/gen_0')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def revision():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT)).decode().strip()


def test_inputs(scene, cfg):
    if scene == 'club':
        data = cfg.data.test
        ids = (ROOT / data.ann_file).read_text().split()
        return ids, [Path(data.dir_lq) / data.filetmpl_lq.format(sid) for sid in ids]
    ids = (ROOT / 'work_dirs/test_final' / (scene + '_ids.txt')).read_text().split()
    return ids, [Path('/DB/data/yujietang/IQ-LUT/face_dataset/test_final') / (sid + '_input.jpg') for sid in ids]


def mask_settings(scene, cfg, split):
    from adaint.semantic_pipeline import semantic_mask_step
    settings = dict(semantic_mask_step(cfg, split, training=False))
    settings.pop('type')
    if split == 'test' and scene != 'club':
        settings.update(foreground_root='/DB/data/yujietang/IQ-LUT/face_dataset/test_final_masks/masks', face_root=None)
    return settings


def prepare():
    import numpy as np
    from PIL import Image
    from adaint.config_utils import load_config
    from adaint.semantic_pipeline import LoadSemanticMasks
    from mmedit.models import build_model
    if SUITE.exists() or TEST.exists() or any(generation(s).parent.exists() for s in SCENES):
        raise FileExistsError('Refusing to overwrite or repeat this experiment prefix')
    jobs = []
    for scene in SCENES:
        cfg = load_config(str(configuration(scene)))
        assert cfg.infer_after_generation is True and cfg.gen == 1 and not cfg.enable_kd
        assert cfg.train_epochs == 100 and cfg.adaint_fix_epochs == 0
        assert not any(cfg.get(k) for k in ('postprocess_init', 'load_from', 'resume_from'))
        assert cfg.model.pretrained is False and not cfg.model.postprocess.freeze_global
        assert cfg.model.range_project.enabled and 'safe_affine' not in cfg.model
        model = build_model(cfg.model, train_cfg=dict(n_fix_iters=0), test_cfg=cfg.test_cfg)
        assert all(p.requires_grad for p in model.parameters())
        del model
        splits = {}
        for split in ('train', 'val', 'test'):
            data = cfg.data[split]
            if split == 'test':
                ids, inputs = test_inputs(scene, cfg)
            else:
                ids = (ROOT / data.ann_file).read_text().split()
                inputs = [Path(data.dir_lq) / data.filetmpl_lq.format(sid) for sid in ids]
                assert all((Path(data.dir_gt) / data.filetmpl_gt.format(sid)).is_file() for sid in ids)
            assert ids and len(ids) == len(set(ids)) and all(p.is_file() for p in inputs)
            loader = LoadSemanticMasks(**mask_settings(scene, cfg, split))
            for sid, path in zip(ids, inputs):
                # Check the actual loader's path convention without a model forward.
                with Image.open(str(path)) as im:
                    size = im.size
                loader(dict(lq=np.zeros((size[1], size[0], 3), dtype=np.uint8), lq_path=str(path)))
            splits[split] = dict(count=len(ids), ids=ids)
        assert not set(splits['train']['ids']) & set(splits['val']['ids'])
        jobs.append(dict(scene=scene, config=str(configuration(scene)), config_sha256=sha(configuration(scene)),
                         output=str(generation(scene)), test_output=str(TEST / scene), splits=splits))
        print('preflight passed:', scene, {k: v['count'] for k, v in splits.items()}, flush=True)
    save(SUITE / 'manifest.json', dict(jobs=jobs, epochs=100, seed=2026, code_revision=revision(),
        baseline_revision='0757473068177b8442cf5384e22df2a746de48e6',
        initialization='scratch; all parameters trainable; no checkpoint or teacher',
        training_export='infer_after_generation=True; built-in native train+val pass only',
        test_export='one native blind pass per scene; no result analysis'))


def train(scene):
    import torch
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.30, 0)
    state = dict(scene=scene, status='training', started=time.time(), code_revision=revision())
    save(SUITE / (scene + '.training.json'), state)
    try:
        sys.argv = ['tools/train_kd_unified.py', str(configuration(scene)), '--seed', '2026']
        runpy.run_path(str(ROOT / 'tools/train_kd_unified.py'), run_name='__main__')
        from adaint.config_utils import load_config
        cfg = load_config(str(configuration(scene)))
        expected = set()
        for split in ('train', 'val'):
            expected.update((ROOT / cfg.data[split].ann_file).read_text().split())
        actual = {p.name[:-len('_result.png')] for p in (generation(scene) / 'train_infer').glob('*_result.png')}
        assert actual == expected, (scene, len(actual), len(expected))
        assert (generation(scene) / 'best.pth').is_file()
        state.update(status='completed', native_train_val_images=len(actual))
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        state.update(finished=time.time(), peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        save(SUITE / (scene + '.training.json'), state)


def infer_test(scene):
    import numpy as np
    import torch
    from PIL import Image
    from adaint.config_utils import load_config
    from adaint.experiment_utils import inference_config, load_enhancer_checkpoint
    from adaint.semantic_pipeline import LoadSemanticMasks
    from mmedit.models import build_model
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.85, 0)
    cfg = inference_config(load_config(str(generation(scene) / 'inference_config.py')))
    assert cfg.model.range_project.enabled
    folder = TEST / scene
    if folder.exists():
        raise FileExistsError('Refusing repeated test inference')
    ids, inputs = test_inputs(scene, cfg)
    loader = LoadSemanticMasks(**mask_settings(scene, cfg, 'test'))
    model = build_model(cfg.model, train_cfg=None, test_cfg=dict(metrics=None, crop_border=0))
    checkpoint = generation(scene) / 'best.pth'
    metadata = load_enhancer_checkpoint(model, str(checkpoint)).get('meta', {})
    model.cuda().eval()
    state = dict(scene=scene, status='running', expected=len(ids), checkpoint=str(checkpoint),
        checkpoint_sha256=sha(checkpoint), selected_epoch=metadata.get('epoch'), code_revision=revision(),
        resolution='native input; one forward per image', records=[])
    save(folder / 'manifest.json', state)
    (folder / 'raw').mkdir()
    (folder / 'inference_config.py').write_bytes((generation(scene) / 'inference_config.py').read_bytes())
    for index, (sid, path) in enumerate(zip(ids, inputs), 1):
        with Image.open(str(path)) as image:
            array = np.asarray(image.convert('RGB')).copy()
        masks = loader(dict(lq=array, lq_path=str(path)))
        x = torch.from_numpy(array.astype(np.float32) / 255.).permute(2, 0, 1)[None].contiguous().cuda()
        mask = torch.from_numpy(masks['foreground_mask']).permute(2, 0, 1)[None].cuda()
        with torch.no_grad():
            output, _, _, aux = model.forward_dummy(x, True, foreground_mask=mask)
            assert torch.isfinite(output).all() and output.min() >= 0 and output.max() <= 1
            raw = aux['raw_output']
            legal = (raw >= 0) & (raw <= 1)
            assert torch.equal(output[legal], raw[legal])
            row = dict(id=sid, input=str(path), size=[array.shape[1], array.shape[0]],
                       final_min=float(output.min()), final_max=float(output.max()),
                       final_clip_fraction=float(((output < 0) | (output > 1)).any(1).float().mean()))
            pixels = (output[0].permute(1, 2, 0).cpu().numpy() * 255.).round().astype(np.uint8)
        destination = folder / 'raw' / (sid + '_result.png')
        Image.fromarray(pixels).save(str(destination))
        row['output'] = str(destination)
        state['records'].append(row)
        save(folder / 'manifest.json', state)
        del output, aux, raw, x, mask, legal
        print('{} test {}/{} {}'.format(scene, index, len(ids), sid), flush=True)
    state.update(status='completed', peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    save(folder / 'manifest.json', state)


def controller(gpu):
    state = dict(status='training', jobs=[], started=time.time(), code_revision=revision())
    children = []
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    for scene in SCENES:
        log = (SUITE / (scene + '.train.console.log')).open('x')
        child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), 'train', '--scene', scene],
            cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        log.close()
        job = dict(scene=scene, pid=child.pid, status='training')
        state['jobs'].append(job)
        children.append((child, job))
    save(SUITE / 'state.json', state)
    for child, job in children:
        code = child.wait()
        job.update(status='trained' if code == 0 else 'failed', train_exit_code=code)
        save(SUITE / 'state.json', state)
    if any(job['status'] == 'failed' for job in state['jobs']):
        state['status'] = 'failed'
        save(SUITE / 'state.json', state)
        raise SystemExit(1)
    state['status'] = 'test_inference'
    save(SUITE / 'state.json', state)
    for scene in SCENES:
        log = (SUITE / (scene + '.test.console.log')).open('x')
        code = subprocess.call([sys.executable, '-B', str(Path(__file__).resolve()), 'test', '--scene', scene],
            cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        log.close()
        job = next(job for job in state['jobs'] if job['scene'] == scene)
        job.update(status='completed' if code == 0 else 'failed', test_exit_code=code)
        save(SUITE / 'state.json', state)
        if code:
            state['status'] = 'failed'
            save(SUITE / 'state.json', state)
            raise SystemExit(code)
    state.update(status='completed', finished=time.time())
    save(SUITE / 'state.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'train', 'test', 'controller'))
    parser.add_argument('--scene', choices=SCENES)
    parser.add_argument('--gpu', type=int, default=7)
    args = parser.parse_args()
    os.chdir(str(ROOT))
    if args.mode == 'prepare':
        prepare()
    elif args.mode == 'train':
        train(args.scene)
    elif args.mode == 'test':
        infer_test(args.scene)
    else:
        controller(args.gpu)
