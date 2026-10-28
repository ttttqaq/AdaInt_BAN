"""Exactly two scratch runs, followed by one native train/test export each."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from pprint import pformat
import runpy
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SUITE = ROOT / 'work_dirs/sa_20261028_suite'
SCENES = ('indoor', 'outdoor')
OLD = dict(indoor='face_indoor_semantic_postprocess_4', outdoor='face_outdoor_semantic_postprocess')


def plain(value):
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(plain(v) for v in value)
    return value


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str))
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def generation(scene):
    return ROOT / 'work_dirs' / ('sa_20261028_' + scene + '_s2026') / 'gen_0'


def prepare():
    from adaint.config_utils import load_config
    if SUITE.exists() or any(generation(s).parent.exists() for s in SCENES):
        raise FileExistsError('Refusing to overwrite this experiment prefix')
    jobs = []
    for scene in SCENES:
        source = ROOT / 'work_dirs' / OLD[scene] / 'gen_0/inference_config.py'
        cfg = plain(copy.deepcopy(load_config(str(source))._cfg_dict))
        model = cfg['model']
        assert model['type'] == 'AiLUTDual' and model['lut_mode'] == 'basis'
        model.pop('gen', None)
        model.pop('kd_loss', None)
        assert model['postprocess']['guidance'] == 'luminance'
        model['postprocess']['freeze_global'] = False
        model['safe_affine'] = dict(enabled=True, knee=.92, raw_weight=.25,
            highlight_weight=1., gradient_weight=.1, top_fraction=.1)
        cfg.update(exp_name=generation(scene).parent.name,
            work_dir=str(generation(scene).parent), gen=1, enable_kd=False,
            train_epochs=100, adaint_fix_epochs=0, checkpoint_interval_epochs=1,
            evaluation_interval_epochs=1, log_interval_epochs=1,
            infer_after_generation=False, postprocess_init=None,
            load_from=None, resume_from=None, train_cfg={})
        cfg['test_cfg'] = copy.deepcopy(cfg.get('validation_test_cfg') or dict(
            metrics=['PSNR', 'deltaE_P3', 'deltaE_P3_foreground', 'deltaE_P3_background'], crop_border=0))
        cfg['data']['workers_per_gpu'] = 2
        for split in ('train', 'val', 'test'):
            cfg['data'].setdefault(split + '_dataloader', {}).update(
                samples_per_gpu=1, workers_per_gpu=2, persistent_workers=True)
        cfg['optimizers']['lr'] = .0001
        cfg['lr_config'] = dict(policy='CosineAnnealing', min_lr=1e-6, by_epoch=False)
        cfg['best_checkpoint_metric'], cfg['best_checkpoint_rule'] = 'PSNR', 'greater'
        cfg['custom_hooks'] = [h for h in cfg.get('custom_hooks', []) if h.get('type') != 'LUTResumeCounterHook']
        path = SUITE / 'configs' / (scene + '.py')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as handle:
            handle.write('\n\n'.join(k + ' = ' + pformat(v, width=100) for k, v in cfg.items()) + '\n')
        jobs.append(dict(scene=scene, config=str(path), config_sha256=sha(path),
            source_config=str(source), source_sha256=sha(source), output=str(generation(scene))))
    save(SUITE / 'manifest.json', dict(jobs=jobs, epochs=100, seed=2026,
        baseline_revision='0757473', architecture='original affine/local/DC, final selective protection',
        augmentation='source brightness/saturation jitter and aligned context geometry retained',
        initialization='scratch, all parameters trainable, no checkpoint or teacher',
        selection='validation PSNR only; one best-checkpoint native train/test pass',
        code_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT)).decode().strip()))


def train(scene):
    import torch
    torch.cuda.set_per_process_memory_fraction(.4, 0)
    started = time.time()
    status = dict(scene=scene, status='running', gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), started=started)
    save(SUITE / (scene + '.training.json'), status)
    try:
        sys.argv = ['tools/train_kd_unified.py', str(SUITE / 'configs' / (scene + '.py')), '--seed', '2026']
        runpy.run_path(str(ROOT / 'tools/train_kd_unified.py'), run_name='__main__')
        status['status'] = 'completed'
    except BaseException as error:
        status.update(status='failed', error=repr(error))
        raise
    finally:
        status.update(finished=time.time(), elapsed_seconds=time.time() - started,
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        save(SUITE / (scene + '.training.json'), status)


def metrics(prediction, target, mask):
    """Native PNG -> GT-size AREA comparison, never a second model forward."""
    import cv2
    import numpy as np
    height, width = target.shape[:2]
    image = cv2.resize(prediction.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA) / 255.
    target = target.astype(np.float32) / 255.
    difference = image - target
    error = np.abs(difference).mean(-1) * 255.
    mse = float(np.square(difference).mean())
    result = dict(PSNR=-10 * np.log10(max(mse, 1e-15)), rgb_mae=float(error.mean()))
    matrix = np.array([[608311/1250200,189793/714400,198249/1000160],
        [35783/156275,247089/357200,198249/2500400], [0.,32229/714400,5220557/5000800]], dtype=np.float32)
    white = np.array([.3127/.329,1.,(1-.3127-.329)/.329], dtype=np.float32)
    def lab(rgb):
        linear = np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4)
        xyz = linear @ matrix.T / white
        f = np.where(xyz > (6/29)**3, np.cbrt(xyz), xyz*(841/108)+4/29)
        return np.stack((116*f[...,1]-16,500*(f[...,0]-f[...,1]),200*(f[...,1]-f[...,2])), -1)
    de = np.linalg.norm(lab(image) - lab(target), axis=-1)
    result['deltaE_P3'] = float(de.mean())
    luma = target @ np.array([.2126,.7152,.0722], dtype=np.float32)
    signed = difference @ np.array([.2126,.7152,.0722], dtype=np.float32) * 255.
    for name, weights in [('foreground', mask), ('background', 1-mask)]:
        keep = weights > .5
        if not keep.any():
            continue
        result['rgb_mae_' + name] = float(error[keep].mean())
        result['deltaE_P3_' + name] = float(de[keep].mean())
        bright = keep & (luma >= np.quantile(luma[keep], .9))
        result['bright_mae_' + name] = float(error[bright].mean())
        result['bright_signed_' + name] = float(signed[bright].mean())
        edges = []
        for axis in (0, 1):
            a = [slice(None)] * 2; b = list(a)
            a[axis], b[axis] = slice(None,-1), slice(1,None)
            selected = bright[tuple(a)] & bright[tuple(b)]
            if selected.any():
                edges.append(float(np.abs(np.diff(difference,axis=axis)).mean(-1)[selected].mean()*255))
        result['bright_gradient_' + name] = sum(edges)/len(edges) if edges else None
    return result


def infer(scene):
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from adaint.config_utils import load_config
    from adaint.experiment_utils import inference_config, load_enhancer_checkpoint
    from adaint.semantic_pipeline import LoadSemanticMasks, semantic_mask_step
    from mmedit.models import build_model
    cfg = inference_config(load_config(str(generation(scene)/'inference_config.py')))
    torch.cuda.set_per_process_memory_fraction(.8, 0)
    model = build_model(cfg.model, train_cfg=None, test_cfg=dict(metrics=None,crop_border=0))
    checkpoint = generation(scene)/'best.pth'
    metadata = load_enhancer_checkpoint(model, str(checkpoint)).get('meta', {})
    model.cuda().eval()
    dest = generation(scene)/'inference'
    if dest.exists():
        raise FileExistsError('Refusing duplicate inference')
    state = dict(scene=scene, status='running', checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        selected_epoch=metadata.get('epoch'), code_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT)).decode().strip(),
        resolution='native input; paired metrics AREA-downsample saved PNG to GT, no second inference',
        p3='raw PNG no ICC; separate p3.py', splits={})
    def read(path):
        with Image.open(str(path)) as im:
            return np.asarray(im.convert('RGB')).copy()
    for split in ('train', 'test'):
        data = cfg.data[split]
        if split == 'test':
            ids = (ROOT/'work_dirs/test_final'/(scene+'_ids.txt')).read_text().split()
            inputs = [Path('/DB/data/yujietang/IQ-LUT/face_dataset/test_final')/(sid+'_input.jpg') for sid in ids]
        else:
            ids = Path(data.ann_file).read_text().split()
            inputs = [Path(data.dir_lq)/data.filetmpl_lq.format(sid) for sid in ids]
        rows = []
        state['splits'][split] = dict(status='running', expected=len(ids), records=rows)
        save(dest/'manifest.json', state)
        settings = semantic_mask_step(cfg, split, training=False);settings.pop('type')
        if split == 'test':
            settings.update(foreground_root='/DB/data/yujietang/IQ-LUT/face_dataset/test_final_masks/masks',face_root=None)
        loader = LoadSemanticMasks(**settings)
        for index, (sid, path) in enumerate(zip(ids, inputs), 1):
            array = read(path)
            masks = loader(dict(lq=array,lq_path=str(path)))
            x = torch.from_numpy(array.astype(np.float32)/255.).permute(2,0,1)[None].contiguous().cuda()
            mask = torch.from_numpy(masks['foreground_mask']).permute(2,0,1)[None].cuda()
            with torch.no_grad():
                output, _, _, aux = model.forward_dummy(x, True, foreground_mask=mask)
                assert torch.isfinite(output).all() and output.min()>=0 and output.max()<=1
                protected = ~aux['protection']['risk']
                stats = dict(raw_min=float(output.min()),raw_max=float(output.max()),
                    raw_high_clip=float((output>1).any(1).float().mean()), raw_low_clip=float((output<0).any(1).float().mean()),
                    proposal_high_clip=float((aux['raw_output']>1).any(1).float().mean()),
                    risk_fraction=float(aux['protection']['risk'].float().mean()),
                    projection_fraction=float(aux['protection']['projected'].float().mean()),
                    protected_max_change=float(((output-aux['raw_output']).abs()*protected).max()))
                assert stats['protected_max_change'] == 0
                pixels=(output[0].permute(1,2,0).cpu().numpy()*255.).round().astype(np.uint8)
            raw = dest/split/'raw'/(sid+'_result.png');raw.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(pixels).save(str(raw))
            del output,aux,x,mask
            row = dict(id=sid,input=str(path),input_size=[array.shape[1],array.shape[0]],raw=str(raw),**stats)
            if split == 'train':
                target=read(Path(data.dir_gt)/data.filetmpl_gt.format(sid))
                weights=cv2.resize(masks['supervision_mask'][...,0],(target.shape[1],target.shape[0]),interpolation=cv2.INTER_NEAREST)
                row['metrics']=metrics(pixels,target,weights)
                old=ROOT/'work_dirs'/OLD[scene]/'gen_0/train_infer'/(sid+'_result.png')
                if old.is_file():row['old_metrics']=metrics(read(old),target,weights)
            else:
                old=ROOT/'work_dirs/test_final/test_final_results'/(sid+'_result.png')
                row['old_new_rgb_mae']=float(np.abs(pixels.astype(float)-read(old).astype(float)).mean())
            rows.append(row)
            save(dest/'manifest.json',state)
            print('{} {} {}/{} {}'.format(scene,split,index,len(ids),sid),flush=True)
        folder=dest/split/'raw'
        subprocess.run([sys.executable,'-B',str(ROOT/'p3.py'),str(folder),'--output',str(folder.with_name('raw_p3'))],check=True)
        from p3 import DISPLAY_P3_ICC
        for row in rows:
            p3=folder.with_name('raw_p3')/Path(row['raw']).name
            with Image.open(row['raw']) as a,Image.open(str(p3)) as b:
                assert not a.info.get('icc_profile') and b.info.get('icc_profile')==DISPLAY_P3_ICC
                assert np.array_equal(np.asarray(a),np.asarray(b))
            row.update(p3=str(p3),pixels_equal=True)
        state['splits'][split]['status']='completed'
        save(dest/'manifest.json',state)
    state.update(status='completed',peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    save(dest/'manifest.json',state)


def controller(gpu):
    state=dict(status='training',jobs=[],started=time.time())
    children=[]
    for scene in SCENES:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
        log=(SUITE/(scene+'.train.console.log')).open('x')
        cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'train','--scene',scene]
        process=subprocess.Popen(cmd,cwd=str(ROOT),env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        log.close();job=dict(scene=scene,pid=process.pid,status='training');state['jobs'].append(job);children.append((process,job,env))
    save(SUITE/'state.json',state)
    for child,job,env in children:
        code=child.wait();job.update(status='trained' if code==0 else 'failed',train_exit_code=code)
        save(SUITE/'state.json',state)
    if any(j['status']=='failed' for j in state['jobs']):
        state.update(status='failed');save(SUITE/'state.json',state);raise SystemExit(1)
    state['status']='inference';save(SUITE/'state.json',state)
    for child,job,env in children:
        log=(SUITE/(job['scene']+'.infer.console.log')).open('x')
        code=subprocess.call([sys.executable,'-B',str(Path(__file__).resolve()),'infer','--scene',job['scene']],cwd=str(ROOT),env=env,stdout=log,stderr=subprocess.STDOUT)
        log.close();job.update(status='completed' if code==0 else 'failed',inference_exit_code=code)
        save(SUITE/'state.json',state)
        if code:state['status']='failed';save(SUITE/'state.json',state);raise SystemExit(code)
    for scene in SCENES:
        for version in ('raw','raw_p3'):
            destination=SUITE/'test_final'/version;destination.mkdir(parents=True,exist_ok=True)
            for path in (generation(scene)/'inference/test'/version).glob('*.png'):
                os.link(str(path),str(destination/path.name))
    state.update(status='completed',finished=time.time());save(SUITE/'state.json',state)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','train','infer','controller'])
    parser.add_argument('--scene',choices=SCENES)
    parser.add_argument('--gpu',type=int,default=7)
    args=parser.parse_args();os.chdir(str(ROOT))
    if args.mode=='prepare':prepare()
    elif args.mode=='train':train(args.scene)
    elif args.mode=='infer':infer(args.scene)
    else:controller(args.gpu)
