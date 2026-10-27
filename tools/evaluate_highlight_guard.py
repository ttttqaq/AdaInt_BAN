"""Paired evaluation, blind inference and validation-only checkpoint selection.

Raw PNGs never embed ICC. --p3 runs the repository's p3.py as a separate step.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def file_hash(path):
    digest = hashlib.sha256()
    with open(str(path), 'rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def finite_means(records):
    def valid(metrics, region, default=1):
        value = metrics.get('_valid_'+region, default)
        return value is not None and math.isfinite(value) and value > 0
    names = sorted({k for row in records for k in row.get('metrics', {})})
    result = {}
    for name in names:
        if name.startswith('_valid_'):
            result[name[7:]+'_valid_count'] = sum(
                valid(row['metrics'], name[7:], 0) for row in records)
            continue
        region = next((suffix for suffix in ('foreground', 'background', 'face')
                       if name.endswith('_'+suffix)), None)
        values = [row['metrics'].get(name) for row in records
                  if region is None or valid(row['metrics'], region)]
        values = [v for v in values if v is not None and math.isfinite(v)]
        result[name] = sum(values)/len(values) if values else None
    return result


def select_guard_checkpoint(records, baseline, work_dir, psnr_tolerance=.1,
                            foreground_color_tolerance=.15,
                            highlight_improvement_tolerance=1e-8,
                            min_relative_improvement=.05,
                            region_over_tolerance=.5/255.,
                            global_color_tolerance=.15):
    """Use only validation records; latest record wins for overwritten epochs."""
    if (not math.isfinite(highlight_improvement_tolerance) or
            highlight_improvement_tolerance < 0):
        raise ValueError('highlight improvement tolerance must be finite and nonnegative')
    for name,value in (('min_relative_improvement',min_relative_improvement),
                       ('region_over_tolerance',region_over_tolerance),
                       ('global_color_tolerance',global_color_tolerance)):
        if not math.isfinite(value) or value<0:
            raise ValueError(name+' must be finite and nonnegative')
    if min_relative_improvement>1:
        raise ValueError('min_relative_improvement must not exceed 1')
    baseline_keys = ('PSNR', 'deltaE_P3', 'deltaE_P3_foreground', 'highlight_over_foreground',
                     'highlight_over_background')
    if any(key not in baseline or baseline[key] is None or
           not math.isfinite(baseline[key]) for key in baseline_keys):
        raise ValueError('baseline requires finite validation PSNR, color and highlight metrics')
    baseline_objective = .5*(baseline['highlight_over_foreground']+
                             baseline['highlight_over_background'])
    constraints=dict(psnr_tolerance=psnr_tolerance,
        foreground_color_tolerance=foreground_color_tolerance,
        global_color_tolerance=global_color_tolerance,
        highlight_improvement_tolerance=highlight_improvement_tolerance,
        min_relative_improvement=min_relative_improvement,
        region_over_tolerance=region_over_tolerance)
    latest = {}
    for row in records:
        if row.get('mode') == 'val' and 'epoch' in row:
            latest[int(row['epoch'])] = row
    candidates, rejected = [], []
    for epoch, row in sorted(latest.items()):
        required = ('PSNR', 'deltaE_P3_foreground', 'deltaE_P3',
                    'highlight_over_foreground', 'highlight_over_background')
        path = Path(work_dir)/('epoch_{}.pth'.format(epoch))
        if not path.is_file() or any(k not in row or row[k] is None or
                                    not math.isfinite(row[k]) for k in required):
            rejected.append(dict(epoch=epoch, reason='missing checkpoint or finite validation metrics',
                                 reasons=['missing checkpoint or finite validation metrics']))
            continue
        objective = .5*(row['highlight_over_foreground']+row['highlight_over_background'])
        improvement=baseline_objective-objective
        reasons=[]
        if row['PSNR'] < baseline['PSNR'] - psnr_tolerance:
            reasons.append('PSNR tolerance')
        if row['deltaE_P3_foreground'] > baseline['deltaE_P3_foreground'] + foreground_color_tolerance:
            reasons.append('foreground color tolerance')
        if row['deltaE_P3'] > baseline['deltaE_P3']+global_color_tolerance:
            reasons.append('global color tolerance')
        for region in ('foreground','background'):
            key='highlight_over_'+region
            if row[key] > baseline[key]+region_over_tolerance:
                reasons.append(region+' highlight regression tolerance')
        if improvement <= highlight_improvement_tolerance:
            reasons.append('no highlight improvement over baseline')
        if improvement+1e-15 < baseline_objective*min_relative_improvement:
            reasons.append('relative highlight improvement below minimum')
        if reasons:
            rejected.append(dict(epoch=epoch,reason='; '.join(reasons),reasons=reasons,
                objective=objective,baseline_objective=baseline_objective,
                metrics={k:row[k] for k in required}))
            continue
        candidates.append(dict(epoch=epoch, checkpoint=str(path), objective=objective,
                               metrics={k:row[k] for k in required}))
    if not candidates:
        return dict(selected=None, winner='baseline', baseline_objective=baseline_objective,
                    constraints=constraints,
                    candidates=[], rejected=rejected,
                    note='No eligible candidate improves baseline highlights within quality tolerances; '
                         'retain original checkpoint. Test is never consulted.')
    selected = min(candidates, key=lambda r:(r['objective'],r['metrics']['deltaE_P3'],
                                             -r['metrics']['PSNR'],r['epoch']))
    return dict(selected=selected, winner='candidate', baseline_objective=baseline_objective,
                constraints=constraints,
                candidates=candidates, rejected=rejected,
                note='Validation only. Objective is equal-region mean positive highlight excess.')


def select_main(args):
    baseline_report = json.loads(args.baseline.read_text())
    baseline = baseline_report.get('summary', baseline_report)
    records = []
    for path in sorted(args.work_dir.glob('*.log.json')):
        for line in path.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    report = select_guard_checkpoint(records, baseline, args.work_dir,
        args.psnr_tolerance, args.foreground_color_tolerance,
        getattr(args, 'highlight_improvement_tolerance', 1e-8),
        getattr(args, 'min_relative_improvement', .05),
        getattr(args, 'region_over_tolerance', .5/255.),
        getattr(args, 'global_color_tolerance', .15))
    report.update(baseline=str(args.baseline.resolve()), baseline_sha256=file_hash(args.baseline),
                  psnr_tolerance=args.psnr_tolerance,
                  foreground_color_tolerance=args.foreground_color_tolerance,
                  highlight_improvement_tolerance=getattr(args, 'highlight_improvement_tolerance', 1e-8))
    destination = args.work_dir/'best_guard.pth'
    if report['selected']:
        shutil.copy2(report['selected']['checkpoint'], str(destination))
        report['saved_checkpoint'] = str(destination)
        report['saved_checkpoint_sha256'] = file_hash(destination)
    elif destination.exists() or destination.is_symlink():
        # This is a derived file from a previous selection, never an original
        # epoch checkpoint. A baseline win must not leave stale candidate weights.
        stale_hash = file_hash(destination) if destination.is_file() else None
        destination.unlink()
        report['removed_stale_best_guard'] = dict(path=str(destination), sha256=stale_hash,
            reason='baseline won; stale derived candidate must not masquerade as current selection')
    (args.work_dir/'best_guard_selection.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


def read_rgb(path):
    import numpy as np
    from PIL import Image
    with Image.open(str(path)) as image:
        return np.asarray(image.convert('RGB')).copy()


def tensor_rgb(array, device):
    """Match FlexibleRescaleToZeroOne + ImageToTensor + collate exactly.

    Normalize with NumPy float32 on CPU before transfer. CUDA division can
    differ by one ULP, and retaining HWC-derived strides selects another
    convolution path; both are avoidable differences from validation loading.
    """
    import numpy as np
    import torch
    normalized=(array.astype(np.float32)/255.).clip(0,1)
    return torch.from_numpy(normalized).permute(2,0,1)[None].contiguous().to(device)


def save_raw_png(output, path):
    """Use a fresh image, with no source ICC metadata and no in-place clamp."""
    from PIL import Image
    array = (output.detach().clamp(0,1)[0].permute(1,2,0).cpu().numpy()*255.).round().astype('uint8')
    path.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(array,mode='RGB').save(str(path))


def verify_p3_export(raw_directory, p3_directory):
    """Verify the separate p3.py step preserved pixels and embedded its profile."""
    import numpy as np
    from PIL import Image
    from p3 import DISPLAY_P3_ICC
    records=[]
    for raw in sorted(Path(raw_directory).glob('*.png')):
        processed=Path(p3_directory)/raw.name
        with Image.open(str(raw)) as source,Image.open(str(processed)) as target:
            if source.info.get('icc_profile'):
                raise RuntimeError('raw inference unexpectedly embedded ICC: '+str(raw))
            if target.info.get('icc_profile')!=DISPLAY_P3_ICC:
                raise RuntimeError('p3.py did not produce the expected Display P3 profile: '+str(processed))
            if not np.array_equal(np.asarray(source.convert('RGB')),np.asarray(target.convert('RGB'))):
                raise RuntimeError('ICC processing changed pixels: '+str(processed))
        records.append(dict(raw=str(raw),p3=str(processed),raw_sha256=file_hash(raw),
                            p3_sha256=file_hash(processed),pixels_equal=True))
    return records


def prepare_masks(cfg, split, image_path, image, device, mask_root=None):
    import torch
    from adaint.semantic_pipeline import LoadSemanticMasks, semantic_mask_step
    settings = semantic_mask_step(cfg, split, training=False)
    settings.pop('type')
    if mask_root is not None:
        settings['foreground_root'] = str(mask_root)
        settings['foreground_suffix'] = '_face_mask'
        settings['face_root'] = None
    data = LoadSemanticMasks(**settings)(dict(lq_path=str(image_path),lq=image))
    masks = {}
    for name in ('foreground_mask','supervision_mask','face_mask'):
        if name in data:
            masks[name] = torch.from_numpy(data[name]).permute(2,0,1)[None].to(device)
    return masks, data['foreground_mask_path']


def cropped(value, border):
    if border:
        if value.shape[-2]<=2*border or value.shape[-1]<=2*border:
            raise ValueError('crop_border removes the entire image')
        return value[...,border:-border,border:-border]
    return value


def raw_metrics(output, mask, crop_border=0, min_area=.0001):
    import torch
    output,mask=cropped(output,crop_border),cropped(mask,crop_border)
    result = dict(raw_high_clip=float((output>1).any(1).float().mean()),
                  raw_low_clip=float((output<0).any(1).float().mean()),
                  raw_min=float(output.min()),raw_max=float(output.max()),
                  nearwhite=float((output>=250./255.).all(1).float().mean()))
    for name, weights in (('foreground',mask),('background',1-mask)):
        mass=weights.sum()
        valid=float(mass)>0 and float(weights.mean())>=min_area
        result['_valid_'+name]=float(valid)
        if valid:
            result['raw_high_clip_'+name]=float(((output>1).any(1,keepdim=True)*weights).sum()/mass)
            result['nearwhite_'+name]=float(((output>=250./255.).all(1,keepdim=True)*weights).sum()/mass)
        else:
            result['raw_high_clip_'+name]=None
            result['nearwhite_'+name]=None
    return result


def paired_metrics(model, output, target, masks, cfg):
    from adaint.highlight_guard import highlight_over_statistics
    values = model.evaluate(output.clone(),target.clone(),
        foreground_mask=masks['foreground_mask'], supervision_mask=masks['supervision_mask'],
        face_mask=masks.get('face_mask'))
    options = cfg.model.get('highlight_guard',{})
    fraction, margin = options.get('top_fraction',.1), options.get('over_margin',2./255.)
    border=model.test_cfg.get('crop_border',0)
    raw,cropped_target=cropped(output,border),cropped(target,border)
    for name,mask in (('foreground',masks['supervision_mask']),('background',1-masks['supervision_mask'])):
        stat=highlight_over_statistics(raw,cropped_target,cropped(mask,border),fraction,margin,
                                      cfg.model.get('region_min_area',.0001))
        for key in ('over_mean','over_mse','over_fraction','signed_mean'):
            value=stat[key]
            values['highlight_'+key+'_'+name]=float(value)
        values['highlight_over_'+name]=float(stat['over_mean'])
        values['_valid_'+name]=float(stat['valid_count']>0)
    return values


def decomposition_metrics(output, aux, mask, crop_border=0, min_area=.0001):
    import torch
    global_output=mask*aux['global_outputs'][0]+(1-mask)*aux['global_outputs'][1]
    before_dc=mask*aux['branch_outputs'][0]+(1-mask)*aux['branch_outputs'][1]
    return dict(global_raw=raw_metrics(global_output,mask,crop_border,min_area),
                local_raw=raw_metrics(before_dc,mask,crop_border,min_area),
                local_abs_mean=float(cropped(before_dc-global_output,crop_border).abs().mean()),
                dc_abs_mean=float(cropped(output-before_dc,crop_border).abs().mean()))


def evaluate_main(args):
    import cv2
    import numpy as np
    import torch
    from adaint.config_utils import load_config
    from adaint.experiment_utils import inference_config,load_enhancer_checkpoint
    from mmedit.models import build_model
    cfg=load_config(str(args.config))
    cfg=inference_config(cfg)
    device=torch.device(args.device)
    if device.type=='cuda' and args.memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction,device)
    evaluation_cfg=copy.deepcopy(cfg.get('validation_test_cfg') or {})
    supported={'PSNR','deltaE_P3','deltaE_P3_top','deltaE_P3_foreground',
               'deltaE_P3_background','deltaL_P3_foreground','deltaAB_P3_foreground'}
    evaluation_cfg['metrics']=[name for name in evaluation_cfg.get('metrics',[]) if name in supported]
    if not evaluation_cfg['metrics']:
        evaluation_cfg['metrics']=['PSNR','deltaE_P3','deltaE_P3_foreground','deltaE_P3_background']
    evaluation_cfg.setdefault('crop_border',0)
    model=build_model(cfg.model,train_cfg=None,test_cfg=evaluation_cfg)
    load_enhancer_checkpoint(model,str(args.checkpoint))
    model=model.to(device).eval()
    if args.mode=='paired':
        dataset=cfg.data[args.split]
        ids=Path(dataset.ann_file).read_text().split()
        inputs=[Path(dataset.dir_lq)/dataset.filetmpl_lq.format(sid) for sid in ids]
        targets=[Path(dataset.dir_gt)/dataset.filetmpl_gt.format(sid) for sid in ids]
    else:
        ids=args.ids_file.read_text().split() if args.ids_file else [
            p.stem[:-6] for p in sorted(args.input_root.glob('*_input.jpg'),key=lambda p:int(p.stem[:-6]))]
        if len(ids)!=len(set(ids)):
            raise ValueError('duplicate test IDs')
        inputs=[args.input_root/(sid+'_input.jpg') for sid in ids]
        targets=[None]*len(ids)
    if args.limit:
        ids,inputs,targets=ids[:args.limit],inputs[:args.limit],targets[:args.limit]
    args.output.mkdir(parents=True,exist_ok=True)
    raw_dir=args.output/'raw'
    records=[]
    border=evaluation_cfg.get('crop_border',0)
    min_area=cfg.model.get('region_min_area',.0001)
    for sid,image_path,target_path in zip(ids,inputs,targets):
        image=read_rgb(image_path)
        original=image
        target=read_rgb(target_path) if target_path else None
        if target is not None and image.shape[:2]!=target.shape[:2]:
            image=cv2.resize(image,(target.shape[1],target.shape[0]),interpolation=cv2.INTER_AREA)
        masks,mask_path=prepare_masks(cfg,args.split if args.mode=='paired' else 'test',
                                     image_path,image,device,args.mask_root if args.mode=='test' else None)
        tensor=tensor_rgb(image,device)
        with torch.no_grad():
            output,_,_,aux=model.forward_dummy(tensor,True,foreground_mask=masks['foreground_mask'])
            if not torch.isfinite(output).all():
                raise RuntimeError('nonfinite output for image '+sid)
            values=raw_metrics(output,masks['supervision_mask'],border,min_area)
            if target is not None:
                values.update(paired_metrics(model,output,tensor_rgb(target,device),masks,cfg))
            row=dict(id=sid,input=str(image_path),input_sha256=file_hash(image_path),
                     mask=mask_path,mask_sha256=file_hash(mask_path),metrics=values)
            if target_path:
                row.update(gt=str(target_path),gt_sha256=file_hash(target_path))
            if args.decompose:
                row['decomposition']=decomposition_metrics(output,aux,masks['foreground_mask'],border,min_area)
            save_raw_png(output,raw_dir/(sid+'_result.png'))
            if args.mode=='paired' and args.native_paired:
                native_masks,_=prepare_masks(cfg,args.split,image_path,original,device)
                native=model.forward_dummy(tensor_rgb(original,device),
                    foreground_mask=native_masks['foreground_mask'])[0]
                row['native_raw_metrics']=raw_metrics(native,native_masks['supervision_mask'],border,min_area)
                save_raw_png(native,args.output/'native_raw'/(sid+'_result.png'))
                del native,native_masks
        records.append(row)
        print('{} {}/{} {}'.format(args.mode,len(records),len(ids),sid),flush=True)
    try:
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT),stderr=subprocess.DEVNULL).decode().strip()
    except subprocess.CalledProcessError:
        revision='unavailable'
    report=dict(mode=args.mode,split=args.split if args.mode=='paired' else 'blind_test',
        count=len(records),config=str(args.config.resolve()),config_sha256=file_hash(args.config),
        checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=file_hash(args.checkpoint),
        code_revision=revision,device=str(device),torch_version=torch.__version__,
        metric_protocol=dict(crop_border=border,region_min_area=min_area,
            tensor_normalization='CPU NumPy float32 / 255, contiguous NCHW, then device transfer',
            top_fraction=cfg.model.get('highlight_guard',{}).get('top_fraction',.1),
            over_margin=cfg.model.get('highlight_guard',{}).get('over_margin',2./255.)),
        summary=finite_means(records),records=records,
        note='Raw metrics precede non-inplace clamp; canonical paired inference is GT-sized with AREA alignment. '
             'Encoded Display P3 numeric RGB. Blind test has no GT metrics. '
             'Summary is a mean of per-image measurements, not pooled by pixel count.')
    (args.output/'evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    names=sorted({k for row in records for k in row['metrics']})
    with (args.output/'metrics.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['id']+names);writer.writeheader()
        for row in records:writer.writerow(dict(id=row['id'],**row['metrics']))
    if args.p3:
        verification=[]
        for directory in [raw_dir]+([args.output/'native_raw'] if args.mode=='paired' and args.native_paired else []):
            subprocess.check_call([sys.executable,str(ROOT/'p3.py'),str(directory),
                                   '--output',str(directory.with_name(directory.name+'_p3'))])
            verification.extend(verify_p3_export(directory,directory.with_name(directory.name+'_p3')))
        (args.output/'p3_verification.json').write_text(json.dumps(verification,indent=2))
    print(json.dumps(report['summary'],indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='mode');sub.required=True
    for mode in ('paired','test'):
        command=sub.add_parser(mode)
        command.add_argument('--config',type=Path,required=True)
        command.add_argument('--checkpoint',type=Path,required=True)
        command.add_argument('--output',type=Path,required=True)
        command.add_argument('--device',default='cuda:0')
        command.add_argument('--memory-fraction',type=float)
        command.add_argument('--p3',action='store_true')
        command.add_argument('--decompose',action='store_true')
        command.add_argument('--limit',type=int)
        if mode=='paired':
            command.add_argument('--split',choices=('train','val'),default='val')
            command.add_argument('--native-paired',action='store_true')
        else:
            command.add_argument('--input-root',type=Path,required=True)
            command.add_argument('--mask-root',type=Path,required=True)
            command.add_argument('--ids-file',type=Path)
    command=sub.add_parser('select')
    command.add_argument('--work-dir',type=Path,required=True)
    command.add_argument('--baseline',type=Path,required=True)
    command.add_argument('--psnr-tolerance',type=float,default=.1)
    command.add_argument('--foreground-color-tolerance',type=float,default=.15)
    command.add_argument('--highlight-improvement-tolerance',type=float,default=1e-8)
    command.add_argument('--min-relative-improvement',type=float,default=.05)
    command.add_argument('--region-over-tolerance',type=float,default=.5/255.)
    command.add_argument('--global-color-tolerance',type=float,default=.15)
    args=parser.parse_args()
    if args.mode=='select':select_main(args)
    else:evaluate_main(args)


if __name__=='__main__':
    main()
