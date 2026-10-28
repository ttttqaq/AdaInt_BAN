"""Recheck four historical test ROIs, with raw stage metrics and P3 crop exports.

    python tools/diagnose_bounded_cases.py --models-json models.json \
      --output work_dirs/bc_suite/case_review --device cuda:0 --memory-fraction .22

models.json is a list of {"name": "bounded", "config": "...", "checkpoint": "..."}.
This tool never uses test GT, chooses checkpoints, or adjusts ROI coordinates.
The glass coordinates are the exact historical [.01,.01,.44,.56] box from
analysis/highlight_guard_20261027/diagnose_cases_remote.py. Metrics intersect
each fixed box with the original feathered foreground mask >.5 (fg) or <=.5
(bg), matching that audit. Exported crops show the whole box, including pixels
outside that intersection; masks are saved separately to make this explicit.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from PIL import Image, ImageCms, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from evaluate_highlight_guard import (file_hash, read_rgb, tensor_rgb, prepare_masks,
                                     save_raw_png, verify_p3_export)
from p3 import DISPLAY_P3_ICC

CASES = (
    dict(id='30', region='sky', bbox=[0., 0., 1., .25], mask_filter='bg'),
    dict(id='16', region='face_highlight', bbox=[.28, .46, .46, .65], mask_filter='fg'),
    dict(id='17', region='wall', bbox=[.37, .01, .9, .36], mask_filter='bg'),
    dict(id='1', region='glass', bbox=[.01, .01, .44, .56], mask_filter='bg'))
STAGES = ('input', 'global', 'local', 'final')
LUMA = np.asarray([.2126, .7152, .0722], dtype=np.float32)
RESAMPLE = getattr(Image, 'Resampling', Image)


def safe_name(name):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(name)) or 'model'


def p3_lab(rgb):
    """NumPy counterpart of adaint.color.display_p3_to_lab (D65, same matrix).

    Kept independent of Torch for synthetic statistic checks. Like the project
    function, the raw variant uses an extended transfer curve outside [0,1].
    The historical ``mean_lab`` field below instead uses clipped RGB explicitly.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    magnitude = np.abs(rgb)
    linear = np.where(magnitude <= .04045, rgb / 12.92,
                      np.sign(rgb) * ((magnitude + .055) / 1.055) ** 2.4)
    matrix = np.asarray([[608311 / 1250200, 189793 / 714400, 198249 / 1000160],
                         [35783 / 156275, 247089 / 357200, 198249 / 2500400],
                         [0., 32229 / 714400, 5220557 / 5000800]])
    xyz = linear @ matrix.T
    white = np.asarray([.3127 / .3290, 1., (1. - .3127 - .3290) / .3290])
    relative = xyz / white
    threshold = (6. / 29.) ** 3
    f = np.where(relative > threshold, np.maximum(relative, threshold) ** (1. / 3.),
                 relative * (841. / 108.) + 4. / 29.)
    return np.stack((116. * f[..., 1] - 16., 500. * (f[..., 0] - f[..., 1]),
                     200. * (f[..., 1] - f[..., 2])), axis=-1)


def gradient_statistics(rgb, selected):
    """Adjacent-pixel variation, NOT an error/quality metric without test GT."""
    rgb_total, y_total, y_square, edges = 0., 0., 0., 0
    y = rgb @ LUMA
    for axis in (0, 1):
        left, right = [slice(None)] * 2, [slice(None)] * 2
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        valid = selected[tuple(left)] & selected[tuple(right)]
        count = int(valid.sum())
        if not count:
            continue
        rgb_difference = np.diff(rgb, axis=axis)[valid]
        y_difference = np.diff(y, axis=axis)[valid]
        rgb_total += float(np.abs(rgb_difference).sum(dtype=np.float64))
        y_total += float(np.abs(y_difference).sum(dtype=np.float64))
        y_square += float(np.square(y_difference).sum(dtype=np.float64))
        edges += count
    return dict(edge_count=edges, rgb_abs_mean=rgb_total / (3 * edges) if edges else None,
                Y_abs_mean=y_total / edges if edges else None,
                Y_rms=(y_square / edges) ** .5 if edges else None)


def roi_statistics(rgb, selected):
    """Raw HWC RGB statistics, with historic aliases and explicit Lab domains."""
    rgb = np.asarray(rgb, dtype=np.float32)
    selected = np.asarray(selected, dtype=bool)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or selected.shape != rgb.shape[:2]:
        raise ValueError('RGB must be H,W,3 and selection must align with H,W')
    if not np.isfinite(rgb).all():
        raise ValueError('nonfinite raw prediction')
    values = rgb[selected]
    count = int(values.shape[0])
    if not count:
        return dict(count=0, valid=False, note='Empty fixed ROI/mask intersection.')
    clipped = np.clip(values, 0, 1)
    y = values @ LUMA
    clipped_lab = p3_lab(clipped).mean(0).tolist()
    return dict(count=count, valid=True, mean_rgb_raw=values.mean(0).tolist(),
                mean_rgb_clipped=clipped.mean(0).tolist(), mean_y_raw=float(y.mean()),
                mean_y_clipped=float((clipped @ LUMA).mean()),
                mean_lab=clipped_lab, mean_lab_clipped=clipped_lab,
                mean_lab_raw_extended=p3_lab(values).mean(0).tolist(),
                any_gt1=float((values > 1).any(1).mean()),
                any_lt0=float((values < 0).any(1).mean()),
                any_ge250=float((clipped >= 250. / 255.).any(1).mean()),
                nearwhite=float((clipped >= 250. / 255.).all(1).mean()),
                y_std=float(y.std(ddof=0)), raw_min=float(values.min()), raw_max=float(values.max()),
                raw_min_rgb=values.min(0).tolist(), raw_max_rgb=values.max(0).tolist(),
                gradient_raw=gradient_statistics(rgb, selected),
                gradient_clipped=gradient_statistics(np.clip(rgb, 0, 1), selected))


def pixel_box(bbox, width, height):
    x0, y0, x1, y1 = bbox
    box = (int(x0 * width), int(y0 * height), int(x1 * width), int(y1 * height))
    if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
        raise ValueError('fixed bbox is empty or outside image: ' + str(box))
    return box


def resolve_model_path(value, json_parent):
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return path.resolve() if path.exists() else (json_parent / path).resolve()


def load_specs(path):
    specs = json.loads(path.read_text())
    if not isinstance(specs, list) or not specs:
        raise ValueError('--models-json must contain a nonempty list')
    seen, names = set(), set()
    for spec in specs:
        if not isinstance(spec, dict) or any(not spec.get(k) for k in ('name', 'config', 'checkpoint')):
            raise ValueError('each model requires nonempty name/config/checkpoint')
        token = safe_name(spec['name'])
        if spec['name'] in names or token in seen:
            raise ValueError('model names (and sanitized filenames) must be unique')
        names.add(spec['name']); seen.add(token)
        for key in ('config', 'checkpoint'):
            spec[key] = str(resolve_model_path(spec[key], path.parent))
    return specs


def flatten_stats(value, prefix=''):
    result = {}
    for key, item in value.items():
        name = prefix + key
        if isinstance(item, dict):
            result.update(flatten_stats(item, name + '_'))
        elif isinstance(item, list):
            for index, number in enumerate(item):
                result[name + '_' + str(index)] = number
        else:
            result[name] = item
    return result


def contact_sheets(output, specs, rows, cell_width):
    """All columns are converted from exported Display P3 into sRGB by LCMS."""
    p3 = ImageCms.ImageCmsProfile(io.BytesIO(DISPLAY_P3_ICC))
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB'))
    result = []
    for case in CASES:
        case_rows = [row for row in rows if row['id'] == case['id']]
        by_name = {row['variant']: row for row in case_rows}
        paths = [('INPUT', output / 'raw_p3' / ('case{}_input.png'.format(case['id'])))]
        paths += [(spec['name'], Path(by_name[spec['name']]['crops_p3']['final'])) for spec in specs]
        top, cell_height = 76, 440
        canvas = Image.new('RGB', (cell_width * len(paths), top + cell_height), '#f5f6f8')
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 10), 'ID {} / {} / bbox {} / statistics filter: {}'.format(
            case['id'], case['region'], case['bbox'], case['mask_filter']), fill='#16202d')
        draw.text((12, 30), 'Crops show the whole box; statistics intersect fg/bg. Variation is not a quality score.', fill='#586477')
        for index, (label, path) in enumerate(paths):
            with Image.open(str(path)) as source:
                rgb = source.convert('RGB')
                embedded = source.info.get('icc_profile')
                if embedded != DISPLAY_P3_ICC:
                    raise RuntimeError('contact input was not exported by p3.py: ' + str(path))
                rgb = ImageCms.profileToProfile(rgb, p3, srgb, outputMode='RGB')
            available_w, available_h = cell_width - 20, cell_height - 45
            scale = min(available_w / rgb.width, available_h / rgb.height)
            image = rgb.resize((max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))), RESAMPLE.LANCZOS)
            x = index * cell_width
            draw.text((x + 10, top), label.encode('ascii', 'replace').decode(), fill='#16202d')
            canvas.paste(image, (x + (cell_width - image.width) // 2, top + 25 + (available_h - image.height) // 2))
        destination = output / ('case{}_{}_contact_srgb.jpg'.format(case['id'], case['region']))
        canvas.save(str(destination), quality=94, icc_profile=srgb.tobytes())
        result.append(dict(id=case['id'], path=str(destination), color_space='sRGB',
                           columns=[label for label, _ in paths], crop_only=True))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-json', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--memory-fraction', type=float, default=.22)
    parser.add_argument('--input-root', type=Path, default=Path('/DB/data/yujietang/IQ-LUT/face_dataset/test_final'))
    parser.add_argument('--mask-root', type=Path, default=Path('/DB/data/yujietang/IQ-LUT/face_dataset/test_final_masks/masks'))
    parser.add_argument('--contact-width', type=int, default=420)
    args = parser.parse_args()
    if not 0 < args.memory_fraction <= 1 or args.contact_width < 120:
        parser.error('memory fraction must be in (0,1]; contact width must be >=120')
    import torch
    from mmedit.models import build_model
    from adaint.config_utils import load_config
    from adaint.experiment_utils import inference_config, load_enhancer_checkpoint
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    specs = load_specs(args.models_json.resolve())
    output = args.output.resolve()
    raw, masks_directory = output / 'raw', output / 'roi_masks'
    raw.mkdir(parents=True, exist_ok=True); masks_directory.mkdir(parents=True, exist_ok=True)
    rows, reference_masks, input_records = [], {}, {}
    for spec in specs:
        cfg = inference_config(load_config(spec['config']))
        model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        load_enhancer_checkpoint(model, spec['checkpoint'])
        model = model.to(device).eval()
        checkpoint_hash, config_hash = file_hash(spec['checkpoint']), file_hash(spec['config'])
        for case in CASES:
            path = args.input_root / (case['id'] + '_input.jpg')
            image = read_rgb(path)
            masks, mask_path = prepare_masks(cfg, 'test', path, image, device, args.mask_root)
            foreground = masks['foreground_mask']
            x = tensor_rgb(image, device)
            height, width = image.shape[:2]
            x0, y0, x1, y1 = pixel_box(case['bbox'], width, height)
            crop = (slice(None), slice(None), slice(y0, y1), slice(x0, x1))
            local_mask = foreground[crop]
            selection = local_mask[0, 0].detach().cpu().numpy()
            selection = selection > .5 if case['mask_filter'] == 'fg' else selection <= .5
            selection_hash = hashlib.sha256(np.ascontiguousarray(selection).tobytes()).hexdigest()
            if case['id'] in reference_masks and reference_masks[case['id']] != selection_hash:
                raise RuntimeError('model configurations produce different ROI mask intersections: ID ' + case['id'])
            reference_masks[case['id']] = selection_hash
            with torch.no_grad():
                final, _, _, aux = model.forward_dummy(x, True, foreground_mask=foreground)
                # Fuse cropped branches instead of allocating extra full-resolution
                # diagnostic tensors. The soft mask is identical to forward fusion.
                global_crop = local_mask * aux['global_outputs'][0][crop] + (1 - local_mask) * aux['global_outputs'][1][crop]
                local_crop = local_mask * aux['branch_outputs'][0][crop] + (1 - local_mask) * aux['branch_outputs'][1][crop]
                stages = {'input': x[crop], 'global': global_crop, 'local': local_crop, 'final': final[crop]}
                statistics, paths = {}, {}
                for stage in STAGES:
                    tensor = stages[stage]
                    values = tensor[0].permute(1, 2, 0).detach().cpu().numpy()
                    statistics[stage] = roi_statistics(values, selection)
                    filename = ('case{}_input.png'.format(case['id']) if stage == 'input' else
                                'case{}_{}_{}.png'.format(case['id'], safe_name(spec['name']), stage))
                    destination = raw / filename
                    if stage != 'input' or case['id'] not in input_records:
                        save_raw_png(tensor, destination)
                    paths[stage] = str(destination)
            input_hash = file_hash(path)
            if case['id'] in input_records and input_records[case['id']]['sha256'] != input_hash:
                raise RuntimeError('input changed during audit: ' + str(path))
            input_records[case['id']] = dict(path=str(path.resolve()), sha256=input_hash)
            mask_output = masks_directory / ('case{}_selection.png'.format(case['id']))
            Image.fromarray(selection.astype(np.uint8) * 255, mode='L').save(str(mask_output))
            row = dict(variant=spec['name'], id=case['id'], region=case['region'], bbox=case['bbox'],
                       pixel_bbox=[x0, y0, x1, y1], image_size=[width, height], mask_filter=case['mask_filter'],
                       mask=str(mask_path), mask_sha256=file_hash(mask_path), selection_sha256=selection_hash,
                       selection_mask=str(mask_output), input=input_records[case['id']],
                       config=spec['config'], config_sha256=config_hash, checkpoint=spec['checkpoint'],
                       sha256=checkpoint_hash, stages=statistics, crops_raw=paths,
                       crops_p3={key: str(output / 'raw_p3' / Path(value).name) for key, value in paths.items()})
            rows.append(row)
            print('{} ID {} {} selected={}'.format(spec['name'], case['id'], case['region'], int(selection.sum())), flush=True)
            del stages, tensor, values, final, aux, global_crop, local_crop, x, foreground, local_mask, masks, image
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    subprocess.check_call([sys.executable, str(ROOT / 'p3.py'), str(raw), '--output', str(output / 'raw_p3')])
    verification = verify_p3_export(raw, output / 'raw_p3')
    sheets = contact_sheets(output, specs, rows, args.contact_width)
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        revision = 'unavailable'
    report = dict(note='Fixed historical test ROIs, no GT, no checkpoint selection or tuning. '
                       'Raw values precede clamp; all-channel nearwhite is different from any-channel overflow. '
                       'Y standard deviation and gradients describe variation, not correctness or recovered detail.',
        protocol=dict(bbox='Normalized [x0,y0,x1,y1]; integer floor endpoints, same as historical audit.',
                      mask='Statistics intersect ROI with feathered foreground_mask>.5 (fg) or <=.5 (bg).',
                      luminance='Encoded Y=.2126R+.7152G+.0722B, not physical linear luminance.',
                      lab='mean_lab / mean_lab_clipped use clipped RGB in Display P3/D65 Lab; raw extended Lab is separate.',
                      gradients='Mean absolute/RMS first differences; both adjacent pixels inside ROI selection.',
                      exports='Crop PNG raw has no ICC; P3 copies generated only by separate p3.py call; decoded pixels verified equal.',
                      contacts='Only exported crop copies, ImageCms Display P3 to sRGB; no inference resizing.'),
        code_revision=revision, device=str(device), memory_fraction=args.memory_fraction,
        models=specs, cases=list(CASES), rows=rows, p3_verification=verification, contact_sheets=sheets)
    (output / 'case_roi_comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    csv_rows = []
    for row in rows:
        for stage, statistics in row['stages'].items():
            item = dict(variant=row['variant'], id=row['id'], region=row['region'], stage=stage,
                        bbox=json.dumps(row['bbox']), checkpoint=row['checkpoint'], checkpoint_sha256=row['sha256'])
            item.update(flatten_stats(statistics)); csv_rows.append(item)
    fields = ['variant', 'id', 'region', 'stage', 'bbox', 'checkpoint', 'checkpoint_sha256']
    fields += sorted({key for row in csv_rows for key in row} - set(fields))
    with (output / 'case_roi_comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(csv_rows)
    print('Saved {}'.format(output / 'case_roi_comparison.json'), flush=True)


if __name__ == '__main__':
    main()
