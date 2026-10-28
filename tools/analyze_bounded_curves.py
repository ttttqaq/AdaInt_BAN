"""Build an offline, ICC-aware image comparison report from saved evaluations.

Example::

    python tools/analyze_bounded_curves.py --suite work_dirs/bc_suite \
        --output work_dirs/bc_suite/report \
        --models legacy-from-scratch=eval/legacy bounded=eval/bounded detail=eval/detail

Model directories contain train/val/test/evaluation.json. Alternatively use
suite/report_manifest.json, e.g. {"models": [{"name": "bounded", "dir": "eval/bounded"}]}.
A model can provide {"splits": {"val": "some/evaluation/directory"}} instead.
Relative manifest/CLI model paths are relative to --suite. --path-map OLD=NEW
can relocate absolute source-image paths in evaluations copied from a server.

Raw overflow diagnostics are copied ONLY from evaluation.json. Additional
GT-top-decile errors use decoded, already clipped/quantized PNG code values;
they cannot measure raw overflow. Native outputs are displayed and aggregated
separately and never resized to manufacture paired GT metrics.
"""
import argparse
import csv
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import quote

import numpy as np
from PIL import Image, ImageCms, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from p3 import DISPLAY_P3_ICC

RESAMPLE = getattr(Image, 'Resampling', Image)
LUMA = np.asarray([.2126, .7152, .0722], dtype=np.float32)
SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB'))
P3 = ImageCms.ImageCmsProfile(io.BytesIO(DISPLAY_P3_ICC))


def safe_name(value):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value)) or 'image'


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def means(records):
    result = {}
    for key in sorted({k for row in records for k in row}):
        values = [row.get(key) for row in records if finite(row.get(key))]
        if values:
            result[key] = float(sum(values) / len(values))
    return result


def read_rgb(path):
    with Image.open(str(path)) as source:
        return np.asarray(source.convert('RGB')).copy()


def highlight_metrics(prediction, target, mask=None, top_fraction=.1, min_area=.0001):
    """Symmetric PNG errors on GT-selected regions, macro-reduced by caller.

    All ties at the GT quantile are retained. Gradient MAE uses RGB first
    differences whose BOTH endpoints are in the selected high-GT region.
    RGB/gradient errors and signed encoded-Y errors are in 8-bit code units.
    """
    if prediction.shape != target.shape:
        raise ValueError('paired PNG and GT dimensions differ; no implicit resizing')
    pred, gt = prediction.astype(np.float32), target.astype(np.float32)
    target_y = gt @ LUMA
    masks = {'global': np.ones(target_y.shape, dtype=bool)}
    if mask is not None:
        masks.update(foreground=mask.astype(bool), background=~mask.astype(bool))
    result = {}
    for name, region in masks.items():
        mass = int(region.sum())
        if not mass or mass / region.size < min_area:
            continue
        threshold = float(np.quantile(target_y[region], 1. - top_fraction))
        selected = region & (target_y >= threshold)
        count = int(selected.sum())
        difference = pred - gt
        prefix = 'png_highlight_' + name + '_'
        result[prefix + 'pixel_count'] = count
        result[prefix + 'rgb_mae_code255'] = float(np.abs(difference[selected]).mean())
        result[prefix + 'signed_Y_code255'] = float((difference @ LUMA)[selected].mean())
        result[prefix + 'Y_mae_code255'] = float(np.abs((difference @ LUMA)[selected]).mean())
        result[prefix + 'nearwhite_fraction'] = float((pred[selected] >= 250).all(-1).mean())
        result[prefix + 'gt_nearwhite_fraction'] = float((gt[selected] >= 250).all(-1).mean())
        result[prefix + 'gt_threshold_Y_code255'] = threshold
        edge_sum, edge_count = 0., 0
        for axis in (0, 1):
            left, right = [slice(None)] * 2, [slice(None)] * 2
            left[axis], right[axis] = slice(None, -1), slice(1, None)
            edges = selected[tuple(left)] & selected[tuple(right)]
            if edges.any():
                gradient_error = np.diff(pred, axis=axis) - np.diff(gt, axis=axis)
                edge_sum += float(np.abs(gradient_error[edges]).sum(dtype=np.float64))
                edge_count += int(edges.sum()) * 3
        result[prefix + 'gradient_mae_code255'] = edge_sum / edge_count if edge_count else None
        result[prefix + 'gradient_scalar_count'] = edge_count
    return result


class Assets:
    """Small thumbnails plus original-P3 links; at most one source is decoded."""
    def __init__(self, output):
        self.output = output
        self.cache = {}
        self.warnings = []
        self.pending_p3 = False
        for directory in ('thumbs_srgb', 'source_raw', 'originals_p3', 'contact_sheets'):
            (output / directory).mkdir(parents=True, exist_ok=True)

    def href(self, path):
        return quote(os.path.relpath(str(path), str(self.output)), safe='/')

    def image(self, path):
        if path is None or not path.is_file():
            return None
        key = str(path.resolve())
        if key in self.cache:
            return self.cache[key]
        stat = path.stat()
        # Retraining can replace a PNG at the same path. A report rerun must
        # never retain an older metadata-only wrapper under that pathname.
        token = hashlib.sha256('{}:{}:{}'.format(key, stat.st_size, stat.st_mtime_ns).encode()).hexdigest()[:18]
        thumb = self.output / 'thumbs_srgb' / (token + '.jpg')
        original = path
        with Image.open(str(path)) as source:
            embedded = source.info.get('icc_profile')
            size = list(source.size)
            rgb = source.convert('RGB')
            if embedded:
                try:
                    profile = ImageCms.ImageCmsProfile(io.BytesIO(embedded))
                except Exception as exc:
                    raise ValueError('invalid ICC profile in {}: {}'.format(path, exc))
            else:
                profile = P3
                # The report follows the project's independent p3.py export
                # contract too. Stage untagged source copies here; one batch
                # invocation below creates all P3 copies without modifying sources.
                suffix = path.suffix.lower()
                suffix = suffix if suffix in ('.jpg', '.jpeg', '.png', '.bmp') else '.png'
                staged = self.output / 'source_raw' / (token + suffix)
                original = self.output / 'originals_p3' / staged.name
                if not original.exists():
                    if path.suffix.lower() == suffix:
                        shutil.copy2(str(path), str(staged))
                    else:
                        rgb.save(str(staged))
                    self.pending_p3 = True
                self.warnings.append('Assumed Display P3 for untagged source: ' + str(path))
            rgb.thumbnail((300, 220), RESAMPLE.LANCZOS)
            rgb = ImageCms.profileToProfile(rgb, profile, SRGB, outputMode='RGB')
            rgb.save(str(thumb), quality=90, icc_profile=SRGB.tobytes())
        result = dict(source=str(path), original=self.href(original), thumb=self.href(thumb),
                      thumbnail_path=str(thumb), size=size,
                      source_icc='embedded' if embedded else 'assumed Display P3')
        self.cache[key] = result
        return result

    def finalize(self):
        """Generate source P3 copies only through the repository's p3.py."""
        if self.pending_p3:
            subprocess.check_call([sys.executable, str(ROOT / 'p3.py'),
                str(self.output / 'source_raw'), '--output', str(self.output / 'originals_p3')],
                stdout=subprocess.DEVNULL)
            for item in self.cache.values():
                if item['source_icc'] == 'assumed Display P3':
                    # A failing batch must not leave a report linking absent files.
                    from urllib.parse import unquote
                    destination = self.output / unquote(item['original'])
                    if not destination.is_file():
                        raise RuntimeError('p3.py did not create source copy: ' + str(destination))


def resolve_path(value, base, mappings):
    if not value:
        return None
    value = str(value)
    for old, new in sorted(mappings, key=lambda item: -len(item[0])):
        if value == old or value.startswith(old.rstrip('/') + '/'):
            value = new.rstrip('/') + value[len(old.rstrip('/')):]
            break
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def model_specs(args, manifest):
    specifications = manifest.get('models', [])
    if isinstance(specifications, dict):
        specifications = [dict(value, name=name) if isinstance(value, dict)
                          else dict(name=name, dir=value) for name, value in specifications.items()]
    result = list(specifications)
    if args.models:
        result = []
        for specification in args.models:
            if '=' not in specification:
                raise ValueError('--models entries must be name=directory')
            name, directory = specification.split('=', 1)
            result.append(dict(name=name, dir=directory))
    if not result:
        raise ValueError('provide --models or a report_manifest.json with models')
    names = [spec['name'] for spec in result]
    if len(names) != len(set(names)):
        raise ValueError('model names must be unique')
    return result


def load_evaluations(specs, suite, splits, mappings, warnings):
    evaluations = {}
    for spec in specs:
        name = spec['name']
        for split in splits:
            directory = spec.get('splits', {}).get(split)
            if directory is None and spec.get('dir') is not None:
                directory = str(Path(spec['dir']) / split)
            if directory is None:
                continue
            directory = resolve_path(directory, suite, mappings)
            path = directory if directory.name == 'evaluation.json' else directory / 'evaluation.json'
            if not path.exists():
                warnings.append('Missing evaluation: ' + str(path))
                continue
            report = json.loads(path.read_text())
            if split in ('train', 'val') and report.get('mode') == 'test':
                raise ValueError('test evaluation assigned to paired split: ' + str(path))
            rows = report.get('records') or []
            ids = [str(row['id']) for row in rows]
            if len(ids) != len(set(ids)):
                raise ValueError('duplicate image IDs: ' + str(path))
            evaluations[(name, split)] = dict(directory=path.parent, report=report,
                                               records={str(row['id']): row for row in rows})
    return evaluations


def output_image(evaluation, record, native=False):
    directory = evaluation['directory']
    key = 'native_output' if native else 'output'
    candidates = []
    if record.get(key):
        candidates.append(Path(record[key]))
    folder = 'native_raw' if native else 'raw'
    for parent in (folder + '_p3', folder):
        candidates.append(directory / parent / (str(record['id']) + '_result.png'))
    return next((p for p in candidates if p.is_file()), None)


def downsample_code(path):
    with Image.open(str(path)) as source:
        return np.asarray(source.convert('RGB').resize((96, 96), RESAMPLE.BILINEAR), dtype=np.float32)


def summarize(rows, specs):
    result = []
    for split, resolution in sorted({(row['split'], row['resolution']) for row in rows}):
        for spec in specs:
            name = spec['name']
            items = [row['models'][name] for row in rows
                     if row['split'] == split and row['resolution'] == resolution and name in row['models']]
            if not items:
                continue
            raw = means([item['raw_metrics'] for item in items])
            png = means([item.get('png_metrics', {}) for item in items])
            result.append(dict(model=name, split=split, resolution=resolution, count=len(items),
                               raw_evaluation=raw, clipped_png_diagnostics=png))
    return result


def write_csv(path, records):
    keys = sorted({key for row in records for key in row})
    first = [key for key in ('split', 'resolution', 'id', 'model', 'count') if key in keys]
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=first + [key for key in keys if key not in first])
        writer.writeheader()
        writer.writerows(clean(records))


def flat_summary(summary):
    result = []
    for item in summary:
        row = {key: item[key] for key in ('model', 'split', 'resolution', 'count')}
        row.update({'eval_' + key: value for key, value in item['raw_evaluation'].items()})
        row.update(item['clipped_png_diagnostics'])
        result.append(row)
    return result


def contact_sheets(rows, model_names, assets):
    sheets = []
    for split, resolution in sorted({(r['split'], r['resolution']) for r in rows}):
        group = [r for r in rows if (r['split'], r['resolution']) == (split, resolution)]
        columns = ['input'] + (['gt'] if resolution == 'paired' and split != 'test' else []) + model_names
        width, height = 230, 190
        for start in range(0, len(group), 10):
            page = group[start:start + 10]
            canvas = Image.new('RGB', (width * len(columns), 30 + height * len(page)), '#f4f5f7')
            draw = ImageDraw.Draw(canvas)
            for index, name in enumerate(columns):
                draw.text((index * width + 8, 8), name.encode('ascii', 'replace').decode(), fill='black')
            for row_number, row in enumerate(page):
                for column, name in enumerate(columns):
                    image = row.get(name) if name in ('input', 'gt') else row['models'].get(name, {}).get('image')
                    x, y = column * width, 30 + row_number * height
                    draw.text((x + 8, y + 2), '{} / {}'.format(split, row['id']), fill='black')
                    if image:
                        with Image.open(image['thumbnail_path']) as source:
                            thumb = source.copy()
                        thumb.thumbnail((width - 12, height - 25), RESAMPLE.LANCZOS)
                        canvas.paste(thumb, (x + (width - thumb.width) // 2, y + 22))
                    else:
                        draw.text((x + 8, y + 45), 'missing', fill='#8b2635')
            path = assets.output / 'contact_sheets' / '{}_{}_{:02d}.jpg'.format(split, resolution, start // 10 + 1)
            canvas.save(str(path), quality=90, icc_profile=SRGB.tobytes())
            sheets.append(dict(split=split, resolution=resolution, href=assets.href(path),
                               first_id=page[0]['id'], last_id=page[-1]['id']))
    return sheets


def html_report(path, payload):
    # JSON is data, including server paths and labels; never interpolate it as HTML.
    encoded = json.dumps(clean(payload), ensure_ascii=False).replace('<', '\\u003c')
    template = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Bounded curves · 图像对照</title>
<style>
:root{color-scheme:light}body{font:14px/1.55 system-ui,sans-serif;margin:0;background:#f5f6f8;color:#17202c}
header,main{max-width:1800px;margin:auto;padding:24px}header{padding-bottom:8px}h1{font-size:27px;margin:0 0 10px}
.note{color:#4b586a;max-width:1250px}.bar{position:sticky;top:0;background:#ffffffed;padding:12px 24px;z-index:2;box-shadow:0 1px 5px #ccd}
select,input,button{font:inherit;padding:5px;margin:3px;border:1px solid #b7c2cf;border-radius:5px}label{margin-right:12px}
table{border-collapse:collapse;width:100%;background:white;font-variant-numeric:tabular-nums}th,td{padding:8px;border-bottom:1px solid #dce2eb;text-align:left}
.scroll{overflow:auto}.sample{background:white;padding:16px;border-radius:9px;margin:18px 0;border:1px solid #dde3ec}.sample h3{margin:0 0 10px}
.grid{display:grid;grid-template-columns:repeat(var(--cols),minmax(160px,1fr));gap:10px;overflow:auto}.tile{min-width:160px;text-align:center}.tile img{width:100%;height:210px;object-fit:contain;background:#edf0f5;border-radius:5px}.tile small{display:block;color:#5e6b7e}
.tag{font-size:12px;padding:3px 7px;background:#e8eef9;border-radius:4px}.selected{border-left:4px solid #3267c0}.metrics{font-size:12px;white-space:pre-wrap;background:#f5f7fb;padding:8px;text-align:left;overflow-wrap:anywhere}a{color:#205caa}summary{cursor:pointer}.missing{color:#9b2b37}
.sheet{display:inline-block;margin:5px;padding:6px;border:1px solid #cdd5df;border-radius:5px}.overlay{display:none;position:fixed;inset:0;background:#101826ed;z-index:5;padding:3vh;overflow:auto}.overlay.open{display:block}.overlay img{max-width:100%;display:block;margin:auto}.overlay button{position:sticky;top:0}.hint{color:#788394}.small{font-size:12px}
</style><header><h1>Bounded curves · 逐图对照</h1>
<p class="note">裁剪前越界率、PSNR 与 P3 色差来自保存的 evaluation.json；新增亮部误差来自已裁剪量化 PNG。两类指标单独呈现。GT 最亮 10% 同时检查双侧 RGB MAE、带符号亮度误差与梯度误差，避免把压暗当作恢复。native 与 paired 分页显示。</p>
<p class="note">联系表缩略图已通过 ImageCms 转为 sRGB；点击图片查看带源 ICC 的原图。未标记源图先复制到 source_raw，再独立调用仓库 p3.py 生成 originals_p3，报告工具不自行嵌入 P3 ICC。浏览器需启用色彩管理。测试集没有 GT，不显示对 GT 的数值结论。</p>
</header><div class="bar"><label>划分 <select id="split"><option value="all">全部</option></select></label><label>分辨率 <select id="resolution"><option value="paired">paired / test</option><option value="native">native（独立）</option><option value="all">全部</option></select></label><label>ID <input id="search" placeholder="例如 1,16,17,30"></label><label>排序 <select id="sort"><option value="id">ID</option><option value="worst">亮部误差最大</option><option value="change">变化最大</option></select></label><label><input type="checkbox" id="selected">仅重点图</label><span id="count"></span><div id="models"></div></div>
<main><h2>汇总</h2><p><a href="summary.json">summary.json</a> · <a href="summary.csv">summary.csv</a> · <a href="per_image_metrics.csv">逐图指标 CSV</a></p><div class="scroll" id="summary"></div><h2>全图联系表</h2><div id="sheets"></div><h2>全部图像</h2><div id="samples"></div><details><summary>来源与限制</summary><pre id="protocol" class="metrics"></pre><pre id="warnings" class="metrics"></pre></details></main><div id="overlay" class="overlay"><button onclick="document.querySelector('#overlay').classList.remove('open')">关闭</button><img id="full"></div>
<script id="data" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.querySelector('#data').textContent),$=s=>document.querySelector(s),esc=s=>String(s??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),fmt=(v,n=4)=>Number.isFinite(v)?v.toFixed(n):'—';
const names=D.models.map(m=>m.name);for(const s of [...new Set(D.rows.map(r=>r.split))])$('#split').insertAdjacentHTML('beforeend',`<option>${esc(s)}</option>`);
for(const n of names)$('#models').insertAdjacentHTML('beforeend',`<label><input type="checkbox" class="model" value="${esc(n)}" checked>${esc(n)}</label>`);
function tile(label,img,metrics,extra=''){if(!img)return `<div class="tile"><b>${esc(label)}</b><p class="missing">缺少图像</p>${extra}</div>`;return `<div class="tile"><b>${esc(label)}</b><a href="${esc(img.original)}" target="_blank"><img loading="lazy" src="${esc(img.thumb)}"></a><small>${img.size.join(' × ')} · <a href="${esc(img.original)}" target="_blank">原图 / P3</a></small>${extra}${metrics?`<div class="metrics">${esc(metrics)}</div>`:''}</div>`}
function brief(m){if(!m)return'';const r=m.raw_metrics,p=m.png_metrics||{};let a=[`raw 越上界 ${fmt(r.raw_high_clip*100,4)}% / 越下界 ${fmt(r.raw_low_clip*100,4)}%`];if(r.PSNR!=null)a.push(`PSNR ${fmt(r.PSNR,3)} dB · 前景 ΔE ${fmt(r.deltaE_P3_foreground,3)}`);for(const region of ['foreground','background','global']){const k='png_highlight_'+region+'_';if(p[k+'rgb_mae_code255']!=null)a.push(`${region} GT亮部: RGB MAE ${fmt(p[k+'rgb_mae_code255'],3)} / signed Y ${fmt(p[k+'signed_Y_code255'],3)} / 梯度 MAE ${fmt(p[k+'gradient_mae_code255'],3)} / 近白 ${fmt(p[k+'nearwhite_fraction']*100,2)}%`)}return a.join('\n')}
function render(){const active=[...document.querySelectorAll('.model:checked')].map(e=>e.value),s=$('#split').value,res=$('#resolution').value,query=$('#search').value.trim().split(/[ ,，]+/).filter(Boolean);let rows=D.rows.filter(r=>(s==='all'||r.split===s)&&(res==='all'||r.resolution===res)&&(!query.length||query.includes(r.id))&&(!$('#selected').checked||r.selected));const mode=$('#sort').value;rows.sort((a,b)=>mode==='id'?(a.split.localeCompare(b.split)||a.id.localeCompare(b.id,undefined,{numeric:true})):(b[mode+'_score']||0)-(a[mode+'_score']||0));$('#count').textContent=rows.length+' 张';$('#samples').innerHTML=rows.map(r=>{const hasgt=r.resolution==='paired'&&r.gt;let tiles=tile('输入',r.input,'');if(hasgt)tiles+=tile('GT',r.gt,'');for(const name of active){const m=r.models[name];tiles+=tile(name,m&&m.image,brief(m))}return `<section class="sample ${r.selected?'selected':''}" id="${esc(r.anchor)}"><h3>${esc(r.split)} · ID ${esc(r.id)} <span class="tag">${esc(r.resolution)}</span> ${r.selected?'<span class="tag">重点</span>':''}</h3>${r.notes.length?`<p class="small missing">${esc(r.notes.join(' · '))}</p>`:''}<div class="grid" style="--cols:${active.length+1+(hasgt?1:0)}">${tiles}</div><details ${r.selected?'open':''}><summary>完整数值与来源</summary><pre class="metrics">${esc(JSON.stringify(r.models,null,2))}</pre></details></section>`}).join('');
const stats=D.summary.filter(r=>(s==='all'||r.split===s)&&(res==='all'||r.resolution===res)&&active.includes(r.model));$('#summary').innerHTML='<table><thead><tr><th>划分 / 分辨率</th><th>模型</th><th>数量</th><th>PSNR</th><th>前景 ΔE</th><th>raw 越上界 %</th><th>前景亮部 RGB MAE</th><th>前景 signed Y</th><th>前景梯度 MAE</th></tr></thead><tbody>'+stats.map(r=>{let p=r.clipped_png_diagnostics,v=r.raw_evaluation,k='png_highlight_foreground_';return `<tr><td>${esc(r.split+' / '+r.resolution)}</td><td>${esc(r.model)}</td><td>${r.count}</td><td>${fmt(v.PSNR,3)}</td><td>${fmt(v.deltaE_P3_foreground,3)}</td><td>${fmt(v.raw_high_clip*100,4)}</td><td>${fmt(p[k+'rgb_mae_code255'],3)}</td><td>${fmt(p[k+'signed_Y_code255'],3)}</td><td>${fmt(p[k+'gradient_mae_code255'],3)}</td></tr>`}).join('')+'</tbody></table>';
$('#sheets').innerHTML=D.contact_sheets.filter(r=>(s==='all'||r.split===s)&&(res==='all'||r.resolution===res)).map(r=>`<a class="sheet" target="_blank" href="${esc(r.href)}">${esc(r.split+' / '+r.resolution+' · '+r.first_id+'–'+r.last_id)}</a>`).join('')}
for(const e of document.querySelectorAll('input,select'))e.addEventListener('input',render);$('#protocol').textContent=JSON.stringify(D.protocol,null,2);$('#warnings').textContent=D.warnings.join('\n');render();
</script></html>'''
    path.write_text(template.replace('__DATA__', encoded), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', nargs='+', metavar='NAME=DIR')
    parser.add_argument('--splits', nargs='+', default=None)
    parser.add_argument('--path-map', action='append', default=[], metavar='OLD=NEW')
    parser.add_argument('--reference', help='reference model for thumbnail-change ranking')
    parser.add_argument('--candidate', help='candidate model for thumbnail-change ranking')
    parser.add_argument('--detail-count', type=int, default=6)
    parser.add_argument('--detail-ids', nargs='*', default=['1', '16', '17', '30'])
    parser.add_argument('--top-fraction', type=float, default=.1)
    args = parser.parse_args()
    if not 0 < args.top_fraction <= 1 or args.detail_count < 0:
        parser.error('top fraction must be in (0,1]; detail count must be nonnegative')
    suite, output = args.suite.resolve(), args.output.resolve()
    manifest_path = suite / 'report_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    specs = model_specs(args, manifest)
    names = [spec['name'] for spec in specs]
    reference, candidate = args.reference or names[0], args.candidate or ('detail' if 'detail' in names else names[-1])
    if reference not in names or candidate not in names:
        parser.error('reference and candidate must be named models')
    splits = args.splits or manifest.get('splits') or ['train', 'val', 'test']
    mapping_data = manifest.get('path_mappings', {})
    mappings = list(mapping_data.items()) if isinstance(mapping_data, dict) else list(mapping_data)
    for item in args.path_map:
        if '=' not in item:
            parser.error('--path-map requires OLD=NEW')
        mappings.append(tuple(item.split('=', 1)))
    warnings = []
    evaluations = load_evaluations(specs, suite, splits, mappings, warnings)
    if not evaluations:
        raise ValueError('no evaluation.json files found')
    output.mkdir(parents=True, exist_ok=True)
    assets, rows, per_image = Assets(output), [], []
    for split in splits:
        ids = sorted({sid for (name, part), ev in evaluations.items() if part == split
                      for sid in ev['records']}, key=lambda sid: (not sid.isdigit(), int(sid) if sid.isdigit() else sid))
        for sid in ids:
            available = [(name, evaluations[(name, split)]) for name in names
                         if (name, split) in evaluations and sid in evaluations[(name, split)]['records']]
            source_record = available[0][1]['records'][sid]
            source_base = available[0][1]['directory']
            input_path = resolve_path(source_record.get('input'), source_base, mappings)
            gt_path = resolve_path(source_record.get('gt'), source_base, mappings)
            mask_path = resolve_path(source_record.get('mask'), source_base, mappings)
            target = read_rgb(gt_path) if gt_path and gt_path.is_file() and split != 'test' else None
            mask = None
            if target is not None and mask_path and mask_path.is_file():
                with Image.open(str(mask_path)) as image:
                    mask = np.asarray(image.convert('L').resize((target.shape[1], target.shape[0]), RESAMPLE.NEAREST)) > 127
            for native in (False, True):
                if native and not any(output_image(ev, ev['records'][sid], True) for _, ev in available):
                    continue
                resolution = 'native' if native else 'paired'
                row = dict(id=sid, split=split, resolution=resolution,
                           anchor='{}-{}-{}'.format(safe_name(split), resolution, safe_name(sid)),
                           input=assets.image(input_path), gt=assets.image(gt_path) if target is not None else None,
                           models={}, notes=[], worst_score=0., change_score=0., selected=False)
                if native:
                    row['notes'].append('原生输入分辨率输出；独立 raw 指标，不计算与 paired GT 的 PNG 误差。')
                if input_path is None or not input_path.is_file():
                    row['notes'].append('输入图路径不可用；可使用 --path-map 映射复制后的源数据。')
                if split != 'test' and target is None:
                    row['notes'].append('GT路径不可用；未计算PNG亮部误差，请核对源数据映射。')
                small = {}
                for name, ev in available:
                    record = ev['records'][sid]
                    for key in ('input_sha256', 'gt_sha256', 'mask_sha256'):
                        if record.get(key) and source_record.get(key) and record[key] != source_record[key]:
                            raise ValueError('{} {} source mismatch ({}) between models'.format(split, sid, key))
                    image_path = output_image(ev, record, native)
                    if native and image_path is None:
                        continue
                    if image_path is None:
                        row['notes'].append('{} 缺少导出PNG；raw数值仍取自evaluation.json。'.format(name))
                    raw = record.get('native_raw_metrics', {}) if native else record.get('metrics', {})
                    item = dict(image=assets.image(image_path), raw_metrics=raw,
                                png_metrics={}, evaluation=str(ev['directory'] / 'evaluation.json'),
                                checkpoint=ev['report'].get('checkpoint'),
                                checkpoint_sha256=ev['report'].get('checkpoint_sha256'))
                    if image_path is not None:
                        small[name] = downsample_code(image_path)
                    if target is not None and not native and image_path is not None:
                        prediction = read_rgb(image_path)
                        if prediction.shape == target.shape:
                            border = int(ev['report'].get('metric_protocol', {}).get('crop_border', 0))
                            if border and min(target.shape[:2]) <= 2 * border:
                                raise ValueError('crop_border removes whole image')
                            sl = (slice(border, -border), slice(border, -border)) if border else (slice(None), slice(None))
                            item['png_metrics'] = highlight_metrics(prediction[sl], target[sl],
                                mask[sl] if mask is not None else None, args.top_fraction,
                                ev['report'].get('metric_protocol', {}).get('region_min_area', .0001))
                        else:
                            row['notes'].append('{} paired output与GT尺寸不同，未计算PNG配对指标。'.format(name))
                        del prediction
                    row['models'][name] = item
                    scores = [value for key, value in item['png_metrics'].items()
                              if key.endswith('rgb_mae_code255') and finite(value)]
                    if scores:
                        row['worst_score'] = max(row['worst_score'], max(scores))
                    flat = dict(split=split, resolution=resolution, id=sid, model=name)
                    flat.update({'eval_' + key: value for key, value in raw.items()})
                    flat.update(item['png_metrics'])
                    per_image.append(flat)
                if reference in small and candidate in small:
                    row['change_score'] = float(np.abs(small[candidate] - small[reference]).mean())
                rows.append(row)
            del target, mask
            print('{} {}: {} models'.format(split, sid, len(available)), flush=True)
    for split, resolution in {(row['split'], row['resolution']) for row in rows}:
        group = [row for row in rows if (row['split'], row['resolution']) == (split, resolution)]
        selected = set(args.detail_ids)
        for field in ('worst_score', 'change_score'):
            selected.update(row['id'] for row in sorted(group, key=lambda r: -r[field])[:args.detail_count])
        for row in group:
            row['selected'] = row['id'] in selected
    summary = summarize(rows, specs)
    protocol = dict(raw='Copied from evaluation.json; never inferred from clipped PNG.',
        png='Decoded encoded-RGB code values, already clipped and quantized. No ICC pixel transform for metrics.',
        highlights='Each GT image/region quantile; all ties retained; per-image means, then equal-image macro means.',
        png_units='RGB MAE, signed Y and first-gradient MAE in 0..255 code units; nearwhite is a fraction.',
        luminance='Encoded Y=.2126R+.7152G+.0722B; not physical linear luminance.',
        gradients='RGB first differences; both endpoints must lie inside the selected GT highlight region.',
        native='Separate resolution group; no resized GT metrics are manufactured.',
        thumbnails='ImageCms source embedded ICC (or explicit assumed Display P3) to sRGB.',
        original_links='Existing tagged source files retained. Untagged copies in source_raw are processed by a separate batched p3.py invocation into originals_p3; this tool never embeds P3 ICC directly.',
        selection='Worst symmetric GT-highlight PNG MAE, largest 96x96 code-RGB change, and configured known IDs.',
        reference=reference, candidate=candidate, top_fraction=args.top_fraction,
        manifest=str(manifest_path) if manifest_path.exists() else None)
    evaluation_sources = [dict(model=name, split=split, evaluation=str(ev['directory'] / 'evaluation.json'),
                               checkpoint=ev['report'].get('checkpoint'), checkpoint_sha256=ev['report'].get('checkpoint_sha256'),
                               reported_summary=ev['report'].get('summary'), metric_protocol=ev['report'].get('metric_protocol'))
                          for (name, split), ev in evaluations.items()]
    assets.finalize()
    payload = dict(protocol=protocol, models=specs, summary=summary, rows=rows,
                   evaluation_sources=evaluation_sources, contact_sheets=contact_sheets(rows, names, assets),
                   warnings=warnings + sorted(set(assets.warnings)))
    (output / 'summary.json').write_text(json.dumps(clean(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    write_csv(output / 'summary.csv', flat_summary(summary))
    write_csv(output / 'per_image_metrics.csv', per_image)
    html_report(output / 'report.html', payload)
    print('Saved {} · {} views'.format(output / 'report.html', len(rows)), flush=True)


if __name__ == '__main__':
    main()
