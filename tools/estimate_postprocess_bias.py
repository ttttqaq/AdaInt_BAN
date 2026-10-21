"""Estimate a fixed RGB baseline and DC bounds using TRAIN IDs only."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from PIL import Image
from adaint.config_utils import load_config


def image_pair(dataset, results, name):
    target = np.asarray(Image.open(Path(dataset.dir_gt) /
        dataset.filetmpl_gt.format(name)).convert('RGB')).astype(np.float32) / 255.
    result = np.asarray(Image.open(results / (name + '_result.png')).convert('RGB'))
    result = cv2.resize(result, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_AREA)
    return result.astype(np.float32) / 255., target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--results', type=Path, required=True,
                        help='existing raw result directory, including train and val IDs')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train = Path(cfg.data.train.ann_file).read_text().split()
    val = Path(cfg.data.val.ann_file).read_text().split()
    if not train or set(train) & set(val):
        raise ValueError('need a nonempty training split disjoint from validation')
    means = []
    for name in train:
        result, target = image_pair(cfg.data.train, args.results, name)
        means.append((target - result).mean((0, 1)))
    means = np.stack(means)
    bias = means.mean(0)
    scores = []
    for name in val:
        result, target = image_pair(cfg.data.val, args.results, name)
        def psnr(prediction):
            mse = float(np.square(prediction - target).mean())
            return float(-10 * np.log10(max(mse, 1e-14)))
        scores.append(dict(id=name, baseline_psnr=psnr(result),
                           fixed_bias_psnr=psnr(np.clip(result + bias, 0, 1))))
    report = dict(fixed_rgb_bias=bias.tolist(), train_count=len(train),
                  train_abs_dc_q95=np.quantile(np.abs(means), .95, axis=0).tolist(),
                  baseline_val_psnr=float(np.mean([r['baseline_psnr'] for r in scores])),
                  fixed_bias_val_psnr=float(np.mean([r['fixed_bias_psnr'] for r in scores])),
                  validation=scores,
                  note='Bias and q95 use training IDs only. RGB is encoded P3; '
                       'AREA alignment to GT. No automatic filtering of bad pairs.')
    output = json.dumps(report, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + '\n')


if __name__ == '__main__':
    main()
