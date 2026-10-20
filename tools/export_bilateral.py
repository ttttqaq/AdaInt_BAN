"""Native-resolution LQ-only export; no GT, reference cache or training augmentation."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaint.config_utils import load_config
from tools.train_kd_unified import infer_and_save_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--out', required=True)
    parser.add_argument('--split', choices=['train','val','both'], default='both')
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg.model.type != 'AiLUTBilateral':
        raise ValueError('export expects an AiLUTBilateral inference config')
    for split in (['train','val'] if args.split == 'both' else [args.split]):
        infer_and_save_split(cfg, cfg.data[split], args.checkpoint, args.out)


if __name__ == '__main__':
    main()
