"""Generate train/val/test annotation files for the custom face dataset.

The dataset directory (see --data-root) is expected to contain:
    1_indoors_data/   {id}_input.jpg + {id}_gt.jpg   (indoor training pairs)
    2_outdoors_data/  {id}_input.jpg + {id}_gt.jpg   (outdoor training pairs)
    3_test_data/      {id}_input.jpg                 (blind test, no GT)

For each training split (indoor / outdoor), the last `--val-ratio` (default
10%) of the sorted numeric ids is held out as a validation set; the rest is
used for training. The two training splits are kept separate on purpose
(indoor and outdoor images are treated as different data distributions and
are never merged). The blind test split is written as-is (no split, no GT
required).

Usage:
    python tools/make_face_annfiles.py \\
        --data-root /DB/data/yujietang/IQ-LUT/face_dataset/all \\
        --out-dir adaint/annfiles/Face
"""
import argparse
import math
import os


def get_ids(folder, suffix):
    files = os.listdir(folder)
    return sorted(set(int(f.split('_')[0]) for f in files if f.endswith(suffix)))


def write_ids(path, ids):
    with open(path, 'w') as f:
        for i in ids:
            f.write(f'{i}\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='/DB/data/yujietang/IQ-LUT/face_dataset/all')
    parser.add_argument('--out-dir', default='adaint/annfiles/Face')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='fraction of each training split held out as '
                        'validation (taken from the END of the sorted id '
                        'list, i.e. the "last 10%%").')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for name, folder in [('indoor', '1_indoors_data'), ('outdoor', '2_outdoors_data')]:
        ids = get_ids(os.path.join(args.data_root, folder), '_gt.jpg')
        n_val = max(1, math.ceil(len(ids) * args.val_ratio))
        train_ids, val_ids = ids[:-n_val], ids[-n_val:]
        write_ids(os.path.join(args.out_dir, f'{name}_train.txt'), train_ids)
        write_ids(os.path.join(args.out_dir, f'{name}_val.txt'), val_ids)
        print(f'[{name}] total={len(ids)} train={len(train_ids)} val={len(val_ids)} '
             f'val_ids={val_ids}')

    test_ids = get_ids(os.path.join(args.data_root, '3_test_data'), '_input.jpg')
    write_ids(os.path.join(args.out_dir, 'test.txt'), test_ids)
    print(f'[test] total={len(test_ids)} (blind, no GT, not split)')


if __name__ == '__main__':
    main()
