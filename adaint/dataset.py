import os.path as osp
from collections import defaultdict
from pathlib import Path

import torch
from mmcv.parallel import DataContainer

from mmedit.datasets.registry import DATASETS
from mmedit.datasets.base_dataset import BaseDataset


class BaseEnhanceDataset(BaseDataset):
    r"""General paired image folder dataset for image enhancement.

    It assumes that the all input images are in directory '/path/to/input'
    (`dir_lq`), and all groundtruth images are in directory '/path/to/gt'
    (`dir_gt`). A GT image has the SAME filename as the corresponding input
    image. The training and testing data are splitted using an annotation
    txt file (`ann_file`).
    Args:
        dir_lq (str): Path to the folder of input images.
        dir_gt (str): Path to the folder of groundtruth images.
        ann_file (str): Path to the annotation txt file.
        pipeline (List[dict | callable]): A sequence of data transformations.
        test_mode (bool, optional): Store `True` when building test dataset.
            Default: `False`.
        filetmpl_lq (str, optional): Template for each filename for input images.
            Default: '{}.jpg'.
        filetmpl_gt (str, optional): Template for each filename for groundtruth images.
            Default: '{}.jpg'.
    """
    
    def __init__(self,
                 dir_lq,
                 dir_gt,
                 ann_file,
                 pipeline,
                 test_mode=False,
                 filetmpl_lq='{}.jpg',
                 filetmpl_gt='{}.jpg'):
        super().__init__(pipeline, test_mode=test_mode)
        
        if not osp.isfile(ann_file):
            raise ValueError('"ann_file" must be a path to annotation txt file.')
        
        self.dir_lq = dir_lq
        self.dir_gt = dir_gt
        self.ann_file = ann_file
        self.filetmpl_lq = filetmpl_lq
        self.filetmpl_gt = filetmpl_gt
        self.data_infos = self.load_annotations()
        
    def load_annotations(self):
        r"""Load annoations for enhancement dataset.

        It loads the LQ and GT image path from the annotation file.
        Each line in the annotation file contains the image name.
        Returns:
            dict: Returned dict for LQ and GT pairs.
        """
        data_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                basename = line.split('\n')[0]
                lq_name = self.filetmpl_lq.format(basename)
                gt_name = self.filetmpl_gt.format(basename)
                data_infos.append(
                    dict(
                        lq_path=osp.join(self.dir_lq, lq_name),
                        gt_path=osp.join(self.dir_gt, gt_name)))
        return data_infos
    
    def evaluate(self, results, logger=None):
        r"""Evaluate with different metrics.

        Args:
            results (list[tuple]): The output of forward_test() of the model.
        Return:
            dict: Evaluation results dict.
        """
        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        results = [res['eval_result'] for res in results]  # a list of dict
        eval_result = defaultdict(list)  # a dict of list

        for res in results:
            for metric, val in res.items():
                eval_result[metric].append(val)
        for metric, val_list in eval_result.items():
            assert len(val_list) == len(self), (
                f'Length of evaluation result of {metric} is {len(val_list)}, '
                f'should be {len(self)}')

        # average the results
        eval_result = {
            metric: sum(values) / len(self)
            for metric, values in eval_result.items()
        }

        return eval_result
    

@DATASETS.register_module()
class FiveK(BaseEnhanceDataset):
    pass


@DATASETS.register_module()
class Face(BaseEnhanceDataset):
    r"""Custom face-retouching dataset for image enhancement.

    Unlike FiveK/PPR10K, the LQ and GT images live in the SAME directory,
    distinguished only by filename suffix (e.g. `{id}_input.jpg` /
    `{id}_gt.jpg`). This is already supported by the base class: just pass
    the same path as both `dir_lq` and `dir_gt`, with `filetmpl_lq` /
    `filetmpl_gt` set to the corresponding suffix templates.

    For the blind test split (no GT available), simply omit any GT-loading
    step from the pipeline; `dir_gt`/`filetmpl_gt` are still required
    arguments but will never be read from disk in that case.
    """
    pass


@DATASETS.register_module()
class PPR10K(BaseEnhanceDataset):
    r"""PPR10K dataset for image enhancement.

    The difference between this class and the base class is that multiple LQ
    images correspond to the same GT image.
    """
    
    def load_annotations(self):
        r"""Load annoations for enhancement dataset.

        It loads the LQ and GT image path from the annotation file.
        Each line in the annotation file contains the image name.
        Returns:
            dict: Returned dict for LQ and GT pairs.
        """
        data_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                basename = line.split('\n')[0]
                lq_name = self.filetmpl_lq.format(basename)
                gt_name = self.filetmpl_gt.format('_'.join(basename.split('_')[:2]))
                data_infos.append(
                    dict(
                        lq_path=osp.join(self.dir_lq, lq_name),
                        gt_path=osp.join(self.dir_gt, gt_name)))
        return data_infos


@DATASETS.register_module()
class PPR10KFolder(BaseEnhanceDataset):
    """Read an existing PPR10K split without creating another holdout split.

    Sources and targets are matched by filename stem, allowing PNG exports
    as well as the original TIFF files. Human masks are optional for training
    and are carried as paths in validation metadata, never used as targets.
    """

    def __init__(self, dir_lq, dir_gt, pipeline, dir_mask=None,
                 test_mode=False, ann_file=None):
        BaseDataset.__init__(self, pipeline, test_mode=test_mode)
        self.dir_lq = dir_lq
        self.dir_gt = dir_gt
        self.dir_mask = dir_mask
        self.ann_file = ann_file
        self.data_infos = self.load_annotations()

    def prepare_test_data(self, idx):
        data = super().prepare_test_data(idx)
        # Validation keeps each image's native size. Collate into lists rather
        # than padding/resizing portraits and landscapes into a shared canvas.
        for key in ('lq', 'gt'):
            if isinstance(data.get(key), torch.Tensor):
                data[key] = DataContainer(data[key], stack=False)
        return data

    @staticmethod
    def _images(directory):
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError('image directory does not exist: {}'.format(directory))
        images = {}
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix.lower() not in (
                    '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'):
                continue
            if path.stem in images:
                raise ValueError('duplicate image stem: {}'.format(path.stem))
            images[path.stem] = str(path)
        return images

    def load_annotations(self):
        sources = self._images(self.dir_lq)
        targets = self._images(self.dir_gt)
        masks = self._images(self.dir_mask) if self.dir_mask is not None else None
        if self.ann_file is None:
            names = sorted(sources)
        else:
            with open(self.ann_file) as handle:
                names = [Path(line.strip()).stem for line in handle if line.strip()]
        if not names:
            raise ValueError('PPR10K split contains no images')
        if len(names) != len(set(names)):
            raise ValueError('PPR10K annotation contains duplicate images')
        data_infos = []
        for name in names:
            for label, mapping in (('source', sources), ('target', targets), ('mask', masks)):
                if mapping is not None and name not in mapping:
                    raise FileNotFoundError('missing {} for PPR10K image {}'.format(label, name))
            info = dict(lq_path=sources[name], gt_path=targets[name])
            if masks is not None:
                info['mask_path'] = masks[name]
            data_infos.append(info)
        return data_infos
