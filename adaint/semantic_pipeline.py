"""Mask loading and aligned crop/context/region sampling for semantic LUTs."""
import copy
from pathlib import Path
import random

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ColorJitter
from mmedit.datasets.registry import PIPELINES

from .calibration_geometry import extract_roi, sample_roi
from .calibration_pipeline import perturb_foreground_exposure


def region_pixels(image, mask, size):
    """Deterministic weighted systematic sampling from ORIGINAL region pixels."""
    weights = mask.reshape(-1).astype(np.float64)
    cumulative = np.cumsum(weights)
    if cumulative[-1] <= 1e-8:
        return np.zeros((size, size, 3), dtype=image.dtype)
    queries = (np.arange(size * size) + .5) * cumulative[-1] / (size * size)
    indices = np.searchsorted(cumulative, queries, side='right').clip(max=len(weights) - 1)
    return image.reshape(-1, 3)[indices].reshape(size, size, 3).copy()


@PIPELINES.register_module()
class LoadSemanticMasks:
    def __init__(self, foreground_root, foreground_suffix='_face_mask',
                 face_root=None, face_suffix='_face_mask', feather_radius=1,
                 perturb_probability=0., perturb_pixels=1, foreground_ignore_boxes=None):
        self.foreground_root = Path(foreground_root)
        self.face_root = Path(face_root) if face_root else None
        self.foreground_suffix, self.face_suffix = foreground_suffix, face_suffix
        if feather_radius < 0 or perturb_pixels < 0 or not 0 <= perturb_probability <= 1:
            raise ValueError('invalid mask feather/perturbation settings')
        self.feather_radius = int(feather_radius)
        self.perturb_probability, self.perturb_pixels = float(perturb_probability), int(perturb_pixels)
        self.indices = {}
        self.foreground_ignore_boxes = foreground_ignore_boxes or {}
        for boxes in self.foreground_ignore_boxes.values():
            for box in boxes:
                if len(box) != 4 or not all(np.isfinite(v) and 0 <= v <= 1 for v in box) or box[2] <= box[0] or box[3] <= box[1]:
                    raise ValueError('mask ignore boxes must be normalized valid rectangles')

    def _load(self, root, suffix, stem, shape):
        key = (str(root), suffix)
        if key not in self.indices:
            if not root.is_dir():
                raise FileNotFoundError('mask directory: ' + str(root))
            index = {}
            for path in sorted(root.rglob('*')):
                if path.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'):
                    continue
                if not path.stem.endswith(suffix):
                    continue
                sample = path.stem[:-len(suffix)] if suffix else path.stem
                if sample in index:
                    raise ValueError('ambiguous mask for ' + sample + ' under ' + str(root))
                index[sample] = path
            self.indices[key] = index
        path = self.indices[key].get(stem)
        if path is None:
            raise FileNotFoundError('mask for ' + stem + ' under ' + str(root) + ' suffix=' + suffix)
        value = mmcv.imread(str(path), flag='grayscale')
        if value is None:
            raise FileNotFoundError(str(path))
        if value.shape != shape:
            value = mmcv.imresize(value, (shape[1], shape[0]), interpolation='nearest')
        # Existing BFE binary mask convention; the fusion soft edge is separate.
        return (value > 127).astype(np.float32)[..., None], str(path)

    def __call__(self, results):
        stem = Path(results['lq_path']).stem
        if stem.endswith('_input'):
            stem = stem[:-len('_input')]
        shape = results['lq'].shape[:2]
        supervision, path = self._load(self.foreground_root, self.foreground_suffix, stem, shape)
        # Explicit, reproducible INPUT-only annotation corrections; never edit source files.
        for left, top, right, bottom in self.foreground_ignore_boxes.get(stem, []):
            x0, y0 = int(np.floor(left * shape[1])), int(np.floor(top * shape[0]))
            x1, y1 = int(np.ceil(right * shape[1])), int(np.ceil(bottom * shape[0]))
            supervision[y0:y1, x0:x1] = 0
        fusion = torch.from_numpy(supervision).permute(2, 0, 1)[None]
        if self.perturb_pixels and random.random() < self.perturb_probability:
            kernel = self.perturb_pixels * 2 + 1
            if random.random() < .5:
                fusion = F.max_pool2d(fusion, kernel, 1, self.perturb_pixels)
            else:
                fusion = -F.max_pool2d(-fusion, kernel, 1, self.perturb_pixels)
        if self.feather_radius:
            fusion = F.avg_pool2d(fusion, 2 * self.feather_radius + 1, 1,
                                  self.feather_radius, count_include_pad=False)
        results.update(foreground_mask=fusion[0].permute(1, 2, 0).numpy(),
                       supervision_mask=supervision, foreground_mask_path=path)
        if self.face_root is not None:
            results['face_mask'], results['face_mask_path'] = self._load(
                self.face_root, self.face_suffix, stem, shape)
        return results


@PIPELINES.register_module()
class SemanticContextAugment:
    def __init__(self, context_size=256, samples_size=64, region_samples_size=32,
                 crop_ratio=(.6, 1.), flip_ratio=.5, brightness=.2, saturation=.2,
                 roi_size=0, roi_margin=.15, require_foreground=False,
                 exposure_probability=0., exposure_max_ev=.3):
        if context_size < 32 or samples_size < 8 or region_samples_size < 1:
            raise ValueError('invalid context/statistics resolution')
        if not 0 < crop_ratio[0] <= crop_ratio[1] <= 1:
            raise ValueError('invalid crop_ratio')
        self.context_size, self.samples_size = int(context_size), int(samples_size)
        self.region_samples_size = int(region_samples_size)
        self.crop_ratio, self.flip_ratio = crop_ratio, float(flip_ratio)
        self.jitter = ColorJitter(brightness=brightness, saturation=saturation)
        if roi_size and roi_size < 32:
            raise ValueError('ROI size must be zero or at least 32')
        self.roi_size, self.roi_margin = int(roi_size), float(roi_margin)
        self.require_foreground = bool(require_foreground)
        if not 0 <= exposure_probability <= 1 or not np.isfinite(exposure_max_ev) or not 0 <= exposure_max_ev <= 1:
            raise ValueError('invalid foreground exposure augmentation')
        self.exposure_probability, self.exposure_max_ev = float(exposure_probability), float(exposure_max_ev)

    def __call__(self, results):
        if results['lq'].shape != results['gt'].shape:
            raise ValueError('SemanticContextAugment must follow ResizeToMatch')
        mask_keys = [key for key in ('foreground_mask', 'supervision_mask', 'face_mask') if key in results]
        aligned_keys = ['lq', 'gt'] + mask_keys
        if random.random() < self.flip_ratio:
            for key in aligned_keys:
                results[key] = np.flip(results[key], axis=1).copy()
        results['lq'] = np.asarray(self.jitter(Image.fromarray(results['lq']))).copy()
        image, mask = results['lq'], results['foreground_mask']
        if self.roi_size:
            image = image.astype(np.float32) / 255
            if random.random() < self.exposure_probability:
                image = perturb_foreground_exposure(image, mask, self.exposure_max_ev)
            results['lq'] = image
        height, width = image.shape[:2]
        context = torch.from_numpy(image).permute(2, 0, 1)[None].float()
        if image.dtype == np.uint8:
            context = context / 255.
        results['lq_context'] = F.interpolate(context, (self.context_size,) * 2,
            mode='bilinear', align_corners=False)[0].permute(1, 2, 0).numpy()
        mask_tensor = torch.from_numpy(mask).permute(2, 0, 1)[None].float()
        results['foreground_context'] = F.interpolate(mask_tensor, (self.context_size,) * 2,
            mode='area')[0].permute(1, 2, 0).numpy()
        ys = np.minimum(((np.arange(self.samples_size) + .5) * height / self.samples_size).astype(int), height - 1)
        xs = np.minimum(((np.arange(self.samples_size) + .5) * width / self.samples_size).astype(int), width - 1)
        results['context_samples'] = image[ys[:, None], xs[None, :]].copy()
        results['foreground_samples'] = mask[ys[:, None], xs[None, :]].copy()
        results['foreground_region_samples'] = region_pixels(image, mask, self.region_samples_size)
        results['background_region_samples'] = region_pixels(image, 1 - mask, self.region_samples_size)
        area = float(mask.mean())
        results['region_area'] = torch.tensor([area, 1 - area], dtype=torch.float32)
        if self.roi_size:
            # This happens BEFORE random crop and BEFORE full-image downsampling.
            roi, roi_mask, box = extract_roi(context, mask_tensor, self.roi_size, self.roi_margin)
            gt = torch.from_numpy(results['gt']).permute(2, 0, 1)[None].float() / 255
            supervision = torch.from_numpy(results['supervision_mask']).permute(2, 0, 1)[None].float()
            values = dict(foreground_roi=roi, foreground_roi_mask=roi_mask,
                foreground_roi_gt=sample_roi(gt, box, self.roi_size),
                foreground_roi_supervision=sample_roi(supervision, box, self.roi_size, True))
            results.update({key: value[0].permute(1, 2, 0).numpy() for key, value in values.items()})
            results['foreground_roi_box'] = box[0]
        crop_h = max(1, int(height * random.uniform(*self.crop_ratio)))
        crop_w = max(1, int(width * random.uniform(*self.crop_ratio)))
        top, left = random.randint(0, height - crop_h), random.randint(0, width - crop_w)
        if self.require_foreground and area > 0:
            # Keep at least half of the foreground; fall back to a full image.
            for attempt in range(16):
                retained = mask[top:top + crop_h, left:left + crop_w].sum()
                if retained >= .5 * mask.sum():
                    break
                top, left = random.randint(0, height - crop_h), random.randint(0, width - crop_w)
            else:
                crop_h, crop_w, top, left = height, width, 0, 0
        results['crop_box'] = torch.tensor([left / width, top / height,
            (left + crop_w) / width, (top + crop_h) / height], dtype=torch.float32)
        for key in aligned_keys:
            results[key] = results[key][top:top + crop_h, left:left + crop_w].copy()
        return results


def semantic_mask_step(cfg, split, training=False):
    settings = dict(cfg.semantic_masks)
    root = settings.pop(split + '_foreground_root', None)
    # Other split roots do not belong to the pipeline constructor.
    face_root = settings.pop(split + '_face_root', None)
    settings = {key: value for key, value in settings.items()
                if not key.endswith('_root') and key != 'region_label'}
    if root is None:
        raise ValueError('semantic_masks.' + split + '_foreground_root is required')
    settings.update(type='LoadSemanticMasks', foreground_root=root, face_root=face_root)
    if not training:
        settings['perturb_probability'] = 0.
    return settings


def configure_semantic_pipeline(cfg):
    if cfg.get('model', {}).get('type') not in ('AiLUTDual', 'AiLUTBilateral'):
        return cfg
    if cfg.get('enable_kd', False) or cfg.get('gen', 1) != 1:
        raise ValueError('AiLUTDual first-stage training requires enable_kd=false and gen=1')
    for split in ('train', 'val', 'test'):
        dataset = cfg.get('data', {}).get(split)
        if not dataset:
            continue
        dataset.type = 'SemanticFace' if dataset.type in ('Face', 'SemanticFace') else dataset.type
        pipeline = dataset.pipeline
        if any(step['type'] == 'LoadSemanticMasks' for step in pipeline):
            continue
        training = split == 'train'
        has_face = bool(cfg.semantic_masks.get(split + '_face_root'))
        if training and cfg.model.get('face_color_weight', 0.) and not has_face:
            raise ValueError('face_color_weight requires separate ' + split + '_face_root')
        mask_keys = ['foreground_mask', 'supervision_mask'] + (['face_mask'] if has_face else [])
        extras = (['lq_context', 'context_samples', 'foreground_context', 'foreground_samples',
                   'foreground_region_samples', 'background_region_samples'] if training else [])
        roi = cfg.model.type == 'AiLUTBilateral'
        if training and roi:
            extras += ['foreground_roi', 'foreground_roi_mask', 'foreground_roi_gt',
                       'foreground_roi_supervision']
        crop = next((step for step in pipeline if step['type'] == 'RandomRatioCrop'), None)
        jitter = next((step for step in pipeline if step['type'] == 'RandomColorJitter'), {})
        flip = next((step for step in pipeline if step['type'] == 'Flip'), {})
        if training and crop is None:
            raise ValueError('semantic training requires RandomRatioCrop in source pipeline')
        updated = []
        inserted = False
        for original in pipeline:
            step = copy.deepcopy(original)
            kind = step['type']
            if kind == 'RandomRatioCrop':
                options = dict(type='SemanticContextAugment',
                    context_size=cfg.model.get('context_size', 256),
                    samples_size=cfg.model.get('stats_pool_size', 64),
                    region_samples_size=cfg.model.get('region_samples_size', 32),
                    crop_ratio=crop['crop_ratio'], flip_ratio=flip.get('flip_ratio', .5),
                    brightness=jitter.get('brightness', 0), saturation=jitter.get('saturation', 0))
                if roi:
                    options.update(roi_size=cfg.model.get('roi_size', 128),
                                   roi_margin=cfg.model.get('roi_margin', .15), require_foreground=True,
                                   exposure_probability=cfg.get('foreground_exposure_probability', .3),
                                   exposure_max_ev=cfg.get('foreground_exposure_max_ev', .3))
                updated.append(options)
            elif kind not in ('Flip', 'RandomColorJitter'):
                if kind in ('FlexibleRescaleToZeroOne', 'ImageToTensor'):
                    if not inserted:
                        updated.append(semantic_mask_step(cfg, split, training))
                        inserted = True
                    step['keys'] = list(step['keys']) + mask_keys + extras
                elif kind == 'Collect':
                    step['keys'] = list(step['keys']) + mask_keys + extras
                    if training:
                        step['keys'] += ['crop_box', 'region_area']
                        if roi:
                            step['keys'].append('foreground_roi_box')
                    step['meta_keys'] = list(step.get('meta_keys', [])) + ['foreground_mask_path']
                    if has_face:
                        step['meta_keys'] += ['face_mask_path']
                updated.append(step)
            # Load before augmentation, after resolution alignment.
            if kind == 'ResizeToMatch':
                updated.append(semantic_mask_step(cfg, split, training))
                inserted = True
        dataset.pipeline = updated
    return cfg
