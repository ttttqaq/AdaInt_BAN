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
                 perturb_probability=0., perturb_pixels=1):
        self.foreground_root = Path(foreground_root)
        self.face_root = Path(face_root) if face_root else None
        self.foreground_suffix, self.face_suffix = foreground_suffix, face_suffix
        if feather_radius < 0 or perturb_pixels < 0 or not 0 <= perturb_probability <= 1:
            raise ValueError('invalid mask feather/perturbation settings')
        self.feather_radius = int(feather_radius)
        self.perturb_probability, self.perturb_pixels = float(perturb_probability), int(perturb_pixels)
        self.indices = {}

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
                 crop_ratio=(.6, 1.), flip_ratio=.5, brightness=.2, saturation=.2):
        if context_size < 32 or samples_size < 8 or region_samples_size < 1:
            raise ValueError('invalid context/statistics resolution')
        if not 0 < crop_ratio[0] <= crop_ratio[1] <= 1:
            raise ValueError('invalid crop_ratio')
        self.context_size, self.samples_size = int(context_size), int(samples_size)
        self.region_samples_size = int(region_samples_size)
        self.crop_ratio, self.flip_ratio = crop_ratio, float(flip_ratio)
        self.jitter = ColorJitter(brightness=brightness, saturation=saturation)

    def __call__(self, results):
        if results['lq'].shape != results['gt'].shape:
            raise ValueError('SemanticContextAugment must follow ResizeToMatch')
        mask_keys = [key for key in ('foreground_mask', 'supervision_mask', 'face_mask') if key in results]
        if random.random() < self.flip_ratio:
            for key in ['lq', 'gt'] + mask_keys:
                results[key] = np.flip(results[key], axis=1).copy()
        results['lq'] = np.asarray(self.jitter(Image.fromarray(results['lq']))).copy()
        image, mask = results['lq'], results['foreground_mask']
        height, width = image.shape[:2]
        context = torch.from_numpy(image).permute(2, 0, 1)[None].float() / 255.
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
        crop_h = max(1, int(height * random.uniform(*self.crop_ratio)))
        crop_w = max(1, int(width * random.uniform(*self.crop_ratio)))
        top, left = random.randint(0, height - crop_h), random.randint(0, width - crop_w)
        results['crop_box'] = torch.tensor([left / width, top / height,
            (left + crop_w) / width, (top + crop_h) / height], dtype=torch.float32)
        for key in ['lq', 'gt'] + mask_keys:
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
    if cfg.get('model', {}).get('type') != 'AiLUTDual':
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
                updated.append(dict(type='SemanticContextAugment',
                    context_size=cfg.model.get('context_size', 256),
                    samples_size=cfg.model.get('stats_pool_size', 64),
                    region_samples_size=cfg.model.get('region_samples_size', 32),
                    crop_ratio=crop['crop_ratio'], flip_ratio=flip.get('flip_ratio', .5),
                    brightness=jitter.get('brightness', 0), saturation=jitter.get('saturation', 0)))
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
