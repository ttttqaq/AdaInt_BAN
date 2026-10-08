"""Joint augmentation with full-image context and normalized crop coordinates."""
import random

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ColorJitter
from mmedit.datasets.registry import PIPELINES


@PIPELINES.register_module()
class ContextAwareAugment:
    def __init__(self, context_size=256, samples_size=64,
                 crop_ratio=(0.6, 1.0), flip_ratio=0.5,
                 brightness=0.2, saturation=0.2):
        if context_size < 32 or samples_size < 8:
            raise ValueError('context_size >= 32 and samples_size >= 8 required')
        if not 0 < crop_ratio[0] <= crop_ratio[1] <= 1:
            raise ValueError('crop_ratio must be in (0, 1]')
        self.context_size = int(context_size)
        self.samples_size = int(samples_size)
        self.crop_ratio = crop_ratio
        self.flip_ratio = float(flip_ratio)
        self.jitter = ColorJitter(brightness=brightness, saturation=saturation)

    def __call__(self, results):
        lq, gt = results['lq'], results['gt']
        if lq.shape != gt.shape:
            raise ValueError('ContextAwareAugment must follow ResizeToMatch')
        if random.random() < self.flip_ratio:
            lq, gt = np.flip(lq, axis=1).copy(), np.flip(gt, axis=1).copy()
        # Apply jitter once to the entire LQ; context and crop then agree.
        lq = np.asarray(self.jitter(Image.fromarray(lq))).copy()
        height, width = lq.shape[:2]
        crop_h = max(1, int(height * random.uniform(*self.crop_ratio)))
        crop_w = max(1, int(width * random.uniform(*self.crop_ratio)))
        top = random.randint(0, height - crop_h)
        left = random.randint(0, width - crop_w)
        context = torch.from_numpy(lq).permute(2, 0, 1).unsqueeze(0).float().div(255.)
        results['lq_context'] = torch.nn.functional.interpolate(
            context, (self.context_size,) * 2, mode='bilinear',
            align_corners=False)[0].permute(1, 2, 0).numpy()
        ys = np.minimum(((np.arange(self.samples_size) + 0.5) *
                         height / self.samples_size).astype(int), height - 1)
        xs = np.minimum(((np.arange(self.samples_size) + 0.5) *
                         width / self.samples_size).astype(int), width - 1)
        results['context_samples'] = lq[ys[:, None], xs[None, :]].copy()
        results['crop_box'] = torch.tensor(
            [left / width, top / height,
             (left + crop_w) / width, (top + crop_h) / height], dtype=torch.float32)
        results['lq'] = lq[top:top + crop_h, left:left + crop_w].copy()
        results['gt'] = gt[top:top + crop_h, left:left + crop_w].copy()
        return results


def configure_context_pipeline(cfg):
    """Wire the context toggle into the training pipeline, keeping old configs intact."""
    model = cfg.get('model', {})
    if model.get('type') != 'AiLUT' or not cfg.get('data', {}).get('train'):
        return cfg
    pipeline = cfg.data.train.pipeline
    if any(step['type'] == 'ContextAwareAugment' for step in pipeline):
        return cfg
    crop = next((step for step in pipeline if step['type'] == 'RandomRatioCrop'), None)
    jitter = next((step for step in pipeline if step['type'] == 'RandomColorJitter'), {})
    flip = next((step for step in pipeline if step['type'] == 'Flip'), {})
    if crop is None:
        raise ValueError('full-image context expects a RandomRatioCrop training pipeline')
    for step in pipeline:
        if step['type'] == 'ResizeToMatch':
            break
    else:
        raise ValueError('full-image context requires paired resolution alignment')
    extra_images = ['lq_context', 'context_samples']
    updated = []
    for step in pipeline:
        if step['type'] == 'RandomRatioCrop':
            updated.append(dict(
                type='ContextAwareAugment',
                context_size=model.get('context_size', 256),
                samples_size=model.get('stats_pool_size', 64),
                crop_ratio=crop['crop_ratio'], flip_ratio=flip.get('flip_ratio', 0.5),
                brightness=jitter.get('brightness', 0),
                saturation=jitter.get('saturation', 0)))
        elif step['type'] not in ('Flip', 'RandomColorJitter'):
            step = dict(step)
            if step['type'] in ('FlexibleRescaleToZeroOne', 'ImageToTensor'):
                step['keys'] = list(step['keys']) + extra_images
            elif step['type'] == 'Collect':
                step['keys'] = list(step['keys']) + extra_images + ['crop_box']
            updated.append(step)
    cfg.data.train.pipeline = updated
    return cfg
