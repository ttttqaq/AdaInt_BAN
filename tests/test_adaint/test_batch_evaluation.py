"""Native-resolution batched enhancement evaluation and per-image results."""
from functools import partial
from unittest.mock import patch

import mmcv
import numpy as np
import pytest
import torch
from mmcv.parallel import MMDataParallel, collate
from PIL import Image
from torch.utils.data import DataLoader

from adaint.dataset import PPR10KFolder
from adaint.model import AiLUT
from mmedit.apis import single_gpu_test, multi_gpu_test
from mmedit.datasets import build_dataloader
from mmedit.datasets.samplers import DistributedSampler
import mmedit.apis.test as test_api


METRICS = ['PSNR', 'deltaE', 'PSNR_hc', 'deltaE_hc']


class PointEnhancer(AiLUT):
    """Exercise the production test path without the CUDA-only LUT kernel."""
    def __init__(self, metrics=METRICS):
        torch.nn.Module.__init__(self)
        self.fp16_enabled = False
        self.test_cfg = mmcv.ConfigDict(metrics=metrics, crop_border=0)
        self.calls = []

    def forward_dummy(self, lq):
        self.calls.append(tuple(lq.shape))
        return lq * 0.8, None, None


def cpu_parallel(model):
    # PyTorch 1.8 cannot construct device_ids=[] when CUDA is available.
    # These regressions explicitly exercise MMCV's CPU scatter path.
    with patch('torch.cuda.is_available', return_value=False):
        return MMDataParallel(model, device_ids=[])


def folder_dataset(tmp_path, count):
    for folder in ('source', 'target', 'mask'):
        (tmp_path / folder).mkdir()
    shapes = [(24, 36), (36, 24), (24, 36), (28, 36)]
    for index in range(count):
        shape = shapes[index % len(shapes)]
        stem = 'sample_{:03d}'.format(index)
        lq = np.full((*shape, 3), 40 + (index % 8) * 17, np.uint8)
        gt = np.rint(lq.astype(np.float64) * 0.9).astype(np.uint8)
        mask = np.full(shape, 255 if index % 3 else 0, np.uint8)
        for folder, value in (('source', lq), ('target', gt), ('mask', mask)):
            Image.fromarray(value).save(str(tmp_path / folder / (stem + '.png')))
    pipeline = [
        dict(type='LoadImageFromFile', key='lq', backend='pillow', channel_order='rgb'),
        dict(type='LoadImageFromFile', key='gt', backend='pillow', channel_order='rgb'),
        dict(type='FlexibleRescaleToZeroOne', keys=['lq', 'gt']),
        dict(type='ImageToTensor', keys=['lq', 'gt']),
        dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path', 'mask_path']),
    ]
    return PPR10KFolder(
        dir_lq=str(tmp_path / 'source'), dir_gt=str(tmp_path / 'target'),
        dir_mask=str(tmp_path / 'mask'), pipeline=pipeline, test_mode=True)


def individual_results(model, dataset):
    results = []
    for index in range(len(dataset)):
        sample = dataset[index]
        results.append(model.forward_test(
            sample['lq'].data.unsqueeze(0), gt=sample['gt'].data.unsqueeze(0),
            meta=[sample['meta'].data]))
    return results


def assert_metrics_equal(actual, expected, abs_tolerance=1e-8):
    assert len(actual) == len(expected)
    for result, reference in zip(actual, expected):
        assert set(result['eval_result']) == set(METRICS)
        for metric in METRICS:
            assert result['eval_result'][metric] == pytest.approx(
                reference['eval_result'][metric], abs=abs_tolerance)


@pytest.mark.parametrize('workers', [0, 2])
def test_native_resolution_batch64_preserves_order_metrics_and_last_sample(tmp_path, workers):
    dataset = folder_dataset(tmp_path, 65)
    model = PointEnhancer().eval()
    expected = individual_results(model, dataset)
    model.calls.clear()
    loader = build_dataloader(
        dataset, samples_per_gpu=64, workers_per_gpu=workers,
        dist=False, shuffle=False, persistent_workers=False)
    results = single_gpu_test(cpu_parallel(model), loader)
    assert_metrics_equal(results, expected)
    assert sorted(call[0] for call in model.calls) == [1, 16, 16, 32]
    assert {call[2:] for call in model.calls} == {(24, 36), (36, 24), (28, 36)}
    for metric, value in dataset.evaluate(results).items():
        assert value == pytest.approx(np.mean([row['eval_result'][metric] for row in expected]))


def test_uniform_tensor_batch_computes_hc_metrics_per_image(tmp_path):
    dataset = folder_dataset(tmp_path, 4)
    model = PointEnhancer().eval()
    samples = [dataset[index] for index in (0, 2)]
    expected = individual_results(model, dataset)
    model.calls.clear()
    result = model.forward_test(
        torch.stack([sample['lq'].data for sample in samples]),
        gt=torch.stack([sample['gt'].data for sample in samples]),
        meta=[sample['meta'].data for sample in samples])
    assert_metrics_equal(result, [expected[0], expected[2]])
    assert model.calls == [(2, 3, 24, 36)]


@pytest.mark.parametrize('iteration', [None, 7])
def test_every_image_is_saved_with_its_native_size(tmp_path, iteration):
    dataset = folder_dataset(tmp_path, 5)
    model = PointEnhancer(metrics=None).eval()
    samples = [dataset[index] for index in range(len(dataset))]
    save_dir = tmp_path / 'outputs'
    results = model.forward_test(
        [sample['lq'].data for sample in samples],
        meta=[sample['meta'].data for sample in samples],
        save_image=True, save_path=str(save_dir), iteration=iteration)
    assert len(results) == len(dataset)
    for index, sample in enumerate(samples):
        stem = 'sample_{:03d}'.format(index)
        path = (save_dir / (stem + '.png') if iteration is None else
                save_dir / stem / (stem + '-000008.png'))
        with Image.open(str(path)) as image:
            assert image.size == tuple(reversed(sample['lq'].data.shape[-2:]))
        assert results[index]['output'].shape == sample['lq'].data.unsqueeze(0).shape
        assert 'gt' not in results[index]


def test_distributed_collection_keeps_image_order_and_removes_sampler_padding(tmp_path, monkeypatch):
    dataset = folder_dataset(tmp_path, 9)
    model = PointEnhancer().eval()
    expected = individual_results(model, dataset)
    rank_state = [1]
    monkeypatch.setattr(test_api, 'get_dist_info', lambda: (rank_state[0], 2))
    monkeypatch.setattr(test_api.dist, 'barrier', lambda: None)
    collection_dir = tmp_path / 'collected'
    for rank in (1, 0):
        rank_state[0] = rank
        sampler = DistributedSampler(
            dataset, num_replicas=2, rank=rank, shuffle=False, samples_per_gpu=3)
        loader = DataLoader(
            dataset, batch_size=3, sampler=sampler,
            collate_fn=partial(collate, samples_per_gpu=3))
        results = multi_gpu_test(
            cpu_parallel(model), loader, tmpdir=str(collection_dir))
        if rank == 1:
            assert results is None
    assert_metrics_equal(results, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='real AiLUT CUDA extension')
def test_cuda_models_batched_metrics_match_individual_evaluation(tmp_path):
    dataset = folder_dataset(tmp_path, 65)
    model = AiLUT(
        n_vertices=9, curve_points=4,
        stats_bins=8, stats_pool_size=16, condition_dim=32,
        head_hidden_dim=16,
        local_resolution=32, local_channels=4, context_size=32,
        test_cfg=mmcv.ConfigDict(metrics=METRICS, crop_border=0)).cuda().eval()
    expected = []
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            expected.append(model.forward_test(
                sample['lq'].data.unsqueeze(0).cuda(),
                gt=sample['gt'].data.unsqueeze(0).cuda(), meta=[sample['meta'].data]))
    loader = build_dataloader(
        dataset, samples_per_gpu=64, workers_per_gpu=0,
        dist=False, shuffle=False, persistent_workers=False)
    results = single_gpu_test(MMDataParallel(model, device_ids=[0]), loader)
    assert_metrics_equal(results, expected, abs_tolerance=0.005)
