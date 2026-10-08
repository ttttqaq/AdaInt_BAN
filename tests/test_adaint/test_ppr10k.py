"""PPR10K folder splits, official HC formulas and refinement integration."""
import mmcv
import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from skimage.color import rgb2lab

from adaint.config_utils import load_config
from adaint.context_transform import configure_context_pipeline
from adaint.dataset import PPR10KFolder
from adaint.experiment_utils import inference_config, resolve_training_schedule
from adaint.metrics_hc import _resize_mask, delta_e_hc, psnr_hc
from adaint.model import AiLUT
from mmedit.core import delta_e, psnr
from mmedit.datasets import build_dataloader, build_dataset
from tools.prepare_lut_experiment import configure_ppr10k_data, prepare


def test_hc_matches_official_weighted_rgb_and_lab_formulas():
    rng = np.random.RandomState(3)
    pred = rng.randint(0, 256, (12, 15, 3)).astype(np.uint8)
    gt = rng.randint(0, 256, pred.shape).astype(np.uint8)
    mask = rng.choice([0, 1, 127, 255], size=pred.shape[:2]).astype(np.uint8)
    weight = np.where(mask == 0, .5, 1.)[..., None]
    # Literal NumPy translation of the official MATLAB calculation.
    weighted_pred = pred.astype(np.float64) / 255 * weight
    weighted_gt = gt.astype(np.float64) / 255 * weight
    expected_psnr = 10 * np.log10(1 / np.mean((weighted_pred - weighted_gt) ** 2))
    pred_lab = rgb2lab(pred[..., ::-1].astype(np.float64) / 255)
    gt_lab = rgb2lab(gt[..., ::-1].astype(np.float64) / 255)
    expected_delta = np.sqrt(np.sum((pred_lab * weight - gt_lab * weight) ** 2, axis=2)).mean()
    assert psnr_hc(pred, gt, mask) == pytest.approx(expected_psnr)
    assert delta_e_hc(pred, gt, mask) == pytest.approx(expected_delta)


@pytest.mark.parametrize('foreground', [False, True])
def test_hc_background_and_foreground_weights(foreground):
    pred = np.full((8, 9, 3), 60, dtype=np.uint8)
    gt = np.full_like(pred, 90)
    mask = np.full(pred.shape[:2], 255 if foreground else 0, dtype=np.uint8)
    expected_gain = 0 if foreground else 20 * np.log10(2)
    assert psnr_hc(pred, gt, mask) == pytest.approx(psnr(pred, gt) + expected_gain)
    assert delta_e_hc(pred, gt, mask) == pytest.approx(delta_e(pred, gt) * (1 if foreground else .5))
    assert psnr_hc(pred, pred, mask) == float('inf')
    assert delta_e_hc(pred, pred, mask) == 0


def test_hc_crop_and_mismatched_mask_resolution():
    pred = np.full((12, 16, 3), 60, dtype=np.uint8)
    gt = np.full_like(pred, 90)
    mask = np.full((6, 8), 255, dtype=np.uint8)
    assert psnr_hc(pred, gt, mask, 2) == pytest.approx(psnr(pred, gt, 2))
    assert delta_e_hc(pred, gt, mask, 2) == pytest.approx(delta_e(pred, gt, 2))
    with pytest.raises(ValueError):
        psnr_hc(pred, gt, mask, 6)


def test_mask_bicubic_samples_antialiasing_and_uint8_rounding():
    # Analytic cubic samples with half-pixel centres and symmetric padding.
    mask = np.array([[0, 0], [255, 255]], dtype=np.uint8)
    enlarged = _resize_mask(mask, (4, 2))
    assert np.array_equal(enlarged[:, 0], [0, 52, 203, 255])
    assert enlarged.dtype == np.uint8
    step = np.repeat(mask, 2, axis=0)
    reduced = _resize_mask(step, (2, 2))
    assert np.array_equal(reduced[:, 0], [14, 241])
    assert np.array_equal(_resize_mask(mask, mask.shape), mask)
    # Resizing along only one dimension must preserve exact background zeros.
    large = np.zeros((360, 540), dtype=np.uint8)
    large[100:250, 150:350] = 255
    small = _resize_mask(large, (358, 540))
    assert small.shape == (358, 540)
    assert np.isfinite(small).all() and (small == 0).any() and (small == 255).any()


def test_folder_split_discovers_all_images_and_checks_pairs(tmp_path):
    for folder in ('source', 'target', 'mask'):
        (tmp_path / folder).mkdir()
    for name in ('10_0', '11_0', '11_1'):
        Image.fromarray(np.zeros((8, 9, 3), np.uint8)).save(str(tmp_path/'source'/(name+'.png')))
        Image.fromarray(np.zeros((8, 9, 3), np.uint8)).save(str(tmp_path/'target'/(name+'.tif')))
        Image.fromarray(np.zeros((8, 9), np.uint8)).save(str(tmp_path/'mask'/(name+'.png')))
    kwargs = dict(dir_lq=str(tmp_path/'source'), dir_gt=str(tmp_path/'target'),
                  dir_mask=str(tmp_path/'mask'), pipeline=[], test_mode=True)
    dataset = PPR10KFolder(**kwargs)
    assert len(dataset) == 3
    assert dataset[0]['mask_path'].endswith('10_0.png')
    (tmp_path/'mask'/'11_1.png').unlink()
    with pytest.raises(FileNotFoundError, match='missing mask'):
        PPR10KFolder(**kwargs)


def test_ppr10k_config_preserves_provided_splits_metrics_and_expert():
    with open('adaint/configs/ppr10k.yaml') as handle:
        base = yaml.safe_load(handle)
    base = configure_ppr10k_data(base, '/dataset', 'c')
    train, test = prepare(base, 'unit_ppr')
    assert train['data']['train']['dir_lq'] == '/dataset/prr_train/source'
    assert train['data']['val']['dir_lq'] == '/dataset/prr_value/source'
    assert train['data']['val']['dir_gt'] == '/dataset/prr_value/target_c'
    assert 'ann_file' not in train['data']['train']
    assert train['model'] == test['model']
    assert test['data']['val'] == train['data']['val']
    expected = ['PSNR', 'deltaE', 'PSNR_hc', 'deltaE_hc']
    assert train['test_cfg']['metrics'] == expected
    assert test['validation_test_cfg']['metrics'] == expected
    cfg = mmcv.Config(train)
    cfg.gpus = 1
    resolve_training_schedule(cfg, 8875)
    assert cfg.iters_per_epoch == 35 and cfg.total_iters == 10500
    assert inference_config(cfg).validation_test_cfg.metrics == expected


def test_ppr10k_full_context_pipeline_collates_different_image_shapes(tmp_path):
    for folder in ('source', 'target_a'):
        (tmp_path/'prr_train'/folder).mkdir(parents=True)
    for index, shape in enumerate(((48, 64), (64, 48))):
        image = np.full((*shape, 3), 100, np.uint8)
        for folder in ('source', 'target_a'):
            Image.fromarray(image).save(str(tmp_path/'prr_train'/folder/('1_{}.png'.format(index))))
    cfg = load_config('adaint/configs/ppr10k.yaml')
    cfg = mmcv.Config(configure_ppr10k_data(cfg._cfg_dict, str(tmp_path), 'a'))
    configure_context_pipeline(cfg)
    context = next(step for step in cfg.data.train.pipeline if step['type'] == 'ContextAwareAugment')
    assert context['brightness'] == 0 and context['saturation'] == 0
    dataset = build_dataset(cfg.data.train)
    loader = build_dataloader(dataset, samples_per_gpu=2, workers_per_gpu=0,
                              dist=False, shuffle=False, persistent_workers=False)
    batch = next(iter(loader))
    assert batch['lq'].shape == (2, 3, 448, 448)
    assert batch['gt'].shape == batch['lq'].shape
    assert batch['lq_context'].shape == (2, 3, 256, 256)
    assert batch['crop_box'].shape == (2, 4)
    assert torch.allclose(batch['lq'], batch['gt'])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='real AiLUT CUDA extension')
def test_ppr10k_paired_model_evaluation_has_all_four_metrics(tmp_path):
    model = AiLUT(
        n_vertices=9, curve_points=4,
        stats_bins=8, stats_pool_size=16, condition_dim=32,
        head_hidden_dim=16,
        local_resolution=32, local_channels=4, context_size=32,
        test_cfg=mmcv.ConfigDict(metrics=['PSNR', 'deltaE', 'PSNR_hc', 'deltaE_hc'],
                                 crop_border=0)).cuda().eval()
    mask = tmp_path/'mask.png'
    Image.fromarray(np.full((24, 36), 255, np.uint8)).save(str(mask))
    lq = torch.rand(1, 3, 24, 36, device='cuda')
    with torch.no_grad():
        result = model(lq, gt=lq*.8, test_mode=True,
                       meta=[dict(mask_path=str(mask))])['eval_result']
    assert set(result) == {'PSNR', 'deltaE', 'PSNR_hc', 'deltaE_hc'}
    assert all(np.isfinite(value) for value in result.values())
    assert result['PSNR_hc'] == pytest.approx(result['PSNR'])
    assert result['deltaE_hc'] == pytest.approx(result['deltaE'])
