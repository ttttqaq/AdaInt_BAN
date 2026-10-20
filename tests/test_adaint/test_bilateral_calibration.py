"""Spatial coordinates, identity, supervised gradient routing and scratch checkpoint selection."""
import json
import random

import numpy as np
import pytest
import torch
from mmcv import ConfigDict

from adaint.bilateral_calibration import (ForegroundLightnessGrid, slice_lightness_grid,
    apply_lightness_residual, multiscale_calibration_loss)
from adaint.calibration_geometry import extract_roi, sample_roi
from adaint.calibration_pipeline import perturb_foreground_exposure
from adaint.calibration_checkpoint import save_calibration_selection, validate_scratch_start
from adaint.color import display_p3_to_lab
from adaint.config_utils import load_config
from adaint.experiment_utils import load_enhancer_checkpoint
from adaint.model_bilateral import AiLUTBilateral
from adaint.semantic_pipeline import SemanticContextAugment
from tests.test_adaint.test_semantic_dual import small_model, batch


def calibration_model(**kwargs):
    options=dict(n_vertices=9,n_ranks=3,stats_bins=8,stats_pool_size=16,condition_dim=32,
        head_hidden_dim=16,local_resolution=32,local_channels=4,context_size=32,
        region_samples_size=8,roi_size=32,calibration_channels=4,color_loss_weight=.1,
        train_cfg=dict(total_iters=101,n_fix_iters=0))
    options.update(kwargs)
    return AiLUTBilateral(**options)


def calibration_batch():
    data=batch()
    roi,mask,box=extract_roi(data['lq'],data['foreground_mask'],32,.15)
    data.update(foreground_roi=roi,foreground_roi_mask=mask,foreground_roi_box=box,
        foreground_roi_gt=sample_roi(data['gt'],box,32),
        foreground_roi_supervision=sample_roi(data['supervision_mask'],box,32,True))
    return data


def test_grid_axes_pixel_centres_and_lightness_are_all_used():
    source=torch.full((1,3,10,12),.4)
    z,y,x=torch.meshgrid(torch.linspace(0,1,8),torch.linspace(0,1,8),torch.linspace(0,1,8))
    grid=(x+2*y+3*z)[None,None].requires_grad_()
    field=slice_lightness_grid(grid,source)
    xx=(torch.arange(12)+.5)/12; yy=(torch.arange(10)+.5)/10
    expected=xx[None,:]+2*yy[:,None]+3*display_p3_to_lab(source)[0,0]/100
    assert torch.allclose(field[0,0],expected,atol=2e-6)
    field.sum().backward();assert grid.grad.abs().sum()>0
    crop=source[...,2:8,3:9]
    result=slice_lightness_grid(grid.detach(),crop,crop_box=source.new_tensor([[.25,.2,.75,.8]]))
    assert torch.allclose(result,field[...,2:8,3:9],atol=2e-6)
    assert not torch.equal(field[...,:6],field[...,6:])


def test_grid_roi_box_outside_padding_and_constant_nonzero_mean():
    source=torch.full((1,3,10,10),.4)
    box=source.new_tensor([[.2,.2,.8,.8]])
    field=slice_lightness_grid(torch.full((1,1,8,8,8),-2.),source,box)
    assert torch.allclose(field[...,2:8,2:8], torch.full((1,1,6,6),-2.), atol=5e-7)
    assert field[...,:2,:].eq(0).all()
    assert field[...,8:,:].eq(0).all()
    assert slice_lightness_grid(torch.ones(1,1,8,8,8),source).mean()==1


@pytest.mark.parametrize('fill',[-.1,0.,.1,.5,1.,1.1])
def test_difference_writeback_is_exact_identity_and_has_finite_black_gradients(fill):
    rgb=torch.full((1,3,8,8),fill,requires_grad=True)
    delta=torch.zeros(1,1,8,8,requires_grad=True)
    result,actual=apply_lightness_residual(rgb,delta)
    assert torch.equal(result,rgb)
    assert actual.abs().max()<1e-6
    result.sum().backward()
    assert torch.isfinite(rgb.grad).all() and torch.isfinite(delta.grad).all()
    if 0 <= fill < 1:
        assert delta.grad.abs().sum()>0


def test_signed_lightness_changes_both_directions_without_forced_chroma_head():
    rgb=torch.full((2,3,8,8),.4)
    delta=torch.tensor([2.,-2.])[:,None,None,None].expand(2,1,8,8)
    result,actual=apply_lightness_residual(rgb,delta)
    assert torch.allclose(actual,delta,atol=1e-3)
    assert (display_p3_to_lab(result)[:,1:]-display_p3_to_lab(rgb)[:,1:]).abs().max()<1e-3


def test_multiscale_supervision_detects_opposing_half_face_errors_and_empty_masks():
    actual=torch.zeros(2,1,32,32,requires_grad=True)
    target=torch.ones_like(actual)*3;target[...,:16]=-3
    mask=torch.ones_like(actual);mask[1]=0
    only_mean,_=multiscale_calibration_loss(actual,target,mask,(1,))
    spatial,errors=multiscale_calibration_loss(actual,target,mask,(1,4,8))
    assert only_mean==0 and spatial>0 and errors[4]>2.99
    spatial.backward()
    assert actual.grad[0,...,:16].mean()>0 and actual.grad[0,...,16:].mean()<0
    assert actual.grad[1].abs().sum()==0
    assert multiscale_calibration_loss(actual,target,torch.zeros_like(mask))[0]==0


def test_exposure_augmentation_is_bidirectional_preserves_background_and_headroom():
    random.seed(2026)
    rgb=np.random.RandomState(2).uniform(.1,.7,(32,40,3)).astype(np.float32)
    rgb[10,11]=1
    mask=np.zeros((32,40,1),np.float32);mask[5:25,7:30]=1
    signs=[]
    for _ in range(40):
        changed=perturb_foreground_exposure(rgb,mask,.3)
        assert np.array_equal(changed[mask[...,0]==0],rgb[mask[...,0]==0])
        assert np.isfinite(changed).all() and changed.min()>=0 and changed.max()<=1
        assert np.all((changed>=1-1e-6).sum(2)<=(rgb>=1-1e-6).sum(2))
        signs.append(float((changed-rgb)[mask[...,0]>0].mean()))
    assert min(signs)<0<max(signs)


def test_flip_crop_and_roi_target_alignment_before_sampling():
    random.seed(8)
    image=np.random.RandomState(3).randint(0,220,(48,64,3)).astype(np.uint8)
    mask=np.zeros((48,64,1),np.float32);mask[4:40,9:44]=1
    result=SemanticContextAugment(context_size=32,samples_size=16,region_samples_size=8,
        roi_size=32,flip_ratio=1,brightness=0,saturation=0,exposure_probability=1)(dict(
        lq=image.copy(),gt=image.copy(),foreground_mask=mask.copy(),supervision_mask=mask.copy()))
    x0,y0,x1,y1=np.rint(result['crop_box'].numpy()*[64,48,64,48]).astype(int)
    assert np.array_equal(result['gt'],np.flip(image,1)[y0:y1,x0:x1])
    gt=torch.from_numpy(np.flip(image,1).copy()).permute(2,0,1)[None].float()/255
    expected=sample_roi(gt,result['foreground_roi_box'][None],32)
    actual=torch.from_numpy(result['foreground_roi_gt']).permute(2,0,1)[None]
    assert torch.equal(expected,actual)


def test_scratch_initialization_is_seeded_and_never_loads_a_checkpoint(monkeypatch):
    def forbidden_load(*args,**kwargs):raise AssertionError('scratch initialization loaded a file')
    monkeypatch.setattr(torch,'load',forbidden_load)
    torch.manual_seed(37);first=calibration_model()
    torch.manual_seed(37);second=calibration_model()
    assert all(torch.equal(v,second.state_dict()[k]) for k,v in first.state_dict().items())
    assert first.cnt_iters==0 and first.n_fix_iters==0
    assert all(p.requires_grad for p in first.parameters())
    assert len(first.local_decoder.outputs)==2
    assert first.foreground_calibration.output.weight.eq(0).all()
    assert first.foreground_calibration.output.bias.eq(0).all()


@pytest.mark.parametrize('scene',['indoor','outdoor','niguang'])
def test_configs_train_from_scratch_without_any_reference_pipeline(scene):
    cfg=load_config('adaint/configs/face_'+scene+'_bilateral_calibration.yaml')
    assert cfg.adaint_fix_epochs==0 and cfg.seed==2026 and cfg.train_epochs==100
    assert cfg.model.pretrained is False and cfg.load_from is None and cfg.resume_from is None
    assert cfg.work_dir=='./work_dirs/face_'+scene+'_bilateral_calibration_scratch'
    assert not any(key in cfg for key in ('calibration_init_from','calibration_basis_config',
                                         'calibration_reference_dir','calibration_selection_tolerances'))
    for split in ('train','val','test'):
        assert not any(s['type']=='LoadCalibrationReference' for s in cfg.data[split].pipeline)
        assert not any('reference_lightness' in s.get('keys',[]) for s in cfg.data[split].pipeline)
    options=next(s for s in cfg.data.train.pipeline if s['type']=='SemanticContextAugment')
    assert options['roi_size']==128 and options['brightness']==0 and options['exposure_probability']==.3
    assert cfg.optimizers.lr==1e-4 and cfg.optimizers.paramwise_cfg.custom_keys.adaint.lr_mult==.1
    old=load_config('adaint/configs/face_'+scene+'_semantic_basis.yaml')
    assert next(s for s in old.data.train.pipeline if s['type']=='SemanticContextAugment').get('roi_size',0)==0


def test_best_selection_only_uses_trained_checkpoints_and_handles_overwritten_resume_records(tmp_path):
    cfg=ConfigDict(work_dir=str(tmp_path),best_checkpoint_metric='PSNR',best_checkpoint_rule='greater')
    for epoch in (1,2):(tmp_path/('epoch_%d.pth'%epoch)).write_text(str(epoch))
    records=[dict(mode='val',epoch=1,PSNR=27.,deltaE_P3_foreground=4.),
             dict(mode='val',epoch=2,PSNR=26.,deltaE_P3_foreground=3.)]
    result=save_calibration_selection(cfg,records,76)
    assert result['training_initialization']=='scratch' and result['progress']==1
    assert (tmp_path/'best.pth').read_text()=='1'
    assert (tmp_path/'best_psnr.pth').read_text()=='1'
    assert (tmp_path/'best_foreground.pth').read_text()=='2'
    assert not (tmp_path/'initial_basis.pth').exists()
    records.append(dict(mode='val',epoch=1,PSNR=25.,deltaE_P3_foreground=2.5))
    result=save_calibration_selection(cfg,records,76)
    assert result['progress']==2 and (tmp_path/'best.pth').read_text()=='2'
    assert (tmp_path/'best_foreground.pth').read_text()=='1'
    cfg.best_checkpoint_metric='deltaE_P3_foreground';cfg.best_checkpoint_rule='less'
    assert save_calibration_selection(cfg,records,76)['progress']==1
    assert 'baseline' not in json.loads((tmp_path/'best_selection.json').read_text())


def test_scratch_start_does_not_write_initial_weights_or_mix_runs(tmp_path):
    cfg=ConfigDict(work_dir=str(tmp_path),load_from=None,resume_from=None)
    validate_scratch_start(cfg)
    assert not list(tmp_path.iterdir())
    cfg.load_from='basis.pth'
    with pytest.raises(ValueError,match='load_from'):validate_scratch_start(cfg)
    cfg.load_from=None;(tmp_path/'epoch_1.pth').touch()
    with pytest.raises(ValueError,match='existing training'):validate_scratch_start(cfg)
    cfg.resume_from=str(tmp_path/'epoch_1.pth');validate_scratch_start(cfg)


CUDA=pytest.mark.skipif(not torch.cuda.is_available(),reason='native AiLUT CUDA')


@CUDA
def test_zero_grid_exactly_preserves_its_complete_basis_and_pure_background():
    basis=small_model(lut_mode='basis').cuda().eval()
    for out in basis.local_decoder.outputs:torch.nn.init.normal_(out.weight,std=.01)
    model=calibration_model().cuda().eval()
    # Copy a randomly created complete basis only for the identity comparison.
    state=model.state_dict();state.update(basis.state_dict());model.load_state_dict(state,strict=True)
    data=calibration_batch()
    with torch.no_grad():
        expected=basis.forward_dummy(data['lq'],foreground_mask=data['foreground_mask'])[0]
        actual,_,_,aux=model.forward_dummy(data['lq'],True,foreground_mask=data['foreground_mask'])
        baseline=data['foreground_mask']*aux['calibration_pre_foreground']+(1-data['foreground_mask'])*aux['branch_outputs'][1]
        assert torch.equal(actual,baseline)
        # Original CUDA inference can differ slightly across separate calls.
        assert torch.allclose(actual,expected,atol=2e-6,rtol=0)
        model.foreground_calibration.output.bias.fill_(.5)
        changed,_,_,changed_aux=model.forward_dummy(data['lq'],True,foreground_mask=data['foreground_mask'])
        bg=(data['foreground_mask']==0).expand_as(actual)
        assert torch.equal(changed[bg],changed_aux['branch_outputs'][1][bg])
        assert torch.allclose(changed[bg],expected[bg],atol=2e-6,rtol=0)
        assert not torch.equal(changed,actual)


@CUDA
def test_auxiliary_only_trains_calibrator_and_main_loss_reaches_all_original_paths():
    model=calibration_model().cuda().eval();data=calibration_batch()
    result=model.forward_train(**data);loss=result['losses']['loss_calibration_multiscale']
    loss.backward()
    assert model.foreground_calibration.output.weight.grad.abs().sum()>0
    assert all(p.grad is None for name,p in model.named_parameters() if not name.startswith('foreground_calibration.'))
    model.zero_grad()
    result=model.forward_train(**data)
    sum(v for k,v in result['losses'].items() if k.startswith('loss_')).backward()
    for group in ['backbone.','branches.foreground.adaint.','branches.background.adaint.',
                  'branches.foreground.lut_generator.','branches.background.lut_generator.',
                  'local_decoder.outputs.0.','local_decoder.outputs.1.']:
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
                   for name,p in model.named_parameters() if name.startswith(group)),group


@CUDA
def test_training_needs_roi_supervision_but_no_reference_and_inference_ignores_gt():
    model=calibration_model().cuda().eval();data=calibration_batch()
    losses=model.forward_train(**data)['losses']
    assert not any('keep_' in key for key in losses)
    assert all(torch.isfinite(value).all() for value in losses.values())
    with pytest.raises(ValueError,match='ROI'):model.forward_train(**{k:v for k,v in data.items() if k!='foreground_roi_gt'})
    with torch.no_grad():
        first=model.forward_dummy(data['lq'],foreground_mask=data['foreground_mask'])[0]
        data['gt'].zero_();data['foreground_roi_gt'].fill_(1)
        second=model.forward_dummy(data['lq'],foreground_mask=data['foreground_mask'])[0]
    assert torch.allclose(first,second,atol=2e-6,rtol=0)


@CUDA
def test_checkpoint_roundtrip_and_partial_old_checkpoint_rejection(tmp_path):
    model=calibration_model().cuda().eval();path=str(tmp_path/'current.pth')
    model.cnt_iters.fill_(17);torch.save(dict(state_dict=model.state_dict()),path)
    restored=calibration_model().cuda().eval();load_enhancer_checkpoint(restored,path)
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())
    torch.save(dict(state_dict=small_model(lut_mode='basis').state_dict()),path)
    with pytest.raises(RuntimeError,match='incompatible'):load_enhancer_checkpoint(restored,path)


@CUDA
def test_calibrator_can_learn_opposing_spatial_changes_and_net_brightening():
    torch.manual_seed(2026)
    module=ForegroundLightnessGrid(32,4).cuda()
    source=torch.full((1,3,32,32),.4,device='cuda');mask=torch.ones(1,1,32,32,device='cuda')
    conditions=[torch.zeros(1,32,device='cuda') for _ in range(2)]
    target=torch.ones_like(mask)*3;target[...,:16]=-1
    optimizer=torch.optim.Adam(module.parameters(),lr=.005)
    for _ in range(90):
        grid=module(source,source,mask,conditions,torch.ones(1,1,1,1,device='cuda'))
        field=slice_lightness_grid(grid,source)
        loss,_=multiscale_calibration_loss(field,target,mask)
        optimizer.zero_grad();loss.backward();optimizer.step()
    assert field[...,:16].mean()< -.5 and field[...,16:].mean()>2
    assert field.mean()>.5 and field.std()>.8


@pytest.mark.parametrize('scene,counts,color_weight',[
    ('indoor',(67,8),.3),('outdoor',(85,10),.1)])
def test_normal_scene_configs_keep_their_original_data_and_loss_weights(scene,counts,color_weight):
    cfg=load_config('adaint/configs/face_'+scene+'_bilateral_calibration.yaml')
    basis=load_config('adaint/configs/face_'+scene+'_semantic_basis.yaml')
    assert cfg.model.type=='AiLUTBilateral' and cfg.adaint_fix_epochs==0
    assert cfg.model.color_loss_weight==color_weight
    assert (cfg.dataset_split.train_count,cfg.dataset_split.val_count)==counts
    assert cfg.dataset_split.test_uses_validation is False
    assert not cfg.semantic_masks.get('foreground_ignore_boxes')
    for split in ('train','val','test'):
        for key in ('dir_lq','dir_gt','ann_file','filetmpl_lq','filetmpl_gt'):
            assert cfg.data[split][key]==basis.data[split][key]
        mask=next(s for s in cfg.data[split].pipeline if s['type']=='LoadSemanticMasks')
        assert mask['foreground_root']==cfg.semantic_masks[split+'_foreground_root']
        references=[s for s in cfg.data[split].pipeline if s['type']=='LoadCalibrationReference']
        assert not references
    assert cfg.data.test.ann_file!=cfg.data.val.ann_file
    assert not any(s.get('key')=='gt' for s in cfg.data.test.pipeline)
    assert not any('gt' in s.get('keys',[]) for s in cfg.data.test.pipeline)
    assert all(key['lr_mult']>0 for key in cfg.optimizers.paramwise_cfg.custom_keys.values())


@pytest.mark.parametrize('explicit_split',[None,'test'])
def test_native_blind_export_selects_test_masks_and_never_loads_gt(monkeypatch,explicit_split):
    from tools import train_kd_unified
    cfg=load_config('adaint/configs/face_indoor_bilateral_calibration.yaml')
    class PipelineCaptured(Exception):pass
    def capture(options):
        step=next(s for s in options.pipeline if s['type']=='LoadSemanticMasks')
        assert step['foreground_root']==cfg.semantic_masks.test_foreground_root
        assert step['foreground_root']!=cfg.semantic_masks.val_foreground_root
        assert not any(s.get('key')=='gt' or 'gt' in s.get('keys',[]) for s in options.pipeline)
        assert not any(s['type'] in ('ResizeToMatch','LoadCalibrationReference','SemanticContextAugment') for s in options.pipeline)
        raise PipelineCaptured()
    monkeypatch.setattr(train_kd_unified,'build_dataset',capture)
    with pytest.raises(PipelineCaptured):
        train_kd_unified.infer_and_save_split(cfg,cfg.data.test,'unused.pth','unused',split=explicit_split)


def test_export_cli_accepts_the_independent_blind_test_split(monkeypatch,tmp_path):
    import sys
    from tools import export_bilateral
    calls=[]
    monkeypatch.setattr(sys,'argv',['export_bilateral.py',
        'adaint/configs/face_outdoor_bilateral_calibration.yaml','student.pth',
        '--out',str(tmp_path),'--split','test'])
    monkeypatch.setattr(export_bilateral,'infer_and_save_split',lambda *a,**k:calls.append((a,k)))
    export_bilateral.main()
    assert len(calls)==1 and calls[0][1]['split']=='test'
    assert calls[0][0][1].ann_file.endswith('Face/test.txt')
