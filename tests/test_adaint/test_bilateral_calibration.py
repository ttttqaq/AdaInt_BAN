"""Spatial coordinates, identity, supervised gradient routing and guarded selection."""
import copy
import json
import random

import numpy as np
import pytest
import torch
from mmcv import ConfigDict

from adaint.bilateral_calibration import (ForegroundLightnessGrid, slice_lightness_grid,
    apply_lightness_residual, multiscale_calibration_loss)
from adaint.calibration_geometry import extract_roi, sample_roi
from adaint.calibration_pipeline import LoadCalibrationReference, perturb_foreground_exposure
from adaint.calibration_reference import fingerprint
from adaint.calibration_checkpoint import choose_guarded_checkpoint
from adaint.color import display_p3_to_lab
from adaint.config_utils import load_config
from adaint.experiment_utils import initialize_bilateral_from_basis, load_enhancer_checkpoint
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
        foreground_roi_supervision=sample_roi(data['supervision_mask'],box,32,True),
        reference_lightness=display_p3_to_lab(data['gt'])[:,:1]/100)
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


def test_reference_flip_crop_and_roi_target_alignment_before_sampling():
    random.seed(8)
    image=np.random.RandomState(3).randint(0,220,(48,64,3)).astype(np.uint8)
    mask=np.zeros((48,64,1),np.float32);mask[4:40,9:44]=1
    reference=np.arange(48*64,dtype=np.float32).reshape(48,64,1)/(48*64)
    result=SemanticContextAugment(context_size=32,samples_size=16,region_samples_size=8,
        roi_size=32,flip_ratio=1,brightness=0,saturation=0,exposure_probability=1)(dict(
        lq=image.copy(),gt=image.copy(),foreground_mask=mask.copy(),supervision_mask=mask.copy(),
        reference_lightness=reference.copy()))
    x0,y0,x1,y1=np.rint(result['crop_box'].numpy()*[64,48,64,48]).astype(int)
    assert np.array_equal(result['reference_lightness'],np.flip(reference,1)[y0:y1,x0:x1])
    assert np.array_equal(result['gt'],np.flip(image,1)[y0:y1,x0:x1])
    gt=torch.from_numpy(np.flip(image,1).copy()).permute(2,0,1)[None].float()/255
    expected=sample_roi(gt,result['foreground_roi_box'][None],32)
    actual=torch.from_numpy(result['foreground_roi_gt']).permute(2,0,1)[None]
    assert torch.equal(expected,actual)


def test_cached_reference_rejects_missing_changed_sources_and_wrong_shape(tmp_path):
    (tmp_path/'1_input.jpg').write_bytes(b'input');(tmp_path/'1_gt.jpg').write_bytes(b'gt')
    record=dict(lq_path=fingerprint(tmp_path/'1_input.jpg'),gt_path=fingerprint(tmp_path/'1_gt.jpg'),cache='1.npy')
    (tmp_path/'manifest.json').write_text(json.dumps({'train_records':{'1':record}}))
    np.save(str(tmp_path/'1.npy'),np.full((10,12,1),.4,np.float32))
    loader=LoadCalibrationReference(str(tmp_path))
    item=dict(lq_path=str(tmp_path/'1_input.jpg'),gt_path=str(tmp_path/'1_gt.jpg'),lq=np.zeros((10,12,3)))
    assert loader(item)['reference_lightness'].mean()==pytest.approx(.4)
    (tmp_path/'1_input.jpg').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):loader(item)
    with pytest.raises(FileNotFoundError):LoadCalibrationReference(str(tmp_path/'missing'))


def test_strict_init_retains_both_original_heads_and_counter_reset(tmp_path):
    basis=small_model(lut_mode='basis');basis.cnt_iters.fill_(100)
    path=str(tmp_path/'basis.pth');torch.save(dict(state_dict=basis.state_dict()),path)
    model=calibration_model();initialize_bilateral_from_basis(model,path)
    for key,value in basis.state_dict().items():
        if key!='cnt_iters':assert torch.equal(value,model.state_dict()[key]),key
    assert model.cnt_iters==0 and all(p.requires_grad for p in model.parameters())
    assert len(model.local_decoder.outputs)==2
    state=dict(basis.state_dict());state.pop('local_decoder.outputs.0.weight')
    torch.save(dict(state_dict=state),path)
    with pytest.raises(RuntimeError,match='missing'):initialize_bilateral_from_basis(model,path)
    torch.save(dict(state_dict=small_model().state_dict()),path)
    with pytest.raises(RuntimeError,match='incompatible'):initialize_bilateral_from_basis(model,path)


def test_new_config_is_joint_training_and_old_configs_do_not_load_references():
    cfg=load_config('adaint/configs/face_niguang_bilateral_calibration.yaml')
    assert cfg.adaint_fix_epochs==0 and cfg.seed==2026
    assert any(s['type']=='LoadCalibrationReference' for s in cfg.data.train.pipeline)
    assert not any(s['type']=='LoadCalibrationReference' for s in cfg.data.val.pipeline)
    options=next(s for s in cfg.data.train.pipeline if s['type']=='SemanticContextAugment')
    assert options['roi_size']==128 and options['brightness']==0 and options['exposure_probability']==.3
    old=load_config('adaint/configs/face_niguang_semantic_basis.yaml')
    assert not any(s['type']=='LoadCalibrationReference' for s in old.data.train.pipeline)
    assert next(s for s in old.data.train.pipeline if s['type']=='SemanticContextAugment').get('roi_size',0)==0


def test_selection_rejects_background_regression_and_uses_compatible_fallback(tmp_path):
    baseline=dict(PSNR=27.,deltaE_P3_foreground=4.,deltaL_P3_foreground=3.,
                  deltaL_P3_background=3.5,deltaL_P3_highlights=3.)
    tolerance=dict(psnr_drop=.05,background_L_increase=.1,highlight_L_increase=.1,foreground_L_increase=.05)
    good=dict(mode='val',epoch=1,PSNR=27.,deltaE_P3_foreground=3.9,deltaL_P3_foreground=2.9,
              deltaL_P3_background=3.5,deltaL_P3_highlights=3.)
    for epoch in (1,2):(tmp_path/('epoch_%d.pth'%epoch)).touch()
    bad=dict(good,epoch=2,deltaE_P3_foreground=3.,deltaL_P3_background=4.)
    result=choose_guarded_checkpoint([good,bad],str(tmp_path),baseline,tolerance,76)
    assert result['progress']==1 and not result['using_baseline_fallback']
    for key,value in [('PSNR',26.),('deltaL_P3_background',4.),('deltaL_P3_foreground',4.),
                      ('deltaL_P3_highlights',4.),('deltaE_P3_foreground',float('nan'))]:
        assert choose_guarded_checkpoint([dict(good,**{key:value})],str(tmp_path),baseline,tolerance,76)['using_baseline_fallback']
    assert choose_guarded_checkpoint([good,dict(good,PSNR=float('nan'))],str(tmp_path),baseline,tolerance,76)['using_baseline_fallback']


CUDA=pytest.mark.skipif(not torch.cuda.is_available(),reason='native AiLUT CUDA')


@CUDA
def test_zero_grid_exactly_preserves_its_complete_basis_and_pure_background(tmp_path):
    basis=small_model(lut_mode='basis').cuda().eval()
    for out in basis.local_decoder.outputs:torch.nn.init.normal_(out.weight,std=.01)
    path=str(tmp_path/'basis.pth');torch.save(dict(state_dict=basis.state_dict()),path)
    model=calibration_model().cuda().eval();initialize_bilateral_from_basis(model,path)
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
def test_gt_and_reference_are_not_inference_inputs_and_missing_training_data_fails():
    model=calibration_model().cuda().eval();data=calibration_batch()
    with pytest.raises(ValueError,match='reference'):model.forward_train(**{k:v for k,v in data.items() if k!='reference_lightness'})
    with torch.no_grad():
        first=model.forward_dummy(data['lq'],foreground_mask=data['foreground_mask'])[0]
        data['gt'].zero_();data['reference_lightness'].fill_(1);data['foreground_roi_gt'].fill_(1)
        second=model.forward_dummy(data['lq'],foreground_mask=data['foreground_mask'])[0]
    assert torch.equal(first,second)


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
