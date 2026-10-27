"""Protocol tests for isolated configs and validation-only guarded selection."""
import argparse
import copy
import json

from tools.run_highlight_guard import guard_variant, prepare_config
from tools.evaluate_highlight_guard import (cropped, finite_means, raw_metrics,
                                            select_guard_checkpoint, select_main)


def source_config():
    return dict(model=dict(type='AiLUTDual',postprocess=dict(freeze_global=False)),
        data=dict(workers_per_gpu=4,train=dict(pipeline=[dict(type='LoadSemanticMasks')]),
                  val=dict(pipeline=[])),
        validation_test_cfg=dict(metrics=['PSNR','deltaE_P3_foreground'],crop_border=0),
        test_cfg=dict(metrics=None),train_cfg=None,
        optimizers=dict(type='Adam',lr=5e-5),lr_config={})


def options():
    return argparse.Namespace(top_fraction=.1,over_margin=2/255.,
        keep_error_threshold=.03,lut_range_weight=.01,final_range_weight=.1,
        over_weight=1.,keep_weight=.1)


def test_saved_inference_config_restores_validation_and_fresh_budget_without_pipeline_duplication():
    source=source_config();before=copy.deepcopy(source)
    cfg=prepare_config(source,'unit','work_dirs/unit','best.pth',{},30,2e-5,2)
    assert source==before
    assert cfg['train_cfg']=={} and cfg['train_epochs']==30
    assert cfg['test_cfg']['metrics'][0]=='PSNR'
    assert 'highlight_over_foreground' in cfg['test_cfg']['metrics']
    assert cfg['data']['train']['pipeline']==source['data']['train']['pipeline']
    assert cfg['postprocess_init']=='best.pth' and cfg['resume_from'] is None
    assert not cfg['infer_after_generation']


def test_variants_have_matched_scope_and_controlled_loss_switches():
    names=['legacy_control','heads_control','range','over','range_over','full','detach']
    variants={n:guard_variant(n,'base.py','base.pth',options()) for n in names}
    assert variants['legacy_control']['trainable_scope']=='all'
    assert all(variants[n]['trainable_scope']=='heads' for n in names[1:])
    assert variants['heads_control']['over_weight']==0
    assert variants['range']['over_weight']==0 and variants['range']['final_range_weight']>0
    assert variants['over']['lut_range_weight']==0 and variants['over']['over_weight']>0
    assert variants['full']['reference_checkpoint']=='base.pth'
    assert variants['range_over']['reference_checkpoint'] is None
    left,right=copy.deepcopy(variants['full']),copy.deepcopy(variants['detach'])
    assert not left.pop('detach_result_condition') and right.pop('detach_result_condition')
    assert left==right


def validation(epoch,psnr=30.,color=3.,fg=2.,over=.02):
    return dict(mode='val',epoch=epoch,PSNR=psnr,deltaE_P3=color,
        deltaE_P3_foreground=fg,highlight_over_foreground=over,
        highlight_over_background=over)


def test_guard_selection_uses_validation_constraints_and_latest_epoch_record(tmp_path):
    for epoch in range(1,5):(tmp_path/('epoch_%s.pth'%epoch)).touch()
    baseline=dict(PSNR=30.,deltaE_P3=3.,deltaE_P3_foreground=2.,
                  highlight_over_foreground=.03,highlight_over_background=.03)
    records=[validation(1,over=.01),validation(2,psnr=29.,over=0),
             validation(3,fg=3.,over=0),validation(4,over=.005),
             validation(4,over=.03),dict(validation(2,over=0),mode='test')]
    selected=select_guard_checkpoint(records,baseline,tmp_path)
    assert selected['selected']['epoch']==1
    assert {r['epoch'] for r in selected['rejected']}=={2,3,4}
    # An absent/invalid validation never falls back to blind-test outcomes.
    assert select_guard_checkpoint([dict(validation(1),mode='test')],baseline,tmp_path)['selected'] is None


def test_guard_selection_ties_use_color_then_psnr(tmp_path):
    for epoch in (1,2):(tmp_path/('epoch_%s.pth'%epoch)).touch()
    baseline=dict(PSNR=30.,deltaE_P3=3.,deltaE_P3_foreground=2.,
                  highlight_over_foreground=.03,highlight_over_background=.03)
    report=select_guard_checkpoint([validation(1,color=3.),validation(2,color=2.9)],baseline,tmp_path)
    assert report['selected']['epoch']==2


def test_finite_means_excludes_invalid_regions_without_diluting_with_zero():
    rows=[dict(metrics=dict(PSNR=30.,highlight_over_foreground=.06,
                            deltaE_P3_foreground=3.,_valid_foreground=1)),
          dict(metrics=dict(PSNR=20.,highlight_over_foreground=0.,
                            deltaE_P3_foreground=0.,_valid_foreground=0))]
    summary=finite_means(rows)
    assert summary['PSNR']==25.
    assert summary['highlight_over_foreground']==.06
    assert summary['deltaE_P3_foreground']==3.
    assert summary['foreground_valid_count']==1


def test_guard_selection_requires_highlight_improvement_beyond_tolerance(tmp_path):
    (tmp_path/'epoch_1.pth').touch()
    baseline=dict(PSNR=30.,deltaE_P3=3.,deltaE_P3_foreground=2.,
                  highlight_over_foreground=.02,highlight_over_background=.02)
    for over in (.03,.02,.02-5e-9):
        report=select_guard_checkpoint([validation(1,over=over)],baseline,tmp_path)
        assert report['selected'] is None and report['winner']=='baseline'
        assert 'no highlight improvement over baseline' in report['rejected'][0]['reasons']
    assert select_guard_checkpoint([validation(1,over=.02-2e-8)],baseline,tmp_path,
                                  min_relative_improvement=0)['selected']


def test_repeated_selection_removes_stale_derived_candidate_when_baseline_wins(tmp_path):
    baseline=tmp_path/'baseline.json'
    baseline.write_text(json.dumps(dict(summary=dict(PSNR=30.,deltaE_P3=3.,deltaE_P3_foreground=2.,
        highlight_over_foreground=.02,highlight_over_background=.02))))
    (tmp_path/'epoch_1.pth').write_bytes(b'epoch candidate')
    (tmp_path/'best_guard.pth').write_bytes(b'stale previous selection')
    (tmp_path/'run.log.json').write_text(json.dumps(validation(1,over=.03))+'\n')
    select_main(argparse.Namespace(baseline=baseline,work_dir=tmp_path,
        psnr_tolerance=.1,foreground_color_tolerance=.15,
        highlight_improvement_tolerance=1e-8))
    assert not (tmp_path/'best_guard.pth').exists()
    assert (tmp_path/'epoch_1.pth').read_bytes()==b'epoch candidate'
    result=json.loads((tmp_path/'best_guard_selection.json').read_text())
    assert result['winner']=='baseline' and result['selected'] is None
    assert result['removed_stale_best_guard']['reason'].startswith('baseline won')


def test_guard_selection_requires_five_percent_and_limits_each_region_and_global_color(tmp_path):
    for epoch in (1,2,3,4):(tmp_path/('epoch_%s.pth'%epoch)).touch()
    baseline=dict(PSNR=30.,deltaE_P3=3.,deltaE_P3_foreground=2.,
                  highlight_over_foreground=.02,highlight_over_background=.02)
    insufficient=validation(1,over=.0195)  # 2.5% only.
    unbalanced=validation(2,over=.01)
    unbalanced['highlight_over_foreground']=.023
    bad_color=validation(3,over=.018,color=3.16)
    accepted=validation(4,over=.019)  # Exactly the pre-registered 5% reduction.
    report=select_guard_checkpoint([insufficient,unbalanced,bad_color,accepted],baseline,tmp_path)
    assert report['selected']['epoch']==4
    reasons={r['epoch']:r['reasons'] for r in report['rejected']}
    assert 'relative highlight improvement below minimum' in reasons[1]
    assert 'foreground highlight regression tolerance' in reasons[2]
    assert 'global color tolerance' in reasons[3]


def test_raw_metrics_respect_crop_border_and_region_min_area():
    import torch
    output=torch.full((1,3,5,5),1.2)
    output[...,1:-1,1:-1]=.5
    mask=torch.zeros((1,1,5,5));mask[...,2,2]=1
    values=raw_metrics(output,mask,crop_border=1,min_area=.2)
    assert values['raw_high_clip']==0 and values['_valid_foreground']==0
    assert values['raw_high_clip_foreground'] is None
    assert raw_metrics(output,mask,crop_border=1,min_area=.1)['_valid_foreground']==1
    assert raw_metrics(output,mask,crop_border=0)['raw_high_clip']>0


def test_raw_save_and_separate_p3_processing_preserve_pixels(tmp_path):
    # The numerical/image dependency check is separate from the pure config
    # tests so the latter remain runnable without a training environment.
    import numpy as np
    from PIL import Image
    from p3 import embed_icc_to_png
    from tools.evaluate_highlight_guard import verify_p3_export
    raw,p3=tmp_path/'raw',tmp_path/'raw_p3'
    raw.mkdir();p3.mkdir()
    array=np.array([[[0,128,255],[17,91,203]]],dtype=np.uint8)
    Image.fromarray(array).save(str(raw/'unit.png'))
    embed_icc_to_png(str(raw/'unit.png'),str(p3/'unit.png'))
    result=verify_p3_export(raw,p3)
    assert len(result)==1 and result[0]['pixels_equal']
