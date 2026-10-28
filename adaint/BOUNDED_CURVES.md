# Bounded LUT and spatial curves, trained from scratch

Implemented on `feature/highlight-guard__from_result-postprocess` at the user's
request, based on `af1616d`. The old model path and checkpoint schema remain
available when `model.bounded_curves.enabled` is false. Old results are preserved.

## Mapping

The two semantic branches retain their conditional features, five signed bases,
and adaptive RGB coordinates. The bases now represent small, distinct logit
residuals. The actual LUT is

`1e-6 + (1-2e-6) * sigmoid(logit(eps+(1-2eps)*identity(vertices)) + residual)`.

The identity follows the current adaptive coordinates; coordinate endpoints are
explicitly zero and one. This bounds actual LUT nodes before interpolation.

The local decoder retains result conditioning, FiLM and coarse/fine features.
It predicts 4-segment RGB curves: six masses per channel, including learnable
black and white reserves. Coarse/fine residual logits are restored with guided
filtering before softmax and cumulative normalization. Each RGB channel is
evaluated separately to limit memory. Monotone knots and interpolation keep
outputs within the numerical interior of [0,1], without a final output clamp.
The near-identity epsilon (.001) initializes endpoints; the numerical margin
(1e-6) is separate. White can be compressed; neither endpoint is fixed at its
initial value. Hard regional centering and all independent DC additions are
removed. Semantic fusion remains a convex mask mixture.

The full bounded model requires basis mode, no frozen scope or AdaInt freeze,
no teacher, and no guard soft penalties. Every model parameter is trainable.
The old `heads` method is retained only for historical checkpoint compatibility.

## Matched experiments

All groups use the original train/validation splits, 100 epochs, seed 2026,
Adam LR 1e-4 with cosine decay to 1e-6, and one image per GPU. No pretrained,
warm-start, load or resume checkpoint is used for a fresh run. Full-context
geometry, masks and crop coordinates are retained; brightness/saturation jitter
is disabled to avoid introducing uint8 clipping. `legacy` is also from scratch
under these same data and training rules, so it is the fair comparison.

| Variant | Change |
| --- | --- |
| legacy | Original unbounded RGB LUT/local/DC architecture |
| bounded | Bounded LUT and spatial curves, original losses |
| detail | bounded plus symmetric GT-highlight RGB MSE and gradient MSE |
| no_result | detail without result-adapter conditioning |
| no_fine | detail without the fine spatial scale |
| legacy_jitter | Optional legacy control retaining source brightness/saturation jitter |

The detail weights are declared starting values (highlight=1, gradient=.1).
GT brightest 10% is selected separately in each semantic region. Positive and
negative RGB errors receive the same cost. Gradient edges require both endpoints
inside the GT-highlight set. Losses normalize within each valid image/region.

Run in the DB94 project root with the `AdaInt_BAN` environment:

```bash
python tools/run_bounded_curves.py --scene outdoor \
  --source-config work_dirs/face_outdoor_semantic_postprocess/gen_0/inference_config.py \
  --suite-dir work_dirs/bc_repro_suite --prefix bc_repro \
  --variants legacy bounded detail no_result no_fine --epochs 100 --seed 2026

python tools/run_bounded_curves.py --scene indoor \
  --source-config work_dirs/face_indoor_semantic_postprocess_4/gen_0/inference_config.py \
  --suite-dir work_dirs/bc_repro_suite --prefix bc_repro \
  --variants legacy bounded detail --epochs 100 --seed 2026

python tools/launch_highlight_guard.py \
  work_dirs/bc_repro_suite/manifest_outdoor.json \
  work_dirs/bc_repro_suite/manifest_indoor.json \
  --gpus 0 1 2 3 4 5 6 7 --min-free-mb 7000 --memory-fraction .22 \
  --output work_dirs/bc_repro_suite/launcher

# Wait for the suite, evaluate each released GPU, and create both scene reports.
python tools/evaluate_bounded_suite.py --suite work_dirs/bc_repro_suite \
  --memory-fraction .22 --min-free-mb 6000
```

Use a fresh prefix/suite; the generator refuses to overwrite experiments. Its
source config supplies architecture and data settings, never source weights.
Each run saves `resolved_config.py`, `inference_config.py`, epoch checkpoints,
validation-PSNR-selected `best.pth`, logs and `runtime_resources.json` under
`work_dirs/<prefix>_<scene>_<variant>_s2026/gen_0/`.

For the augmentation control, use a separate suite/prefix and
`--variants legacy_jitter`; this is the same original architecture and scratch
budget, with only the source photometric jitter retained. For the endpoint
initialization control, use another separate suite/prefix with
`--variants bounded --epsilon .01`. This changes the shared global/local
near-identity prior and is not a global-only ablation. Neither control uses
validation or blind-test images during training.

## Inference and evidence

```bash
python tools/evaluate_highlight_guard.py paired \
  --config work_dirs/bc_repro_outdoor_detail_s2026/gen_0/inference_config.py \
  --checkpoint work_dirs/bc_repro_outdoor_detail_s2026/gen_0/best.pth \
  --split val --device cuda:0 --output work_dirs/bc_repro_suite/detail/val \
  --decompose --p3
```

Use `--split train` for paired training evaluation. A full outdoor blind-test
command is:

```bash
CUDA_VISIBLE_DEVICES=7 python tools/evaluate_highlight_guard.py test \
  --config work_dirs/bc_repro_outdoor_detail_s2026/gen_0/inference_config.py \
  --checkpoint work_dirs/bc_repro_outdoor_detail_s2026/gen_0/best.pth \
  --input-root /DB/data/yujietang/IQ-LUT/face_dataset/test_final \
  --mask-root /DB/data/yujietang/IQ-LUT/face_dataset/test_final_masks/masks \
  --ids-file work_dirs/test_final/outdoor_ids.txt --device cuda:0 \
  --memory-fraction .22 --decompose --p3 \
  --output work_dirs/bc_repro_suite/detail/test
```

Indoor inference uses its own config/checkpoint and `indoor_ids.txt`.
Blind test never uses GT, training, or checkpoint selection. Canonical
paired evaluation AREA-aligns input to GT; native exports are distinct evidence.
Add `--native-paired` to paired evaluation to also export full-input-resolution
images into `native_raw/` and independent `p3.py` copies into `native_raw_p3/`.
Run native exports sequentially on a card with sufficient free memory: their
12-megapixel spatial coefficient fields require much more memory than training
crops or the canonical GT-size evaluation. Canonical PSNR always uses the
paired GT size; native range statistics are reported separately.

Training and inference save raw PNGs without ICC. `--p3` independently calls
`p3.py` into `raw_p3/`, then checks the expected Display P3 profile and exact
decoded-pixel equality. Numerical range is measured BEFORE PNG clipping and
quantization. Zero out-of-range output alone does not establish correct
highlights: compare symmetric error, signed brightness, GT gradient and images.

`tools/analyze_bounded_curves.py --suite <suite> --output <suite>/report`
reads `report_manifest.json` to build quantitative and per-image HTML evidence.
Old warm-start experiments are historical references, not matched scratch
controls. Final conclusions must distinguish these comparisons and report
regressions, compute cost and unresolved cases.

Checkpoint selection uses paired validation only, with test images excluded.
The fixed 100th-epoch train/validation diagnostic answers a separate question
about final fitting behavior; it does not silently replace validation-selected
checkpoints. A strong validation value at an early epoch can coexist with poor
training-set reproduction, so both are necessary for interpreting this dataset.

Learning curves can be generated with `tools/plot_bounded_training.py`. It needs
Matplotlib only for plotting; `--no-plots` writes summaries using the standard
library. When plotting copied server logs locally, repeated `--path-map OLD=NEW`
arguments map work-directory prefixes without changing manifests or provenance.

## DB94 results, 2026-10-28

Eleven 100-epoch scratch runs completed (main eight, two original-jitter
controls, one outdoor epsilon=.01 control). Canonical best-validation outdoor
PSNR was 27.5839 for matched legacy and 27.7826 for detail; training PSNR was
29.6911 and 29.7636. The historical original was 25.8580 validation / 29.7405
training. The larger validation gain over that historical model must not all be
attributed to this structure: disabling photometric jitter is a large contributor.

At the fixed 100th epoch, outdoor legacy/detail train PSNR was 30.7549/29.8321,
and indoor legacy/detail was 32.9598/32.4457. Thus the new bounded structure has
not achieved superior matched-budget training fitting. All new bounded variants
had exactly zero upper/lower raw exceedance on their evaluated train/val/test
images, but GT-relative highlight errors, foreground color tradeoffs and visible
background color failures remain. Zero exceedance does not prove correct tone.

Indoor validation-selected detail was epoch 4 (train27.7895/val29.7435), while
the old `_4` was train31.9867/val29.9169. At epoch100, detail improved training
to32.4457 but validation declined to29.1996. It is not an overall replacement
for `_4`; indoor added detail losses also did not improve final training PSNR
over plain bounded. Outdoor removing the fine scale lost .404dB validation;
removing result conditioning lost only .069dB and improved foreground Delta E,
so no_result remains a simpler candidate rather than a proven unnecessary part.
These are one-seed observations, not significance or generalization claims.

Main evidence is `work_dirs/bc_20261028_suite/`:

- `final_review/{outdoor,indoor}/report/report.html`: all image comparisons.
- `combined_experiment_summary.{json,csv}` and `combined_selection.json`:
  matched controls, final-epoch diagnostics and validation-only selection.
- `case_diagnostics/`: fixed test cases 1,16,17,30; no test GT or retuning.
- `verification/verification_summary.json`: pre-cleanup audit of 2,245 raw/P3
  PNG pairs, zero errors; `final_regression.log`: 133 tests passed.
- `verification/metadata_snapshot.json`: all configurations, runtime evidence,
  checkpoint hashes and evaluation/export records preserved before cleanup.

At the user's request no further experiments were added, and poor-trial weights
and redundant epoch checkpoints were deleted after evaluation and evidence backup.
Retained weights are outdoor legacy best, detail best/epoch100, no_result best;
indoor legacy best, bounded best/epoch100. Other variants retain configuration,
metrics and display previews. See `verification/checkpoint_cleanup.json` and
the artifact cleanup record for exact deletion/retention lists. Do not rerun an
orchestrator against the intentionally pruned old suite; reproduce with a fresh
prefix. Existing historical models and data were not changed.
