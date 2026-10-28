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
```

Use a fresh prefix/suite; the generator refuses to overwrite experiments. Its
source config supplies architecture and data settings, never source weights.
Each run saves `resolved_config.py`, `inference_config.py`, epoch checkpoints,
validation-PSNR-selected `best.pth`, logs and `runtime_resources.json` under
`work_dirs/<prefix>_<scene>_<variant>_s2026/gen_0/`.

## Inference and evidence

```bash
python tools/evaluate_highlight_guard.py paired \
  --config work_dirs/bc_repro_outdoor_detail_s2026/gen_0/inference_config.py \
  --checkpoint work_dirs/bc_repro_outdoor_detail_s2026/gen_0/best.pth \
  --split val --device cuda:0 --output work_dirs/bc_repro_suite/detail/val \
  --decompose --p3
```

Use `--split train` for paired training evaluation. Test mode uses the same
evaluator's `test` subcommand, test image/mask roots and the existing scene ID
files. Blind test never uses GT, training, or checkpoint selection. Canonical
paired evaluation AREA-aligns input to GT; native exports are distinct evidence.

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
