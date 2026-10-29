# Range projection with the original affine enhancer

Parent: `feature/result-postprocess__from_basis-lut`, `0757473`.
The two basis LUTs, result adapter, multiscale affine fields, regional centering,
mask fusion and DC calibration are unchanged. The optional `model.range_project`
adds no trainable parameters or checkpoint state keys. Disabled is the default.

When enabled, the complete DC-corrected raw proposal Z is projected onto [0,1]
with per-channel clamp. Every already legal channel is unchanged, including
normal .96/.98/1.0 highlights. There is no fixed shoulder, no source-anchored
common-step fallback, and nothing unbounded is added after projection.

The original main RGB reconstruction directly supervises Z; existing final
RGB/P3/region losses and regularizers remain. Additional terms are actual-Z
range distance (.1), foreground/background difficult raw RGB MSE (.1), and
GT-matching difficult-neighborhood gradient MSE (.01). Difficult pixels are the
top 10% of detached RGB-error plus range-distance scores per image and region.
Edges with either selected endpoint are included within the semantic region.
All legal GTs are fixed points of the projection and have zero range penalty.
The projection guarantees range; image quality requires experiment evidence.

The three YAMLs `face_{indoor,outdoor,club}_semantic_range_project.yaml` specify
100 epochs, seed 2026 through the runner, Adam 1e-4/cosine to 1e-6, no frozen
parameters, no pretrained/warm-start/resume/teacher, and retain source data and
augmentation settings. Club's 142/16/30 split and person masks are copied from
the existing club dataset adaptation, not regenerated. Indoor/outdoor retain
their luminance-guided experiment settings. All use infer_after_generation=true.

```bash
python -B tools/run_range_project.py prepare
python -B tools/run_range_project.py controller --gpu 7
```

Preparation validates files/masks and refuses existing experiment prefixes.
Outputs are `work_dirs/rp_20261029_{indoor,outdoor,club}_s2026/gen_0/`.
The unchanged trainer exports each best checkpoint's native train+val images
once into `train_infer/`. There is no second train inference pass. Blind test
outputs are `work_dirs/test_final_range_project_20261029/{scene}/raw/`, with
matching inference configs and execution manifests. PNGs have no added ICC.
The controller performs no result analysis, ablation, automatic retry or repeat.

Checks: `python -B -m pytest -o addopts= -p no:cacheprovider -q tests/test_range_project.py`.
