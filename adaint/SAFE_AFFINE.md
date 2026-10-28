# Safe affine enhancement

Based on `feature/result-postprocess__from_basis-lut` (`0757473`). The original
two basis LUT mappings, result adapter, coarse/fine affine fields, regional
centering, mask fusion and DC calibration are retained. `model.safe_affine`
defaults to disabled and adds no parameters/state keys.

When enabled, the final unbounded proposal passes through a C1 highlight
shoulder in linear Display P3. Its three channels share one gain. The encoded
knee is .92; this is a declared first-experiment value, not a tuned optimum.
Negative/out-of-range candidates then use the largest feasible common RGB step
from the original input, with a floating-point interior allowance on active
steps. Pixels with no shoulder or range violation bypass all conversions
exactly. The endpoint operation occurs after DC; nothing is added afterwards.
It guarantees range, not correct exposure or recovery of already clipped input.

The normal final RGB/P3/regional objectives are retained. Added supervision is
symmetric GT-region brightest-10% RGB MSE (weight 1), RGB-gradient MSE (.1), and
raw-proposal RGB MSE in the detached risk set (.25), preserving a direct gradient
when the projection saturates. No teacher/frozen scope or bounded LUT replacement
is used. All model parameters train from scratch.

The authorized run is exactly indoor/outdoor once each: 100 epochs, seed 2026,
Adam 1e-4/cosine to 1e-6, source geometric/photometric augmentations retained.
The sources are the saved `_4`/outdoor configs, never their weights. `best.pth`
uses validation PSNR only. Native train/test images are each inferred once from
that checkpoint; no second canonical forward or endpoint re-evaluation is run.
Paired image metrics resize native saved PNGs to GT with AREA interpolation.

Run from the DB94 project with the AdaInt_BAN interpreter:

```bash
python tools/run_safe_affine.py prepare
python tools/run_safe_affine.py controller --gpu 7
```

Preparation refuses an existing prefix. Outputs are
`work_dirs/sa_20261028_{indoor,outdoor}_s2026/gen_0/`; inference has
`inference/{train,test}/{raw,raw_p3}` plus a manifest. Raw PNGs have no ICC;
`p3.py` is called separately and its pixels/profile are checked. The two scene
test sets are hardlinked into `work_dirs/sa_20261028_suite/test_final/`.

Necessary checks: `python -m pytest -o addopts= -q tests/test_safe_affine.py`.
The explicit addopts override prevents this legacy repository from adding the
entire unrelated tests directory automatically.
