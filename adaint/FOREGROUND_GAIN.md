# 前景双向空间亮度残差

DB94：`/DB/data/yujietang/IQ-LUT/image_enhance/AdaInt_BAN`。
仍在 `feature/spatial-tone__from_basis-lut` 分支，本次父提交为 `03d18a1`。

## 本次结构

```text
原双路 RGB LUT + 原区域零均值局部仿射（全部冻结）
  ├─ 前景基础结果 → 新增单通道空间增益 → 前景结果
  └─ 背景基础结果
                         ↓
                    原 mask 融合
```

新增配置：`adaint/configs/face_niguang_foreground_gain.yaml`。
新模型不构建多曲线预测头、空间混合权重头或背景 tone。
历史 `spatial_tone_enabled` 模式仍可加载已有 checkpoint 和复现旧实验；它与新模式互斥。
未改动历史配置、数据、mask 或旧实验目录。

`ForegroundSpatialGain` 复用原局部解码器冻结的 shared 特征。新增轻量 adapter、
前景条件 FiLM 和一个单通道输出头；同时输入原图/基础增强结果的线性亮度、
3×3/7×7 亮度均值、局部对比度、亮度差和前景 mask。
基础增强亮度由同一个完整上下文网格上的 LUT + 局部仿射输出计算，不能从训练裁剪独立生成。

原局部分辨率为 128，复用的特征网格为 64×64。增益 logits 先做完整上下文引导滤波，
再按原 crop_box 上采样，最后转为有界对数增益：

`log(g) = log(gain_max) * tanh(logit)`，默认 `gain_max=2`，即增益范围 `[0.5, 2]`。

在 Display P3 线性 RGB 上用共同增益调整前景，保留公共色域增益限制。
输出头零初始化，残差回写确保初始输出与 baseline 完全一致，包括原有扩展 RGB 值。
近黑像素不需要除以亮度；纯黑保持黑色。新增增益不做区域零均值约束。

冻结 backbone、两路 LUT/AdaInt、原局部特征与输出头，原模块始终 eval（关闭 dropout），
包括 runner 每次调用 train() 后和从 checkpoint 续训后。
仅新增 `foreground_gain.*` 参数可训练，共 **16,625** 个；总参数 **2,302,649** 个。
`adaint_fix_epochs=0` 不会在本模式中解冻 AdaInt：这里固定的是整个原模型、持续整个训练。

背景支路逐值保持基线。融合 mask 等于 0 的像素保持基线；羽化边带仍会按原软 mask 混入前景改动。

## 损失与监控

- 保留原 RGB、P3、区域监督及原 LUT/局部正则。冻结参数对应正则仍记录，但不会更新原模型。
- 新增前景 L* Charbonnier 监督，作用于最终融合结果，按逐图有效前景面积归一化。
  `region_lightness_weight=0.1`，`foreground_lightness_weight=1`。
- `shadow_weight=0`，取消额外暗部加权；`background_lightness_weight=0`，不生成新增背景明度损失。
- 实际应用的对数增益幅度正则 0.001；实际完整上下文增益图的边缘感知平滑正则 0.0001。
  不使用曲线斜率或曲线混合权重正则。
- 日志新增 `gain_foreground_mean/min/max`、`gain_foreground_log_std`、
  `gain_foreground_delta_L`、`loss_gain_magnitude`、`loss_gain_smooth`。
  保留前景 L*/ab、背景/高光 L*、PSNR 等指标，默认仍按 PSNR 选择 best。

## 运行训练

在 DB94 执行：

```bash
cd /DB/data/yujietang/IQ-LUT/image_enhance/AdaInt_BAN
git switch feature/spatial-tone__from_basis-lut
source /DB/rhome/yujietang/miniconda3/etc/profile.d/conda.sh
conda activate AdaInt_BAN

CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=2 python tools/train_kd_unified.py \
  adaint/configs/face_niguang_foreground_gain.yaml --seed 2026
```

GPU 编号按实际可用设备调整。本次短检查在共享 GPU 上限额运行，不代表 GPU 5 处于空闲。
必须使用上面的 unified 入口，它会处理 baseline 初始化、训练预算、保存和推理。

配置从 `work_dirs/face_niguang_semantic_basis/gen_0/best.pth` 加载原模型，
只允许新增 `foreground_gain.*` 参数缺失，其他参数名和形状必须全部匹配。
旧 tone checkpoint 不能作为 `gain_init_from` 使用；新运行计数和优化器从零开始。
没有初始化来源的 gain 训练会明确报错，避免冻结随机基线。

默认 100 epochs、batch=1、Adam LR=1e-4，cosine 最低 LR=1e-6；
原 76/9 划分、增强和 mask 机制保持不变，BAN 关闭，没有预热后全量解冻阶段。
配置内 seed=2026；命令行未传 --seed 时使用此值，传入时以命令行为准。

输出目录：`work_dirs/face_niguang_foreground_gain/gen_0/`：
`epoch_N.pth`、`latest.pth`、`best.pth`、日志、`resolved_config.py`、`inference_config.py`；
结束后在 `train_infer/` 导出训练和验证图片。不会覆盖三个已有实验目录。

续训：

```bash
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=2 python tools/train_kd_unified.py \
  adaint/configs/face_niguang_foreground_gain.yaml --seed 2026 \
  --resume-from work_dirs/face_niguang_foreground_gain/gen_0/latest.pth
```

续训恢复完整 gain 模型、优化器和计数，跳过 baseline 重新初始化；原模型始终冻结。

单图推理：

```bash
CUDA_VISIBLE_DEVICES=5 python adaint/demo.py \
  work_dirs/face_niguang_foreground_gain/gen_0/inference_config.py \
  work_dirs/face_niguang_foreground_gain/gen_0/best.pth \
  /path/to/123_input.jpg /path/to/123_result.png
```

沿用原 mask 约定，需要配置 test_foreground_root 下存在 `123_face_mask.png`。
模型不自动转换输入 ICC，也不改变原导出色彩约定。

## 实际验证（2026-10-19）

```bash
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=2 python -m pytest -o addopts="" -q \
  tests/test_adaint/test_foreground_gain.py \
  tests/test_adaint/test_spatial_tone.py tests/test_adaint/test_semantic_dual.py

CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=2 python -m pytest -o addopts="" -q \
  tests/test_adaint/test_basis_ablation.py \
  -k "not test_only_lut_generation_changes_in_existing_production_files"

CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=2 python tools/smoke_semantic_dual.py \
  adaint/configs/face_niguang_foreground_gain.yaml \
  --steps 2 --max-side 256 --check-mask-coverage
```

- 新 gain/历史 tone/双路测试 41 项通过，basis 回归 13 项通过，共 54 项。
  排除 1 项历史静态断言，它要求整个模型 forward/loss 与旧 LUT-only 任务逐字相同；未修改该旧测试。
- 验证恒等初始化、首步重建梯度、双向增益及范围、共同通道比例、裁剪一致性、空区域、
  原局部零均值、背景保持、冻结与 eval、严格 checkpoint/续训及旧模型兼容。
- 真实数据 mask 覆盖 76/9/9，完成 2 步优化与验证，所有损失/梯度/指标有限。
- 正式入口完成 2 张训练图、1 张验证图的 2 epochs（4 updates），原生分辨率；
  保存的两个 checkpoint 中，全部 71 个旧模型 state 项（不计计数器）与 basis best 完全相等，
  新增输出头已更新，seed 元数据为 2026。
- 从 epoch_1 恢复正式训练入口后，计数/优化器正常，原模型仍完全冻结；
  完成 best 选择、推理配置保存和 3 张 750×1000 满分辨率导出。
- 短检查产物：`/tmp/adaint-foreground-gain-sanity-so479ls_/run/gen_0/`。

没有启动 100 epochs 主实验。以上确认实现与运行链路正常，增强收益仍需正式训练结果验证。

## 改动文件

- `foreground_gain.py`：新增前景空间增益适配器。
- `semantic_modules.py`：局部解码器可选返回原共享特征，不增加旧模型参数。
- `model_semantic.py`：新增模式、前景校正、固定训练模式、损失和监控。
- `experiment_utils.py`：严格 baseline→gain 初始化。
- `tools/train_kd_unified.py`：新初始化入口、固定基线调度日志、配置 seed 回退。
- `tools/smoke_semantic_dual.py`：支持新模式真实数据检查。
- 新 YAML、新 gain 测试和本文；历史说明增加新方案入口。
