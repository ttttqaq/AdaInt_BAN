# 双路 LUT 后接空间自适应亮度 tone

DB94 仓库：`/DB/data/yujietang/IQ-LUT/image_enhance/AdaInt_BAN`。
新分支：`feature/spatial-tone__from_basis-lut`，父提交：`33a4890`。

## 方法与修改范围

每路执行：RGB basis LUT → 原有区域零均值局部仿射 → 新增亮度 tone，最后按原有前景 mask 融合。
保留原 LUT、AdaInt、局部模块、RGB/P3 损失、数据划分和增强流程。旧配置默认
`spatial_tone_enabled=false`、`region_lightness_weight=0`，保留原 checkpoint schema 和行为。

新增 `spatial_tone.py`：

- 采用 Display P3 的 sRGB 传递函数和 P3/D65 Y 系数，在线性 RGB 上计算亮度。
  抽查 ID 10、11 的输入与 GT，四张图片 ICC 均为 Display P3；读取管线仍不执行 ICC 转换。
- 每路由独立区域条件头预测 3 条 16 段曲线。正增量归一化累积，端点为 0、1；
  均匀增量与 softmax 增量混合，对斜率施加硬上限 4，默认最小斜率为 0.8。
- 共享轻量空间网络读取 RGB、线性亮度、多尺度亮度和 mask，使用独立区域 FiLM/输出头。
  在完整上下文预测低分辨率 logits，经过引导滤波和同坐标上采样，再做 softmax。
- 用曲线凸组合产生目标 Y，通过共同线性 RGB 增益回写，增益硬上限为 4。
  近黑像素使用首段斜率；纯黑保持黑色。新增模块不施加区域零均值约束。
- 对在色域内的输入沿 RGB 射线限制共同增益，避免新增逐通道裁剪造成偏色。
  原 LUT 已有的负值/超色域值在恒等初始化时保留，最终导出仍沿用原有处理。

`model_semantic.py` 新增最终融合输出上的区域 L* Charbonnier 监督：
前景/背景逐图分别按有效加权面积归一化，空/微小区域跳过。暗部权重仅由输入
L*/100 决定，默认从 1 到 3，L*/100 ≥ 0.5 时权重为 1；目标始终是 GT。
保留原有损失，新增过大增益、过陡曲线、空间权重跳变三个正则。
日志记录两路 tone 增益均值/最大值及原局部修正幅度。

验证新增 `deltaL_P3_background` 和 `deltaL_P3_highlights`；高光定义为 GT L* ≥ 80，
无高光图像不参与该指标均值。保留 PSNR、前景明度/色度等原指标，仍按 PSNR 选 best。

`experiment_utils.py` 和 `train_kd_unified.py` 新增 `tone_init_from`：只允许新增
`spatial_tone.*` 参数缺失，所有旧参数名/形状必须严格匹配；不加载旧优化器，计数清零。
`load_from`/`resume_from` 仍严格要求完整模型匹配。曲线初始化为恒等，空间头用微小非对称初始化。
计数随 checkpoint 保存；续训根据计数恢复 tone-only/joint 阶段。

## 正式训练

在 DB94 执行：

```bash
cd /DB/data/yujietang/IQ-LUT/image_enhance/AdaInt_BAN
git switch feature/spatial-tone__from_basis-lut
source /DB/rhome/yujietang/miniconda3/etc/profile.d/conda.sh
conda activate AdaInt_BAN

CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=2 python tools/train_kd_unified.py \
  adaint/configs/face_niguang_semantic_spatial_tone.yaml --seed 2026
```

默认从现有 `work_dirs/face_niguang_semantic_basis/gen_0/best.pth` 初始化原模型。
100 epochs、单图 batch、训练 76 张、验证 9 张，BAN 关闭。
前 3 epochs（228 iterations）固定全部原模型参数并关闭原模块 dropout，仅训练 tone；
随后 97 epochs 联合微调。原参数起始 LR 为 1e-5，tone 为 1e-4，AdaInt 为 1e-6，
沿用 cosine 调度。`adaint_fix_epochs=0`，因为 tone 预热阶段已经冻结两路 AdaInt。

结果目录：`work_dirs/face_niguang_semantic_spatial_tone/gen_0/`，包括
`epoch_N.pth`、`latest.pth`、`best.pth`、`resolved_config.py`、`inference_config.py`；
结束后 `train_infer/` 导出训练集和验证集满分辨率结果。

续训（同一配置、同一工作目录；自动跳过 baseline 初始化）：

```bash
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=2 python tools/train_kd_unified.py \
  adaint/configs/face_niguang_semantic_spatial_tone.yaml --seed 2026 \
  --resume-from work_dirs/face_niguang_semantic_spatial_tone/gen_0/latest.pth
```

单图推理（mask 仍按原输入 ID 从配置的 test_foreground_root 加载）：

```bash
CUDA_VISIBLE_DEVICES=5 python adaint/demo.py \
  work_dirs/face_niguang_semantic_spatial_tone/gen_0/inference_config.py \
  work_dirs/face_niguang_semantic_spatial_tone/gen_0/best.pth \
  /path/to/123_input.jpg /path/to/123_result.png
```

## 消融和关键参数

复制新 YAML 并为每个实验设置独立 `exp_name`/`work_dir`，避免覆盖结果。
以下四组应使用同一 baseline checkpoint、数据划分、随机种子、微调 LR 与 100 epochs 预算：

| 组别 | spatial_tone_enabled | region_lightness_weight | tone_curves | tone_warmup_epochs |
| --- | --- | --- | --- | --- |
| A，原结构微调对照 | false | 0 | 忽略 | 0 |
| B，仅新增明度监督 | false | 0.1 | 忽略 | 0 |
| C，每路单条亮度曲线 | true | 0.1 | 1 | 3 |
| D，空间混合亮度曲线 | true | 0.1 | 3 | 3 |

历史 baseline checkpoint 可以直接提供原模型推理参照；严格的微调对照应另跑 A，
记录 C/D 预热与联合优化步数不同。单条曲线时 softmax 恒为 1，空间网络不影响输出。
原 `face_niguang_semantic_basis.yaml` 完全保留，可重跑原从头训练实验。

新参数位于 YAML 的 model 中：`tone_segments/resolution/channels` 控制曲线段数与空间网络；
`tone_max_gain/max_slope` 为硬限制，`tone_gain_limit/slope_limit` 为软正则阈值；
`tone_gain_factor/slope_factor/spatial_factor` 默认 0.001/0.001/0.0001。
`foreground_lightness_weight/background_lightness_weight` 默认各为 1，
由 `region_lightness_weight=0.1` 统一缩放。`shadow_weight=2` 控制额外暗部权重。

## 已完成验证（2026-10-19）

```bash
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=2 python -m pytest -o addopts="" -q \
  tests/test_adaint/test_spatial_tone.py tests/test_adaint/test_semantic_dual.py

CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=2 python -m pytest -o addopts="" -q \
  tests/test_adaint/test_basis_ablation.py \
  -k "not test_only_lut_generation_changes_in_existing_production_files"

CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=2 python tools/smoke_semantic_dual.py \
  adaint/configs/face_niguang_semantic_spatial_tone.yaml \
  --steps 2 --max-side 256 --check-mask-coverage
```

- 机制与双路回归 28 项通过；basis 回归 13 项通过。
  排除 1 项历史静态断言：它要求模型全部 forward/loss 与 `fcaf8bc` 字节相同，
  仅适用于当时只替换 LUT 生成器的任务，与本次明确新增方法不相容；未修改该旧测试。
- mask 覆盖 train/val/test = 76/9/9；真实数据 1 步 tone-only、1 步 joint 和验证通过，
  所有损失、梯度和指标有限。新模型 2,412,206 个参数。
- 正式入口用 2 张真实训练图、1 张验证图完成 2 epochs（4 updates），原生分辨率，
  成功保存 checkpoint/推理配置、选择 best，并导出 3 张 750×1000 结果。
  产物：`/tmp/adaint-spatial-tone-sanity-fk9d9huu/run/gen_0/`。
- 正式入口从该预热末尾的 `epoch_1.pth` 恢复训练，正确恢复计数/优化器，
  在 iteration 2 进入 joint 阶段，并再次完成验证、best 选择与原生分辨率导出。

尚未运行 100 epochs 主实验。这些验证确认实现与训练链路正常，不能证明增强质量已提升。
当前 test 复用 validation，保持原项目约定；比较收益时需结合前景 L*/ab、高光、PSNR，
以及满分辨率光晕和暗部噪声，后续再用独立场景检验泛化。
