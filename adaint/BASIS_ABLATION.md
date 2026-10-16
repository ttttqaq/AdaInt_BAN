# Tone / basis LUT 消融实验

DB94 工作目录：
/DB/data/yujietang/IQ-LUT/image_enhance/AdaInt_BAN

分支：experiment/basis-lut__from_semantic-dual
父分支：feature/semantic-dual__from_p3-local-color
父提交：fcaf8bc

消融分支现已切换到原 AdaInt_BAN 目录，此前新建的独立目录已删除。原有 feature/semantic-dual__from_p3-local-color 分支及其 3f4ca96 提交完整保留，包含 color_loss_weight=0.05 和 p3.py 参数。本消融仍基于 fcaf8bc，按确认使用 color_loss_weight=0.3。

## 实验对比

| 模式 | LUT 生成方式 | 配置 |
| --- | --- | --- |
| tone 对照，默认 tone_residual | 在本路 AdaInt 节点上计算 RGB tone curve，生成基础 LUT，再加归一化残差基的加权和 | face_indoor_semantic_dual.yaml / face_outdoor_semantic_dual.yaml |
| basis 消融，basis | 直接预测权重，将本路可学习的原始 basis LUT 加权求和，得到完整 LUT | face_indoor_semantic_basis.yaml / face_outdoor_semantic_basis.yaml |

basis 模式的公式为 L_r = sum_k w_{r,k} B_{r,k}，再用本路 AdaInt 节点查询原始 RGB。没有额外的 tone 基础表、外加恒等映射或基表 RMS 归一化。

参考 [AdaInt 官方 LUTGenerator](https://github.com/ImCharlesY/AdaInt/blob/main/adaint/model.py) 和本仓库 main 分支：融合系数为可正可负的无约束权重，bank 初始为一张恒等 LUT 加其余零 LUT，系数头输出 bias 初始为 1。第一张恒等基表也参与训练。

仅迁移完整 basis LUT 的生成公式和 bank 初始化。系数预测头继续使用现方案原有的两层 PredictionHead 及小幅输出权重初始化；不同时替换成 main 的单层系数头。AdaInt 节点预测、独立双分支、区域条件、局部修正和损失都沿用现方案。

这是“当前 tone＋残差基”与“直接 basis 融合”的对照。当前 tone 版本本身也有残差基，因此实验并非纯 tone-only 与 basis-only 的比较。

## 限定改动范围

现有生产文件只改两处：

1. adaint/semantic_modules.py：新增 BasisLUT；RegionMapping 根据 lut_mode 选择生成方式。
2. adaint/model_semantic.py：AiLUTDual 接收并向两路传递 lut_mode，默认仍为 tone_residual。

其余新增文件：

- adaint/configs/face_indoor_semantic_basis.yaml
- adaint/configs/face_outdoor_semantic_basis.yaml
- tests/test_adaint/test_basis_ablation.py
- adaint/BASIS_ABLATION.md（本说明）

forward_dummy、forward_train、forward_test、区域颜色统计、局部 decoder、所有损失、mask 管线、训练脚本、原配置与 main 分支均未修改。新增测试会比较父提交对应代码，检查这些范围保持一致。

模型构建时保持与 tone 对照相同的随机数消耗顺序；basis 模式构建完即移除 tone 模块，不保留其参数或计算。相同随机种子下，LUT 生成器以外的骨干、条件编码器、AdaInt、局部 decoder 等所有共同参数的初始值逐项相同，避免仅因模块创建顺序变化而产生额外随机初始化差异。

默认设置下参数量：

- tone_residual：2,364,200
- basis：2,286,024
- 差值：78,176，来自两路 tone 预测头。

每路仍为 5 张 33³ basis LUT；没有为补齐参数量而扩大其他模块。

## 如何训练

在 DB94 原目录切换到消融分支后运行。

```bash
cd /DB/data/yujietang/IQ-LUT/image_enhance/AdaInt_BAN
git switch experiment/basis-lut__from_semantic-dual
source /DB/rhome/yujietang/miniconda3/etc/profile.d/conda.sh
conda activate AdaInt_BAN
```

室内 basis 消融：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_kd_unified.py \
  adaint/configs/face_indoor_semantic_basis.yaml --seed 2026
```

室内 tone 对照，使用同一目录中的原始配置：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_kd_unified.py \
  adaint/configs/face_indoor_semantic_dual.yaml --seed 2026
```

两份室内配置均使用 color_loss_weight=0.3、100 epochs、5 epochs 的 AdaInt 冻结、相同数据划分和其他超参数。GPU 0 是本次验证使用的设备，正式运行可选空闲 GPU。建议两组均从头训练，固定同一 seed。不要将旧目录 color_loss_weight=0.05 的结果直接当作本组 0.3 的唯一对照。

室外将配置替换为：

- basis：adaint/configs/face_outdoor_semantic_basis.yaml
- tone：adaint/configs/face_outdoor_semantic_dual.yaml

室外两组均沿用 color_loss_weight=0.1。

结果目录均位于原 AdaInt_BAN 目录下：

- work_dirs/face_indoor_semantic_basis/gen_0
- work_dirs/face_indoor_semantic_dual/gen_0
- work_dirs/face_outdoor_semantic_basis/gen_0
- work_dirs/face_outdoor_semantic_dual/gen_0

训练流程沿用现有入口，保存 checkpoint、best.pth、配置和日志；训练结束使用 best.pth 对训练/验证图像导出满分辨率 train_infer/ 结果。本次目录迁移不移动或删除历史结果；复跑已有实验时请用 --work-dir 指定新的输出目录，避免混用旧日志。

## 参数与 checkpoint

新增参数仅为 YAML 中的：

```yaml
model:
  type: AiLUTDual
  lut_mode: basis        # basis 或 tone_residual
```

原 tone 配置未写 lut_mode 时自动使用 tone_residual。修改模式时也需使用不同 work_dir，避免混用实验日志与 checkpoint。

- model.n_ranks：每路 basis 数量，当前 5。
- model.n_vertices：每轴节点数，当前 33。
- model.curve_points：tone 模式的曲线段数。basis 模式没有 tone 参数或前向计算；消融时保留相同配置值，以维持共同模块的种子初始化顺序。
- 其他训练、损失、区域与 mask 参数仍按原 SEMANTIC_DUAL.md 调节；做本组消融时保持两组一致。

tone 与 basis 的模型结构不同，不能跨模式 --resume-from。basis 可从自身 checkpoint 续训：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_kd_unified.py \
  adaint/configs/face_indoor_semantic_basis.yaml --seed 2026 \
  --resume-from work_dirs/face_indoor_semantic_basis/gen_0/latest.pth
```

对比全图 PSNR、P3 全图/困难网格 ΔE、前景/背景 ΔE、前景明度与色度指标，并同时记录参数量。若正式实验存在随机波动，应使用相同的一组多个 seed 复核。

## 验证命令

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 python -m pytest -o addopts="" -q \
  tests/test_adaint/test_basis_ablation.py \
  tests/test_adaint/test_semantic_dual.py

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 python tools/smoke_semantic_dual.py \
  adaint/configs/face_indoor_semantic_basis.yaml \
  --steps 2 --max-side 256 --check-mask-coverage
```

使用 -o addopts="" 避免仓库 setup.cfg 将全部旧测试自动加入。之前已有两项旧配置测试仍断言 color_loss_weight=0.01，与实际基线配置不一致；本次没有改动这些测试或原配置。

## 本次验证结果（2026-10-16）

- 61 项相关回归通过，排除 2 项父分支已有的过期 color_loss_weight=0.01 断言。
- 新增 basis 检查覆盖：融合与 main 的 Linear bank 等价、权重可带符号、无 RMS 归一化、无额外 tone/恒等项、bank 初始化、所有基表的梯度、共同参数种子初始化一致、默认 tone 路径与 checkpoint schema 保持、配置唯一变量、冻结/解冻、局部中心化、裁剪一致性及空区域。
- 室内真实数据的两个优化步骤及验证通过，全部损失/梯度/指标有限。
- 室内 mask 覆盖检查：训练 67、验证 8、测试 64 张均完整。
- tools/train_kd_unified.py 完成 2 张训练图、1 张验证图的一轮完整检查，包含 best.pth 选取、保存、重新加载及 3 张满分辨率图像导出。训练/验证计算缩小至 128×128，仅用于流程验证。
- 完整入口检查文件位于 DB94 的 /tmp/adaint-basis-runner-tu1yzrjx/work_dir/gen_0；真实数据短训练日志为 /tmp/adaint_basis_ablation_smoke.log。
- 原有分支及其室内 YAML、p3.py 参数保留在 3f4ca96；消融分支已迁回原目录，新建的独立目录已删除。
- 没有运行 100 epochs 正式消融实验，当前验证不能用来判断 tone 与 basis 哪个效果更好。
