# AdaInt semantic dual fusion

服务器：DB94
仓库：/DB/data/yujietang/IQ-LUT/image_enhance/AdaInt_BAN
分支：feature/semantic-dual__from_p3-local-color
基线：feature/p3-local-color__from_global-affine，6b3959d

## 实现的方法

共享一套未预训练 TPAMI backbone。前景、背景分别拥有 ConditionEncoder、16 段 tone curve、33 节点/轴的 AdaInt、5 张归一化残差 LUT 基及系数头。各自用原始 RGB 查表，各自接受局部仿射修正，最终按空间 mask 混合输出；不同 AdaInt 坐标的 LUT 不按数组下标混合。

局部模块共享 shallow/deep/fusion 空间特征提取，分开 FiLM 和四系数输出头。每路修正的 RGB 均值在该路完整上下文 mask 内计算并减去，随后再融合。中心化是在低分辨率完整上下文网格上定义的近似约束；不声称高分辨率输出严格区域零均值。

输入条件包括共享 CNN 全图特征、全图颜色统计、区域归一化颜色统计、区域面积和有效性。区域统计包含 RGB/亮度直方图、均值、标准差、亮暗比例、饱和度统计。64×64 全图中心采样与 32×32 区域补充采样结合；补充采样直接从原图按照区域 mask 的累计权重确定性取样，不计算黑底抠图直方图。空区域回退到全图统计，小区域通过 region_min_area 控制可靠性。

最终融合结果接受原始 RGB MSE、原始 P3 全图与困难网格监督，以及逐图按有效区域面积归一化的前景/背景 RGB、P3 监督。两路 LUT 正则各按自己的非均匀节点间隔计算后取平均，维持原始正则量级。可选独立脸部/皮肤监督及边界梯度监督，默认关闭。BAN 保持关闭。

现有 mask 覆盖人脸、头发、颈部及明显肤色区域，未覆盖衣服和身体。默认实验是该前景区域/其余区域双分支，不是完整人物分割。foreground_mask 用于融合，supervision_mask 是未羽化的监督 mask；可另提供独立 face_mask，默认不会把含头发的前景 mask 当作纯皮肤 mask。

## 代码位置

- adaint/model_semantic.py：AiLUTDual 主模型、空间融合、逐区域损失、双路冻结/解冻、区域评估。
- adaint/semantic_modules.py：区域颜色统计、独立映射模块、共享空间特征的双头局部修正。
- adaint/semantic_pipeline.py：mask 按输入 ID 加载、分开融合与监督 mask、同步翻转/裁剪/上下文/采样、可选窄边羽化和扰动。
- adaint/semantic_dataset.py：SemanticFace，区域评估只平均有效样本，记录有效样本数；完全没有有效区域时报告 NaN。
- adaint/configs/face_indoor_semantic_dual.yaml 与 face_outdoor_semantic_dual.yaml：新增可运行配置，保持基线数据划分与训练预算。
- adaint/config_utils.py、adaint/__init__.py、adaint/experiment_utils.py：配置与注册、训练预算和严格 checkpoint 兼容性。
- adaint/model.py：评估入口传递 mask，支持按图像尺寸分组的 mask 与区域指标。
- tools/train_kd_unified.py：训练后满分辨率推理同步加载 mask；可配置最佳 checkpoint 指标。
- tools/smoke_semantic_dual.py：真实数据短训练/验证和 mask 覆盖检查。
- tests/test_adaint/test_semantic_dual.py：融合、统计、采样、裁剪、梯度、空区域、checkpoint、评估和选模测试。

BFE_v12 仅作为参考，没有改动其文件，也不需要运行时导入 BFE。

## 训练

在 DB94 执行：

```bash
cd /DB/data/yujietang/IQ-LUT/image_enhance/AdaInt_BAN
git switch feature/semantic-dual__from_p3-local-color
source /DB/rhome/yujietang/miniconda3/etc/profile.d/conda.sh
conda activate AdaInt_BAN

CUDA_VISIBLE_DEVICES=5 python tools/train_kd_unified.py \
  adaint/configs/face_indoor_semantic_dual.yaml --seed 2026
```

室外训练将配置替换为 adaint/configs/face_outdoor_semantic_dual.yaml。GPU 5 是实现验证时空闲的设备，实际运行可自行修改。两份配置均为 100 epochs，前 5 epochs 冻结两路 AdaInt，单图 batch，BAN 关闭。默认工作目录分别为 work_dirs/face_indoor_semantic_dual/gen_0 与 work_dirs/face_outdoor_semantic_dual/gen_0。训练结束保存 best.pth，并在 train_infer/ 输出训练集和验证集的满分辨率增强结果。

续训：

```bash
CUDA_VISIBLE_DEVICES=5 python tools/train_kd_unified.py \
  adaint/configs/face_indoor_semantic_dual.yaml --seed 2026 \
  --resume-from work_dirs/face_indoor_semantic_dual/gen_0/latest.pth
```

单图推理（需要配置 test_foreground_root 下有该输入 ID 对应的 mask）：

```bash
CUDA_VISIBLE_DEVICES=5 python adaint/demo.py \
  work_dirs/face_indoor_semantic_dual/gen_0/inference_config.py \
  work_dirs/face_indoor_semantic_dual/gen_0/best.pth \
  /path/to/123_input.jpg /path/to/123_result.png
```

mask 命名默认为 123_face_mask.png；支持 jpg/jpeg/png/bmp/tif/tiff，递归索引，但同一 ID 多个文件会报错。新图片需提供从输入得到的 mask，缺失 mask 会明确报错，不会静默用肤色阈值替代。

原单路 checkpoint 不能用 --load-from 或 --resume-from 加载成双路模型，严格加载会拒绝。默认从头训练；没有实现旧单路 checkpoint 的自动扩展或迁移。

## 参数调节

两份新 YAML 的 model 部分：

| 参数 | 默认 | 用途 |
| --- | --- | --- |
| local_enabled | true | 保留双头局部修正；false 可做无局部模块消融 |
| use_region_condition | true | 是否使用区域颜色信息；false 可做全图条件对照 |
| region_samples_size | 32 | 每路补充采样边长，每路 1024 个原始颜色样本 |
| region_sample_mix | 0.5 | 区域补充统计在区域条件中的混合比例；整图网格漏采时自动完全使用补充统计 |
| region_min_area | 0.0001 | 可靠区域/有效区域最低面积比例；训练监督按 crop，局部修正按完整上下文 |
| foreground_rgb_weight / background_rgb_weight | 0.1 / 0.1 | 最终结果逐图归一化区域 MSE 权重 |
| foreground_color_weight / background_color_weight | 0.1 / 0.1 | 区域 P3 ΔE/100 权重 |
| face_color_weight | 0.0 | 独立脸部/皮肤 mask 上 ΔE/100 权重；启用前需提供独立 mask 目录 |
| boundary_gradient_weight | 0.0 | 边界带相对 GT 的 RGB 梯度误差权重 |
| color_loss_weight | 室内 0.3，室外 0.1 | 沿用基线当前值的全图/困难网格 P3 权重 |
| color_grid_size / color_top_fraction / color_hard_weight | 16 / 0.1 / 0.25 | 保留 16×16 网格最困难 10% 监督及其占比 |
| n_vertices / n_ranks / curve_points | 33 / 5 / 16 | 每路节点数、残差基数及 tone 段数 |
| smooth_factor / monotonicity_factor / global_magnitude_factor | 沿用基线 | 每路 LUT 正则取平均 |
| local_magnitude_factor / local_smoothness_factor | 沿用基线 | 各区域局部修正幅度与边缘感知平滑正则 |

semantic_masks 部分：

| 参数 | 用途 |
| --- | --- |
| train_foreground_root / val_foreground_root / test_foreground_root | 三个划分的融合与区域监督 mask 目录 |
| train_face_root / val_face_root / test_face_root | 可选独立脸部/皮肤监督 mask 目录，默认 null |
| foreground_suffix / face_suffix | 输入 ID 后的 mask 文件名后缀，默认 _face_mask |
| feather_radius | 窄边均值羽化半径，默认 1；0 使用硬边 |
| perturb_probability / perturb_pixels | 训练融合 mask 的轻微腐蚀/膨胀概率及半径，默认 0 / 1；验证和推理自动关闭扰动 |
| region_label | 记录 mask 语义标签；改变它不会自动改变分割内容 |

室内 train/val_foreground_root 已设为：
/DB/data/yujietang/IQ-LUT/face_dataset/all/indoor_face_masks/masks

室外 train/val 使用对应 outdoor_face_masks/masks；两份配置的 blind test 使用 test_face_masks/masks。换完整人像 mask 时只需更换 foreground 目录及后缀，并根据需要单独提供脸部/皮肤 mask。可在 test_cfg.metrics 中增加 deltaE_P3_face，但对应验证目录必须有独立 face mask。

训练与选模参数位于 YAML 顶层及 optimizers/data：

- train_epochs、adaint_fix_epochs、checkpoint_interval_epochs、evaluation_interval_epochs、log_interval_epochs。
- optimizers.lr、optimizers.paramwise_cfg.custom_keys.adaint.lr_mult 同时作用于两路节点头。
- data.train_dataloader.samples_per_gpu。原训练 crop 尺寸随图变化，默认 1；增大 batch 需要先将图像/crop 固定为同一尺寸，并同步 mask。
- best_checkpoint_metric: PSNR、best_checkpoint_rule: greater 是默认选模方式。可改为 deltaE_P3_foreground 和 less。区域指标同时记录，需结合整体 PSNR、背景及前景明度/色度检查。
- infer_after_generation 控制训练结束后满分辨率导出。

P3 损失仍沿用基线 Display P3/D65 数值约定。读取图片不会自动进行 ICC 色域转换，为文件添加 P3 ICC 也不等同于转换像素色域；本次没有改变该约定。

## 验证与范围

```bash
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=2 python tools/smoke_semantic_dual.py \
  adaint/configs/face_indoor_semantic_dual.yaml \
  --steps 2 --max-side 256 --check-mask-coverage

CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=2 python -m pytest -o addopts="" -q \
  tests/test_adaint/test_semantic_dual.py
```

-o addopts="" 避免 setup.cfg 默认 addopts=tests/ 自动带入全部旧 MMEditing 测试。原仓库存在缺失 BasicVSR 配置的测试，以及仍断言 color_loss_weight=0.01 的过期基线配置测试（当前室内/室外实际为 0.3/0.1），没有为通过这些测试而修改基线参数。

本次只做机制测试、真实数据短训练和小规模完整入口检查；没有启动 100 epochs 主实验。不能据此宣称 PSNR 或人像颜色质量提升，需与基线在相同数据、预算及色彩约定下正式对照。

实际验证结果（2026-10-14）：

- 相关测试 47 passed，2 deselected；排除的是基线已有的过期颜色权重断言。
- 室内 mask 覆盖：训练 67 张、验证 8 张、测试 64 张。
- 真实室内数据完成 2 个优化步骤，所有损失、梯度和指标有限。
- 使用 2 张真实训练图、1 张验证图完成 1 epoch 的 tools/train_kd_unified.py 全流程检查；训练/验证计算缩小至 128×128，仅用于入口检查。
- 成功保存 epoch_1.pth、best.pth、resolved_config.py 和 inference_config.py，并完成训练后满分辨率导出，3 张结果均为 3000×4000。
- 保存的推理配置与双路 checkpoint 成功调用 adaint/demo.py 的单图推理函数，输出 4000×3000 的原图分辨率。
- 两路 LUT、tone、节点及局部输出头均有有限且非零的任务梯度；完整上下文区域中心化、裁剪一致性、空区域、mask 后融合和独立节点检查通过。
- 模型参数量为 2,364,200。
- 验证文件位于 DB94 的 /tmp/adaint-semantic-sanity-hq2pxtdf/verified_work_dir/gen_0，仅用于短流程检查；不是正式训练产物。
