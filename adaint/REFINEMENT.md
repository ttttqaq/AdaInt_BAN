# DB94 LUT 改进：代码与实验指南

远程项目：`/DB/data/yujietang/IQ-LUT/image_enhance/AdaInt_BAN`。
新分支：`codex/lut-refinement__from_lut-architecture-v2`，基于 V2 的 `953565ae4e9d9e0fd72b19104c0cecfc8a69ee85`。
实现目标是兼顾视觉效果、PSNR 与色差；以下是功能验证及实验建议，尚未完成完整训练，不能据此宣称性能优于 V2。

## 1. 固定的架构

通过 `model.type` 选择模型，不设置架构模块的独立布尔开关。

| model.type | 内容 |
| --- | --- |
| `AiLUTRefinementBase` | 原 V2 架构及参数形状；所有新增方法关闭时作为 V2 对照，已验证前向一致。 |
| `AiLUTArchitectureV3` | 固定加入 tone 后统计条件编码，以及浅/深层融合、亮度引导的空间权重解码器。默认两个残差 LUT。 |
| `AiLUTGlobalLocalV3` | 固定采用一个全局残差 LUT 和两个独立、有界、可正负变化的局部残差 LUT。 |

V3 先用原始条件预测单调 tone 曲线，再对实际 tone 变换后的像素计算统计，与原条件融合，交给 AdaInt、LUT 系数和空间引导。先 tone 后统计，避免把 T(mean(x)) 当成 mean(T(x))。

新空间解码器默认从 128×128 输入提取 64×64 浅层和 32×32 深层特征，融合后输出 64×64 logits，再以亮度引导滤波上采样。V3 在滤波后 softmax，保持权重非负、和为 1。默认 `guided_filter_radius=2`、`guided_filter_epsilon=0.001`；滤波本身属于固定架构。

GlobalLocal 输出为 `tone + global_residual + gate1*local_residual1 + gate2*local_residual2`。两个 gate 独立预测，默认 `local_gate_scale=0.5`，用 tanh 限制到 [-0.5, 0.5]。单调正则约束四种极端 gate 组合的有效 LUT，允许局部残差正负修正。该约束不保证包含内容相关空间权重的完整模型严格单调。

V3 和 GlobalLocal 使用新参数结构，需要重新训练；不能直接加载 V2 checkpoint。加载器严格检查增强网络参数，剥离冻结教师的参数。模型选项一致时，原 V2 checkpoint 可加载到 RefinementBase 用于推理。

## 2. 独立开关及参数

下表参数位于 YAML 的 `model` 下，新增方法在模板中默认全关。

| 方法 | 开关 | 默认值与建议 |
| --- | --- | --- |
| 对角单调 | `use_diagonal_monotonicity` | 只约束 dR/dR、dG/dG、dB/dB。`monotonicity_factor=2`，内部已校正通道平均尺度，**不要再除以 3**；可试 1/2/4。 |
| KD 后期衰减 | `use_kd_decay` | `kd_decay_start_ratio=0.3`、`kd_decay_end_ratio=1`、`kd_final_weight=0.1`；前 30% 不衰减，后线性降至 10%。末倍率可试 0.05/0.1/0.2。 |
| 可靠性 KD | `use_confidence_kd` | `confidence_temperature=0.05`；按教师对 GT 的逐像素 RGB 误差加权 exp(-error/temperature)。0.02 更严格，0.1 更宽松。 |
| 分层像素统计 | `use_stratified_statistics` | `stats_pool_size=64`；采实际像素，减少平均缩图抹平暗部/高光统计。可试 64/96。 |
| 边缘感知平滑 | `use_edge_aware_smoothness` | 保留 `use_spatial_smoothness=true`。`spatial_smoothness_factor=0.001`、`edge_smoothness_scale=10`；scale 可试 5/10/20，越大越允许边缘处变化。 |
| Lab 辅助损失 | `use_lab_loss` | 可微 sRGB/D65 Lab 的归一化 L1，`lab_loss_weight=0.05`；可试 0.01/0.05/0.1。 |
| RGB MSE 辅助损失 | `use_rgb_mse_loss` | L1 主损失保留，`rgb_mse_loss_weight=0.1`；可试 0.1/0.5/1，进一步偏向 PSNR。 |
| 低频色调监督 | `use_low_frequency_loss` | 池化后的 Lab L1，`low_frequency_loss_weight=0.1`、`low_frequency_size=32`；权重可试 0.05/0.1/0.2。 |
| 整图上下文/局部监督 | `use_full_image_context` | 顶层 `context_size=256`；完整配对图提供条件和统计，局部 crop 监督；空间引导按 crop 坐标对齐。 |

第 0 代没有教师，两个 KD 方法从第 1 代生效。KD 实际权重是顶层 `kdloss_weight`（默认 1）乘衰减倍率，再按需乘像素可靠性。可靠性权重不重新归一化至均值 1。

整图上下文使用同一次 LQ 翻转/颜色扰动生成缩略图、采样像素和 crop；GT 只同步几何变换。训练仍先 ResizeToMatch 对齐 GT；推理直接输入完整图。

原 V2 最佳训练配置的 `use_interval_floor=false`、`use_flexible_tone_endpoints=false` 保留。生成器保证训练和测试模型选项一致，修正原 V2 测试配置端点选项不一致的使用问题。

## 3. 环境与生成配置

以下命令在 DB94 上执行：

```bash
cd /DB/data/yujietang/IQ-LUT/image_enhance/AdaInt_BAN
source /DB/rhome/yujietang/miniconda3/etc/profile.d/conda.sh
conda activate AdaInt_BAN
git switch codex/lut-refinement__from_lut-architecture-v2
```

已准备 `adaint/configs/generated/v3_ready.yaml`（V3 主组合）与 `global_local_ready.yaml`（同方法组合的全局/局部架构），各自有匹配的 `_test.yaml`，可直接训练。这些名称已经使用；下面的命名示例用于生成新的独立实验。

也可把 python 替换为 `/DB/rhome/yujietang/miniconda3/envs/AdaInt_BAN/bin/python`。

生成 V2 对照、单项方法、纯 V3 架构及主组合：

```bash
python tools/prepare_lut_experiment.py --name v2_control --preset baseline --gen 4 --epochs 100
python tools/prepare_lut_experiment.py --name v2_diag --preset diagonal --gen 4 --epochs 100
python tools/prepare_lut_experiment.py --name v3_arch_only --preset v3 --gen 4 --epochs 100
python tools/prepare_lut_experiment.py --name v3_integrated --preset all --gen 4 --epochs 100
```

每次生成 `adaint/configs/generated/NAME.yaml` 与 `NAME_test.yaml`，输出到独立 `work_dirs/NAME`；已有同名文件拒绝覆盖，调参请换新名称。

`all` = V3 + diagonal + kd_decay + stratified + edge_smoothness + lab + full_context；不包括可靠性 KD、MSE、低频损失或 GlobalLocal。

在指定架构上组合任意方法并改参数：

```bash
python tools/prepare_lut_experiment.py --name v3_lab005 --architecture v3 --enable diagonal kd_decay stratified edge_smoothness lab --set model.lab_loss_weight=0.05 model.kd_final_weight=0.1 --gen 4 --epochs 100
```

简写对应上表开关：`diagonal`、`kd_decay`、`confidence_kd`、`stratified`、`edge_smoothness`、`lab`、`mse`、`low_frequency`、`full_context`。单项 preset 只启用该方法；`--enable` 可累计方法。

## 4. 推荐实验顺序

先做单项消融定位收益，再保留有效方法累计。以下相邻步骤只改一项；顺序不代表必然逐步提升。

| 顺序 | 架构 | --enable |
| --- | --- | --- |
| 0 | v2 | 无，baseline |
| 1 | v2 | diagonal |
| 2 | v2 | diagonal kd_decay |
| 3 | v2 | diagonal kd_decay stratified |
| 4 | v2 | diagonal kd_decay stratified edge_smoothness |
| 5 | v3 | 与步骤 4 相同，只改变 model.type |
| 6 | v3 | diagonal kd_decay stratified edge_smoothness lab |
| 7 | v3 | diagonal kd_decay stratified edge_smoothness lab full_context |
| 8，可选 | global_local | 与步骤 7 相同，只改变 model.type |

GlobalLocal 示例：

```bash
python tools/prepare_lut_experiment.py --name v3_global_local --architecture global_local --enable diagonal kd_decay stratified edge_smoothness lab full_context --set model.local_gate_scale=0.5 --gen 4 --epochs 100
```

可靠性 KD、MSE、低频损失在最终有效组合上分别追加 `confidence_kd` / `mse` / `low_frequency`，每次换实验名。局部修正尺度先 0.5，必要时试 0.25。

预算有限时，非 KD 方法可统一 `--gen 1` 初筛；KD 方法必须至少 `--gen 2`。正式比较保持相同数据划分、代数、epoch 和 seed，不把 1 代与 4 代差异归因于某个方法。

## 5. 训练与推理

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_kd_unified.py adaint/configs/generated/v3_integrated.yaml --gpus 1 --seed 3407
```

默认 4 代、每代 100 epoch、前 5 epoch 固定 AdaInt、每 epoch 保存/验证一次。迭代数按实际图片数与 batch 换算：1000 张训练图、单 GPU、batch=1，每代 100000 iter。扩充数据后更新 `adaint/annfiles/Face/indoor_train.txt` 和验证划分；只新增图片不会自动增加样本。

变比例裁剪产生不同大小的图像，建议先保留 samples_per_gpu=1。增加 batch 前要保证大小可拼接。workers_per_gpu=0 时，同时设 `data.train_dataloader.persistent_workers=false`、`data.val_dataloader.persistent_workers=false`。

每代保存 iter_*.pth、best.pth、日志、resolved_config.py、inference_config.py。新模板保存 optimizer。教师使用上一代保存的推理配置；新推理配置不含 KD 教师，避免递归加载。

最终代验证集评估并保存图片：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py work_dirs/v3_integrated/gen_3/inference_config.py work_dirs/v3_integrated/gen_3/best.pth --eval-val --out work_dirs/v3_integrated/gen_3/val_metrics.pkl --save-path work_dirs/v3_integrated/gen_3/val_images
```

blind test 集原分辨率推理：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py work_dirs/v3_integrated/gen_3/inference_config.py work_dirs/v3_integrated/gen_3/best.pth --save-path work_dirs/v3_integrated/gen_3/test_images
```

4 代最终目录是 gen_3，2 代是 gen_1，1 代是 gen_0。也可用生成的 NAME_test.yaml 做 blind test；普通推理优先使用每代 inference_config.py，不使用带教师的 resolved_config.py。

项目 deltaE 默认是 CIE76，越低越好；Lab 辅助损失不等于直接优化 CIEDE2000。best.pth 仍按验证 PSNR 选择，同时查看它的 deltaE 和图片。需要不同颜色折中，可选另一 iter_*.pth 传给同一测试命令。

新模板 `infer_after_generation=false`，避免扩大数据后每代自动导出大量满分辨率图；需要时生成配置加入 `--set infer_after_generation=true`。

从第 2 代开始或恢复第 2 代的具体迭代：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_kd_unified.py adaint/configs/generated/v3_integrated.yaml --gpus 1 --seed 3407 --start-gen 2
CUDA_VISIBLE_DEVICES=0 python tools/train_kd_unified.py adaint/configs/generated/v3_integrated.yaml --gpus 1 --seed 3407 --start-gen 2 --resume-from work_dirs/v3_integrated/gen_2/iter_50000.pth
```

前代 best.pth 必须存在。恢复使用相同配置和预算，衰减随保存的 cnt_iters 继续；本代既有验证日志参与 best 选择。后续代从头训练，不重复加载 resume checkpoint。恢复钩子对齐旧 MMCV 的迭代号与模型完成步数，避免多跑一步；恢复不保证数据顺序与随机扰动逐位一致。
`--no-validate` 不适用于该训练器：后续教师依赖验证 best。验证间隔必须是 checkpoint 间隔的正整数倍，且训练预算能覆盖至少一次验证。

## 6. 诊断与实现文件

额外记录 `guide_spatial_std`、`residual_abs_mean`、`output_change`、`target_change`，有教师时记录 `kd_multiplier`；这些值不计入总 loss。用于观察权重是否近乎常量、残差是否过弱、输出增强量是否长期低于目标增强量。

| 文件（相对项目根目录） | 改动 |
| --- | --- |
| adaint/model_refinement.py | 三种模型、固定架构、新损失/正则/衰减、诊断日志。 |
| adaint/context_transform.py | 整图上下文、采样统计、crop 坐标对齐。 |
| adaint/loss.py | 可靠性 KD、冻结教师、正确设备及严格 checkpoint 加载。 |
| adaint/experiment_utils.py | 动态预算、可重载配置、严格增强网络加载。 |
| adaint/config_utils.py、adaint/__init__.py | 注册及上下文管线配置。 |
| adaint/configs/face_indoor_refinement.yaml | V2 对照、方法参数及训练模板。 |
| tools/prepare_lut_experiment.py | 命名实验、方法组合、参数覆盖及匹配测试配置。 |
| tools/train_kd_unified.py | 按 epoch 预算、教师配置、恢复与可选满分辨率推理。 |
| tools/test.py | 严格加载、--eval-val、CUDA 推理。 |
| tests/test_adaint/test_refinement.py | 数学行为、配置、坐标、真实 CUDA 前向及梯度测试。 |

## 7. 验证

- DB94 原环境 Python 3.7.10 / PyTorch 1.8.1+cu111 / MMCV 1.3.17，真实 AiLUT CUDA 扩展。
- 15 项针对性测试通过：V2 对照前向一致；新架构梯度有限且非零；Lab 参考色；统计；坐标；单调尺度；KD；动态预算；配置保存重载；严格 checkpoint；恢复计数对齐。
- 真实数据两代、每代 2 iter 功能训练通过；V3 主架构及所有独立训练方法同时启用，教师加载、验证、best.pth 保存均成功。不是完整训练效果对比。
- 3000×4000 原图推理通过，保存 PNG 分辨率一致；独立 --eval-val 评估、图片及指标保存通过。
- 真实断点恢复通过：从已完成 1 iter 的 checkpoint 恢复，只补跑 1 iter 到总计 2 iter，计数与训练进度一致。
- 通用 MMEditing 测试有既有环境问题（缺少 BasicVSR 配置、LMDB 二进制符号不兼容）；未改动相关依赖，不宣称全套通过。

复现针对性测试：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m pytest -o addopts='' -q tests/test_adaint/test_refinement.py
```

`-o addopts=''` 覆盖仓库默认收集整个 tests/ 的设置。没有搜索文献，也未开始完整训练或推送远程分支。
