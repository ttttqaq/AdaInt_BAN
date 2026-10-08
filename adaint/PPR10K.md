# A100：PPR10K 新模型训练与验证

项目：`/remote-home/share/yujietang/image_enhance/my_code/AdaInt_BAN`。
分支：`feature/lut-refinement__from_lut-architecture-v2`。
数据目录实际名称是 `PRR10K`；配置与指标沿用数据集的正式名称 `PPR10K`。

## 数据划分

直接使用 `/remote-home/share/yujietang/dataset/PRR10K` 的现成目录：

```text
PRR10K/
  prr_train/              # 8,875 张训练图
    source/*.png
    target_a/*.png
    target_b/*.png
    target_c/*.png
  prr_value/              # 2,286 张验证图
    source/*.png
    target_a/*.png
    target_b/*.png
    target_c/*.png
  masks_360p/*.png        # 所有图像对应的人像区域掩码
```

`PPR10KFolder` 按文件名主干匹配输入、GT 与掩码，读取目录里的全部图片，不从训练目录划分验证集，也不依赖旧的 train/test.txt。两个目录的图片和组编号均无交集。`data.test` 与 `data.val` 都指向现成的 `prr_value`，没有额外的盲测集。

当前输入与 GT 是 8-bit RGB PNG。训练采用随机裁剪、448×448 固定尺寸、同步翻转，不添加颜色扰动；整图上下文开启时，条件来自整图，crop 坐标与局部监督保持对齐。验证不裁剪或缩放，在提供的 360p 图像上计算指标。

## 模型与默认预算

- `ppr10ka_v3.yaml`：`AiLUTArchitectureV3`，专家 a；方法开关与学习率沿用你当前的实验设置。
- `ppr10ka_global_local.yaml`：模型为 `AiLUTGlobalLocalV3`，保留其当前方法开关与学习率。
- `ppr10k_refinement.yaml`：`AiLUTRefinementBase` 的 V2 对照模板，新训练方法默认关闭。

代数、训练 epoch、batch、冻结周期、保存／验证／日志周期、学习率、方法开关及导图选项，以正在使用的 YAML 为准。验证与独立测试 batch 可通过各自 dataloader 的 samples_per_gpu 设置，例如 64。训练步数只在内部自动换算；各代 best.pth 按验证集 PSNR 选择，并作为下一代教师。

## 按 epoch 调参与日志

修改正在使用的训练配置，例如 `/remote-home/share/yujietang/image_enhance/my_code/AdaInt_BAN/adaint/configs/generated/ppr10ka_v3.yaml`。基础模板用于以后生成新实验；它的修改不会自动改写已有实验配置。

```yaml
gen: 4
train_epochs: 100
adaint_fix_epochs: 5
checkpoint_interval_epochs: 1
evaluation_interval_epochs: 5
log_interval_epochs: 1
```

依次表示代数、每代训练 epoch、AdaInt 冻结 epoch、保存周期、评估周期和训练日志周期。`log_interval_epochs: 5` 表示每 5 epoch 打印训练信息；每次验证仍会打印四项评估指标。周期使用整数，评估周期仍须为保存周期的正整数倍。

配置已删除 `total_iters`、`n_fix_iters`、各处 iteration interval、`by_epoch` 与底层 workflow。无需手动同步步数；修改 epoch 或 batch 后，训练入口自动推导底层参数。保存的 `resolved_config.py` 与 `inference_config.py` 也只暴露 epoch 周期参数。

日志示例：

```text
Epoch [5/100] ... loss_recons: ... loss: ...
Epoch(val) [5/100] ... PSNR: ... deltaE: ... PSNR_hc: ... deltaE_hc: ...
```

训练 loss 为日志周期内按样本数量加权的平均值，评估不会覆盖当轮训练信息。新权重命名为 `epoch_1.pth`、`epoch_2.pth` 等；`best.pth` 和 `latest.pth` 保留。已有 iteration checkpoint 及旧 JSON 日志仍可用于恢复与最佳权重选择。

## 开始训练

```bash
ssh A100
source /home/yujietang/miniconda3/etc/profile.d/conda.sh
conda activate AdaInt_BAN
project_root=/remote-home/share/yujietang/image_enhance/my_code/AdaInt_BAN
cd "$project_root"

CUDA_VISIBLE_DEVICES=4 python "$project_root/tools/train_kd_unified.py" \
  "$project_root/adaint/configs/generated/ppr10ka_v3.yaml" --gpus 1 --seed 3407
```

GPU 4 是本次功能测试使用的卡；正式运行时可换成空闲 GPU。GlobalLocal 训练只需更换配置：

```bash
CUDA_VISIBLE_DEVICES=4 python "$project_root/tools/train_kd_unified.py" \
  "$project_root/adaint/configs/generated/ppr10ka_global_local.yaml" --gpus 1 --seed 3407
```

输出分别位于项目根目录下的 `work_dirs/ppr10ka_v3/gen_0` 至 `gen_3`，或 `work_dirs/ppr10ka_global_local/gen_0` 至 `gen_3`。每代保存 checkpoint、日志、`resolved_config.py`、`inference_config.py` 和 `best.pth`。每代训练后是否批量导图由该实验的 infer_after_generation 控制；本次保留已有取值。

## 四项评估指标

参考官方代码：`/remote-home/share/yujietang/image_enhance/code_repository/PPR10K/code_3DLUT/calculate_metrics.m`。

令 w=1（mask 非零，人像区域），w=0.5（mask 为零，背景）：

- `PSNR`：完整图像的普通 RGB PSNR。
- `deltaE`：完整图像的 CIE76 Lab 色差。
- `PSNR_hc`：对 RGB 图像乘 w 后计算 PSNR。因此背景在 MSE 中的系数是 0.25。
- `deltaE_hc`：在 Lab 空间对图像乘 w 后计算 CIE76 距离。因此背景色差的系数是 0.5。

HC 指标仍对整张图像取平均，不除以人像面积或权重和。输出与 GT 使用模型现有的 8-bit 图像量化方式；非零掩码值均作为人像。若掩码尺寸不一致，使用独立实现的 MATLAB 风格 bicubic 抗锯齿缩放，保留整数掩码的舍入与边界处理。全量检查确认输入与 GT 均一致；9 张验证图的掩码高度相差 1–2 像素，会进入此缩放路径。

多图验证保留每张图的原始尺寸：loader 不堆叠不同尺寸的图像，模型在当前 batch 内按相同尺寸分组执行前向计算，再逐图计算四项指标并保存图片。结果恢复原始样本顺序，指标仍按图片数量平均；最后不足一个 batch 的样本也会被保留。这不会对横图、竖图进行统一缩放或填充，也不会改变训练增强。

训练时四项指标自动写入每代验证日志。训练完成后可重新评估并保存图片：

```bash
CUDA_VISIBLE_DEVICES=4 python "$project_root/tools/test.py" \
  "$project_root/work_dirs/ppr10ka_v3/gen_3/inference_config.py" \
  "$project_root/work_dirs/ppr10ka_v3/gen_3/best.pth" \
  --eval-val \
  --out "$project_root/work_dirs/ppr10ka_v3/gen_3/val_metrics.pkl" \
  --save-path "$project_root/work_dirs/ppr10ka_v3/gen_3/val_images"
```

也支持生成的匹配测试配置，例如 `--eval-val` 配合 `ppr10ka_v3_test.yaml`。普通推理优先用各代 `inference_config.py`，避免加载蒸馏教师。

## 专家 b/c、架构和预算调整

生成新的实验名；已有配置不会被覆盖。例如专家 b：

```bash
python "$project_root/tools/prepare_lut_experiment.py" \
  --dataset ppr10k --target b --name ppr10kb_v3 --preset all --gen 4 --epochs 100
CUDA_VISIBLE_DEVICES=4 python "$project_root/tools/train_kd_unified.py" \
  "$project_root/adaint/configs/generated/ppr10kb_v3.yaml" --gpus 1 --seed 3407
```

`--target c` 使用专家 c。`--architecture v2/v3/global_local` 选择架构；`--preset baseline` 关闭新增训练方法，`--enable` 可独立选择方法。`all` 不包括可靠性 KD、RGB MSE 和低频损失；需要时添加 `--enable confidence_kd mse low_frequency`。

可用 `--epochs`、`--gen` 和 `--set data.train_dataloader.samples_per_gpu=4` 调整预算。`--data-root` 可更换遵循同一目录结构的数据根目录。通过生成器选择专家，勿仅修改 YAML 顶层 target 字段而遗漏实际 GT 路径。

恢复示例：

```bash
CUDA_VISIBLE_DEVICES=4 python "$project_root/tools/train_kd_unified.py" \
  "$project_root/adaint/configs/generated/ppr10ka_v3.yaml" --gpus 1 --seed 3407 \
  --start-gen 2 --resume-from "$project_root/work_dirs/ppr10ka_v3/gen_2/epoch_50.pth"
```

## 修改文件

路径均相对于上述项目根目录：

| 文件 | 用途 |
| --- | --- |
| adaint/dataset.py、adaint/__init__.py | 新增并注册按现成目录加载的 PPR10KFolder。 |
| adaint/metrics_hc.py | 按官方定义实现 PSNR_hc 与 deltaE_hc。 |
| adaint/model_architecture_v2.py | 新模型公共评估入口接收 mask_path，支持四项指标。 |
| adaint/context_transform.py | 无颜色扰动步骤时，整图上下文不额外添加颜色扰动。 |
| adaint/config_utils.py | 兼容旧 MMEditing 对 Resize 尺寸 tuple 的要求。 |
| adaint/epoch_hooks.py | 按 epoch 打印训练／验证日志，保存 epoch checkpoint，转换底层提示。 |
| mmedit/core/evaluation/eval_hooks.py | 评估前保留 epoch 训练日志。 |
| adaint/experiment_utils.py、tools/test.py | 保存并恢复验证指标，--eval-val 保留 HC 评估。 |
| tools/prepare_lut_experiment.py | 选择数据集、专家及数据路径，生成匹配训练／评估配置。 |
| adaint/configs/ppr10k_refinement.yaml | PPR10K 基础模板。 |
| adaint/configs/generated/ppr10ka_v3*.yaml、ppr10ka_global_local*.yaml | 两种可运行模型及匹配评估配置，共四份。 |
| tests/test_adaint/test_epoch_logging.py | epoch 换算、导出配置、日志平均与 checkpoint 兼容测试。 |
| tests/test_adaint/test_ppr10k.py | 指标公式、数据匹配、配置预算、batch 与 CUDA 评估测试。 |
| adaint/PPR10K.md | 本运行说明。 |

## 验证

29 项 epoch 日志、PPR10K 与原 refinement 针对性测试通过，包含官方指标公式、掩码缩放及真实 CUDA 模型评估。复现命令：

```bash
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 python -m pytest -o addopts='' -q \
  "$project_root/tests/test_adaint/test_epoch_logging.py" \
  "$project_root/tests/test_adaint/test_ppr10k.py" \
  "$project_root/tests/test_adaint/test_refinement.py"
```

真实 PPR10K 数据功能训练已通过：V3 与 GlobalLocal 各运行 2 代、每代 2 iter，训练 batch=2，验证样本包含 1432_0（掩码尺寸不同）。教师加载、反向传播、best.pth 保存、--eval-val 四项指标及 PNG 导出均正常。完整训练没有启动，这些短训练指标不代表最终模型效果。

Batch 256 后续验证：在 A100 80GB 的 GPU 4 上，V3 和 GlobalLocal 均完成真实 batch=256 的普通训练及带教师蒸馏各一步。CUDA 峰值预留显存分别约 20.6 GiB / 25.2 GiB；单 GPU 预算换算测试通过。测试未启动完整训练。

Epoch 日志改造验证：29 项针对性测试通过。真实 PPR10K 小样本两代、每代 3 epoch 验证了训练／验证日志同时输出、epoch 权重保存、最佳权重选择和教师加载；从 epoch_1.pth 恢复仅继续第 2、3 个 epoch，独立验证仍保留四项指标。未启动完整训练。
