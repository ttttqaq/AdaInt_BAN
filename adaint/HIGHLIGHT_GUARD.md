# Highlight Guard：第一轮训练约束与受限微调

仓库：`/DB/data/yujietang/IQ-LUT/image_enhance/AdaInt_BAN`

方法分支：`feature/highlight-guard__from_result-postprocess`，直接基线为
`feature/result-postprocess__from_basis-lut` 的 `0757473`。

本轮保留已有 `AiLUTDual`、两路 Basis LUT、AdaInt、结果条件、粗／细局部修正和 DC。
实现内容是可选高光训练约束、有限参数微调、正常区域输出保持，以及统一的诊断与选模协议。
没有实现新的有界局部结构、额外 LUT 专家、天空分割网络或新的颜色空间。
这些约束是软损失，不保证所有输出都处于 `[0,1]`，实际收益须由配对验证和视觉复核确定。

## 2026-10-27 已完成实验与复现入口

室外七组、室内两组均已完成追加 30 epochs；清单、训练资源、选模与图像证据保存于
`work_dirs/hg_20261027_suite/`。本次实际权重为 `over=1`、`lut_range=1`、
`final_range=10`、`keep=0.1`，来自训练样本初始损失量级校准；范围项覆盖了脚本的通用默认值。
验证集按预设质量容差选择室外 `range_over` 第 3 轮，文件为
`work_dirs/hg_20261027_outdoor_range_over_s2026/gen_0/best_guard.pth`。
`best.pth` 仍表示 PSNR 最佳权重，两者不应混用。

推荐先使用已有候选结果 `work_dirs/hg_20261027_suite/candidates/range_over/`。
重训时使用全新前缀／suite，以下命令与实际主要设置一致，不覆盖已完成实验：

```bash
CUDA_VISIBLE_DEVICES=3 ADAINT_CUDA_MEMORY_FRACTION=0.22 \
python tools/run_highlight_guard.py \
  --scene outdoor \
  --source-config work_dirs/face_outdoor_semantic_postprocess/gen_0/inference_config.py \
  --source-checkpoint work_dirs/face_outdoor_semantic_postprocess/gen_0/best.pth \
  --suite-dir work_dirs/hg_repro_20261027_suite --prefix hg_repro_20261027 \
  --variants range_over --epochs 30 --seed 2026 --lr 0.00002 \
  --over-weight 1 --lut-range-weight 1 --final-range-weight 10 \
  --keep-weight 0.1 --execute range_over
```

推理使用已保存的 `inference_config.py` 与 `best_guard.pth`；最终统计以
`suite/candidates/range_over/val/evaluation.json` 的独立复核为准。
训练日志使用五位小数，旧模型区域采样还存在 GPU 浮点 CDF 的微小重复前向差异，
因而日志与独立复核数值可能略有不同。测试图没有 GT，不计算测试 PSNR 或 GT 色差。
天空偏色和强反射并未被本轮完全解决，不能把范围软罚称为硬边界保证。

室内默认继续使用原 `face_indoor_semantic_postprocess_4`。室内新实验仅作回归检查；
虽然过亮量下降，存在个别图像质量取舍，未替换原 checkpoint。

## 兼容性与室内保护

`model.highlight_guard` 可省略。默认四项新增损失权重均为 `0`，
`trainable_scope='all'`，`detach_result_condition=false`。该状态不增加学习参数或 buffer，
旧 checkpoint 的增强器参数键和形状保持兼容；旧模型的前向和原损失路径保持原样。

室内生产结果继续使用
`work_dirs/face_indoor_semantic_postprocess_4/gen_0/best.pth`、其保存的
`inference_config.py` 及原预后处理。室内新实验只用于回归检查，不自动替换该模型。
公共函数改动仍需回放旧路径，不能仅凭新开关默认关闭就声明室内逐像素不变。

正常区域参考模型仅在 `keep_weight>0` 的训练前向中惰性加载，保持 `eval()` 和无梯度。
其构造保护 CPU RNG，跟随输入设备；它不注册到增强器的参数或 `state_dict` 中。
保存的推理配置移除参考路径、训练损失权重、冻结范围及梯度切断设置，只保留诊断所需的
`top_fraction`、`over_margin`。新 checkpoint 的正常推理不依赖参考模型文件。

本轮每个训练实验使用一张 GPU（`--gpus 1`），不同实验可以使用不同 GPU。
参考模型的惰性非注册实现不应配合单进程、多设备的 `DataParallel` 复制使用；
不要把跨卡并行实验误写成同一 anchor 模型的多设备复制。

## 配置与实际训练行为

```yaml
model:
  type: AiLUTDual
  # 其他参数从对应实验保存的 inference_config.py 继承。
  highlight_guard:
    over_weight: 0.0
    lut_range_weight: 0.0
    final_range_weight: 0.0
    keep_weight: 0.0
    top_fraction: 0.1
    over_margin: 0.00784313725490196  # 2/255，编码亮度代理单位
    keep_error_threshold: 0.03      # 逐像素 RGB RMSE 上限，不是 MSE
    reference_config: null
    reference_checkpoint: null
    trainable_scope: all            # all 或 heads
    detach_result_condition: false
```

`heads` 精确定义为只开放以下已有参数：

- 两路 `branches.<foreground|background>.lut_generator.head.output` 的权重和偏置；
- `local_decoder.outputs` 的粗尺度输出卷积；
- 若存在，`local_decoder.refinement_outputs` 的细尺度输出卷积。

backbone、条件编码器、AdaInt、Basis LUT、融合系数 head 的隐藏层、空间特征提取、FiLM、
结果适配器、金字塔特征和 DC 参数均冻结。冻结特征模块保持 `eval()`，避免 Dropout
继续改变上下文；训练步也不会因原 AdaInt 日程而将其重新解冻。
`heads` 与 `postprocess.freeze_global=true` 含义冲突，配置会拒绝该组合。

`detach_result_condition=true` 只对送入结果适配器的两路全局上下文结果停止梯度。
前向数值和参数量不变；`a×G`、中心化网格和 DC 的其他梯度路径仍然存在，
因此它是局部梯度路径消融，不是全局／局部完全解耦。

## 四项新增损失

所有损失使用保存裁剪之前的原始浮点预测。亮度代理沿用项目定义
`Y=0.2126R+0.7152G+0.0722B`，输入是编码 RGB；它不是线性光的物理亮度。

| 日志损失名 | 未乘权重时的定义与量纲 | 归一化 |
| --- | --- | --- |
| `loss_guard_over` | GT 区域亮部上 `relu(Y_pred-Y_GT-over_margin)^2`，编码亮度平方 | 每图、每区域先按选中像素归一化；有效图平均，前景／背景再等权平均 |
| `loss_guard_lut_range` | 完整融合 LUT 数值的 `relu(L-1)^2+relu(-L)^2`，RGB 数值平方 | 表内所有标量平均，两路 LUT 再平均；不作用于可任意重缩放的单独系数 |
| `loss_guard_final_range` | 最终 raw RGB 的相同范围软罚，RGB 数值平方 | 所有图像、通道及像素的标量平均 |
| `loss_guard_keep` | 学生与冻结参考的 RGB MSE，RGB 数值平方 | 每图、每个有效正常区域先归一化，前景／背景再等权平均 |

高光集合由每张 GT 在各自前景／背景区域内的亮度分位数确定，默认最亮 10%。
阈值只看 GT，不看预测；阈值处的相同亮度全部保留，平坦区域不会任意挑选像素。
区域小于 `region_min_area` 或为空时不参与平均；整个 batch 都为空时返回有限、可反向的零。

正常区域保持项只接受同时满足以下条件的参考像素：

1. 参考 RGB 每通道都在 `[0,1]`；
2. 参考与 GT 的逐像素 RGB RMSE 不超过 `keep_error_threshold`，默认 `.03`，约为 `7.65/255`；
3. 该像素不属于 GT 高光集合中“参考比 GT 亮超过 `over_margin`”的热点。

因此它不会无条件蒸馏旧模型全部输出。参考、GT 和选择 mask 都停止梯度；
GT 仅用于训练／配对验证，不参与无 GT 测试推理。它也不启用多代 BAN。

各损失归一化不同，不能仅比较配置中的权重大小判断实际作用。
应记录未加权量、加权损失、训练稳定性及验证质量。下面 CLI 的权重是可覆盖的实验起点，
不表示已搜索得到最优值；正式分析以该次生成配置和 manifest 为准。

## 诊断指标与选模

训练配置生成器在原 PSNR／P3 指标之外增加：

- `raw_high_clip`：裁剪前至少一个 RGB 通道大于 `1` 的像素比例；
- `raw_high_clip_foreground`、`raw_high_clip_background`：相应硬监督区域内的比例；
- `highlight_over_foreground`、`highlight_over_background`：GT 分位数亮部内
  `relu(Y_pred-Y_GT-over_margin)` 的均值，未平方，单位为编码亮度。

raw 指标在调用 `tensor2img` 之前从独立副本统计，避免 CPU 原地裁剪污染范围记录。
区域指标只平均有效样本；裁边和最小区域面积规则与配对评估一致。
任一通道越界、三通道近白及视觉上失去高光细节是不同概念，不能互相代替。

训练脚本的 `best.pth` 仍由验证 PSNR 最大值选择。额外的 `select` 子命令只读取
验证日志及原模型的验证评估报告，按以下预先声明的默认条件选择 `best_guard.pth`：

- PSNR 相比原模型下降不超过 `0.1 dB`；
- 前景和全图 P3 色差增加各不超过 `0.15`；
- 每个区域的高光过亮增加不超过 `0.5/255`；
- 前景／背景等权高光目标绝对改善超过 `1e-8`，且相对改善至少 5%。

满足门槛的候选中，先选高光目标最低者，再按全图色差、PSNR、epoch 打破平分。
没有合格候选时保留原模型，不生成新的 `best_guard.pth`；若此前有旧的派生
`best_guard.pth`，工具删除该派生副本并在 JSON 中记录，原 epoch checkpoint 不受影响。
这些门槛可由 CLI 覆盖，但应在查看盲测结果前固定并保留记录。

测试集没有 GT，不计算测试 PSNR／P3 对 GT 指标，也不使用专业反馈截图训练或选模。

## 准备同预算实验

以下命令在 DB94 执行。CUDA 卡号只是示例，运行前查看实际剩余显存。

```bash
cd /DB/data/yujietang/IQ-LUT/image_enhance/AdaInt_BAN
source /DB/rhome/yujietang/miniconda3/etc/profile.d/conda.sh
conda activate AdaInt_BAN
git switch feature/highlight-guard__from_result-postprocess

python tools/run_highlight_guard.py \
  --scene outdoor \
  --source-config work_dirs/face_outdoor_semantic_postprocess/gen_0/inference_config.py \
  --source-checkpoint work_dirs/face_outdoor_semantic_postprocess/gen_0/best.pth \
  --suite-dir work_dirs/hg_20261027_suite \
  --prefix hg_20261027 \
  --variants legacy_control heads_control range over range_over full detach \
  --epochs 30 --seed 2026 --lr 0.00002
```

该命令只生成配置和 `manifest_outdoor.json`，不启动训练，且拒绝覆盖同名已有配置／实验。
初始预算为追加 30 epochs、LR `2e-5`、seed `2026`、单图 batch，余弦最低 LR `1e-6`。
各候选使用相同数据划分、来源 checkpoint、随机种子及追加预算。

| variant | 参数范围 | 新增损失／路径 |
| --- | --- | --- |
| `legacy_control` | 原模型全部参数 | 无新增损失，同预算续训对照 |
| `heads_control` | 上述 heads | 无新增损失，隔离受限微调影响 |
| `range` | heads | 融合 LUT 与最终输出范围软罚 |
| `over` | heads | 单侧亮部损失 |
| `range_over` | heads | 两种范围软罚＋亮部损失 |
| `full` | heads | 上述全部＋正常区域保持 |
| `detach` | heads | 与 full 相同，另切断结果条件梯度 |

CLI 可用 `--over-weight`、`--lut-range-weight`、`--final-range-weight`、`--keep-weight`、
`--top-fraction`、`--over-margin`、`--keep-error-threshold` 指定起始设置。
manifest 记录来源配置／checkpoint、代码提交、数据标注文件 hash、预算和实际命令。

室内回归实验使用同一准备命令，将 `--scene` 改为 `indoor`，两个来源路径改为
`work_dirs/face_indoor_semantic_postprocess_4/gen_0/` 下对应文件，并指定需要的 variants。
它生成 `manifest_indoor.json` 和全新工作目录，不覆盖室内 `_4`。

## 训练、并行与恢复

```bash
CUDA_VISIBLE_DEVICES=0 ADAINT_CUDA_MEMORY_FRACTION=0.25 \
  python tools/train_kd_unified.py \
  work_dirs/hg_20261027_suite/configs/hg_20261027_outdoor_full_s2026.py \
  --seed 2026 --gpus 1
```

`ADAINT_CUDA_MEMORY_FRACTION` 是可选的每进程 PyTorch 显存比例上限，须结合实测峰值和
卡上其他任务设置；它不表示预留显存。不同 variants 可在其他卡上分别运行同类命令，
不要修改其他任务的进程或重复使用同一个工作目录。

该实验的输出目录为
`work_dirs/hg_20261027_outdoor_full_s2026/gen_0/`，包括 epoch checkpoint、日志、
`resolved_config.py`、`inference_config.py` 和 PSNR 选择的 `best.pth`。
`postprocess_init` 只迁移增强器参数并清零计数，不恢复旧优化器。
本轮生成配置关闭训练结束后的自动批量推理，推理由下述评估工具单独执行。

恢复同一实验：

```bash
CUDA_VISIBLE_DEVICES=0 ADAINT_CUDA_MEMORY_FRACTION=0.25 \
  python tools/train_kd_unified.py \
  work_dirs/hg_20261027_suite/configs/hg_20261027_outdoor_full_s2026.py \
  --seed 2026 --gpus 1 \
  --resume-from work_dirs/hg_20261027_outdoor_full_s2026/gen_0/latest.pth
```

恢复时使用同一生成配置、scope 和优化器设置。`resume_from` 优先于 warm start，
严格恢复增强器和优化器／计数；需要保持参考损失时仍须保留原参考配置和 checkpoint。
这些参考路径只在训练时需要。

## 配对评估、选模与原生分辨率导出

先用原模型生成一次同协议验证基准：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/evaluate_highlight_guard.py paired \
  --config work_dirs/face_outdoor_semantic_postprocess/gen_0/inference_config.py \
  --checkpoint work_dirs/face_outdoor_semantic_postprocess/gen_0/best.pth \
  --split val --device cuda:0 --memory-fraction 0.25 \
  --output work_dirs/hg_20261027_suite/baseline_outdoor_val --decompose --p3

python tools/evaluate_highlight_guard.py select \
  --work-dir work_dirs/hg_20261027_outdoor_full_s2026/gen_0 \
  --baseline work_dirs/hg_20261027_suite/baseline_outdoor_val/evaluation.json
```

查看 `best_guard_selection.json` 的 `winner`。只有候选合格时，下例的
`best_guard.pth` 才存在；若原模型获胜，继续使用上述原模型配置和 checkpoint。

```bash
CUDA_VISIBLE_DEVICES=0 python tools/evaluate_highlight_guard.py paired \
  --config work_dirs/hg_20261027_outdoor_full_s2026/gen_0/inference_config.py \
  --checkpoint work_dirs/hg_20261027_outdoor_full_s2026/gen_0/best_guard.pth \
  --split val --device cuda:0 --memory-fraction 0.25 \
  --output work_dirs/hg_20261027_outdoor_full_s2026/gen_0/eval_val \
  --decompose --native-paired --p3
```

将 `--split val` 改为 `--split train`，并使用单独的 `--output .../eval_train`，
即可检查训练集。正式配对指标先用 AREA 将 input 对齐 GT 尺寸；
`--native-paired` 另存 input 原生分辨率结果，不能把两种分辨率结果混为同一指标。

每个评估目录包含 `evaluation.json`、逐图 `metrics.csv`、无 ICC 的 `raw/`；
原生导出另存 `native_raw/`。`--decompose` 记录融合全局／局部阶段的范围与修正幅度。
summary 按有效样本的逐图指标平均，不是按全数据像素池化。

## 无 GT 测试与独立 P3 处理

室外测试 ID 使用已有划分文件，不把室内图片交给室外候选当作相同模型评估。

```bash
CUDA_VISIBLE_DEVICES=0 python tools/evaluate_highlight_guard.py test \
  --config work_dirs/hg_20261027_outdoor_full_s2026/gen_0/inference_config.py \
  --checkpoint work_dirs/hg_20261027_outdoor_full_s2026/gen_0/best_guard.pth \
  --input-root /DB/data/yujietang/IQ-LUT/face_dataset/test_final \
  --mask-root /DB/data/yujietang/IQ-LUT/face_dataset/test_final_masks/masks \
  --ids-file work_dirs/test_final/outdoor_ids.txt \
  --device cuda:0 --memory-fraction 0.25 \
  --output work_dirs/hg_20261027_outdoor_full_s2026/gen_0/test_outdoor \
  --decompose --p3
```

`--p3` 不在训练／推理保存函数里嵌入 ICC。工具先写无 ICC PNG，再独立调用：

```bash
python p3.py \
  work_dirs/hg_20261027_outdoor_full_s2026/gen_0/test_outdoor/raw \
  --output work_dirs/hg_20261027_outdoor_full_s2026/gen_0/test_outdoor/raw_p3
```

工具验证 raw 没有 ICC、P3 版本嵌入仓库的 Display P3 profile，且两者解码像素完全相同，
结果写入 `p3_verification.json`。统一查看 `raw_p3/` 作视觉对照；
数值评估始终基于编码 RGB 和保存前浮点输出，不能从已裁剪 PNG 推断原始越界率。

## 检查

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -o addopts= -p no:cacheprovider \
  tests/test_adaint/test_highlight_guard_losses.py \
  tests/test_adaint/test_highlight_guard_model.py \
  tests/test_adaint/test_highlight_guard_tools.py -q
```

测试覆盖旧提交前向／损失数值回放、heads 冻结、梯度切断、教师隔离与推理自含、
GT 相对高光选择、RMSE 正常区域保持、raw 范围梯度、裁剪前诊断、验证选模和独立 ICC 导出。
正式实验还需记录资源峰值、配对质量、测试视觉结果及失败情况；通过功能测试不等于画质改善。
