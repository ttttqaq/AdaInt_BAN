# 前景双边网格明度校准（联合训练）

代码起点：`experiment/basis-lut__from_semantic-dual` 的 `1d3898a`。新分支：`feature/bilateral-calibration__from_basis-lut`。保留该起点已有的保存格式设置，不改旧实验配置、原始图像、掩码或历史结果。

## 方法和训练范围

- 完整继承 basis best：共享骨干、前景/背景 RGB basis LUT、两路 AdaInt、原前景/背景局部仿射头均保留，全部参数从训练开始就可更新，`adaint_fix_epochs: 0`。没有冻结教师网络装在学生内。
- 在裁剪、缩小全图之前提取原图前景 ROI（128×128，正方形、15% 边距）。小型多尺度编码器读取输入、当前完整前景基图、掩码、P3/D65 L*、局部明度差、坐标及前景/背景条件，预测 8×8 空间节点 × 8 明度节点的单通道网格。
- 按原生像素的位置和输入明度查询 ΔL*，范围 ±8，允许净提亮、净压暗，不做前景零均值。原 RGB 路径继续学习颜色，新头只修正前景明度。差分回写保证网格为零时与完整 basis 输出逐像素一致；必要时做保持明度的色域压缩。
- 最终输出继续使用原 RGB、Lab 和前景/背景损失。完整 ROI 的 1×1、4×4、8×8 分块比较实际明度变化和所需变化；该辅助目标只更新校准器，当前基图和目标停止梯度，原路径仍通过最终 GT 损失更新。
- 训练前缓存原 basis 在 **76 张训练图** 上的 L*/100。原结果接近训练 GT 的位置使用软输出保持约束，背景权重大于前景。这固定的是参考输出，不是学生参数；验证、导出均不读取缓存和 GT。
- 30% 的训练样本增加温和前景曝光扰动：线性 P3 内的常量、方向随机渐变或宽亮斑/阴影，最大 ±0.3 EV。背景像素、GT 不变，限制新增裁剪；原全图亮度/饱和度扰动关闭。

Adam 初始学习率：新校准器 `1e-4`，共享骨干/背景分支 `2e-6`，前景分支/原局部头 `5e-6`；余弦衰减至 `1e-6`，没有冻结阶段。训练 30 轮，每轮评估和保存。

## 运行

在 DB94 的项目根目录执行；下面以空闲 GPU 6 为例。

```bash
ssh DB94
cd /DB/data/yujietang/IQ-LUT/image_enhance/AdaInt_BAN
conda activate AdaInt_BAN

# 首次生成；缓存已存在时检查其来源和数据，不重复推理。
CUDA_VISIBLE_DEVICES=6 python tools/prepare_bilateral_reference.py \
  adaint/configs/face_niguang_bilateral_calibration.yaml

CUDA_VISIBLE_DEVICES=6 python tools/train_kd_unified.py \
  adaint/configs/face_niguang_bilateral_calibration.yaml \
  --seed 2026 --deterministic
```

初始化 checkpoint：`work_dirs/face_niguang_semantic_basis/gen_0/best.pth`。加载时严格要求全部原权重存在且形状一致，包括原前景局部头；只有新增校准器允许没有预训练权重。新网格输出层零初始化。

参考文件保存在 `work_dirs/face_niguang_bilateral_calibration/reference/`，包含训练 L*、源 checkpoint SHA256、图像/掩码路径和文件指纹、模型/掩码配置、同规则基线验证指标。输入数据、划分或基线改动时会拒绝旧缓存，应在新配置里更换 `calibration_reference_dir` 和 `work_dir`。

默认使用 76 张训练图、9 张验证图；`data.test` 复用验证集，不是独立测试集。ID 82 的掩码顶部误检以配置中的 ignore box 排除，原掩码文件未改；缓存准备程序也会用这一规则重新评估原 basis，避免跨掩码比较。

## 权重和结果

训练结果在 `work_dirs/face_niguang_bilateral_calibration/gen_0/`：

| 文件 | 含义 |
|---|---|
| `epoch_N.pth` | 每轮学生权重、优化器和计数 |
| `initial_basis.pth` | 以新模型结构保存的完整 basis + 零校准，用作兼容的回退权重 |
| `best_psnr.pth` | 整图 PSNR 最优的训练 checkpoint |
| `best_foreground.pth` | 前景 ΔE 最优的训练 checkpoint，不附带背景通过保证 |
| `best.pth` | 满足下述约束后的前景最优权重，或显式回退初始 basis |
| `best_selection.json` | 各 best 的来源、基线、阈值、合格轮次、是否回退 |
| `inference_config.py` / `training_config.py` | 可重用的推理/训练配置 |
| `train_infer/` | 默认 `best.pth` 对训练与验证全部 85 张原生尺寸图的导出，携带输入 P3 ICC |

默认合格条件预先固定：PSNR 相对同规则原 basis 最多降低 0.05 dB；背景/高光平均 L* 绝对误差最多增加 0.10；前景 L* 误差最多增加 0.05；前景 ΔE 必须优于 basis。在合格 checkpoint 中选前景 ΔE 最低者。高光指 GT L*≥80 的像素。全部阈值在 YAML 的 `calibration_selection_tolerances` 中固定。

**先查看 `best_selection.json` 的 `using_baseline_fallback`。若为 true，`best.pth` 和默认导出就是初始 basis，不能将其当作新模型训练后的改进结果。** 训练后的前景最优权重仍保存在 `best_foreground.pth`，可用下面命令导出，检查新方法具体学到了什么。

```bash
CUDA_VISIBLE_DEVICES=6 python tools/export_bilateral.py \
  work_dirs/face_niguang_bilateral_calibration/gen_0/inference_config.py \
  work_dirs/face_niguang_bilateral_calibration/gen_0/best_foreground.pth \
  --out work_dirs/face_niguang_bilateral_calibration/gen_0/foreground_infer \
  --split both
```

导出使用输入 RGB 和前景掩码，按输入自身尺寸处理，不加载 GT、不匹配 GT 尺寸、不使用训练缓存。默认包含训练/验证两部分；`--split train` 或 `val` 可单独导出。PNG 带输入的 Display P3 ICC，RGB 数值不做额外颜色空间转换。

训练中断时，使用相同配置和运行目录恢复。例如：

```bash
CUDA_VISIBLE_DEVICES=6 python tools/train_kd_unified.py \
  adaint/configs/face_niguang_bilateral_calibration.yaml \
  --resume-from work_dirs/face_niguang_bilateral_calibration/gen_0/epoch_10.pth \
  --seed 2026 --deterministic
```

恢复会严格检查完整学生结构，恢复优化器和迭代计数，并读取先前验证记录参与 best 选择。旧 basis 只通过 `calibration_init_from` 初始化，不能直接作为新结构的 `--resume-from`。

## 诊断

日志除原指标外，记录背景/高光 L* 误差、1/4/8 分块明度校准误差、实际 ΔL* 的均值/绝对值/标准差、网格幅值和触界比例、前景/背景参考质量权重。比较训练/验证时分别统计；重点查看净补偿、空间差异以及背景漂移，不用 85 张混合平均代替验证表现。

零初始化一致、梯度、坐标映射、受控正负修正和短程训练检查只能验证实现正确，不代表正式训练必然提升。新结果仍应与同掩码、同 P3 协议的原 basis 逐图比较。

实现验证中，零网格与**同一次前向里的完整基图**逐像素相同。分别运行原模型与新模型时，原 CUDA 推理本身存在约 `1e-6` 的浮点波动；原模型重复运行也会出现该量级差异，因此独立调用比较使用 `2e-6` 的绝对容差，不能把跨调用的浮点差异归为校准效果。

## 本次实现验证（2026-10-20）

DB94 原环境（Python 3.7、PyTorch 1.8.1、原生 CUDA AiLUT）下，相关测试共 65 项通过。正式启动器以 ID 1/31 两张训练图和 ID 82 一张验证图运行 2 轮，再从第 1 轮恢复；检查确认总计数为 4、优化器包含全部学生参数且学习率非零，骨干、双路 LUT/AdaInt、原双路局部头及新编码器均有实际参数更新。三个样本均成功导出 750×1000 的 P3 PNG。

隔离检查在 `/tmp/adaint_bilateral_sanity_20261020/`，未占用正式训练目录。完整 76 张训练参考缓存和 9 张同规则基线验证已准备好；尚未运行正式 30 轮实验。短程检查的默认 best 按规则回退到 basis，训练后的前景 best 也已单独导出；这些检查不用于判断方法提升。
