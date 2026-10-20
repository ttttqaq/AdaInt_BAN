# 前景双边网格：从零联合训练

分支：`feature/bilateral-calibration__from_basis-lut`。当前版本使用模型默认初始化直接训练，不加载预训练 basis、不准备参考缓存、不提前保存初始权重。输入仍采用项目已有的 P3 数值协议；推理输出不附加 ICC，由用户通过 `p3.py` 决定后处理。

## 训练内容

共享骨干、前景/背景 RGB basis LUT、两路 AdaInt、原前景/背景局部仿射头，以及前景双边网格校准器全部参与训练。`pretrained: false`、`load_from: null`、`resume_from: null`、`adaint_fix_epochs: 0`。沿用模型的随机网络初始化、恒等 LUT 和零校准初始化，均不读取已有实验权重。

结构继续使用完整 ROI 的多尺度编码、8×8×8 单通道网格，在原分辨率按位置和输入明度查询正负 ΔL*（±8）。零网格保持当前完整 basis 路径的输出。颜色由原 RGB 路径学习。

保留最终 RGB/Lab、前景/背景监督、原 LUT/局部正则，以及完整 ROI 的 1×1、4×4、8×8 分块明度监督。该辅助目标使用当前前景路径的停止梯度结果；原路径仍由最终 GT 损失更新。这不需要另一个模型或缓存。

旧版依赖预训练输出的软保持损失、基线阈值选模和初始 basis 回退均已移除。背景与高光误差仍正常记录，用于检查训练结果。

从零训练默认 100 轮。Adam 基础学习率 `1e-4`，AdaInt 为 `1e-5`，其余模块为 `1e-4`，余弦衰减至 `1e-6`。每轮保存、验证，全程无冻结阶段。保留温和双向前景曝光扰动（概率 0.3、最大 ±0.3 EV）。

## 三套数据配置

| 场景 | 配置（`adaint/configs/` 下） | 训练 / 验证 | 结果目录（`work_dirs/` 下） |
|---|---|---|---|
| 室内常态 | `face_indoor_bilateral_calibration.yaml` | 67 / 8 | `face_indoor_bilateral_calibration_scratch` |
| 室外常态 | `face_outdoor_bilateral_calibration.yaml` | 85 / 10 | `face_outdoor_bilateral_calibration_scratch` |
| 逆光 | `face_niguang_bilateral_calibration.yaml` | 76 / 9 | `face_niguang_bilateral_calibration_scratch` |

沿用原数据划分、输入/GT、掩码和区域损失权重；三套分别训练。室内、室外各有原来的 64 张无 GT 测试图；逆光 `data.test` 复用验证集。逆光 ID 82 的 ignore box 仅用于逆光配置。已有历史结果及缓存不删除，新训练使用 `_scratch` 目录。

## 直接训练

在 DB94 项目根目录执行，以 GPU 6、室内为例：

```bash
ssh DB94
cd /DB/data/yujietang/IQ-LUT/image_enhance/AdaInt_BAN
conda activate AdaInt_BAN

CUDA_VISIBLE_DEVICES=6 python tools/train_kd_unified.py \
  adaint/configs/face_indoor_bilateral_calibration.yaml \
  --seed 2026 --deterministic
```

室外、逆光分别替换为 `face_outdoor_bilateral_calibration.yaml`、`face_niguang_bilateral_calibration.yaml`。室外同样可以直接训练，不需要先训练旧 basis。参考缓存准备脚本已移除。

如果希望单独开一次新运行，使用 `--work-dir work_dirs/自定义新目录`；该目录下会生成 `gen_0/`。已有训练输出的目录需显式恢复，或选择新目录，避免将不同运行的验证记录混在一起。

## 权重、选模与导出

以室内为例，结果在 `work_dirs/face_indoor_bilateral_calibration_scratch/gen_0/`：

| 文件 | 含义 |
|---|---|
| `epoch_N.pth` | 当轮模型、优化器和训练计数 |
| `best_psnr.pth` | 验证整图 PSNR 最优权重 |
| `best_foreground.pth` | 验证前景 ΔE 最优权重 |
| `best.pth` | 按 `best_checkpoint_metric` / `best_checkpoint_rule` 选择，默认 PSNR 最优 |
| `best_selection.json` | 所选指标、分数和实际训练 checkpoint 来源，无旧基线回退 |
| `resolved_config.py` / `inference_config.py` | 保存的训练 / 推理配置 |
| `train_infer/` | 默认 best 对训练及验证全部图片的原尺寸 PNG，不含 ICC |

默认自动导出：室内 75 张、室外 95 张、逆光 85 张。常态输入为 3000×4000、GT 为 750×1000；训练/验证沿用 `ResizeToMatch`，导出按输入自身尺寸，不读取 GT。

单独导出训练后的前景最优权重：

```bash
CUDA_VISIBLE_DEVICES=6 python tools/export_bilateral.py \
  work_dirs/face_indoor_bilateral_calibration_scratch/gen_0/inference_config.py \
  work_dirs/face_indoor_bilateral_calibration_scratch/gen_0/best_foreground.pth \
  --out work_dirs/face_indoor_bilateral_calibration_scratch/gen_0/foreground_infer \
  --split both
```

`--split` 支持 `train`、`val`、`both`（训练+验证）、`test`。独立测试使用自己的掩码根目录，输出请放到单独的 `test_infer/`，避免与训练集重复 ID 覆盖。

训练结束的自动导出及 `tools/export_bilateral.py` 都不复制、嵌入输入 P3 ICC，也不自动调用 `p3.py`。需要标签时自行使用项目原有 `p3.py`；该文件保持不变。模型内部 P3/Lab 数值计算与输出文件是否携带 ICC 是两个独立设置。

## 断点恢复

训练过程中正常保存 checkpoint；继续同一次运行时显式指定，例如：

```bash
CUDA_VISIBLE_DEVICES=6 python tools/train_kd_unified.py \
  adaint/configs/face_indoor_bilateral_calibration.yaml \
  --resume-from work_dirs/face_indoor_bilateral_calibration_scratch/gen_0/epoch_10.pth \
  --seed 2026 --deterministic
```

恢复会读取这次训练保存的模型、优化器和计数。新的从零训练运行不需要任何预先保存的权重。

## 本次验证

相关测试 72 项通过。室内、室外、逆光均以真实图片完成 2 轮短程从零训练；逆光还检查了断点恢复。确认全部参数进入优化器且学习率非零，骨干、双路 LUT/AdaInt、原双路局部头和网格编码器均有实际更新。网络参数名称、形状和默认初始化与改动前一致。

共核验 11 张自动/独立测试导出的 PNG，输入均有 P3 ICC，而输出均不含 ICC，尺寸保持原输入。临时验证位于 `/tmp/adaint_scratch_20261020/`；正式 100 轮训练尚未启动。这些短程检查用于验证实现与运行流程。
