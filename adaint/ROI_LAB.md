# 前景 ROI-Lab：全模型联合微调

分支 `feature/roi-lab__from_basis-lut` 从用户指定的
`experiment/basis-lut__from_semantic-dual`（`33a4890`）建立。
新配置是 `adaint/configs/face_niguang_roi_lab.yaml`，模型类型为 `AiLUTROILab`。
原 basis、室内、室外配置仍使用原来的模型与数据流程。

## 改动内容

1. 保留共享 backbone、前景/背景条件编码器、双路 RGB LUT 和双 AdaInt，全部参与训练。
2. 保留并训练背景的原局部仿射修正及共享局部特征层。
3. 去掉旧前景局部仿射的 FiLM/输出头，以原图前景 ROI 的轻量编码器和三通道 Lab 残差替代。
4. ROI 特征的掩码池化结果经零初始化投影进入前景 LUT 条件；空间特征与前景 LUT 结果共同预测 ΔL*、Δa*、Δb*。
5. 在完整 ROI 的前景掩码上去除残差均值，让局部头主要处理区域内部差异。整脸平均曝光与颜色由可训练的前景 LUT 调整。
6. 保留现有 RGB/P3/区域损失；增加完整前景 ROI 的平均明度误差、中心化明度误差，以及实际 Lab 修正的幅度和平滑正则。

没有多曲线 tone、背景 tone 或单独训练 gain 的冻结阶段。

### ROI、颜色和训练/推理一致性

- 默认 ROI 为 128×128、边界扩张 15%；取原始完整输入中包含前景的方形区域，越界部分补边，保持原图几何比例。
- 训练时先做配对翻转、提取完整 ROI 和完整 ROI GT，再随机裁剪。ROI 不是从 64×64 或 256×256 全图特征上放大。
- 裁剪至少保留原前景的一半；采样失败时使用完整图像。
- 训练和推理共用 ROI 采样与坐标映射；同一完整 ROI 场映射到训练裁剪或原生整图，不在每个裁剪上单独去均值。
- GT ROI 只用于监督。GT 不进入 ROI 编码、条件向量或预测路径；无 GT 的原生分辨率导出已验证。
- Lab 使用项目已有的 Display P3/D65 定义，不使用 D50 Lab。L* 先限制在可显示范围，再沿等线性亮度的灰轴压缩色度，避免分别裁剪三个 RGB 通道。
- 请求的局部残差在 ROI 前景上零均值，默认上限为 ±8 L*、±4 a*/b*。插值、显示范围限制与色域处理后，实际均值可能不为零，因此日志同时记录请求均值和实际均值；不声称与 LUT 完全解耦。
- 原前景仿射模块被替换，所以初始化不保证逐像素等于旧模型。新投影/输出头零初始化，兼容的原权重继承；没有保留不参与训练的旧前景头。

总参数 **2,308,984**，全部可训练。新 ROI 模块有 39,536 个参数，替换旧前景头的 16,576 个参数，相对 basis 净增 22,960 个。

## 训练设置

从 `work_dirs/face_niguang_semantic_basis/gen_0/best.pth` 初始化，默认训练 30 个 epoch，seed=2026。

| 参数组 | 初始学习率 |
| --- | ---: |
| backbone | 5e-6 |
| 原条件编码器、LUT、AdaInt、背景局部模块 | 1e-5 |
| 新 ROI 编码器、投影与局部头 | 5e-5 |

Adam + 余弦调度。`adaint_fix_epochs: 0`；新模型发现非零冻结预算会直接报错。
没有对原模型调用强制 eval，也没有 detach 原模型特征。
零初始化头可能让部分上游参数第一步梯度为零，这属于初始化的链式梯度行为；参数始终开启梯度，后续实际更新已验证。

新配置关闭 input-only brightness/saturation=0.2，保留配对几何增强。原 backbone 的训练模式与 Dropout 行为保留。

### 82 号掩码与公平比较

新配置中的 `semantic_masks.foreground_ignore_boxes` 对 82 号使用输入图核查后的保守排除区域 `[0, 0, 1, 0.42]`。
它去除头顶上方明显的窗外误检部分（原 750×1000 mask 中为 4,355 像素），同步用于融合、ROI、监督与评价，**不改写原 mask 文件**。
这只是已确认区域的保守清理，不是完整重新精标。

因此新实验应与同一掩码规则下的基线比较，不能直接与旧日志的前景 L* MAE=3.03261 比较。
已经用原 basis best 对 9 张验证图重算，保存于：

`work_dirs/face_niguang_roi_lab/baseline_same_masks.json`

| 同掩码基线指标 | 数值 |
| --- | ---: |
| PSNR | 26.68775 |
| 前景 ΔE | 4.60213 |
| 前景 L* MAE | 2.89906 |
| 前景 ab 误差 | 3.01848 |
| 背景 L* MAE | 3.64283 |

这组指标来自未更新的原 basis checkpoint，不是新架构的结果。微小浮点推理差异可能影响最后几位。
若修改了 mask 规则、划分或评价设置，应再次运行：

```bash
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=2 python tools/evaluate_roi_baseline.py \
  adaint/configs/face_niguang_roi_lab.yaml
```

## 运行训练

在 DB94 执行，GPU 编号按实际使用情况替换：

```bash
cd /DB/data/yujietang/IQ-LUT/image_enhance/AdaInt_BAN
git switch feature/roi-lab__from_basis-lut
source /DB/rhome/yujietang/miniconda3/etc/profile.d/conda.sh
conda activate AdaInt_BAN

CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=2 python tools/train_kd_unified.py \
  adaint/configs/face_niguang_roi_lab.yaml --seed 2026
```

配置中的 `roi_init_from` 自动继承兼容原权重、丢弃明确替换的前景头，并从 0 重置训练计数。原优化器状态不导入。
迁移加载仅允许这组明确的结构变化，其他键缺失/多余或形状不匹配会报错。

断点恢复：

```bash
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=2 python tools/train_kd_unified.py \
  adaint/configs/face_niguang_roi_lab.yaml --seed 2026 \
  --resume-from work_dirs/face_niguang_roi_lab/gen_0/latest.pth
```

恢复使用完整 ROI 模型 checkpoint，严格核验键/形状，恢复计数和优化器；不会重新用 basis 覆盖断点权重。

输出目录：`work_dirs/face_niguang_roi_lab/gen_0/`。

- `epoch_*.pth`、`latest.pth`：逐轮保存和恢复。
- `best.pth`：默认按验证前景 ΔE 最小选择，同时考虑明度与色度。
- `best_psnr.pth`：另存全图 PSNR 最优轮，便于检查前景与全图之间的取舍。
- `best_selection.json`：记录两种选择的分数、轮次与来源。
- `train_infer/{id}_result.png`：使用 best 对训练和验证图进行原生输入分辨率导出，不加载 GT；有 P3 ICC 的输入会把该配置写入输出 PNG，不改变 RGB 数值。
- `resolved_config.py`、`inference_config.py`、日志：保存实际配置、分组学习率、seed 与提交信息。

日志中的单个 lr 可能显示第一个 backbone 参数组的值，其余组按上表的倍率更新。
可在新配置的 `optimizers.paramwise_cfg.custom_keys` 修改倍率，所有组都保持非零学习率。

重点观察 `roi_mean_lightness_error`、`roi_spatial_lightness_error`、`roi_delta_L_spatial_std`、
`roi_requested_mean_abs`、`roi_actual_mean_abs`，以及原有验证前景 L*/ab、背景明度、高光和 PSNR。
不能仅凭图像更亮或一个前景指标变好，判断整体方法成功。

## 已完成的验证与限制

- 相关测试共 53 项通过（52 项模型/旧配置/训练日志回归，加 1 项新增空间拟合机制检查）。
- 包含 P3/D65 往返、黑色/色域边界有限梯度、ROI 原始细节、越界与空前景、裁剪坐标、GT 不进入预测、严格初始化与完整 checkpoint 加载。
- 验证原 backbone、两路条件/LUT/AdaInt、背景局部层和新 ROI 模块都获得梯度并实际更新。
- 受控空间拟合检查能同时对同一 ROI 的暗侧提亮、亮侧压暗，避免仅凭“有梯度”判断机制工作。
- 正式 `train_kd_unified.py` 入口使用真实图 1/31（其中 31 的 GT 尺寸不同）、验证图 82，完成 2 epochs/4 updates，再从 epoch_1 恢复完成余下训练。
- 恢复后的计数、三组初始学习率、原参数更新、完整 strict reload、三张 750×1000 带 P3 ICC 的导出均通过。
- 验证目录：`/tmp/adaint-roi-lab-sanity-v3rl06q5`；功能短跑不是性能实验，不能据此判断新方法收益。
- 仓库 `setup.cfg` 默认 `addopts=tests/` 会额外运行整个旧 MMEditing 测试库。首次运行有 350 passed/8 failed，失败涉及缺失的旧模型配置、已有旧配置断言、LMDB 环境；未改动这些无关部分。
- 本次相关检查应显式覆盖默认收集参数：

```bash
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=2 python -m pytest -o addopts= -q \
  tests/test_adaint/test_roi_lab.py \
  tests/test_adaint/test_semantic_dual.py \
  tests/test_adaint/test_basis_ablation.py \
  tests/test_adaint/test_epoch_logging.py
```

完整 30 轮主实验没有启动。数据仍为 76 张训练、9 张验证；test 复用 validation，不是独立测试集。
