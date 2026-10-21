# 可消融的全局结果感知后处理

DB94：`/DB/data/yujietang/IQ-LUT/image_enhance/AdaInt_BAN`

分支：`feature/result-postprocess__from_basis-lut`，基于 `experiment/basis-lut__from_semantic-dual` 的 `1d3898a`。原分支、旧训练配置、数据划分、原有RGB/P3/区域损失权重、LUT生成公式、CUDA扩展及历史结果未修改。

## 实现内容

- `adaint/postprocess.py`：结果条件适配器、有界整图/区域RGB校准、多尺度系数细化、RGB向量引导滤波。
- `adaint/model_semantic.py`：仅集成上述后处理、对应幅度/平滑项与诊断日志；支持冻结全局模型并关闭其Dropout。
- `adaint/experiment_utils.py`：严格、明确允许新模块的warm start；旧增强器参数缺失或形状不符仍拒绝加载。
- `tools/train_kd_unified.py`：warm start与消融命令行覆盖，保存初始化来源；resume继续恢复完整权重与优化器。
- `tools/smoke_semantic_dual.py`：短流程检查支持相同初始化/冻结方式。
- `tools/estimate_postprocess_bias.py`：只从训练集估计固定RGB偏移对照及修正范围，在验证集报告其实际效果。
- 三份新YAML：室内、逆光、室外；原Basis与tone配置保持原样。
- `tests/test_adaint/test_postprocess.py`：旧代码数值回放、迁移、冻结、梯度、RGB数值稳定性、裁剪/空区域、序列化等。

既有Basis测试中“禁止修改整个forward”的源码锁改为继续保护全局映射与旧decoder；新测试用固定父提交 `1d3898a` 的模型作数值回放，验证关闭开关时保持原行为。

## YAML开关与推荐起点

新配置推荐先运行核心组合：**结果条件＋整图校准，冻结全局，保留原空间修正**。所有方法均已实现，但不默认把全部方法同时开启当成最佳方案。

```yaml
train_epochs: 30
adaint_fix_epochs: 0
postprocess_init: work_dirs/face_niguang_semantic_basis/gen_0/best.pth
model:
  # 其他原模型参数保持原配置
  postprocess:
    result_condition: true
    global_dc: true
    regional_dc: false
    global_dc_limit: 0.08
    regional_dc_limit: 0.04
    calibration_hidden: 64
    multiscale: false
    guidance: luminance
    freeze_global: true
    fixed_rgb_bias: null
optimizers:
  type: Adam
  lr: 0.00005
  # 其他优化器项沿用原配置
```

| 开关 | 推荐首次值 | 作用 |
| --- | --- | --- |
| `result_condition` | true | 空间decoder显式读取两路完整上下文的全局结果和mask；零初始化适配器保持迁移起点 |
| `global_dc` | true | 每图预测3维、有正负方向的整图RGB校准 |
| `regional_dc` | false | 额外预测前景/背景相对校准，可独立开启 |
| `global_dc_limit` | 0.08 | `d0=limit*tanh(head)`，归一化编码RGB单位，每通道最大绝对值 |
| `regional_dc_limit` | 0.04 | 区域差异向量d1的最大绝对值，不是完整像素总修正的边界 |
| `calibration_hidden` | 64 | 低容量校准预测头的隐藏宽度 |
| `multiscale` | false | 原64系数场保持，额外细化分支融合细/中/粗特征，默认生成128系数场 |
| `guidance` | luminance | 可改为rgb，采用正则化3×3 RGB协方差求解的向量引导滤波 |
| `freeze_global` | true | 冻结backbone、两路条件/LUT/AdaInt，保留局部与校准模块训练；全局子模块维持eval |
| `fixed_rgb_bias` | null | 固定RGB偏移对照，格式`[R,G,B]`；与学习的global_dc互斥 |

`global_dc_limit=0.08`和`regional_dc_limit=0.04`均不是经过正式效果搜索的最优值。训练集平均残差绝对值的95%分位：室内约 `[.0173,.0139,.0171]`，逆光约 `[.0673,.0644,.0589]`，因此0.08可作为两组共用的保守起始边界。区域0.04是较小的实验起点，默认关闭。

默认核心模型总参数2,299,979，其中冻结全局后可训练75,299；全部功能开启时总参数2,317,230，可训练92,550。原Basis为2,286,024参数。参数不等于计算量：RGB引导和额外系数网格仍会增加推理开销，应在空闲GPU正式计时。

### 前向和初始化语义

原本的空间修正仍按每路完整上下文区域去均值。额外校准在两路融合后加入：

```text
Y = m*(G_fg + U_fg) + (1-m)*(G_bg + U_bg) + d0 + (m-pi)*d1
pi = 完整上下文mask均值，而不是随机crop均值
```

d0控制整图平均；d1控制区域相对变化，区域过小或为空时关闭d1。校准头和新增系数细化头的输出层均零初始化。校准能量采用原 `local_magnitude_factor`，新增多尺度系数沿用原 `local_smoothness_factor`；未新增暗部专用损失或固定肤色规则。

`result_condition`只控制空间decoder的结果条件输入；校准头只要启用，就始终读取原图、全局结果与mask。默认局部输入仍为128，因此基础系数为64、细化系数为128；若改变 `model.local_resolution`，两者随之变化。

保留luminance引导时，从旧checkpoint迁移到结果条件/校准/多尺度结构，初始输出可保持原值。可选模块保留独立的随机数分配上下文，相同seed下共有参数初始化不因开关而改变。切换到rgb会改变原系数场的滤波方式，初始输出不承诺与旧luminance版本相同；这正是需要单独验证的消融。

RGB引导提供颜色边缘条件，并不生成高频纹理；没有实现锐化、去噪、Retinex或像素MLP，也没有自动清理错误数据配对。

## 开始训练

当前不自动启动正式训练；先选择空闲GPU。下面GPU0只是命令示例。

```bash
cd /DB/data/yujietang/IQ-LUT/image_enhance/AdaInt_BAN
git switch feature/result-postprocess__from_basis-lut
source /DB/rhome/yujietang/miniconda3/etc/profile.d/conda.sh
conda activate AdaInt_BAN
```

先跑逆光核心组合：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_kd_unified.py \
  adaint/configs/face_niguang_semantic_postprocess.yaml --seed 2026 \
  --work-dir work_dirs/face_niguang_post_core_seed2026
```

对应室内实验：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_kd_unified.py \
  adaint/configs/face_indoor_semantic_postprocess.yaml --seed 2026 \
  --work-dir work_dirs/face_indoor_post_core_seed2026
```

室外使用 `adaint/configs/face_outdoor_semantic_postprocess.yaml`，需先具备对应Basis baseline的 `best.pth`。warm start路径可在YAML顶层 `postprocess_init` 修改；模型类型、基表数、节点数等必须匹配。

全部功能同时开启，使用独立结果目录：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_kd_unified.py \
  adaint/configs/face_niguang_semantic_postprocess.yaml --seed 2026 \
  --work-dir work_dirs/face_niguang_post_all_seed2026 \
  --cfg-options model.postprocess.regional_dc=True \
    model.postprocess.multiscale=True model.postprocess.guidance=rgb
```

默认配置已经开启result_condition与global_dc，并冻结全局；上面的覆盖再启用regional_dc、多尺度和RGB引导。

## 消融方式

每组从同一原始Basis best初始化，固定相同seed、30轮追加预算和数据规则。原空间模块在所有组都继续训练，不能拿旧best直接当“同等训练预算”的A组。

| 组 | 在核心配置后添加的`--cfg-options` | 建议目录后缀 |
| --- | --- | --- |
| A 原空间继续训练 | `model.postprocess.result_condition=False model.postprocess.global_dc=False` | `post_A_seed2026` |
| B 只加整图校准 | `model.postprocess.result_condition=False` | `post_B_seed2026` |
| C 只加空间结果条件 | `model.postprocess.global_dc=False` | `post_C_seed2026` |
| D 核心组合 | 无覆盖 | `post_core_seed2026` |
| D+区域 | `model.postprocess.regional_dc=True` | `post_region_seed2026` |
| D+多尺度 | `model.postprocess.multiscale=True` | `post_multiscale_seed2026` |
| D+RGB引导 | `model.postprocess.guidance=rgb` | `post_rgb_seed2026` |
| 全开 | 三项均覆盖 | `post_all_seed2026` |

例如A组：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_kd_unified.py \
  adaint/configs/face_niguang_semantic_postprocess.yaml --seed 2026 \
  --work-dir work_dirs/face_niguang_post_A_seed2026 \
  --cfg-options model.postprocess.result_condition=False model.postprocess.global_dc=False
```

这些参数也可直接在各自YAML副本中修改。`--cfg-options`只允许后处理与相关训练项，不用于覆盖数据管线；数据或上下文管线改动应编辑源YAML再重新加载。

### 固定RGB偏移对照

估计只使用训练ID，验证集仅用于报告效果：

```bash
python tools/estimate_postprocess_bias.py \
  adaint/configs/face_niguang_semantic_basis.yaml \
  --results work_dirs/face_niguang_semantic_basis/gen_0/train_infer_niguang \
  --output /tmp/niguang_fixed_bias.json
```

室内把配置换为室内Basis，结果目录换为 `train_infer`。本次训练集固定bias对照：室内验证29.8079→29.7913dB，逆光26.6837→26.4318dB，说明简单固定偏移没有取得收益。此工具不使用GT oracle来生成验证图的逐图偏移，不自动过滤坏配对。

## 输出、评价、恢复和后续联合微调

输出仍为 `--work-dir/gen_0/`，保留epoch/latest/best checkpoint、配置和日志。训练完成使用best对训练+验证图作满分辨率 `train_infer/` 导出。

重新评价新模型：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  work_dirs/face_niguang_post_core_seed2026/gen_0/inference_config.py \
  work_dirs/face_niguang_post_core_seed2026/gen_0/best.pth --eval-val
```

恢复训练时使用本次保存的resolved配置，避免忘记当初CLI覆盖的开关；明确提供原始根工作目录，避免重复拼接gen_0：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_kd_unified.py \
  work_dirs/face_niguang_post_all_seed2026/gen_0/resolved_config.py \
  --resume-from work_dirs/face_niguang_post_all_seed2026/gen_0/latest.pth \
  --work-dir work_dirs/face_niguang_post_all_seed2026 --seed 2026
```

`postprocess_init`是迁移起点：保留旧空间权重，只允许新增模块缺失，计数清零，优化器从头开始。`--resume-from`是完整恢复：不再次warm start，恢复模型、优化器和计数。推理和resume仍要求同结构checkpoint，不能任意改变开关后非严格加载。

冻结版本有效后可选做小学习率联合微调；这是另一组实验，不覆盖冻结组：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_kd_unified.py \
  adaint/configs/face_niguang_semantic_postprocess.yaml --seed 2026 \
  --work-dir work_dirs/face_niguang_post_joint_seed2026 \
  --cfg-options postprocess_init=work_dirs/face_niguang_post_core_seed2026/gen_0/best.pth \
    model.postprocess.freeze_global=False optimizers.lr=0.00001 train_epochs=10
```

这个例子对全模型使用较小学习率。若进一步配置不同模块lr_mult，需注意MMCV按最长匹配键覆盖；已有adaint键的lr_mult为0.1。不要将解冻/优化器策略改变伪装为相同协议下的纯结构对照。

从零训练也支持：在YAML中将 `postprocess_init: null`、`freeze_global: false`，并恢复100轮/5轮AdaInt冻结预算。首轮更推荐已有best上的后处理验证，避免把全局训练波动混进结论。

## 日志与验证结果（2026-10-21）

新增日志包括整图/区域校准绝对均值、输出越界比例和校准幅度正则；原 `local_correction_abs_mean`继续描述空间修正。训练输出不强制clamp，指标/保存沿用原路径。

在DB94现有Python3.7/PyTorch1.8.1环境中完成：

- 相关CPU回归：44 passed、9 CUDA tests skipped；包括真实AiLUT CPU扩展的前向/反向、旧提交数值回放、冻结参数不变、迁移、空mask、裁剪一致性、RGB稳定性、共有参数初始化一致、严格checkpoint与配置序列化。
- 室内、逆光真实数据各2次优化步骤和验证，损失、梯度与指标有限，mask覆盖分别为67/8/64和76/9/9。
- 全功能开启的真实CPU IterBasedRunner：2张训练图、1张验证图、1轮，训练/验证缩小到128长边；成功保存epoch_1、选择best、保存推理配置/验证图、严格重载，并恢复优化器和计数。
- 完整runner检查产物：`/tmp/adaint-postprocess-cpu-bktzjaw2`。
- GPU均繁忙，因此未执行CUDA回归、GPU耗时测试或正式30轮效果实验。短检查指标不代表新方法画质提升。

建议室内、逆光使用相同结构/选择规则验证，再扩展室外。现有逆光test复用验证集，不是独立测试；也没有自动处理之前诊断到的错误配对/黑边监督。
