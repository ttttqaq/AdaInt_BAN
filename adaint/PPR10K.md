**本分支 PPR10K 训练和验证**

项目位置：`/remote-home/share/yujietang/image_enhance/my_code/AdaInt_BAN`。分支：`feature/global-affine__from_lut-refinement`。唯一模型为 `AiLUT`，详细架构见 [ARCHITECTURE.md](ARCHITECTURE.md)。

默认继续使用现有 PNG 文件和划分：`/remote-home/share/yujietang/dataset/PRR10K/prr_train/source` 配对 `target_a`；验证位于 `prr_value/source`、`prr_value/target_a`，HC mask 位于 `masks_360p`。没有额外启用官方五种增强或更换数据版本。

**从头训练单代**

```bash
ssh A100
source /home/yujietang/miniconda3/etc/profile.d/conda.sh
conda activate AdaInt_BAN
cd /remote-home/share/yujietang/image_enhance/my_code/AdaInt_BAN

git branch --show-current
python tools/prepare_lut_experiment.py --dataset ppr10k --target a --name ppr10ka_global_affine_01
CUDA_VISIBLE_DEVICES=1 python tools/train_kd_unified.py adaint/configs/generated/ppr10ka_global_affine_01.yaml --gpus 1 --seed 42
```

GPU 1 在本次验证时空闲，运行前用 `nvidia-smi` 选择空闲卡。模板：300 epoch，前 5 epoch 固定 AdaInt，每 1 epoch 保存，每 15 epoch 验证；batch 256，验证 batch 64，Adam LR `1e-3`，AdaInt LR 倍率 `0.1`，cosine 最低 LR `1e-6`。这些数据/优化预算继承原来实际 V3 配置，没有做周期或学习率调参。8875 张训练图时为每 epoch 35 次更新，总计 10500 次。默认 `gen: 1`、不启用教师。

结果目录：`work_dirs/ppr10ka_global_affine_01/gen_0/`。最终 `best.pth` 按逐图平均验证 PSNR 选择，保存对应 `resolved_config.py`、无教师的 `inference_config.py` 和日志。默认关闭每代结束后全训练集图片导出，避免长时间额外推理；可通过 `--set infer_after_generation=true` 开启。

不同实验使用新的 `--name`；生成器会拒绝覆盖同名配置。更换专家使用 `--target b` 或 `--target c`，数据根目录使用 `--data-root /path/to/PRR10K`。只需降低显存占用时可设置 `--set data.train_dataloader.samples_per_gpu=64`，实际迭代数会根据 batch 自动重算。

**同架构 BAN**

```bash
python tools/prepare_lut_experiment.py --dataset ppr10k --target a --name ppr10ka_global_affine_ban_01 --gen 4
CUDA_VISIBLE_DEVICES=1 python tools/train_kd_unified.py adaint/configs/generated/ppr10ka_global_affine_ban_01.yaml --gpus 1 --seed 42
```

`--gen 4` 自动启用教师，顺序产生 `gen_0` 到 `gen_3`。教师使用上一代验证最佳模型，MSE 权重从 0.1 衰减到 0。它与单代训练使用同一个模型类，没有独立的 KD 架构。

**恢复本次新架构训练**

```bash
CUDA_VISIBLE_DEVICES=1 python tools/train_kd_unified.py adaint/configs/generated/ppr10ka_global_affine_01.yaml --gpus 1 --seed 42 --start-gen 0 --resume-from work_dirs/ppr10ka_global_affine_01/gen_0/latest.pth
```

恢复 BAN 某一代时使用该实验配置、正确的 `--start-gen N` 和 `gen_N/latest.pth`。恢复后计数器与 runner 对齐，AdaInt 冻结及教师衰减进度继续。旧 V3/GlobalLocal/main 架构 checkpoint 不可用于本架构恢复。

**验证与导出**

```bash
CUDA_VISIBLE_DEVICES=1 python tools/test.py work_dirs/ppr10ka_global_affine_01/gen_0/inference_config.py work_dirs/ppr10ka_global_affine_01/gen_0/best.pth --eval-val
CUDA_VISIBLE_DEVICES=1 python tools/test.py work_dirs/ppr10ka_global_affine_01/gen_0/inference_config.py work_dirs/ppr10ka_global_affine_01/gen_0/best.pth --eval-val --save-path work_dirs/ppr10ka_global_affine_01/val_images
```

`--eval-val` 恢复保存的验证指标：PSNR、deltaE、PSNR_hc、deltaE_hc。不同原生尺寸图像按形状分组，保持每张图指标及输出顺序。单图增强可运行 `python adaint/demo.py <inference_config.py> <best.pth> <input.png> <output.png>`。

**日志含义**

`lut_weight_abs_mean` 观察实际线性系数幅度；`lut_weight_batch_std` 观察同一 batch 的图像条件变化（batch=1 时自然为 0）；`local_coefficient_spatial_std` 观察四系数空间变化；`local_correction_abs_mean` 观察局部修正对输出的实际影响；`global_change`、`output_change`、`target_change` 分别观察全局、最终、目标相对输入的变化。旧 `guide_spatial_std` 不再适用。仅凭系数方差非零不能证明提升 PSNR。
