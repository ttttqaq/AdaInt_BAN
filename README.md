**AdaInt-BAN：单架构 tone/LUT 与局部修正分支**

本分支 `feature/global-affine__from_lut-refinement` 只保留 `AiLUT` 一个增强模型：整图条件预测 tone、AdaInt 与完整 33³ 全局 LUT；原始 RGB 坐标查询后添加直接学习的局部仿射修正。BAN 复用相同架构的冻结教师。

- [完整架构、改前对比与训练目标](adaint/ARCHITECTURE.md)
- [PPR10K 训练、恢复和验证命令](adaint/PPR10K.md)
- 数据模板：`adaint/configs/ppr10k.yaml`、`adaint/configs/face_indoor.yaml`
- 实验配置生成：`python tools/prepare_lut_experiment.py --dataset ppr10k --name <new_experiment_name>`
- 测试：`CUDA_VISIBLE_DEVICES=<free_gpu> python -m pytest -o addopts= tests/test_adaint -q`

当前 A100 已有 `AdaInt_BAN` conda 环境、MMCV/MMEditing 和 AiLUT CUDA 扩展。本次修改不安装或升级依赖。新的模型与旧架构 checkpoint 不兼容，请从头训练；已有实验日志、数据、预训练文件与 Git 历史保留。

基础项目：[AdaInt-BAN](https://github.com/zrk2/AdaInt-BAN)、[AdaInt](https://github.com/ImCharlesY/AdaInt)。本分支的结构改动和原论文实现存在差别，尚未完成全量 PPR10K 训练验证，不能视为复现论文指标。
