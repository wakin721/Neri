# 本地实验模型接入软件

2026-11-14。新增可选分类模型 **Neri1.2preview**（原显示名“Neri 实验模型 B/16（本地数据）”）。它使用 11 月 14 日相机目录隔离划分的 44,472 张训练支持、45 个已知物种，以及完整的 C 校准数据；没有把 E 加入训练。此次更名保留原模型清单路径、检查点、校准集与配方，便于沿用已有设置和人工反馈。

在软件的分类模型列表刷新后选择该名称，继续选择探测模型并使用照片识别。原有 Neri1.1 双门控模型可以继续选择。安装本接入更新不会更改默认模型选择；已有识别结果需要重新识别才会使用实验模型。

实验配方为 0.55×逐行标准化的基数校正 Within＋0.45×逐行标准化的层级 Ridge，再做 tau=0.5 自适应整事件融合。拒识独立采用事件 Top-75% margin 与径向证据融合，并在 C Unknown 上做物种等权 Macro90 校准。软件不会额外套用单框独立门控或把事件自动改为目标轨迹聚合。

模型清单和 checkpoint 明确记录 `inference_recipe=experiment_full`、`head_type=ridge_within_experiment`；旧软件会拒绝这个新 head type，避免把实验模型误当双门控运行。旧的双门控模型仍默认使用 `production_dual`，不要求重导出。

预览和校验页显示“实验事件”分数，设置页显示实验事件门限。实验模型没有独立单框门限；融合分类分数和原始 Ridge 分数保持区分。人工确认后的自动补图、撤销／改标签、重新校准和发布继续保留实验配方。完整校准包同时保留裁图去重身份与原图身份，原图属于 C 时不能作为新的训练补图。

检测框上的“分类证据”由该框在序列融合前，对最终候选物种的融合分类分数减去最强竞争物种分数，再经 sigmoid 映射到 0–1。它是每框的相对支持，不是校准后的正确概率；当事件最终物种与当前框自身证据不一致时，可以低于 0.5。事件接纳分数继续单独决定拒识，不会因显示分数低而再次拒绝。相同证据的框仍可能得到相同数值。

旧记录没有保存这项每框证据时，原数值标为“事件接纳”；重新识别后才会生成新的分类证据。无需重训或更换模型。显示修复可通过 `scripts/deploy_dinov3_confidence_fix.ps1` 仅更新相关后端和已编译界面，保留现有模型、校准包和设置。

## 真实数据验证

软件分类接口与冻结实验在 C 的 14,998 条和 E 的 14,477 条查询上，分类、接纳决定完全一致，门控最大误差均为 0。导出核验记录位于：

`E:/Files/Neri/runs/local_protocol_comparison_20261114/experiment_deployment/export_report.json`

本地 E 上实验配方的已知正确接纳为 58.82%、已知误拒为 38.80%、独立未知宏拒识为 85.71%。接入保持实验行为，不声称拒识质量已达到 90%。这次接入限 DINOv3-B/16，不将 L/16 的历史实验成绩当成本地部署成绩。

## 复现与部署

从已经冻结、核验的本地比较导出：

```powershell
& 'D:/Application/Neri/toolkit/python.exe' -X utf8 scripts/export_local_dinov3_experiment.py
```

输出为 `res/model/cls/user/dinov3-experiment-local.neri.json` 和相邻资产目录，包含 checkpoint、离线编码器、官方源码及完整校准包。部署后推理和自动学习不依赖实验项目目录。

通用 NPZ 训练入口也支持：

```powershell
python scripts/train_dinov3_ridge_within.py fit --bank bank.npz --calibration calibration.npz --encoder-weights encoder.pth --encoder-code dinov3-code --manifest res/model/cls/user/my-experiment.neri.json --inference-recipe experiment_full --display-name "我的实验模型"
```

使用现有可回退安装脚本部署新增模型、后端和编译好的界面：

```powershell
./scripts/update_dinov3_installation.ps1 -InstallationRoot D:/Application/Neri -ModelManifest res/model/cls/user/dinov3-experiment-local.neri.json
```

不传 `-SelectModel` 时保留原设置。脚本核验、备份并复制完整校准包，避免出现“可以推理但不能自动学习”的安装。

安装后可用 `scripts/verify_dinov3_experiment_installation.py` 对安装目录的真实后端重放 C/E，检查模型列表和自动学习校准可用性，并用实际编码器重放 32 张照片。核验结果保存在本轮 `experiment_deployment` 目录。
