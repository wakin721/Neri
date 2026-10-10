# 2026-10-09 本地实验与修复记录

本页整理已有运行记录。模型、特征、数据库、原始预测、安装暂存和回退副本保存在本机 `runs/`；该目录默认忽略。已受 Git 跟踪的历史训练文件继续跟踪。

## Seq Memory 与 Within-Seq 对照

两种实现使用同一冻结 DINOv2 编码器及配对建库、校准、测试样本，分别在辅助未知集校准 Val90。下表是三个预定划分的等权均值，只比较单帧条件；原始汇总还包含序列平均控制。

| 评估池 | Seq 已知误拒 | Within 已知误拒 | Seq 未知召回 | Within 未知召回 |
|---|---:|---:|---:|---:|
| 本地已校验留出 | 15.77% | 33.93% | 77.23% | 86.59% |
| CCT20 同域 | 31.19% | 37.22% | 90.07% | 91.70% |
| CCT20 跨相机 | 45.97% | 57.38% | 91.39% | 94.06% |
| iWildCam | 53.53% | 45.50% | 89.54% | 88.53% |

结果随数据域变化。本地和 CCT20 中，Within-Seq 提高未知召回，同时增加已知误拒；iWildCam 中，Within-Seq 的已知误拒较低。两种实现的分类 k 不同，闭集分类差异不能单独归因于拒识算法。Val90 是辅助未知校准目标，不保证测试未知召回达到 90%。

本地物种轮换使用 30 类、三个相机划分，每个划分做 30 次角色轮换：20 Known、5 辅助 Unknown、5 新 Unknown。90 轮中 Seq Memory 已知微平均误拒较低 57 轮，Within-Seq 较低 33 轮；Within 减 Seq 的平均误拒差为 +3.68 个百分点。Memory k3 控制与 Within 共享分类数组，仅比较拒识分支。各轮共享图像和物种，不能当作 90 个独立样本；测试 ROC 的同等召回指标仅供回顾诊断，不能用于选择部署阈值。

归档汇总见 [多数据集汇总](seq-vs-within-summary.csv)、[物种轮换汇总](unknown-rotation-summary.csv)、[阈值敏感性汇总](unknown-rotation-sensitivity.csv)。原重算记录见 [多数据集验证](seq-vs-within-verification.json) 和 [物种轮换验证](unknown-rotation-verification.json)。这些是历史记录，迁移后的脚本检查结果应另存为新报告。

## 分类头校准和 Flutter 修补

42 Known / 43 Unknown 的历史划分以原始已校验裁剪数为依据：少于 10 张的物种全部保留为 Unknown 校准样本。Seq Memory 和 Within-Seq 都拒识 111/123 张 Unknown（90.24%），对应已知误拒分别为 23.45% 和 40.22%。这是内部校准，包含赤麻鸭单相机例外，不是独立泛化测试。此处记录两个历史配置，不表示当前默认模型必须采用其中一个。

Flutter 最大化崩溃修补针对指定引擎版本的空父节点访问；正式补丁及版本哈希检查已在 `scripts/flutter_accessibility_hotfix.py` 中跟踪。原验证覆盖 4,098 组模拟和 4,098 组本机执行。验证工具读取原版与修补版 DLL，不安装或改写 DLL；升级引擎后须重新判断补丁适用性。

`runs/within_seq_merge_check/workspace-before-nightly/` 保存切换分支前的训练文件，其中旧模型、manifest 和审计记录不同于当前受跟踪版本。该副本属于回退档案，应保留，不能按重复文件删除。历史合并检查 README 中的失败状态早于后续 DEPLOYMENT 记录，不能据此判断当前分支是否仍未合并。

## 工具与数据依赖

迁移后的工具保留历史算法、固定种子、样本数量检查与数值边界。它们提供可配置路径，但仍依赖对应历史数据；不是接受任意数据集的通用训练接口。训练、校准和 benchmark 还需要单独的 `Neri_plus` checkout 提供 `pipeline`、`training` 模块及冻结编码器。运行前使用对应环境安装项目数值依赖；修补验证另需 capstone、pefile、unicorn 及 Windows x64。

| 工具 | 用途 |
|---|---|
| [train_reviewed_seq_memory.py](../../../scripts/train_reviewed_seq_memory.py) | 建立与验证已校验序列 Memory；输出路径不再多嵌套一层 |
| [calibrate_seq_memory.py](../../../scripts/calibrate_seq_memory.py) | 历史低样本物种 Val90 校准 |
| [verify_seq_memory_calibration.py](../../../scripts/verify_seq_memory_calibration.py) | 通过指定软件运行时重放校准 |
| [benchmark_seq_vs_within.py](../../../scripts/benchmark_seq_vs_within.py) | 本地、iWildCam、CCT20 的配对评测 |
| [benchmark_unknown_rotations.py](../../../scripts/benchmark_unknown_rotations.py) | 本地 90 轮物种角色轮换；复用同目录的配对评测模块 |
| [verify_seq_vs_within_results.py](../../../scripts/verify_seq_vs_within_results.py) | 从保存的预测独立重算多数据集指标 |
| [verify_unknown_rotation_results.py](../../../scripts/verify_unknown_rotation_results.py) | 重算轮换、隔离、阈值及指标 |
| [analyze_unknown_rotation_sensitivity.py](../../../scripts/analyze_unknown_rotation_sensitivity.py) | 从既有阈值扫描生成独立 CSV、JSON、HTML 汇总 |
| [validate_flutter_accessibility_hotfix.py](../../../scripts/validate_flutter_accessibility_hotfix.py) | 版本限定修补的模拟和 Windows 本机验证 |

在项目根目录运行，各工具的 `--help` 列出所需参数。创建实验的工具要求新输出目录；验证工具默认只打印结果，传入 `--report-file` 时拒绝覆盖已有文件。敏感性工具输出到新目录，保留原报告和协议。

```powershell
# 只读重算既有轮换结果；数据移动后可用 --metadata-file 指定原 snapshot。
python -B scripts/verify_unknown_rotation_results.py --run-dir runs/local_unknown_rotation_20261009

# 从既有阈值扫描产生新的敏感性汇总，不重跑模型。
python -B scripts/analyze_unknown_rotation_sensitivity.py --run-dir runs/local_unknown_rotation_20261009 --output-dir runs/rotation_sensitivity_replay

# 训练路径示例：用实际 Neri_plus 路径替换变量，输出必须是新目录。
$sourceRoot = 'PATH_TO_NERI_PLUS'
python -B scripts/train_reviewed_seq_memory.py --source-root $sourceRoot --output-dir runs/reviewed_memory_replay
python -B scripts/calibrate_seq_memory.py --source-root $sourceRoot --prior-dir runs/seq_memory_reviewed_20261009 --output-dir runs/val90_replay
```

原始实验脚本仍保留在被忽略的运行目录中，供历史审计。安装、缓存修复及合并过程脚本不作为长期工具提交；模型和备份应按项目的模型分发或独立归档流程保存。

## 2026-10-10 迁移检查

九个工具的语法与 `--help` 入口通过；五个创建实验的入口均拒绝已有输出目录。多数据集验证重算 50 行结果和 398 项指标/校准检查；物种轮换验证重算 270 行结果、6,480 项检查及 1,080 行敏感性扫描。校准验证通过当前仓库运行时重放全部 4,404 个样本，最大分数差约 1.19e-7，仍拒识 111/123 张 Unknown。敏感性汇总与原 JSON 完全一致，原报告和协议保留。修补验证再次通过 4,098 组模拟、4,098 组本机执行及 DLL 加载检查。

本次没有重新提取图像特征、完整训练或运行全部 benchmark；上述检查覆盖迁移入口、保存结果重算、分类运行时与修补验证。Git 忽略规则生效，原受跟踪的七个历史训练文件继续受跟踪。
