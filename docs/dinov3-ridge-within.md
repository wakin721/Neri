# Neri1.1（DINOv3＋Ridge＋Within）

本实现将 2026-11-07 的联合实验迁入 `system/dinov3`，软件模型选择器通过版本化 `.neri.json` 加载。冻结 DINOv3 ViT-B/16 的 CLS 特征；Ridge 做分类，Within 提供拒识证据，两者支持旧物种补图和新增物种。

## 在软件中使用

先导出模型，把 manifest 和其相邻的 assets 文件夹一起保存在 `res/model/cls/user`，刷新分类模型列表，选择 **Neri1.1**，同时选择探测模型。照片检测完成后，软件统一重算完整序列，更新结果和本地结果数据库；处理中显示的是临时单图结果。缺少拍摄时间的照片按独立事件处理，不引入历史。

## 组件安装与云端分发

环境维护中的 **Neri1.1（DINOv3 ViT-B/16）** 提供安装、修复和删除。安装会从模型分发服务的 `/Neri_Data/Model/DINOv3` 递归下载组件，校验 `*.neri.json` 契约、checkpoint 与编码器 SHA-256、官方 `encoder-code` 目录，然后原子替换 `res/model/DINOv3`。已下载组件会在分类模型列表中以 NeriCloud 来源出现；删除组件不会删除共享 PyTorch 环境或人工校验记录。

云端目录应放置导出的同代文件：`dinov3-*.neri.json` 以及它引用的 `*-assets-*/` 目录（`model.neri-model`、`encoder.pth`、`encoder-code/`）。模型服务公开 `GET /api/models/v1/dinov3/manifest` 与 `POST /api/models/v1/dinov3/direct`，桌面下载器使用同一套字节数与哈希校验，失败时保留原有可用组件。

Neri1.1 是软件显示的模型名称。部署文件中的 `backend=dinov3`、编码器架构、既有文件路径和校验值继续保留，便于加载已有权重和训练记录。更名不改变模型参数或门限。

完整序列推理需要全部照片到齐。照片按目录相机和 30 分钟事件分组，同相机过去 7 天、最多 64 个其他事件的代表特征参与视觉历史适配（温度 0.1，残差权重 0.1）。严格排除未来、同事件和重复内容。不支持视频；含视频时选择跳过视频。

在 30 分钟相机事件内部，系统进一步按**同一目标轨迹（`track_id`）**分组：同图多目标框严格禁止落入同一轨迹，跨帧目标通过空间重叠／中心位移与特征余弦相似度一对一关联。

分类使用层级平衡 Ridge 和按目标轨迹分组、内容去重的自适应序列融合。Within 使用冻结白化、事件均值、相机内最大值、相机间 Top-3 与类中心各占 50%。拒识采用**两阶段门控**：
1. **单框独立门控（`box_known_score >= box_threshold`）**：每框先根据自身的 `2*top1-top2` 与白化范数（以 C 中已知类单框 median/MAD 标准化）满足独立接纳先决条件；未通过单框门控的背景框被隔离，既不能借用同图／同事件动物的序列分数被接纳，也不会拉低真实目标的序列证据。
2. **同目标轨迹序列门控（`sequence_known_score >= sequence_threshold`）**：通过单框先决条件的目标在同一轨迹内计算 Top-75% 的 `2*top1-top2` 与序列均值径向证据；两阶段联合门限在 C 辅助未知类上满足物种等权 90% 拒识目标（Macro90）。

输出将**每框识别证据**（`raw_det_conf`、`box_known_score`、`box_threshold`、`box_accepted`、`within_score`、`raw_ridge_score`、`ridge_score`）与**序列门控分数**（`sequence_known_score`、`sequence_threshold`、`sequence_accepted`、`track_id`、`event_id`）分开保存并在检测框详情中分别显示。界面 `conf` 使用每框自身综合门控差值的 sigmoid，范围为 0–1，用于展示，不是分类概率。拒识框保留为 Unknown，候选项保留 Ridge 分类和 Within 证据。

预览和校验中的 Neri1.1 置信度滑块已改为只读的“DINO 拒识判定分数”：显示各框本次识别记录的单框／序列接纳分数及门限，分数越低越容易被拒识，两级均通过才接纳。设置中的当前模型学习状态显示当前校准门限；自动学习重新校准后会更新这些值。探测置信度与 IOU 仍控制探测阶段。已经产生的 DINO 框在显示和导出时直接遵循最终接纳／拒识结果，不再被旧的 0–1 分类置信度筛选隐藏，拒识框也不会因此被归为空照片；人工确认结果优先。重新检测使用探测设置，不会把拒识门限用作探测置信度。

## 从已有实验导出

在项目根目录运行：

```powershell
python scripts/export_dinov3_experiment.py --source F:/files/python/Neri_plus --manifest res/model/cls/user/dinov3-ridge-within.neri.json
```

读取初始 Bank、原有冻结几何、已校验的编码特征，以及 C 的冻结历史。不会用 E 选择支持图、标准化参数或门限。导出的是初始支持库；实验中的补图预算和种子不依据 E 自动挑选。权重与官方模型源码、许可证复制到部署目录，部署后不依赖实验项目路径或联网下载。

## 自己训练和补图

输入为不含 pickle 的 NPZ：`features[N,768]`、`labels[N]`、`cameras[N]`、`sequences[N]` 和标量 `encoder_sha256`。特征必须由相同冻结编码器提取，预处理为 224×224 bicubic contain、RGB(124,116,104) padding、ImageNet normalization，输出归一化 CLS，排除 register tokens。C 可另含 `contents`、`sites`、`adapted_features`；提供 `datetimes[N]` 时命令自动重建 C 内的因果历史，已有 `adapted_features` 时优先使用它。省略两者表示 C 没有可用历史，应确认这符合实际部署数据。Bank 可带 `origin` 和 `transform` 复用已冻结几何。`cameras` 用于门控/径向序列分组，`sites` 用于分类序列分组，在历史实验中分别对应 location 和 camera_component。

```powershell
python scripts/train_dinov3_ridge_within.py fit --bank bank.npz --calibration calibration.npz --encoder-weights encoder.pth --encoder-code dinov3-code --manifest res/model/cls/user/my-dinov3.neri.json
python scripts/train_dinov3_ridge_within.py update --supports trusted-supplements.npz --calibration calibration.npz --manifest res/model/cls/user/my-dinov3.neri.json
```

补图必须来自人工确认的训练支持，不应把评估集或自动预测写入训练库。`update` 同时更新 Ridge 缓存和 Within 事件统计，保持原编码器、中心与白化变换不变。每次更新清空旧校准，然后用完整 C 重新计算标准化及门限。新增物种后，C 中已成为已知类的样本不能继续作为辅助未知类。发布采用新 checkpoint 加原子 manifest 替换；旧资源代次保留，方便恢复。

## Python 接口

```python
from system.dinov3.model import DinoV3RidgeWithinModel

model = DinoV3RidgeWithinModel().fit(features, labels, cameras, sequences)
model.calibrate(c_features, c_labels, c_cameras, c_sequences)
result = model.predict(query_features, q_cameras, q_sequences, contents=q_hashes)
model.partial_fit(extra_features, extra_labels, extra_cameras, extra_sequences)
# 更新后必须重新 calibrate，才能得到接纳/拒识判定。
model.save("joint.neri-model")
restored = DinoV3RidgeWithinModel.load("joint.neri-model")
```

`scores()` 返回 Ridge 原始分数，`predict()` 返回分类、接纳标记、门控分数、门限及两条分数矩阵。`adapted_features` 接收历史适配后的原始特征，由模型执行固定白化；省略时使用当前原始查询。模型文件保存原始支持、事件统计、冻结几何、校准和来源信息，加载后可以继续补图。

旧模型的 registry/feedback 数据不能直接用于 DINOv3。除上述命令和模型 API 外，校验界面现在提供人工补图入口：

1. 在检测框详情中确认物种，或对整张照片填写人工物种和数量，软件保存校验后自动在后台学习。整图只有一个明确物种、且有效框数不少于 1 并不超过人工数量时，符合条件的裁图一同加入学习；人工数量与裁图数分别保存。例如人工标 4 只、检测出 2 框时，统计仍为 4 只，学习使用 2 张裁图。
2. 在设置的“检测设置 → 当前模型学习状态”查看等待、学习中、已完成或失败状态，以及已学习样本、物种和最近更新时间。软件合并连续确认，使用原编码器提取裁图，更新 Ridge 与 Within，并用完整 C 重新校准、原子发布；无需额外点击更新按钮，后台计算期间可继续校验。
3. 新模型在下一次识别时加载；已有识别结果需重新识别。更新期间已加载的任务继续使用其原模型。

新导出和训练的模型会携带完整校准输入及 SHA-256。旧模型缺少这些数据时，需先通过“导入校准数据”选择同一冻结编码器生成的 NPZ，字段与上述训练输入一致；必须包含 `encoder_sha256`，以及原有的 `adapted_features` 或 `datetimes`（若校准使用历史上下文）。不能用预测结果或误拒对照池代替完整 C。校准集中的新已知物种会自动改为已知角色；若已无辅助未知类，更新将失败并保留原模型。

学习在本机执行；模型改进计划的上传行为仍由原设置控制。重复确认同一框不会叠加，相同裁图只计一次，标签冲突时阻止发布。源文件改变、编码器/校准哈希不符或学习失败时保留原模型和人工校验，在设置中显示原因并可重试。新学习记录保存冻结特征，改标签或撤销后会自动从原始模型重放当前有效记录，替换或移除旧学习结果。旧版已学习但未保存特征的记录需要重新导出模型后才能改动。后端重启时恢复未完成的学习。

照片框标签使用简短显示编号、物种和置信度；长事件／轨迹 ID 仅作内部身份，标签限制为一行并保持在照片范围内。

预览和校验页的显示开关与 DINO 分数栏保持单行；多框或窄窗口可横向拖动查看。说明与完整门限比较通过悬停提示查看，不再占用图片区域的高度。

序列待汇总时，摘要置信度旁显示信息图标，悬停或长按查看临时结果说明；汇总完成或人工确认后该图标隐藏，不再单独增加一行文字。

### 整图标注与批量学习

单物种整图标注会更新框上的物种名称。框数超过人工数量、数量无效或裁图无效时，框只带“整图推定”标签，不获得训练资格；多物种时不猜测各框归属、不做整图学习，可通过逐框确认提供训练支持。数量为 0 或标为空时，撤回对应学习记录；没有检测框时仅保存人工图级标注。批量操作对每张照片分别判定，不能把整组连拍的数量相加。

人工图级物种和数量保存在 `image_annotation`，后续逐框操作不会用框数覆盖它。框上的 `label_source` 区分整图传播和逐框确认，`learning_eligible` 独立记录学习资格。用户明确排除的误检框与“不参与学习”的框不因整图标注重新加入。改变数量或改为多物种后，会撤回不再合格的既有学习支持；撤回整图标注时恢复原有框标签。数量条件只是快捷归属规则，仍应逐框排除背景等误检，不能由数量关系证明每个框都正确。

## 本机接入验证

2026-11-07：相关 Python 回归共 119 项通过；四个前端变更文件格式检查通过，Dart 分析无错误或警告，仅有原有的五条代码风格提示。Windows Release 已成功编译；模型选择和视频设置的 Flutter widget 回归共 21 项通过。

使用已审核实验的 C 集进行运行时对照：13,165 条查询的 Ridge 分类、最终接纳/拒识均完全一致，门控最大绝对误差 `1.55468007800863e-6`，门限差 `3.7102336492012e-7`。相同 C 查询的历史索引逐项一致。打包后的冻结编码器重放 32 张 Bank 图片，特征最大绝对误差为 0。校验没有使用 E 拟合或选择参数。

本机模型包含 51 类、16,454 张初始支持和 342 张辅助未知类校准图片。生成的模型资产不纳入 Git，可用上述命令复现；原有实验目录保持独立。

## 使用当前已校验照片重训

`train_reviewed_dinov3.py` 读取当前 `data/by_species/00已校验`，用目录名作为人工确认的物种标签；旧目录清单里的自动标签不会覆盖当前标签。从原始连拍目录恢复相机、时间与事件身份，并冻结文件大小、修改时间。内容 SHA-256 与特征在一次读盘中同时生成。

```powershell
python scripts/train_reviewed_dinov3.py --source F:/files/python/Neri_plus --out runs/dinov3_reviewed_20261107 --stage prepare
python scripts/train_reviewed_dinov3.py --source F:/files/python/Neri_plus --out runs/dinov3_reviewed_20261107 --stage extract
python scripts/train_reviewed_dinov3.py --source F:/files/python/Neri_plus --out runs/dinov3_reviewed_20261107 --stage fit
```

当前用户选择的规则：原始已校验图片数不少于 10 的 45 类进入分类库；少于 10 的 42 类用于辅助未知类校准。完全相同的裁图去重，标签冲突的相同裁图隔离。相机优先划分 Bank/C/E，再保证各已知类的事件覆盖；所有相机共享例外列在 `partition.json`。连拍、原始图及相同内容不跨集合。模型在读取 E 预测结果前保存并冻结；E 只有已知类，未知拒识率属于 C 校准诊断。

输出 `training_report.json`、`calibration_replay.npz` 和 `deployment/dinov3-reviewed.neri.json`。完整提取可以从检查点续跑；恢复时先重放 32 张已缓存图片。编码器保持 float32，不进行梯度训练。

### 已知物种的误拒对照池

训练时默认从已预留的 C 集中，每个已知物种抽取最多 **50 个独立连拍、每个连拍一张**，作为**应被接纳**的对照。固定种子、优先覆盖不同相机，每个连拍／内容／原始图在同一物种内最多取一张；不足 50 个独立事件时全取，并在报告中列出数量。这些图片与 Bank、E 隔离，仍保留原来的已知物种标签。名单在模型拟合前保存，模型来源信息记录名单哈希。

对照池用于按图片、按物种检查误拒。它使用完整校准序列的上下文和已冻结分数，保留现有的未知类门限选择规则。报告同时列出完整 C 集中所有已知类照片的误拒率，作为全量参考。输出位于 `rejection_validation`：`selection.json` 是固定名单，`pool.jsonl` 和 `samples.csv` 包含照片路径、应接纳标记、预测与误拒结果，`per_species.csv` 汇总抽样及全量数据的物种数量和误拒，`report.json` 汇总误拒率，`known_acceptance_controls.npz` 保存抽样特征和完整序列下的原分数。

```powershell
# 新训练默认每类最多 50 个独立连拍；可以显式指定数量。
python scripts/train_reviewed_dinov3.py --source F:/files/python/Neri_plus --out runs/new_training --stage all --known-probes-per-species 50
# 已有训练可直接生成对照报告，使用已冻结的 C 集预测。
python scripts/train_reviewed_dinov3.py --source F:/files/python/Neri_plus --out runs/dinov3_reviewed_20261107 --stage check-rejection --validation-out runs/dinov3_reviewed_20261107/rejection_validation_50
```

调整抽样数量或种子时，用 `--validation-out` 指定另一个输出目录，保留已有固定名单。这是校准集诊断，不代表独立测试结果。
