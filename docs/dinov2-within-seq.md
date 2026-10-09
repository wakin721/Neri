# Within-Seq 批量分类与聚类接口

新增 `POST /api/dinov2/discovery/within-seq`，按提供的开放世界生物多样性监测文稿重实现 Frozen Memory + Within-Seq + Val90 + HDBSCAN 路线。代码位于 `system/dinov2/within_seq.py`。

该方法同时用于显式传入特征的批处理接口和 Nightly 默认推理头，部署范围见文末。仓库中没有文稿对应的原始训练导出、初始特征库和辅助未知校准集，因此不能声称与论文实验模型数值一致；旧 Memory checkpoint 也不能直接作为 Within-Seq checkpoint 使用。

## 安装及输入

```sh
python -m pip install -r requirements-within-seq.txt
```

向接口提交 JSON 对象，字段如下：

| 字段 | 内容 |
| --- | --- |
| `bank.features` | 初始 Known 库，N × 768 的原始 L2 归一化 DINOv2 ViT-B/14 CLS 特征 |
| `bank.labels` | N 个已知物种名称，至少两类 |
| `bank.camera_ids`、`bank.sequence_ids` | 与每行特征对应的相机和序列 ID |
| `auxiliary_unknown.features` | 独立的辅助未知类别图像特征，M × 768 |
| `auxiliary_unknown.camera_ids`、`auxiliary_unknown.sequence_ids` | 辅助未知数据的相机和序列 ID |
| `query.features` | 待处理混合批次特征，Q × 768，不提供真实类别 |
| `query.camera_ids`、`query.sequence_ids` | 待处理批次的相机和序列 ID |
| `target_unknown_recall` | 可选，默认 0.90，范围 (0, 1] |

相机 ID 应在整个数据集内稳定且唯一；序列以 `(camera_id, sequence_id)` 标识，三个数据池不得共享序列。每个数据池都必须非空。特征应来自一致的文稿预处理（letterbox 224）与编码器；接口只校验维度、有限性、范数和 ID，无法证明特征的实际来源或辅助集的类别独立性。

不能传入旧 `memory_head.npz` 的已中心化 `features`；旧文件也不包含重建此方法所需的全部序列元数据。准备好上述 JSON 后可调用：

```sh
curl --fail-with-body -X POST http://127.0.0.1:8000/api/dinov2/discovery/within-seq \
  -H 'Content-Type: application/json' --data-binary @within-seq-request.json
```

端口以实际后端配置为准。接口在每次请求中重新构建初始模型并校准，适合批量实验；尚无跨请求模型缓存或持久化特征上传。

## 算法约定

1. 分类分支固定 `k=3`，采用 Neri Memory 家族的类别均衡中心化、每相机最大相似度、前 k 相机均值与类别中心各占 0.5。文稿没有完整导出该分类头，因此这部分是明确的重实现假设。
2. 拒识分支使用类别 → 相机 → 序列 → 图像的层次等权重，估计类内协方差；以 0.1 向 `trace(cov)/768 * I` 收缩，通过精度矩阵 Cholesky 变换后归一化。中心与白化只由初始 Known 库估计。
3. 序列记忆先平均原始单位特征并归一化，再白化和归一化。自适应 k 由不同序列最近邻距离的最大相对间隔确定，范围 1–16，使用第 17 个邻居确定最后一个间隔；并列选择较小 k。距离下限为 `1e-6`。
4. 拒识分数为 `2 * s_top1 - s_top2`，再在查询序列内对图像分数取平均。分类结果仍来自独立的固定 k 分类分支。
5. Val90 对辅助未知图像对应的序列平均分数取第 `ceil(0.9 * M)` 个顺序统计量，并用 `nextafter(value, +inf)` 构造阈值。严格 `score < threshold` 为未知；并列可能使辅助未知召回超过目标。校准按图像计数，长序列保留其图像权重。
6. 仅对被拒识序列的**原始、未白化**特征均值归一化后聚类，使用外部 `hdbscan` 包的 `min_cluster_size=5`、`min_samples=3`、`allow_single_cluster=False`、Euclidean 距离和 EOM。不要求预先给定簇数。少于 5 个候选序列全部记为噪声。

查询相似度按 128 行分批计算，类别和相机索引在模型构建时预计算，避免逐图重复构建索引。HDBSCAN 的输入是序列均值，减少连拍图像对聚类规模的影响。

## 输出和当前范围

所有逐图输出均与 `query.features` 顺序一致：`known_species` 为分类头预测，`species` 将拒识项标为 `Unknown`；`selected_k` 为拒识分支选取的 k；`image_knownness`、`sequence_knownness` 为聚合前后分数。响应同时提供阈值和辅助未知集上的实际召回率。

`novel_cluster` 中 `null` 表示已接受图像，`-1` 表示拒识但被聚类判为噪声，非负整数表示本次请求内的候选簇编号。编号不能跨请求用于物种身份。`review_budget_images = ceil(0.03 * 拒识图像数)`，分母包含噪声；这里只计算预算，不执行标注选择或注册。

本次未实现文稿全部九种聚类对比、中心/边缘人工审核调度、M0→M2 增量记忆更新和新类注册闭环。启用后续增量更新时必须保留初始中心及白化，并在同一辅助未知池重新校准；至少两个不同序列的人工确认标签才可创建新类，不能把聚类簇标签自动传播为训练真值。

## 验证

```sh
python -m pip install pytest httpx
OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_dinov2_within_seq.py
```

测试覆盖层次权重、白化协方差、固定 k 分类、k 的边界和并列、序列聚合、Val90 阈值、数据池泄漏检查、小样本噪声处理、真实 HDBSCAN 分离和 HTTP 接口。测试不替代在原始实验数据上的精度与吞吐评估。


## Nightly 默认推理头

Nightly 的 `res/dinov2/memory_head.npz` 是 `memory_within_seq`，安装 DINOv2 组件时默认使用它。按序列筛选之前的已校验裁剪图片数划分类别：少于 10 张的物种全部作为 Unknown，恰好 10 张仍为 Known。本次 42 类 Known 使用 3,311 个序列代表特征建库；43 类 Unknown 的全部 123 张图片退出示例库、分类中心、质心和白化估计，仅用于阈值校准。不能按序列去重后的库大小判定物种角色；例如红翅旋壁雀有 12 张已校验图片，尽管建库只有一个序列，仍属于 Known。

编码器冻结，不做梯度训练。分类固定 k=3、类别中心权重 0.5；拒识使用从新 Known 库重建并冻结的中心和白化、自适应 k，以及 Val90。Val90 表示使辅助未知校准图像的微平均拒识召回至少达到 90%，不是把分数阈值设为 0.90。校准遵循桌面默认的独立单帧查询方式，123 张 Unknown 分数取第 `ceil(0.9*123)=111` 阶边界，采用边界之上的 float32 可表示值。最终阈值为 **0.592520534992218**，严格低于阈值时拒识。已保存 checkpoint 的真实运行时重放全部 4,404 个校准样本，并改变分块大小核对决策。

Unknown 校准拒识 111/123 张（90.24%）；4,281 张 Known 校准图像的误拒率 **40.22%**，闭集准确率 92.48%。将 Unknown 按相机和原始采集序列聚合、保持阈值不变时，诊断召回为 91.06%。本次使用稀疏物种内部校准，替代旧 85 类头的 CCT20 辅助校准；两次数据划分不同，不能直接比较旧 19.50% 和新 40.22% 误拒率。用户选择优先拒识未知，保留 Val90，无 Known 误拒率约束。

校准与建库按裁剪 ID、原图、内容哈希、burst 和 acquisition 隔离。赤麻鸭只有一个相机，为保留该 Known 类允许与部分 Unknown 共用相机，但原图和采集组仍隔离；其他 Known 类不使用 Unknown 所在相机。Known 校准覆盖 40/42 类，红嘴鸥与赤麻鸭缺少校准样本。低样本物种曾属于旧模型；此次全部变换从 Known 专用库重建。以上是内部校准结果，没有独立测试集，不能保证现场未知物种召回达到 90%。完整角色计数和模型哈希见 `res/dinov2/memory_head_manifest.json`。

可从本机冻结的 `<10` 划分和同一编码器原始特征复现（输出到新目录）：

```sh
python scripts/export_within_seq.py --source-dir runs/seq_memory_val90_lt10_20261009 --output-dir runs/within_seq_val90_lt10_replay/deployment
```

导出器校验类别角色、特征行 ID 和校准隔离，并重建中心、质心和白化。历史 `train_seq_memory.ps1` 和 `verify_seq_memory.py` 是旧 Memory/已知误拒率校准流程，不用于本次 Within-Seq Val90 导出。

常规单张图像或缺少可信序列 ID 的检测裁剪按独立单帧序列评分，避免把不同动物或任意探测器批次混合。Python 分类入口可显式提供 `camera_ids` 和 `sequence_ids`，会先逐图评分，再跨整个请求聚合；分块大小不会改变结果。`classify_event` 对实际事件内每张裁剪的拒识分数平均，分类采用事件平均特征。桌面尚未在跨图片任务中自动恢复连拍序列，因此不会声称默认跨文件批处理已实现完整连拍聚合。

人工确认的 Memory 证据可参与分类和白化空间匹配，但不重新拟合初始中心及白化；临时类别仍不能自动转为正式类别。阈值保持初始 Val90，不在未知生产图像上自动调整，动态扩展后的召回需另行评估。

新默认头使用新的模型指纹，历史反馈和注册数据保留在旧指纹目录中，不自动将未审核记录迁移到新模型。更新软件不会覆盖用户自行选择的旧模型；已有组件可通过重新安装 DINOv2 组件启用新的默认头。
