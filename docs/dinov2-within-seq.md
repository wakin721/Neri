# Within-Seq 批量分类与聚类接口

新增 `POST /api/dinov2/discovery/within-seq`，按提供的开放世界生物多样性监测文稿重实现 Frozen Memory + Within-Seq + Val90 + HDBSCAN 路线。代码位于 `system/dinov2/within_seq.py`。

这是显式传入特征的批处理接口，尚未接入桌面 UI 的默认推理和物种注册流程。仓库中没有文稿对应的原始训练导出、初始特征库和辅助未知校准集，因此不能声称与论文实验模型数值一致，也不能直接替换现有 Memory checkpoint。

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
