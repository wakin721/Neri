# DINOv2 HDBSCAN 新类发现（5%）

`POST /api/dinov2/discovery/hdbscan` 提供批量新类发现。HDBSCAN 是软件唯一的聚类算法，使用 `scikit-learn>=1.3,<2`，不需要预设簇数或随机种子。旧的加权 KMeans 接口已移除。

| 字段 | 含义 |
|---|---|
| `classification_model_path` | 已安装的 Memory 分类头对应模型路径 |
| `calibration_features` | 独立已知校准事件的冻结 DINOv2 单位向量，形状 `(N, 768)` |
| `query_features` | 待发现混合批次的单位向量，形状 `(M, 768)` |
| `calibration_sequence_ids`, `query_sequence_ids` | 与上述每行对应的全局唯一连拍序列 ID，不同相机的序列也必须区分 |
| `calibration_camera_ids`, `query_camera_ids` | 可选，需同时提供；提供时密度计算排除同相机邻居 |
| `min_cluster_size` | 最小独立事件数，默认 4，至少 2 |
| `min_samples` | 密度估计的近邻数（包含自身），默认与 `min_cluster_size` 相同，至少 1 |

`known_species[i]` 是第 `i` 行 Memory 最佳已知类；`novel_cluster[i] == -1` 表示保留已知分类，`-2` 表示被拒识但尚未形成密度簇的噪声，非负整数表示新类簇。`rejected_mask` 明确给出拒识结果，`rejected_count` 包含噪声图像，`noise_count` 为噪声图像数，`occupied_clusters` 只统计非负簇 ID。响应还包含 `algorithm: "hdbscan"`、有效密度参数、阈值和校准集实测误拒率。输入数组必须严格按行对齐。旧参数 `n_clusters`、`seed` 会返回校验错误。

流程使用当前 Memory 头的冻结类别分数（标准头包含类别质心混合），不修改已安装模型。局部密度取待发现批次里排除同序列（有相机元数据时也排除同相机）后的第 5 个近邻；密度比和 Memory 分数分别按已知校准集的中位数及 MAD 标准化，再融合为新颖度。阈值只从已知校准分数按 5% 保守秩选取，等于阈值时保留已知分类。

拒识池按连拍序列合并为平均单位特征，每个序列贡献一个独立事件，在 CL2N 特征空间执行欧氏距离 HDBSCAN（EOM，允许单簇）。序列标签映射回每张图像；事件数不足或密度不足时返回噪声，不强制形成新类簇，也不退回其他算法。

注册库和人工反馈的原型生成共用 HDBSCAN，使用原有特征空间和学习阶段原型数量上限。原型为受支持簇的算术均值；超出上限时优先保留事件数最多的簇，有受支持簇时排除噪声。原型生成使用 0.2 的簇合并距离（单位向量下对应余弦相似度 0.98）。少量已标注证据或全部为噪声时，只保留一个物种均值摘要供现有学习质量门限评估，不将其作为发现的新类簇。已有模型原型和反馈代次不会在此次代码更新中被重写，后续原型更新才使用 HDBSCAN；在线匹配、独立事件分组和分类门限保持现有规则。

调用方必须提供独立的已知校准样本和完整待发现批次。现有默认 Memory 模型的部署阈值是 4% 校准结果，不能直接用作本方法的 5% 门限。若未提供相机 ID，接口仅排除同序列近邻，不推断相机信息。

算法和参数定义参见 [scikit-learn HDBSCAN 官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html)。
