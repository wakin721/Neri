# 内置 Neri1.2preview LiteRT FP32 CPU

> [!NOTE]
> **2026-11-16 变更：** `Neri1.2preview·LiteRT` 权重与分片不再上传 GitHub，也不再内置于 Windows 轻量发行包，改为通过 NeriCloud（`runtime=cpu`）按需下载安装。发行包与 CI 仅内置并校验 MegaDetector V6 `MDV6-yolov10-c.tflite` 探测器。`scripts/prepare_bundled_classifier.py` 保留为显式本地开发工具，仅在本地保留分片时用于手动组装调试。以下为 2026-11-15 历史记录。

2026-11-15。Windows 轻量包新增内置 45 类照片分类器，显示名为 `Neri1.2preview·LiteRT`，与内置 MegaDetector 配合运行，不需要 PyTorch。新安装默认选择这两个内置模型；用户明确选择“不使用”时保留该设置。

## 运行行为

模型继续使用 `backend=dinov3` 的完整事件分类协议，通过 `encoder_backend=litert` 选择 CPU 编码器。分类头、完整校准包和 `experiment_full` 配方与已验证模型相同。编码器输入输出均为 FP32，忽略探测器 FP16 偏好。图内已经包含 L2 归一化，运行时验证单位范数并原样保留输出，避免二次归一化扰动校准。

照片流程保留裁图收集、历史适配、完整事件融合、拒识和人工反馈更新。原版 PyTorch 编码器继续可用。该分类器的清单仍声明跳过视频。

`encoder_sha256` 保留原始编码器的特征来源；实际 `.tflite` 另有 `encoder_litert_sha256`，并加入分类缓存指纹和学习特征缓存校验。学习发布保留 LiteRT 后端和离线权重引用，不需要原始编码器源码。

## 资源与构建

内置分类模型位于 `res/model/cls/bundled`，独立于 `user`、`sync` 和可选云端组件目录。云端同步不会删除它。

- FP32 编码器：342,720,832 字节。
- 分类头：151,610,423 字节。
- 完整校准包：86,748,709 字节。
- 原始编码器 SHA-256：`73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c`。
- LiteRT 编码器 SHA-256：`dd2e4f5eb531512747e121a2e4149e0208ce9f37ce730305e76270267cd4971d`。

`res/bundled-models/Neri1.2preview-litert-fp32` 保存 9 个不超过 80 MiB 的源分片及逐文件、逐分片 SHA-256 索引。执行：

```powershell
python scripts/prepare_bundled_classifier.py
```

即可离线还原资源。Windows 打包和 CI 自动执行这一步，源分片不会重复进入发行包。打包对白名单中的这一个内置清单、分类头和校准包开放例外；任意用户或同步目录中的 PyTorch 权重、分类清单、NumPy 资产仍被排除。发行校验要求探测器和分类器同时存在，并核对全部资产校验和。

## 验证与交付

- DINOv3、模型管理及 LiteRT 后端回归：415 项通过，另有 18 项子测试通过。
- 模型选择及视频设置 Flutter 测试：24 项通过。
- Windows release 编译通过。
- 实际便携包不含 PyTorch；内置探测器与 45 类分类器真实推理通过。
- 使用包内 Python 对三张来自独立相机目录的本地动物照片完成探测、分类、完整事件汇总，未安装或导入 PyTorch。
- 真实设置接口首次启动选择内置探测器和 FP32 分类器；依赖就绪。
- Python 环境污染隔离和维护模块导入检查通过。
- 最终 ZIP 全部 20,858 个条目解压及 CRC 校验通过；解压后的包内 Python 再次通过真实探测/分类推理与环境隔离检查。

本地交付文件为 `dist/Neri_v3.1.1_alpha2_x64_lite_builtin_fp32_20261115.zip`。构建与验证记录保存在 `runs/builtin-litert-release-20261115`。
