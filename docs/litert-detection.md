# 无 PyTorch 的 LiteRT 探测与分类

当运行环境没有 `torch`、`torchvision` 或 `ultralytics` 时，Neri 可以使用 LiteRT（`ai-edge-litert`）完成照片探测和快速视频抽帧探测，不导入 PyTorch 或 TensorFlow。

## 使用方式

1. 把 `.tflite` 或 `.litert` 探测模型放入 `res/model/detect/user`，或由 NeriCloud 同步到 `res/model/detect/sync`。
2. 可选：在同目录放置 `<模型名>.labels.txt`，格式为 `name,index`，例如：

   ```text
   animal,0
   person,1
   vehicle,2
   ```

3. 启动后，模型列表会显示带 **LiteRT** 标记的模型。新安装默认选择内置探测器；需要照片物种分类时，在“环境维护 → Neri1.2preview”选择 CPU 下载版本并安装，再选择 **Neri1.2preview·LiteRT** 分类器。已有设置中的明确“不使用”选择会保留。

发行包仅在 `res/model/detect/bundled` 附带 MegaDetector V6 `MDV6-yolov10-c.tflite`（动物 / 人 / 车辆）。45 类 Neri1.2preview LiteRT FP32 分类模型不再内置发行包，也不再上传至 GitHub，由用户在“设置 -> 环境维护”通过 NeriCloud 按需下载（自动硬件探测或显式选择 CPU/GPU）。`res/model/detect/bundled` 独立于云端同步目录。

## 支持范围

- 照片：完整支持，输出检测框、置信度和类别。
- 视频：支持“快速识别”抽帧和“跳过视频”；不支持完整视频追踪。
- 分类：可搭配从 NeriCloud 下载的 Neri1.2preview·LiteRT，保留历史适配、完整事件融合、拒识、人工反馈及重新校准流程。该版本编码器始终使用 CPU FP32，不受探测器 FP16 偏好影响。选择该分类器时使用照片识别、跳过视频。
- 模型头：支持 MegaDetector `[1,N,6]` end2end 输出和 YOLO `[1,4+C,N]` / `[1,N,4+C]` 原始输出，后者在主机侧执行按类别 NMS。

## 环境检查

`/api/settings` 返回 `pytorch_available`、`litert_available` 和 `detection_ready`。前端据此启用开始按钮并显示当前为 LiteRT 探测。只要 LiteRT 运行库可用且至少存在一个 LiteRT 探测模型，`detection_ready` 就为真，即使 `missing_yolo_dependencies` 中仍列出 PyTorch 相关包。

分类模型元数据通过 `encoder_backend=litert` 标识 CPU 运行时，`backend=dinov3` 继续标识完整事件分类协议。

同源 Neri1.2preview 的 CPU/GPU 版本通过 `learning_family_id` 共用学习档案，补图和新增物种随版本切换保留。下载和切换入口统一位于环境维护，默认自动检测。迁移、撤销与存储说明见 [CPU/GPU 共享学习](shared-learning-cpu-gpu-20261115.md)。

## 打包验证与本地开发工具

Windows 打包与 CI（`scripts/package_windows.ps1`、`.github/workflows/build-windows-x64.yml`）仅打包并校验内置 `MDV6-yolov10-c.tflite` 探测器，不再依赖仓库模型分片。`scripts/package_litert_resources.py` 与 `scripts/verify_bundled_litert.py` 会强制校验：发行包包含 `MDV6-yolov10-c.tflite` 且通过 LiteRT CPU 推理，同时不包含 `model/cls/` 分类器资产、`bundled-models/` 分片或 PyTorch。

`scripts/prepare_bundled_classifier.py` 保留为显式本地开发工具：当开发者本地工作区存在 `res/bundled-models/Neri1.2preview-litert-fp32` 分片时，可手动执行 `python scripts/prepare_bundled_classifier.py` 进行离线还原与本地调试；分片与还原产物均由 `.gitignore` 忽略，不进入版本库或发行包。
