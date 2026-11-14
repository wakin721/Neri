# 无 PyTorch 的 LiteRT 探测

当运行环境没有 `torch`、`torchvision` 或 `ultralytics` 时，Neri 可以使用 LiteRT（`ai-edge-litert`）完成照片探测和快速视频抽帧探测，不导入 PyTorch 或 TensorFlow。

## 使用方式

1. 把 `.tflite` 或 `.litert` 探测模型放入 `res/model/detect/user`，或由 NeriCloud 同步到 `res/model/detect/sync`。
2. 可选：在同目录放置 `<模型名>.labels.txt`，格式为 `name,index`，例如：

   ```text
   animal,0
   person,1
   vehicle,2
   ```

3. 启动后，模型列表会显示带 **LiteRT** 标记的探测模型。没有 PyTorch 时，设置接口会自动清空分类模型并优先选择 LiteRT 探测模型。

仓库在 `res/model/detect/sync` 附带 MegaDetector V6 `MDV6-yolov10-c.tflite`（动物 / 人 / 车辆）作为默认离线 LiteRT 探测器。

## 支持范围

- 照片：完整支持，输出检测框、置信度和类别。
- 视频：支持“快速识别”抽帧和“跳过视频”；不支持完整视频追踪。
- 分类：LiteRT 探测器暂不搭配分类模型。需要 Neri1.1 物种分类时仍须安装 PyTorch；没有 PyTorch 时输出通用类别（动物 / 人 / 车辆）。
- 模型头：支持 MegaDetector `[1,N,6]` end2end 输出和 YOLO `[1,4+C,N]` / `[1,N,4+C]` 原始输出，后者在主机侧执行按类别 NMS。

## 环境检查

`/api/settings` 返回 `pytorch_available`、`litert_available` 和 `detection_ready`。前端据此启用开始按钮并显示当前为 LiteRT 探测。只要 LiteRT 运行库可用且至少存在一个 LiteRT 探测模型，`detection_ready` 就为真，即使 `missing_yolo_dependencies` 中仍列出 PyTorch 相关包。
