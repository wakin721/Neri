"""Generic CPU LiteRT detector used when PyTorch/Ultralytics are unavailable.

Supports the two detector heads used by Neri models:

* end-to-end ``[1, N, 6]`` rows ``x1,y1,x2,y2,score,class`` (MegaDetector V6)
* raw YOLO ``[1, 4+C, N]`` or ``[1, N, 4+C]`` with host-side class-aware NMS

No torch, torchvision, ultralytics or tensorflow imports are used.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

INPUT_EDGE = 640
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.5
DEFAULT_MAX_DETECTIONS = 300
DEFAULT_CLASS_NAMES = {0: "animal", 1: "person", 2: "vehicle"}

COCO_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)


@dataclass(frozen=True)
class LiteRtDetection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int


def _labels_path(path: Path) -> Path | None:
    candidates = (
        path.with_suffix(".labels.txt"),
        path.with_name(path.stem + ".labels.json"),
        path.parent / "labels.txt",
    )
    return next((item for item in candidates if item.is_file()), None)


def _parse_label_lines(text: str) -> dict[int, str]:
    names: dict[int, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.replace("\t", ",").split(",")]
        if len(parts) < 2:
            continue
        left, right = parts[0], parts[1]
        if left.isdigit():
            names[int(left)] = right
        elif right.isdigit():
            names[int(right)] = left
    return names


def read_model_classes(path: Path, *, class_count: int | None = None) -> dict[int, str]:
    """Read a sidecar label map without loading the interpreter."""
    path = Path(path)
    labels = _labels_path(path)
    if labels is not None:
        if labels.suffix.lower() == ".json":
            try:
                payload = json.loads(labels.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                payload = None
            if isinstance(payload, dict):
                return {int(key): str(value) for key, value in payload.items()}
            if isinstance(payload, list):
                return {index: str(value) for index, value in enumerate(payload)}
        else:
            names = _parse_label_lines(labels.read_text(encoding="utf-8"))
            if names:
                return names
    if class_count == len(COCO_NAMES):
        return {index: name for index, name in enumerate(COCO_NAMES)}
    if class_count in (None, 3):
        return dict(DEFAULT_CLASS_NAMES)
    return {index: f"class_{index}" for index in range(class_count or 0)}


def letterbox_rgb(rgb, size: int):
    """Return model input and exact padding/resampling geometry."""
    import cv2
    import numpy as np

    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("LiteRT 输入必须是 uint8 RGB 图像")
    height, width = rgb.shape[:2]
    if min(height, width) < 1:
        raise ValueError("LiteRT 输入尺寸无效")
    gain = min(size / width, size / height)
    resized_w = max(1, min(size, round(width * gain)))
    resized_h = max(1, min(size, round(height * gain)))
    left, top = (size - resized_w) // 2, (size - resized_h) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top + resized_h, left:left + resized_w] = cv2.resize(
        rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR
    )
    return canvas, (left, top, resized_w / width, resized_h / height)


def _nms(boxes, scores, class_ids, iou_threshold, max_detections):
    import numpy as np

    keep: list[int] = []
    for class_id in np.unique(class_ids):
        order = np.flatnonzero(class_ids == class_id)
        order = order[np.argsort(-scores[order], kind="stable")]
        while len(order):
            best = int(order[0])
            keep.append(best)
            if len(keep) >= max_detections:
                return keep[:max_detections]
            rest = order[1:]
            if not len(rest):
                break
            intersection = np.maximum(
                0,
                np.minimum(boxes[best, 2:], boxes[rest, 2:])
                - np.maximum(boxes[best, :2], boxes[rest, :2]),
            ).prod(axis=1)
            area_best = (boxes[best, 2:] - boxes[best, :2]).prod()
            area_rest = (boxes[rest, 2:] - boxes[rest, :2]).prod(axis=1)
            union = area_best + area_rest - intersection
            order = rest[
                intersection / np.maximum(union, np.finfo(np.float32).eps)
                <= iou_threshold
            ]
    return keep[:max_detections]


def _decode_end2end(output, confidence, classes):
    import numpy as np

    shape = output.shape
    if len(shape) != 3 or shape[0] != 1:
        raise ValueError("LiteRT end2end 输出维度无效")
    if shape[2] == 6:
        rows = output[0]
    elif shape[1] == 6:
        rows = output[0].T
    else:
        raise ValueError("LiteRT end2end 输出列数无效")
    rows = rows.astype(np.float32, copy=False)
    if not np.isfinite(rows).all():
        raise ValueError("LiteRT 输出包含非有限值")
    selected = rows[rows[:, 4] >= confidence]
    if classes is not None:
        selected = selected[np.isin(np.rint(selected[:, 5]).astype(np.int64), list(classes))]
    if not len(selected):
        return np.empty((0, 6), dtype=np.float32)
    return np.column_stack(
        (selected[:, :4], selected[:, 4], np.rint(selected[:, 5]).astype(np.int64))
    ).astype(np.float32)


def _decode_raw(output, confidence, iou, classes, max_detections):
    import numpy as np

    shape = output.shape
    if len(shape) != 3 or shape[0] != 1:
        raise ValueError("LiteRT YOLO 输出维度无效")
    channels_first = shape[1] < shape[2]
    channels = shape[1] if channels_first else shape[2]
    anchors = shape[2] if channels_first else shape[1]
    class_count = channels - 4
    if class_count < 1:
        raise ValueError("LiteRT YOLO 输出类别数无效")
    values = output[0]
    if channels_first:
        scores = values[4:, :]
    else:
        scores = values[:, 4:].T
    if not np.isfinite(values).all():
        raise ValueError("LiteRT 输出包含非有限值")
    class_ids = scores.argmax(axis=0)
    confidences = scores[class_ids, np.arange(anchors)]
    selected = np.flatnonzero(confidences >= confidence)
    if classes is not None:
        selected = selected[np.isin(class_ids[selected], list(classes))]
    if not len(selected):
        return np.empty((0, 6), dtype=np.float32)
    if channels_first:
        xywh = values[:4, selected].T.copy()
    else:
        xywh = values[selected, :4].copy()
    xywh[:, :2] = xywh[:, :2] - xywh[:, 2:] / 2
    xywh[:, 2:] = xywh[:, :2] + xywh[:, 2:]
    keep = _nms(
        xywh,
        confidences[selected],
        class_ids[selected],
        iou,
        max_detections,
    )
    return np.column_stack(
        (
            xywh[keep],
            confidences[selected][keep],
            class_ids[selected][keep],
        )
    ).astype(np.float32)


class LiteRtDetector:
    def __init__(
        self,
        path: Path,
        *,
        confidence: float = DEFAULT_CONFIDENCE,
        iou: float = DEFAULT_IOU,
        max_detections: int = DEFAULT_MAX_DETECTIONS,
        interpreter_factory=None,
    ):
        import numpy as np

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        factory = interpreter_factory
        if factory is None:
            from ai_edge_litert.interpreter import Interpreter

            factory = Interpreter
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.max_detections = int(max_detections)
        self.interpreter = factory(
            model_path=str(path),
            num_threads=min(4, os.cpu_count() or 1),
        )
        self.interpreter.allocate_tensors()
        inputs = self.interpreter.get_input_details()
        outputs = self.interpreter.get_output_details()
        if len(inputs) != 1 or not outputs:
            raise ValueError("LiteRT 模型输入/输出数量无效")
        input_shape = tuple(int(value) for value in inputs[0]["shape"])
        if len(input_shape) != 4 or input_shape[0] != 1:
            raise ValueError("LiteRT 模型输入维度无效")
        if input_shape[1] == 3:
            self.channels_first = True
            self.input_size = input_shape[2]
            if input_shape[3] != 3:
                raise ValueError("LiteRT 模型输入通道无效")
        elif input_shape[3] == 3:
            self.channels_first = False
            self.input_size = input_shape[1]
        else:
            raise ValueError("LiteRT 模型输入通道无效")
        self.input_dtype = inputs[0]["dtype"]
        if self.input_dtype not in (np.float32, np.uint8):
            raise ValueError("LiteRT 模型输入类型不受支持")
        self.input_index = inputs[0]["index"]
        self.output_details = outputs[0]
        self.output_index = self.output_details["index"]
        output_shape = tuple(int(value) for value in self.output_details["shape"])
        if len(output_shape) != 3:
            raise ValueError("LiteRT 模型输出维度无效")
        if output_shape[2] == 6:
            self.output_kind = "end2end"
            class_count = 3
        elif output_shape[1] == 6:
            self.output_kind = "end2end"
            class_count = 3
        else:
            self.output_kind = "raw"
            class_count = min(output_shape[1], output_shape[2]) - 4
            if class_count < 1:
                raise ValueError("LiteRT 模型输出类别数无效")
        self.names = read_model_classes(path, class_count=class_count)

    def predict(self, rgb, *, confidence=None, iou=None, classes=None):
        import numpy as np

        size = int(self.input_size)
        canvas, (left, top, gain_x, gain_y) = letterbox_rgb(rgb, size)
        if self.input_dtype == np.uint8:
            tensor = canvas[None]
        else:
            tensor = canvas.astype(np.float32)[None] / np.float32(255.0)
        if self.channels_first:
            tensor = tensor.transpose(0, 3, 1, 2).copy()
        self.interpreter.set_tensor(self.input_index, tensor)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_index)
        if output.dtype != np.float32:
            scale, zero = self.output_details.get("quantization", (0.0, 0))
            if not scale:
                raise ValueError("LiteRT 量化输出缺少 scale")
            output = (output.astype(np.float32) - zero) * scale
        conf = self.confidence if confidence is None else float(confidence)
        threshold = self.iou if iou is None else float(iou)
        if self.output_kind == "end2end":
            rows = _decode_end2end(output, conf, classes)
        else:
            rows = _decode_raw(
                output, conf, threshold, classes, self.max_detections
            )
        if not len(rows):
            return []
        height, width = rgb.shape[:2]
        if float(np.max(np.abs(rows[:, :4]))) <= 2.0:
            rows[:, :4] *= size
        rows[:, [0, 2]] = (rows[:, [0, 2]] - left) / gain_x
        rows[:, [1, 3]] = (rows[:, [1, 3]] - top) / gain_y
        rows[:, [0, 2]] = np.clip(rows[:, [0, 2]], 0, width)
        rows[:, [1, 3]] = np.clip(rows[:, [1, 3]], 0, height)
        return [
            LiteRtDetection(
                float(row[0]),
                float(row[1]),
                float(row[2]),
                float(row[3]),
                float(row[4]),
                int(row[5]),
            )
            for row in rows
            if row[2] > row[0] and row[3] > row[1]
        ]
