"""CPU-only YOLO26 LiteRT inference. No torch/ultralytics/TensorFlow imports.

The pinned official yolo-flutter-app w8a32 export has RGB float32 input
and raw normalized xywh + 80 class scores, with class-aware NMS on the host.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import zipfile

INPUT_EDGE = 640
CLASS_COUNT = 80
ANCHOR_COUNT = 8400
CONFIDENCE_THRESHOLD = 0.15
IOU_THRESHOLD = 0.5


def _metadata(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo("metadata.json")
        if info.file_size > 64 * 1024:
            raise ValueError("privacy_model_metadata_too_large")
        metadata = json.loads(archive.read(info))
    if (metadata.get("task") != "detect" or metadata.get("end2end") is not False
            or metadata.get("imgsz") != [INPUT_EDGE, INPUT_EDGE]):
        raise ValueError("privacy_model_metadata_mismatch")
    names = metadata.get("names")
    if not isinstance(names, dict) or set(names) != {str(i) for i in range(CLASS_COUNT)}:
        raise ValueError("privacy_model_classes_mismatch")
    return metadata


def letterbox(rgb):
    """Return float32 RGB NHWC input plus exact padding/resampling geometry."""
    import cv2
    import numpy as np

    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("privacy_input_invalid")
    height, width = rgb.shape[:2]
    if min(height, width) < 1:
        raise ValueError("privacy_input_invalid")
    gain = min(INPUT_EDGE / width, INPUT_EDGE / height)
    resized_w = max(1, min(INPUT_EDGE, round(width * gain)))
    resized_h = max(1, min(INPUT_EDGE, round(height * gain)))
    left, top = (INPUT_EDGE - resized_w) // 2, (INPUT_EDGE - resized_h) // 2
    canvas = np.full((INPUT_EDGE, INPUT_EDGE, 3), 114, dtype=np.uint8)
    canvas[top:top + resized_h, left:left + resized_w] = cv2.resize(
        rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    tensor = canvas.astype(np.float32)[None] / np.float32(255.0)
    return tensor, (left, top, resized_w / width, resized_h / height)


def decode_detections(output, geometry, classes, max_detections):
    """Validate the pinned raw head, suppress duplicates, undo letterboxing.

    Return Nx6 rows: x1,y1,x2,y2,confidence,class_id in source-image pixels.
    Pick the most likely of ALL classes before privacy filtering, so an animal
    with a low person score is not turned into a false person detection.
    """
    import numpy as np

    if output.shape != (1, CLASS_COUNT + 4, ANCHOR_COUNT) or output.dtype != np.float32:
        raise ValueError("privacy_detection_shape_mismatch")
    if not np.isfinite(output).all():
        raise ValueError("privacy_detection_nonfinite")
    scores = output[0, 4:, :]
    if np.any(scores < 0) or np.any(scores > 1):
        raise ValueError("privacy_detection_scores_invalid")
    class_ids = scores.argmax(axis=0)
    confidence = scores[class_ids, np.arange(ANCHOR_COUNT)]
    selected = np.flatnonzero((confidence >= CONFIDENCE_THRESHOLD) & np.isin(class_ids, list(classes)))
    if not len(selected):
        return np.empty((0, 6), dtype=np.float32)
    # NumPy advanced indexing can move the anchor axis to the front. Make the
    # desired (N,4) layout explicit instead of depending on that special case.
    xywh = output[0, :4, :][:, selected].T.copy()
    if np.any(xywh[:, 2:] <= 0) or np.any(np.abs(xywh) > 4):
        raise ValueError("privacy_detection_coordinates_invalid")
    xywh *= INPUT_EDGE
    boxes = np.concatenate((xywh[:, :2] - xywh[:, 2:] / 2,
                            xywh[:, :2] + xywh[:, 2:] / 2), axis=1)
    ids = class_ids[selected]
    confidences = confidence[selected]
    area = (boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
    keep = []
    for class_id in np.unique(ids):
        order = np.flatnonzero(ids == class_id)
        order = order[np.argsort(-confidences[order], kind="stable")]
        while len(order):
            best = int(order[0])
            keep.append(best)
            if len(keep) >= max_detections:
                raise ValueError("privacy_detection_saturated")
            rest = order[1:]
            if not len(rest):
                break
            intersection = np.maximum(0, np.minimum(boxes[best, 2:], boxes[rest, 2:])
                                      - np.maximum(boxes[best, :2], boxes[rest, :2])).prod(axis=1)
            union = area[best] + area[rest] - intersection
            order = rest[intersection / np.maximum(union, np.finfo(np.float32).eps) <= IOU_THRESHOLD]
    result = np.column_stack((boxes[keep], confidences[keep], ids[keep])).astype(np.float32)
    left, top, gain_x, gain_y = geometry
    result[:, [0, 2]] = (result[:, [0, 2]] - left) / gain_x
    result[:, [1, 3]] = (result[:, [1, 3]] - top) / gain_y
    return result


class LiteRtDetector:
    def __init__(self, path: Path, *, classes, max_detections):
        import numpy as np
        from ai_edge_litert.interpreter import Interpreter

        metadata = _metadata(Path(path))
        self.names = {int(key): name for key, name in metadata["names"].items()}
        self.classes = tuple(classes)
        self.max_detections = max_detections
        self.interpreter = Interpreter(model_path=str(path), num_threads=min(4, os.cpu_count() or 1))
        self.interpreter.allocate_tensors()
        inputs = self.interpreter.get_input_details()
        outputs = self.interpreter.get_output_details()
        input_shapes = {(1, INPUT_EDGE, INPUT_EDGE, 3), (1, 3, INPUT_EDGE, INPUT_EDGE)}
        if (len(inputs) != 1 or len(outputs) != 1
                or tuple(inputs[0]["shape"]) not in input_shapes
                or inputs[0]["dtype"] != np.float32
                or tuple(outputs[0]["shape"]) != (1, CLASS_COUNT + 4, ANCHOR_COUNT)
                or outputs[0]["dtype"] != np.float32):
            raise ValueError("privacy_model_tensor_mismatch")
        self.input_index = inputs[0]["index"]
        self.output_index = outputs[0]["index"]
        self.channels_first = tuple(inputs[0]["shape"])[1] == 3

    def predict(self, rgb):
        tensor, geometry = letterbox(rgb)
        if self.channels_first:
            tensor = tensor.transpose(0, 3, 1, 2).copy()
        self.interpreter.set_tensor(self.input_index, tensor)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_index)
        return decode_detections(output, geometry, self.classes, self.max_detections)
