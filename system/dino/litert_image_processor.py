"""ImageProcessor-compatible LiteRT detector for no-PyTorch deployments."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .litert_detector import LiteRtDetector, read_model_classes

logger = logging.getLogger(__name__)

_TRANSLATIONS = {"animal": "动物", "person": "人", "vehicle": "车辆"}


class _Box:
    __slots__ = ("conf", "cls", "xyxy")

    def __init__(self, detection):
        self.conf = np.asarray(detection.confidence, dtype=np.float32)
        self.cls = np.asarray(detection.class_id, dtype=np.int64)
        self.xyxy = np.asarray(
            [[detection.x1, detection.y1, detection.x2, detection.y2]],
            dtype=np.float32,
        )


class _Boxes:
    def __init__(self, detections):
        self._items = [_Box(item) for item in detections]
        self.conf = np.asarray(
            [item.confidence for item in detections], dtype=np.float32
        )
        self.cls = np.asarray(
            [item.class_id for item in detections], dtype=np.int64
        )
        self.conf = self.conf.reshape(-1)
        self.cls = self.cls.reshape(-1)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, index):
        return self._items[index]


class _Result:
    def __init__(self, boxes, names, path: str):
        self.boxes = boxes
        self.names = dict(names)
        self.path = path
        self.candidates_data: dict[int, list] = {}
        self.selected_candidates_data: dict[int, dict] = {}
        self.classification_filtered_boxes: set[int] = set()


class _ModelHandle:
    def __init__(self, names, path: str):
        self.names = dict(names)
        self.path = path


class LiteRtImageProcessor:
    """Photo/fast-video detector backed only by ``ai-edge-litert``."""

    def __init__(
        self,
        model_path: str,
        *,
        interpreter_factory=None,
    ):
        self.model_path = str(model_path)
        self.detector = LiteRtDetector(
            Path(model_path), interpreter_factory=interpreter_factory
        )
        self.names = dict(self.detector.names)
        self.model = _ModelHandle(self.names, self.model_path)
        self.cls_model = None
        self.dino_classifier = None
        self.dinov3_classifier = None
        self.translation_dict = dict(_TRANSLATIONS)
        self._last_runtime_gc = 0.0

    @staticmethod
    def _preprocess_image(image):
        if image is None or image.size == 0:
            return None
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        return np.ascontiguousarray(image)

    def preload_batch_data(self, img_paths: List[str]):
        processed = []
        original_rgb = []
        valid_indices = []
        for index, path in enumerate(img_paths):
            image = cv2.imread(str(path))
            prepared = self._preprocess_image(image)
            if prepared is None:
                continue
            valid_indices.append(index)
            processed.append(prepared)
            original_rgb.append(cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB))
        if not processed:
            return None
        return valid_indices, processed, original_rgb

    def preload_frame_data(self, frames: List[Any]) -> Tuple:
        processed = [self._preprocess_image(frame) for frame in frames]
        if any(frame is None for frame in processed):
            raise ValueError("视频帧预处理失败")
        original_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in processed]
        return list(range(len(processed))), processed, original_rgb

    def detect_batch_species(
        self,
        img_paths: List[str],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        confidence = float(kwargs.get("conf", 0.25))
        iou = float(kwargs.get("iou", 0.5))
        classes = kwargs.get("classes")
        preloaded = kwargs.get("preloaded_data") or self.preload_batch_data(img_paths)
        empty = {
            "物种名称": "",
            "物种数量": "",
            "detect_results": None,
            "最低置信度": None,
        }
        if not preloaded:
            return [dict(empty) for _ in img_paths]
        valid_indices, _processed, original_rgb = preloaded
        output: List[Dict[str, Any]] = []
        for source_index in range(len(img_paths)):
            if source_index not in valid_indices:
                output.append(dict(empty))
                continue
            position = valid_indices.index(source_index)
            detections = self.detector.predict(
                original_rgb[position],
                confidence=confidence,
                iou=iou,
                classes=classes,
            )
            result = _Result(
                _Boxes(detections),
                self.names,
                str(img_paths[source_index]),
            )
            counts: dict[str, int] = {}
            confidences: list[float] = []
            for detection in detections:
                raw_name = self.names.get(detection.class_id, str(detection.class_id))
                name = self.translation_dict.get(raw_name, raw_name)
                counts[name] = counts.get(name, 0) + 1
                confidences.append(float(detection.confidence))
            output.append(
                {
                    "物种名称": ",".join(counts) if counts else "空",
                    "物种数量": (
                        ",".join(str(counts[name]) for name in counts)
                        if counts
                        else "空"
                    ),
                    "detect_results": [result],
                    "最低置信度": (
                        f"{min(confidences):.3f}" if confidences else None
                    ),
                    "confidence_priority": kwargs.get(
                        "confidence_priority", "detection"
                    ),
                }
            )
        return output

    def detect_video_species(self, *_args, **_kwargs):
        raise ValueError(
            "LiteRT 探测暂不支持完整视频追踪，请使用快速识别或跳过视频。"
        )

    def load_cls_model(self, _model_path: str) -> None:
        raise ValueError(
            "LiteRT 探测暂不支持搭配分类模型；请取消分类模型或安装 PyTorch。"
        )

    def load_dino_classifier(self, _classifier) -> None:
        self.load_cls_model("")

    def cleanup_runtime_cache(self, *_args, **_kwargs) -> None:
        return None
