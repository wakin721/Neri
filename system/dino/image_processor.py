"""DINOv3-aware image processor adapter.

The adapter can be constructed without importing Ultralytics. The generic YOLO
processor is loaded lazily only when detector work actually needs it.
"""
from __future__ import annotations

import concurrent.futures
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ImageProcessor:
    def __init__(self, model_path: Optional[str]):
        self.model_path = model_path or None
        self.model = None
        self.cls_model = None
        self._legacy = None
        self.dino_classifier = None
        self.dinov3_classifier = None
        self._dinov3_collector = None
        self._dinov3_current_paths = ()
        self._photo_memory_guard = None

        # Preserve eager model loading when the normal YOLO runtime is present,
        # but do not make importing/constructing the DINOv3 adapter depend on it.
        if self.model_path:
            try:
                self._ensure_legacy()
            except ModuleNotFoundError as exc:
                if exc.name != "ultralytics":
                    raise

    def _ensure_legacy(self):
        legacy = self._legacy
        if legacy is not None:
            return legacy

        from system.image_processor_legacy import ImageProcessor as LegacyImageProcessor

        legacy = LegacyImageProcessor(self.model_path)
        self._legacy = legacy
        self.model = legacy.model
        self.cls_model = legacy.cls_model
        return legacy

    def __getattr__(self, name: str):
        if name == "_legacy":
            raise AttributeError(name)
        legacy = self._ensure_legacy()
        return getattr(legacy, name)

    def load_cls_model(self, model_path: str) -> None:
        legacy = self._ensure_legacy()
        legacy.load_cls_model(model_path)
        self.cls_model = legacy.cls_model

    def load_dino_classifier(self, classifier) -> None:
        if getattr(classifier, "backend", None) != "dinov3":
            raise ValueError("Only DINOv3 classification models are supported.")
        from system.dinov3.processor import DinoV3CropCollector

        self.dino_classifier = classifier
        self.dinov3_classifier = classifier
        self._dinov3_collector = DinoV3CropCollector()

    def preload_batch_data(self, img_paths):
        legacy = self._ensure_legacy()
        legacy._photo_memory_guard = self._photo_memory_guard
        return legacy.preload_batch_data(img_paths)

    def release_batch_images(self):
        if self._legacy is not None:
            self._legacy.release_batch_images()

    def _classify_dino_crops(
        self,
        det_results,
        original_imgs_rgb,
        *,
        source_indices=None,
    ):
        if self.dino_classifier is None:
            return [{} for _ in det_results], [{} for _ in det_results]
        crop_tasks = []
        for result_index, result in enumerate(det_results):
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            original_rgb = original_imgs_rgb[result_index]
            for box_index, box in enumerate(boxes):
                crop_tasks.append((result_index, box_index, box, original_rgb))

        crops = []
        crop_map = []
        if crop_tasks:
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                for item in executor.map(self._crop_single_box, crop_tasks):
                    if item is None:
                        continue
                    result_index, box_index, crop = item
                    crops.append(crop)
                    crop_map.append((result_index, box_index))

        candidate_maps = [{} for _ in det_results]
        selected_maps = [{} for _ in det_results]
        if not crops:
            return candidate_maps, selected_maps

        predictions = self.dino_classifier.classify_crops(crops, array_color="rgb")
        if len(predictions) != len(crop_map):
            raise RuntimeError(
                "DINOv3 classifier returned a different number of predictions than crops"
            )
        from system.dinov3.processor import DinoV3CropRecord

        for prediction, (result_index, box_index) in zip(predictions, crop_map):
            box = det_results[result_index].boxes[box_index]
            detection_confidence = float(box.conf.item())
            bbox = tuple(float(value) for value in box.xyxy.tolist()[0])
            candidate = prediction.as_candidate(
                detection_confidence=detection_confidence
            )
            candidate.update({"predicted_species": prediction.species})
            candidate_maps[result_index][box_index] = [candidate]
            selected_maps[result_index][box_index] = (
                candidate if prediction.accepted else None
            )
            source_index = (
                source_indices[result_index]
                if source_indices is not None
                else result_index
            )
            path = (
                self._dinov3_current_paths[source_index]
                if source_index < len(self._dinov3_current_paths)
                else getattr(
                    det_results[result_index],
                    "path",
                    f"crop-image:{source_index}",
                )
            )
            candidate["sequence_pending"] = True
            candidate["sequence_status"] = "pending"
            self._dinov3_collector.record_crop(
                DinoV3CropRecord(
                    image_path=str(path),
                    box_index=box_index,
                    bbox=bbox,
                    detection_confidence=detection_confidence,
                    raw_embedding=prediction.embedding.copy(),
                    initial_prediction=prediction,
                    initial_candidate=dict(candidate),
                )
            )
        return candidate_maps, selected_maps

    def detect_batch_species(
        self,
        img_paths: List[str],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if self.dino_classifier is None:
            return self._ensure_legacy().detect_batch_species(img_paths, **kwargs)
        self._dinov3_current_paths = tuple(img_paths)
        self._dinov3_collector.discard_paths(img_paths)

        if self.model is None and self.model_path:
            try:
                legacy = self._ensure_legacy()
                self.model = legacy.model
            except ModuleNotFoundError as exc:
                if exc.name != "ultralytics":
                    raise

        if self.model is None:
            raise ValueError("联合分类模型必须同时选择探测模型。")

        preloaded_data = kwargs.get("preloaded_data")
        loaded = preloaded_data or self.preload_batch_data(img_paths)
        if not loaded:
            return [
                {
                    "物种名称": "",
                    "物种数量": "",
                    "detect_results": None,
                    "最低置信度": None,
                    "error": "图片无法读取或解码，请检查文件是否为空或损坏。",
                }
                for _ in img_paths
            ]
        valid_indices, processed_imgs, original_imgs_rgb = loaded
        if not processed_imgs:
            return [
                {
                    "物种名称": "",
                    "物种数量": "",
                    "detect_results": None,
                    "最低置信度": None,
                    "error": "图片无法读取或解码，请检查文件是否为空或损坏。",
                }
                for _ in img_paths
            ]

        use_fp16 = bool(kwargs.get("use_fp16", False))
        encoder = getattr(self.dino_classifier, "encoder", None)
        if encoder is not None and hasattr(encoder, "use_fp16"):
            encoder.use_fp16 = bool(
                use_fp16 and getattr(encoder, "device", None) == "cuda"
            )
        device_name, use_fp16 = self._determine_device(use_fp16)
        self._sync_device(device_name)
        detect_started = time.perf_counter()
        det_results = self.model(
            processed_imgs,
            augment=kwargs.get("augment", True),
            agnostic_nms=kwargs.get("agnostic_nms", True),
            imgsz=kwargs.get("imgsz", 1920),
            half=use_fp16,
            device=device_name,
            iou=kwargs.get("iou", 0.3),
            conf=kwargs.get("conf", 0.25),
            classes=kwargs.get("classes"),
            save=False,
        )
        self._sync_device(device_name)
        detect_elapsed = time.perf_counter() - detect_started
        classify_started = time.perf_counter()
        candidate_maps, selected_maps = self._classify_dino_crops(
            det_results,
            original_imgs_rgb,
            source_indices=valid_indices,
        )
        logger.info(
            "DINOv3 batch timing: images=%d detect=%.3fs classify=%.3fs fp16=%s",
            len(processed_imgs),
            detect_elapsed,
            time.perf_counter() - classify_started,
            getattr(encoder, "use_fp16", False),
        )

        output: List[Dict[str, Any]] = []
        result_iter = iter(det_results)
        candidate_iter = iter(candidate_maps)
        selected_iter = iter(selected_maps)
        valid_set = set(valid_indices)
        for source_index in range(len(img_paths)):
            if source_index not in valid_set:
                output.append(
                    {
                        "物种名称": "",
                        "物种数量": "",
                        "detect_results": None,
                        "最低置信度": None,
                        "error": "图片无法读取或解码，请检查文件是否为空或损坏。",
                    }
                )
                continue
            result = next(result_iter)
            candidates = next(candidate_iter)
            selected = next(selected_iter)
            counts: Dict[str, int] = {}
            confidences: list[float] = []
            boxes = getattr(result, "boxes", None)
            if boxes is not None:
                for index, _box in enumerate(boxes):
                    if not hasattr(result, "candidates_data"):
                        result.candidates_data = {}
                    result.candidates_data[index] = candidates.get(index, [])
                    chosen = selected.get(index)
                    if chosen is None:
                        if not hasattr(result, "classification_filtered_boxes"):
                            result.classification_filtered_boxes = set()
                        result.classification_filtered_boxes.add(index)
                        continue
                    if not hasattr(result, "selected_candidates_data"):
                        result.selected_candidates_data = {}
                    result.selected_candidates_data[index] = chosen
                    name = str(chosen["name"])
                    counts[name] = counts.get(name, 0) + 1
                    confidences.append(float(chosen["conf"]))
            output.append(
                {
                    "物种名称": ",".join(counts.keys()) if counts else "空",
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
                        "confidence_priority",
                        "classification",
                    ),
                }
            )
        return output
