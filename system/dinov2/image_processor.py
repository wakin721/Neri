"""DINOv2-aware image processor adapter.

DINOv2 state can be constructed without importing Ultralytics. The generic YOLO
processor is loaded lazily only when detector/classification work actually needs
it. This keeps DINOv2 checkpoint/runtime tests independent from the optional YOLO
runtime while preserving the existing detector implementation.
"""
from __future__ import annotations

import concurrent.futures
import uuid
from typing import Any, Dict, List, Optional

from .classifier import DinoV2Observation


class ImageProcessor:
    def __init__(self, model_path: Optional[str]):
        self.model_path = model_path or None
        self.model = None
        self.cls_model = None
        self._legacy = None
        self.dinov2_classifier = None
        self._dinov2_observations: list[DinoV2Observation] = []

        # Preserve eager model loading when the normal YOLO runtime is present,
        # but do not make importing/constructing the DINOv2 adapter depend on it.
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

    def load_dinov2_classifier(self, classifier) -> None:
        self.dinov2_classifier = classifier
        self._dinov2_observations = []

    def drain_dinov2_observations(self):
        observations = tuple(self._dinov2_observations)
        self._dinov2_observations = []
        return observations

    def _classify_dinov2_crops(self, det_results, original_imgs_rgb):
        if self.dinov2_classifier is None:
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

        predictions = self.dinov2_classifier.classify_crops(crops, array_color="rgb")
        if len(predictions) != len(crop_map):
            raise RuntimeError(
                "DINOv2 classifier returned a different number of predictions than crops"
            )
        for prediction, (result_index, box_index) in zip(predictions, crop_map):
            box = det_results[result_index].boxes[box_index]
            detection_confidence = float(box.conf.item())
            bbox = tuple(float(value) for value in box.xyxy.tolist()[0])
            observation_id = uuid.uuid4().hex
            candidate = prediction.as_candidate(
                detection_confidence=detection_confidence
            )
            candidate.update(
                {
                    "observation_id": observation_id,
                    "predicted_species": prediction.species,
                }
            )
            candidate_maps[result_index][box_index] = [candidate]
            selected_maps[result_index][box_index] = (
                candidate if prediction.accepted else None
            )
            self._dinov2_observations.append(
                DinoV2Observation(
                    result_index=result_index,
                    box_index=box_index,
                    embedding=prediction.embedding,
                    accepted=prediction.accepted,
                    species=prediction.species,
                    source=prediction.source,
                    registry_id=prediction.registry_id,
                    registration_status=prediction.registration_status,
                    known_score=prediction.known_score,
                    threshold=prediction.threshold,
                    detection_confidence=detection_confidence,
                    observation_id=observation_id,
                    best_known_species=prediction.best_known_species,
                    bbox=bbox,
                    squared_distance=prediction.squared_distance,
                    class_margin=prediction.class_margin,
                    adjusted_distance_score=prediction.adjusted_distance_score,
                    score_threshold=prediction.score_threshold,
                    registry_action=prediction.registry_action,
                )
            )
        return candidate_maps, selected_maps

    def detect_batch_species(
        self,
        img_paths: List[str],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if self.dinov2_classifier is None:
            return self._ensure_legacy().detect_batch_species(img_paths, **kwargs)

        if self.model is None and self.model_path:
            try:
                legacy = self._ensure_legacy()
                self.model = legacy.model
            except ModuleNotFoundError as exc:
                if exc.name != "ultralytics":
                    raise

        if self.model is None:
            raise ValueError("DINOv2 分类模型必须同时选择探测模型。")

        self._dinov2_observations = []
        preloaded_data = kwargs.get("preloaded_data")
        loaded = preloaded_data or self.preload_batch_data(img_paths)
        if not loaded:
            return [
                {
                    "物种名称": "",
                    "物种数量": "",
                    "detect_results": None,
                    "最低置信度": None,
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
                }
                for _ in img_paths
            ]

        use_fp16 = bool(kwargs.get("use_fp16", False))
        device_name, use_fp16 = self._determine_device(use_fp16)
        self._sync_device(device_name)
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
        candidate_maps, selected_maps = self._classify_dinov2_crops(
            det_results, original_imgs_rgb
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
                        "confidence_priority", "classification"
                    ),
                }
            )
        return output
