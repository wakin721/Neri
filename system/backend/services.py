"""Active backend service facade with DINOv2-only DINO routing.

The large legacy service module remains a temporary implementation detail for
non-DINO media/job behavior. Its DINO hook globals are replaced here so no old
DINOv3 model is loaded or executed.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import services_legacy as _legacy

for _name in dir(_legacy):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_legacy, _name)


def _dinov2_manifest_payload(model_path: str | Path | None):
    if not model_path:
        return None
    path = Path(model_path).expanduser()
    if not path.is_file() or not path.name.endswith(".neri.json"):
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    backend = payload.get("backend")
    if backend == "dinov3":
        raise ValueError("DINOv3 模型已不再支持；请安装或选择 DINOv2 模型。")
    return payload if backend == "dinov2" else None


def _validate_dinov2_job_options(
    model_path: str | None,
    classification_model_path: str | None,
    video_mode: str,
) -> None:
    payload = _dinov2_manifest_payload(classification_model_path)
    if payload is None:
        return
    if not model_path:
        raise ValueError("DINOv2 分类模型必须同时选择探测模型。")
    if video_mode == "all":
        raise ValueError("DINOv2 暂不支持视频完整识别，请使用快速识别或跳过视频。")


def _persist_dinov2_observations(*args, **kwargs):
    from .dinov2_feedback_service import persist_runtime_observations
    return persist_runtime_observations(*args, **kwargs)


def _load_detector(model_path: str | None, classification_model_path: str | None = None):
    from system.dinov2.image_processor import ImageProcessor

    resolved_model_path = _resolve_model_path(model_path) if model_path else None
    resolved_classification_path = _resolve_classification_model_path(
        classification_model_path
    )
    if classification_model_path and resolved_classification_path is None:
        raise FileNotFoundError(f"分类模型文件不存在: {classification_model_path}")
    if resolved_model_path is None and resolved_classification_path is None:
        raise ValueError("探测模型和分类模型至少需要选择一个。")

    payload = _dinov2_manifest_payload(resolved_classification_path)
    if payload is not None and resolved_model_path is None:
        raise ValueError("DINOv2 分类模型必须同时选择探测模型。")

    detector = ImageProcessor(
        str(resolved_model_path) if resolved_model_path is not None else None
    )
    if resolved_classification_path is not None:
        if payload is not None:
            from system.dinov2.runtime import load_dinov2_model

            runtime = load_dinov2_model(resolved_classification_path)
            detector.load_dinov2_classifier(runtime.classifier)
            detector.dinov2_registry = runtime.registry
            detector.dinov2_feedback = runtime.feedback
            detector.dinov2_runtime = runtime
            # Temporary cleanup alias for the frozen generic job loop only.
            # This does not enable DINOv3 inference and is deleted in Task 12.
            detector.dinov3_runtime = runtime
        else:
            detector.load_cls_model(str(resolved_classification_path))
    if (
        detector.model is None
        and getattr(detector, "cls_model", None) is None
        and getattr(detector, "dinov2_classifier", None) is None
    ):
        raise RuntimeError("未能加载所选的探测模型或分类模型。")
    return detector


def _learnable_observations_for_file(
    classification_model_path: str,
    file_path: Path,
):
    from .dinov2_feedback_service import _open_feedback_state, _path_identity

    feedback, _center = _open_feedback_state(classification_model_path)
    try:
        target = _path_identity(file_path)
        rows = feedback._conn.execute(
            "SELECT id,payload FROM observations ORDER BY id"
        ).fetchall()
        result = []
        for row in rows:
            try:
                payload = json.loads(row["payload"])
            except (TypeError, json.JSONDecodeError):
                continue
            if _path_identity(payload.get("source_path", "")) != target:
                continue
            observation_id = str(row["id"] or "")
            if observation_id:
                result.append(feedback.get_observation(observation_id))
        return result
    finally:
        feedback.close()


def _checkpoint_species_for_model(classification_model_path: str):
    from .dinov2_registry_service import load_checkpoint_for_model
    from system.dinov2.runtime import DinoV2ManifestError

    try:
        return set(load_checkpoint_for_model(classification_model_path).classes)
    except (DinoV2ManifestError, FileNotFoundError):
        return None


def _record_validation_registry_feedback(
    classification_model_path: str,
    observation,
    operation_id: str,
    confirmed_species: str,
):
    from .dinov2_feedback_service import record_registry_species_feedback

    return record_registry_species_feedback(
        classification_model_path,
        observation,
        operation_id,
        confirmed_species,
    )


def _record_validation_feedback(
    classification_model_path: str,
    observation,
    operation_id: str,
    action: str,
    confirmed_species: str | None,
):
    from .dinov2_feedback_service import (
        _affected_learning_species,
        _open_feedback_state,
    )

    feedback, feature_center = _open_feedback_state(classification_model_path)
    try:
        record = feedback.record_feedback(
            observation.id,
            operation_id=operation_id,
            action=action,
            confirmed_species=confirmed_species,
        )
        for species in sorted(_affected_learning_species(record)):
            feedback.recompute_species(species, feature_center)
        return record
    finally:
        feedback.close()


# These assignments redirect global lookups inside the frozen generic service
# implementation. No DINOv3 checkpoint/encoder/runtime is loaded through them.
_legacy._validate_dinov3_job_options = _validate_dinov2_job_options
_legacy._dinov3_manifest_payload = _dinov2_manifest_payload
_legacy._persist_dinov3_observations = _persist_dinov2_observations
_legacy._load_detector = _load_detector
_legacy._learnable_observations_for_file = _learnable_observations_for_file
_legacy._checkpoint_species_for_model = _checkpoint_species_for_model
_legacy._record_validation_registry_feedback = _record_validation_registry_feedback
_legacy._record_validation_feedback = _record_validation_feedback
