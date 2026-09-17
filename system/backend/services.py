"""Active backend service facade with DINOv2-only DINO routing.

The large legacy service module remains a temporary implementation detail for
non-DINO media/job behavior. Its DINO hook globals are replaced here so no old
DINOv3 model is loaded or executed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

from . import services_legacy as _legacy

for _name in dir(_legacy):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_legacy, _name)


class ProcessingJobManager(_legacy.ProcessingJobManager):
    """Persist resumable detection progress without making UI updates disk-bound."""

    _resume_checkpoint_interval_seconds = 1.0

    def __init__(self, *args, **kwargs) -> None:
        self._resume_checkpoint_at: dict[str, float] = {}
        super().__init__(*args, **kwargs)

    def _should_checkpoint_detection_progress(
        self,
        job_id: str,
        changes: dict[str, object],
    ) -> bool:
        if "processed" not in changes or "results" not in changes:
            return False
        try:
            next_processed = int(changes["processed"])
        except (TypeError, ValueError):
            return False
        with self._lock:
            request = self._job_requests.get(job_id)
            current = self._jobs.get(job_id)
            return bool(
                request is not None
                and request.options.enable_detection
                and current is not None
                and next_processed > current.processed
            )

    def _mutate_job(
        self,
        job_id: str,
        *,
        persist: bool = True,
        **changes: object,
    ) -> None:
        if not persist and self._should_checkpoint_detection_progress(job_id, changes):
            now = time.monotonic()
            last = self._resume_checkpoint_at.get(job_id)
            if last is None or now - last >= self._resume_checkpoint_interval_seconds:
                persist = True
                self._resume_checkpoint_at[job_id] = now
        super()._mutate_job(job_id, persist=persist, **changes)

    def _save_state_unlocked(self) -> None:
        """Atomically replace job_state.json so forced restarts keep the last checkpoint."""
        state_path = job_state_path()
        temp_path = state_path.with_name(f"{state_path.name}.tmp")
        try:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "jobs": {
                    job_id: job.model_copy(update={"active": False}).model_dump()
                    for job_id, job in self._jobs.items()
                },
                "requests": {
                    job_id: request.model_dump()
                    for job_id, request in self._job_requests.items()
                    if job_id in self._jobs
                },
            }
            temp_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(temp_path, state_path)
        except Exception:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


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


# These assignments redirect DINO hooks inside the generic service implementation.
_legacy._validate_dinov2_job_options = _validate_dinov2_job_options
_legacy._dinov2_manifest_payload = _dinov2_manifest_payload
_legacy._persist_dinov2_observations = _persist_dinov2_observations
_legacy._load_detector = _load_detector
_legacy._learnable_observations_for_file = _learnable_observations_for_file
_legacy._checkpoint_species_for_model = _checkpoint_species_for_model
_legacy._record_validation_registry_feedback = _record_validation_registry_feedback
_legacy._record_validation_feedback = _record_validation_feedback


# Make the active facade self-contained: callers that import services directly
# must receive the same indexed/batched DINOv2 behavior as the FastAPI app.
# The installer is idempotent via marker attributes on each patched function.
import sys as _sys
from .runtime_patches import install_runtime_patches as _install_runtime_patches

_install_runtime_patches(_sys.modules[__name__])
del _install_runtime_patches
del _sys
