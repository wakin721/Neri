"""Backend helpers for opening the registry associated with a DINOv3 model."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import uuid

import cv2
import numpy as np

from system.config import SUPPORTED_VIDEO_EXTENSIONS
from system.dinov3.checkpoint import DinoV3Checkpoint, load_checkpoint
from system.dinov3.feedback import (
    FeedbackObservation,
    HumanFeedbackStore,
    feedback_path_for_registry,
)
from system.dinov3.registry import (
    RegistrationConditionError,
    SpeciesRegistry,
    registry_path_for_fingerprint,
)
from system.dinov3.runtime import DinoV3ManifestError, resolve_dinov3_manifest
from system.dinov3.state import default_dinov3_state_root


def load_checkpoint_for_model(classification_model_path: str) -> DinoV3Checkpoint:
    manifest_path = resolve_dinov3_manifest(classification_model_path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DinoV3ManifestError(f"Invalid DINOv3 manifest: {manifest_path}") from exc
    checkpoint_name = payload.get("checkpoint") if isinstance(payload, dict) else None
    if not isinstance(checkpoint_name, str) or not checkpoint_name.strip():
        raise DinoV3ManifestError("DINOv3 manifest checkpoint is missing")
    return load_checkpoint(manifest_path.parent / checkpoint_name)


def open_registry_for_model(classification_model_path: str) -> SpeciesRegistry:
    checkpoint = load_checkpoint_for_model(classification_model_path)
    path = registry_path_for_fingerprint(
        default_dinov3_state_root(),
        checkpoint.fingerprint,
    )
    return SpeciesRegistry(path, model_fingerprint=checkpoint.fingerprint)


def _checkpoint_species(checkpoint: DinoV3Checkpoint, value: str) -> str | None:
    wanted = str(value).strip().casefold()
    if not wanted:
        return None
    return next(
        (species for species in checkpoint.classes if str(species).strip().casefold() == wanted),
        None,
    )


def _candidate_feedback_observations(
    registry: SpeciesRegistry,
    registration_id: int,
    *,
    checkpoint: DinoV3Checkpoint,
) -> list[FeedbackObservation]:
    detail = registry.get(registration_id)
    if detail.status != "candidate":
        raise RegistrationConditionError("Only Candidate species can be changed this way")

    events = registry.list_events(registration_id)
    # Both methods are backed by _event_rows(... ORDER BY id), so event metadata
    # and embeddings have a stable one-to-one order.
    embeddings = registry._embeddings(registration_id)  # noqa: SLF001
    if not events or len(events) != len(embeddings):
        raise RegistrationConditionError("Candidate has no complete events to process")

    observations: list[FeedbackObservation] = []
    for event, embedding in zip(events, embeddings, strict=True):
        started_at = event.get("started_at")
        captured_at = None
        if started_at:
            try:
                captured_at = datetime.fromisoformat(str(started_at))
            except ValueError:
                captured_at = None
        source_path = str(event.get("source_path") or "")
        raw_bbox = event.get("bbox")
        bbox = (
            tuple(float(value) for value in raw_bbox)
            if isinstance(raw_bbox, list) and len(raw_bbox) == 4
            else (0.0, 0.0, 0.0, 0.0)
        )
        observations.append(
            FeedbackObservation(
                id=(
                    f"registry-event:{checkpoint.fingerprint}:"
                    f"{registration_id}:{int(event['id'])}"
                ),
                source_path=source_path,
                media_kind=(
                    "video"
                    if Path(source_path).suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
                    else "image"
                ),
                box_index=0,
                frame_index=(
                    int(event["frame_index"])
                    if event.get("frame_index") is not None
                    else None
                ),
                timestamp_seconds=(
                    float(event["timestamp_seconds"])
                    if event.get("timestamp_seconds") is not None
                    else None
                ),
                bbox=bbox,
                camera_id=str(event.get("camera_id") or "unknown"),
                captured_at=captured_at,
                predicted_species="Unknown",
                best_known_species=None,
                accepted=False,
                prediction_source="registry_candidate",
                registry_id=registration_id,
                known_score=0.0,
                threshold=float(checkpoint.threshold),
                embedding=embedding,
            )
        )
    return observations


def _open_candidate_feedback(
    registry: SpeciesRegistry,
    checkpoint: DinoV3Checkpoint,
) -> HumanFeedbackStore:
    return HumanFeedbackStore(
        feedback_path_for_registry(registry.path),
        model_fingerprint=checkpoint.fingerprint,
        checkpoint_classes=checkpoint.classes,
        threshold=checkpoint.threshold,
    )


def _feature_center_array(checkpoint: DinoV3Checkpoint) -> np.ndarray:
    center = checkpoint.feature_center
    if center is None:
        raise RegistrationConditionError("DINOv3 checkpoint is missing feature_center")
    if hasattr(center, "detach"):
        center = center.detach().cpu()
    if hasattr(center, "numpy"):
        center = center.numpy()
    result = np.asarray(center, dtype=np.float32)
    if result.shape != (768,) or not np.isfinite(result).all():
        raise RegistrationConditionError("DINOv3 checkpoint has an invalid feature_center")
    return result


def merge_registry_candidate_into_checkpoint(
    registry: SpeciesRegistry,
    classification_model_path: str,
    registration_id: int,
    checkpoint_species: str,
) -> dict[str, object]:
    """Move one Candidate's independent events into a same-name checkpoint overlay."""
    checkpoint = load_checkpoint_for_model(classification_model_path)
    detail = registry.get(registration_id)
    if detail.status != "candidate":
        raise RegistrationConditionError("Only Candidate species can be merged")
    canonical_species = _checkpoint_species(checkpoint, checkpoint_species)
    if canonical_species is None:
        raise RegistrationConditionError("Matching checkpoint species was not found")
    if detail.common_name.strip().casefold() != canonical_species.strip().casefold():
        raise RegistrationConditionError(
            "Candidate identity must match the checkpoint species before merging"
        )

    observations = _candidate_feedback_observations(
        registry,
        registration_id,
        checkpoint=checkpoint,
    )
    operation_id = f"registry-merge:{registration_id}:{uuid.uuid4().hex}"
    feedback = _open_candidate_feedback(registry, checkpoint)
    feedback_written = False
    try:
        for observation in observations:
            feedback.persist_observation(observation)
            feedback.record_feedback(
                observation.id,
                operation_id=operation_id,
                action="update",
                confirmed_species=canonical_species,
            )
            feedback_written = True
        feedback.recompute_species(canonical_species, _feature_center_array(checkpoint))
        try:
            registry.delete(registration_id)
        except Exception:
            if feedback_written:
                feedback.revert_operation(operation_id)
                feedback.recompute_species(
                    canonical_species,
                    _feature_center_array(checkpoint),
                )
            raise
    except Exception:
        if feedback_written:
            active = feedback._conn.execute(  # noqa: SLF001
                "SELECT reverted FROM feedback_operations WHERE operation_id=?",
                (operation_id,),
            ).fetchone()
            if active is not None and not bool(active[0]):
                feedback.revert_operation(operation_id)
                feedback.recompute_species(
                    canonical_species,
                    _feature_center_array(checkpoint),
                )
        raise
    finally:
        feedback.close()

    return {
        "merged": True,
        "registration_id": registration_id,
        "checkpoint_species": canonical_species,
        "event_count": len(observations),
    }


def discard_registry_candidate_as_empty(
    registry: SpeciesRegistry,
    classification_model_path: str,
    registration_id: int,
) -> dict[str, object]:
    """Mark every Candidate event as empty/false detection, then remove the Candidate."""
    checkpoint = load_checkpoint_for_model(classification_model_path)
    observations = _candidate_feedback_observations(
        registry,
        registration_id,
        checkpoint=checkpoint,
    )
    operation_id = f"registry-empty:{registration_id}:{uuid.uuid4().hex}"
    feedback = _open_candidate_feedback(registry, checkpoint)
    feedback_written = False
    try:
        for observation in observations:
            feedback.persist_observation(observation)
            feedback.record_feedback(
                observation.id,
                operation_id=operation_id,
                action="empty",
            )
            feedback_written = True
        try:
            registry.delete(registration_id)
        except Exception:
            if feedback_written:
                feedback.revert_operation(operation_id)
            raise
    except Exception:
        if feedback_written:
            active = feedback._conn.execute(  # noqa: SLF001
                "SELECT reverted FROM feedback_operations WHERE operation_id=?",
                (operation_id,),
            ).fetchone()
            if active is not None and not bool(active[0]):
                feedback.revert_operation(operation_id)
        raise
    finally:
        feedback.close()

    return {
        "discarded": True,
        "registration_id": registration_id,
        "event_count": len(observations),
    }


def _read_registry_frame(event: dict[str, object]) -> np.ndarray:
    source = Path(str(event.get("source_path") or "")).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"例图源文件不存在: {source}")
    if source.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
        capture = cv2.VideoCapture(str(source))
        try:
            frame_index = event.get("frame_index")
            timestamp_seconds = event.get("timestamp_seconds")
            if frame_index is not None:
                capture.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
            elif timestamp_seconds is not None:
                capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_seconds) * 1000.0)
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok or frame is None:
            raise FileNotFoundError(f"无法读取例图视频帧: {source}")
        return frame
    frame = cv2.imread(str(source))
    if frame is None:
        raise FileNotFoundError(f"无法读取例图: {source}")
    return frame


def render_media_example(
    *,
    source_path: str,
    bbox,
    frame_index: int | None = None,
    timestamp_seconds: float | None = None,
) -> bytes:
    values = list(bbox) if bbox is not None else []
    if len(values) != 4:
        raise FileNotFoundError("该历史事件没有裁切框信息")
    event = {
        "source_path": source_path,
        "bbox": values,
        "frame_index": frame_index,
        "timestamp_seconds": timestamp_seconds,
    }
    frame = _read_registry_frame(event)
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in values)
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width <= 0 or box_height <= 0:
        raise FileNotFoundError("该历史事件的裁切框无效")
    pad_w = int(box_width * 0.1)
    pad_h = int(box_height * 0.1)
    left = max(0, int(x1) - pad_w)
    top = max(0, int(y1) - pad_h)
    right = min(width, int(x2) + pad_w)
    bottom = min(height, int(y2) + pad_h)
    if right <= left or bottom <= top:
        raise FileNotFoundError("该历史事件的裁切框超出图像范围")
    crop = frame[top:bottom, left:right]
    crop_h, crop_w = crop.shape[:2]
    if crop_h != crop_w:
        size = max(crop_h, crop_w)
        pad_top = (size - crop_h) // 2
        pad_bottom = size - crop_h - pad_top
        pad_left = (size - crop_w) // 2
        pad_right = size - crop_w - pad_left
        crop = cv2.copyMakeBorder(
            crop,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=[114, 114, 114],
        )
    ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("裁切例图编码失败")
    return encoded.tobytes()


def render_registry_example(
    registry: SpeciesRegistry, registration_id: int, event_id: int
) -> bytes:
    event = next(
        (item for item in registry.list_events(registration_id) if item["id"] == event_id),
        None,
    )
    if event is None:
        raise FileNotFoundError("DINOv3 registry event not found")
    return render_media_example(
        source_path=str(event.get("source_path") or ""),
        bbox=event.get("bbox"),
        frame_index=(
            int(event["frame_index"]) if event.get("frame_index") is not None else None
        ),
        timestamp_seconds=(
            float(event["timestamp_seconds"])
            if event.get("timestamp_seconds") is not None
            else None
        ),
    )
