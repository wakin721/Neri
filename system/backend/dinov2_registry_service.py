"""Backend helpers for DINOv2 checkpoint-scoped Registry state."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import uuid

import cv2
import numpy as np

from system.config import SUPPORTED_VIDEO_EXTENSIONS
from system.dinov2.checkpoint import DinoV2Checkpoint, load_checkpoint
from system.dinov2.feedback import FeedbackObservation, HumanFeedbackStore, feedback_path_for_registry
from system.dinov2.registry import RegistrationConditionError, SpeciesRegistry, registry_path_for_fingerprint
from system.dinov2.registry_examples import (
    persist_feedback_observation_example,
    persist_registry_event_example,
    read_registry_event_example,
    store_feedback_observation_example,
)
from system.dinov2.runtime import DinoV2ManifestError, resolve_dinov2_manifest
from system.dinov2.state import default_dinov2_state_root


def load_checkpoint_for_model(classification_model_path: str) -> DinoV2Checkpoint:
    manifest = resolve_dinov2_manifest(classification_model_path)
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DinoV2ManifestError(f"Invalid DINOv2 manifest: {manifest}") from exc
    checkpoint_name = payload.get("checkpoint") if isinstance(payload, dict) else None
    if not isinstance(checkpoint_name, str) or not checkpoint_name.strip():
        raise DinoV2ManifestError("DINOv2 manifest checkpoint is missing")
    return load_checkpoint(manifest.parent / checkpoint_name)


def open_registry_for_model(classification_model_path: str) -> SpeciesRegistry:
    checkpoint = load_checkpoint_for_model(classification_model_path)
    return SpeciesRegistry(
        registry_path_for_fingerprint(default_dinov2_state_root(), checkpoint.fingerprint),
        model_fingerprint=checkpoint.fingerprint,
        prototype_norm_power=checkpoint.prototype_norm_power,
    )


def _checkpoint_species(checkpoint: DinoV2Checkpoint, value: str) -> str | None:
    wanted = str(value).strip().casefold()
    if not wanted:
        return None
    return next((species for species in checkpoint.classes if str(species).strip().casefold() == wanted), None)


def _candidate_feedback_observations(registry: SpeciesRegistry, registration_id: int, *, checkpoint: DinoV2Checkpoint) -> list[FeedbackObservation]:
    detail = registry.get(registration_id)
    if detail.status != "candidate":
        raise RegistrationConditionError("Only Candidate species can be changed this way")
    events = registry.list_events(registration_id)
    embeddings = registry._embeddings(registration_id)  # noqa: SLF001
    if not events or len(events) != len(embeddings):
        raise RegistrationConditionError("Candidate has no complete events to process")
    observations: list[FeedbackObservation] = []
    for event, embedding in zip(events, embeddings, strict=True):
        captured_at = None
        started_at = event.get("started_at")
        if started_at:
            try:
                captured_at = datetime.fromisoformat(str(started_at))
            except ValueError:
                pass
        source_path = str(event.get("source_path") or "")
        raw_bbox = event.get("bbox")
        bbox = tuple(float(v) for v in raw_bbox) if isinstance(raw_bbox, list) and len(raw_bbox) == 4 else (0.0, 0.0, 0.0, 0.0)
        observations.append(FeedbackObservation(
            id=f"registry-event:{checkpoint.fingerprint}:{registration_id}:{int(event['id'])}",
            source_path=source_path,
            media_kind="video" if Path(source_path).suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS else "image",
            box_index=0,
            frame_index=int(event["frame_index"]) if event.get("frame_index") is not None else None,
            timestamp_seconds=float(event["timestamp_seconds"]) if event.get("timestamp_seconds") is not None else None,
            bbox=bbox,
            camera_id=str(event.get("camera_id") or "unknown"),
            captured_at=captured_at,
            predicted_species="Unknown",
            best_known_species=None,
            accepted=False,
            prediction_source="registry_candidate",
            registry_id=registration_id,
            known_score=0.0,
            threshold=float(checkpoint.rejection.cosine_threshold),
            embedding=embedding,
        ))
    return observations


def _open_candidate_feedback(registry: SpeciesRegistry, checkpoint: DinoV2Checkpoint) -> HumanFeedbackStore:
    return HumanFeedbackStore(
        feedback_path_for_registry(registry.path),
        model_fingerprint=checkpoint.fingerprint,
        checkpoint_classes=checkpoint.classes,
        rejection=checkpoint.rejection,
        prototype_norm_power=checkpoint.prototype_norm_power,
    )


def _feature_center_array(checkpoint: DinoV2Checkpoint) -> np.ndarray:
    center = checkpoint.feature_center.detach().cpu().numpy() if hasattr(checkpoint.feature_center, "detach") else np.asarray(checkpoint.feature_center)
    result = np.asarray(center, dtype=np.float32)
    if result.shape != (768,) or not np.isfinite(result).all():
        raise RegistrationConditionError("DINOv2 checkpoint has an invalid feature_center")
    return result


def merge_registry_candidate_into_checkpoint(registry: SpeciesRegistry, classification_model_path: str, registration_id: int, checkpoint_species: str) -> dict[str, object]:
    checkpoint = load_checkpoint_for_model(classification_model_path)
    detail = registry.get(registration_id)
    if detail.status != "candidate":
        raise RegistrationConditionError("Only Candidate species can be merged")
    canonical = _checkpoint_species(checkpoint, checkpoint_species)
    if canonical is None:
        raise RegistrationConditionError("Matching checkpoint species was not found")
    if detail.common_name.strip().casefold() != canonical.strip().casefold():
        raise RegistrationConditionError("Candidate identity must match the checkpoint species before merging")
    observations = _candidate_feedback_observations(registry, registration_id, checkpoint=checkpoint)
    operation_id = f"registry-merge:{registration_id}:{uuid.uuid4().hex}"
    feedback = _open_candidate_feedback(registry, checkpoint)
    written = False
    try:
        for observation in observations:
            feedback.persist_observation(observation)
            event_id = int(observation.id.rsplit(":", 1)[1])
            cached = read_registry_event_example(registry.path, event_id)
            if cached is not None:
                store_feedback_observation_example(feedback.path, observation.id, cached)
            else:
                persist_feedback_observation_example(feedback.path, observation.id, source_path=observation.source_path, bbox=observation.bbox, frame_index=observation.frame_index, timestamp_seconds=observation.timestamp_seconds)
            feedback.record_feedback(observation.id, operation_id=operation_id, action="update", confirmed_species=canonical)
            written = True
        feedback.recompute_species(canonical, _feature_center_array(checkpoint))
        registry.delete(registration_id)
    except Exception:
        if written:
            try:
                feedback.revert_operation(operation_id)
                feedback.recompute_species(canonical, _feature_center_array(checkpoint))
            except Exception:
                pass
        raise
    finally:
        feedback.close()
    return {"merged": True, "registration_id": registration_id, "checkpoint_species": canonical, "event_count": len(observations)}


def discard_registry_candidate_as_empty(registry: SpeciesRegistry, classification_model_path: str, registration_id: int) -> dict[str, object]:
    checkpoint = load_checkpoint_for_model(classification_model_path)
    observations = _candidate_feedback_observations(registry, registration_id, checkpoint=checkpoint)
    operation_id = f"registry-empty:{registration_id}:{uuid.uuid4().hex}"
    feedback = _open_candidate_feedback(registry, checkpoint)
    written = False
    try:
        for observation in observations:
            feedback.persist_observation(observation)
            feedback.record_feedback(observation.id, operation_id=operation_id, action="empty")
            written = True
        registry.delete(registration_id)
    except Exception:
        if written:
            try:
                feedback.revert_operation(operation_id)
            except Exception:
                pass
        raise
    finally:
        feedback.close()
    return {"discarded": True, "registration_id": registration_id, "event_count": len(observations)}


def _read_registry_frame(event: dict[str, object]) -> np.ndarray:
    source = Path(str(event.get("source_path") or "")).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"例图源文件不存在: {source}")
    if source.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
        capture = cv2.VideoCapture(str(source))
        try:
            if event.get("frame_index") is not None:
                capture.set(cv2.CAP_PROP_POS_FRAMES, float(event["frame_index"]))
            elif event.get("timestamp_seconds") is not None:
                capture.set(cv2.CAP_PROP_POS_MSEC, float(event["timestamp_seconds"]) * 1000.0)
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


def render_media_example(*, source_path: str, bbox, frame_index: int | None = None, timestamp_seconds: float | None = None) -> bytes:
    values = list(bbox) if bbox is not None else []
    if len(values) != 4:
        raise FileNotFoundError("该历史事件没有裁切框信息")
    frame = _read_registry_frame({"source_path": source_path, "bbox": values, "frame_index": frame_index, "timestamp_seconds": timestamp_seconds})
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = map(float, values)
    box_width, box_height = x2 - x1, y2 - y1
    if box_width <= 0 or box_height <= 0:
        raise FileNotFoundError("该历史事件的裁切框无效")
    pad_w, pad_h = int(box_width * 0.1), int(box_height * 0.1)
    left, top = max(0, int(x1) - pad_w), max(0, int(y1) - pad_h)
    right, bottom = min(width, int(x2) + pad_w), min(height, int(y2) + pad_h)
    if right <= left or bottom <= top:
        raise FileNotFoundError("该历史事件的裁切框超出图像范围")
    crop = frame[top:bottom, left:right]
    ch, cw = crop.shape[:2]
    if ch != cw:
        size = max(ch, cw)
        pt, pl = (size - ch) // 2, (size - cw) // 2
        crop = cv2.copyMakeBorder(crop, pt, size - ch - pt, pl, size - cw - pl, cv2.BORDER_CONSTANT, value=[114, 114, 114])
    ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("裁切例图编码失败")
    return encoded.tobytes()


def render_registry_example(registry: SpeciesRegistry, registration_id: int, event_id: int) -> bytes:
    cached = read_registry_event_example(registry.path, event_id)
    if cached is not None:
        return cached
    event = next((item for item in registry.list_events(registration_id) if item["id"] == event_id), None)
    if event is None:
        raise FileNotFoundError("DINOv2 registry event not found")
    bbox = event.get("bbox")
    persist_registry_event_example(registry.path, event_id, source_path=str(event.get("source_path") or ""), bbox=bbox or (), frame_index=int(event["frame_index"]) if event.get("frame_index") is not None else None, timestamp_seconds=float(event["timestamp_seconds"]) if event.get("timestamp_seconds") is not None else None)
    cached = read_registry_event_example(registry.path, event_id)
    return cached if cached is not None else render_media_example(source_path=str(event.get("source_path") or ""), bbox=bbox, frame_index=int(event["frame_index"]) if event.get("frame_index") is not None else None, timestamp_seconds=float(event["timestamp_seconds"]) if event.get("timestamp_seconds") is not None else None)
