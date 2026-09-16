"""DINOv2 observation persistence and human-feedback helpers."""
from __future__ import annotations

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from system.dinov2.checkpoint import load_checkpoint
from system.dinov2.events import camera_id_for_path
from system.dinov2.feedback import (
    FeedbackObservation,
    HumanFeedbackStore,
    feedback_path_for_registry,
)
from system.dinov2.registry import registry_path_for_fingerprint
from system.dinov2.runtime import resolve_dinov2_manifest
from system.dinov2.state import default_dinov2_state_root

logger = logging.getLogger(__name__)
_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


def _parse_capture_time(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    for candidate in (text, text.replace("Z", "+00:00")):
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _indexed(values: Sequence | None, index: int):
    if values is None or index < 0 or index >= len(values):
        return None
    return values[index]


def _media_kind(path: Path) -> str:
    return "video" if path.suffix.lower() in _VIDEO_SUFFIXES else "image"


def _path_identity(value: str | Path) -> str:
    text = str(value or "").strip()
    try:
        return str(Path(text).expanduser().resolve()).replace("\\", "/").casefold()
    except Exception:
        return text.replace("\\", "/").casefold()


def persist_runtime_observations(
    detector,
    paths: list[Path],
    items: list,
    input_path: Path,
    *,
    source_paths: list[Path] | None = None,
    frame_indices: Sequence[int | None] | None = None,
    timestamp_seconds: Sequence[float | None] | None = None,
) -> None:
    """Persist one inference batch, draining DINOv2 observations exactly once."""
    feedback = getattr(detector, "dinov2_feedback", None)
    registry = getattr(detector, "dinov2_registry", None)
    drain = getattr(detector, "drain_dinov2_observations", None)
    if (feedback is None and registry is None) or not callable(drain):
        return
    observations = drain()
    if not observations:
        return
    roots = source_paths or paths
    camera_root = input_path if input_path.is_dir() else input_path.parent
    for observation in observations:
        try:
            index = int(observation.result_index)
            if (
                index < 0
                or index >= len(paths)
                or index >= len(items)
                or index >= len(roots)
            ):
                logger.warning(
                    "Ignoring out-of-range DINOv2 observation index: %s", index
                )
                continue
            source_path = Path(roots[index])
            item = items[index]
            captured_at = _parse_capture_time(getattr(item, "date_taken", None))
            camera_id = camera_id_for_path(source_path, camera_root)
            frame_index = _indexed(frame_indices, index)
            timestamp = _indexed(timestamp_seconds, index)
            observation_id = str(getattr(observation, "observation_id", "") or "")
            if feedback is not None and observation_id:
                feedback.persist_observation(
                    FeedbackObservation(
                        id=observation_id,
                        source_path=str(source_path),
                        media_kind=_media_kind(source_path),
                        box_index=int(observation.box_index),
                        frame_index=(
                            int(frame_index) if frame_index is not None else None
                        ),
                        timestamp_seconds=(
                            float(timestamp) if timestamp is not None else None
                        ),
                        bbox=tuple(float(value) for value in observation.bbox),
                        camera_id=camera_id,
                        captured_at=captured_at,
                        predicted_species=str(observation.species),
                        best_known_species=(
                            str(observation.best_known_species)
                            if observation.best_known_species
                            else None
                        ),
                        accepted=bool(observation.accepted),
                        prediction_source=str(observation.source),
                        registry_id=(
                            int(observation.registry_id)
                            if observation.registry_id is not None
                            else None
                        ),
                        known_score=float(observation.known_score),
                        threshold=float(observation.threshold),
                        embedding=observation.embedding,
                    )
                )
            if registry is None or observation.accepted:
                continue
            registry.record_unknown(
                observation.embedding,
                camera_id=camera_id,
                captured_at=captured_at,
                source_path=str(source_path),
                bbox=tuple(float(value) for value in observation.bbox),
                frame_index=int(frame_index) if frame_index is not None else None,
                timestamp_seconds=(
                    float(timestamp) if timestamp is not None else None
                ),
                candidate_kind=(
                    "new_mode_candidate"
                    if getattr(observation, "registry_action", "candidate")
                    == "new_mode_candidate"
                    else "candidate"
                ),
            )
        except Exception as exc:
            logger.warning("Failed to persist DINOv2 observation: %s", exc)


def _open_feedback_state(classification_model_path: str):
    """Open checkpoint-scoped DINOv2 feedback without loading the encoder."""
    manifest = resolve_dinov2_manifest(classification_model_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    checkpoint_name = payload.get("checkpoint")
    if not isinstance(checkpoint_name, str) or not checkpoint_name.strip():
        raise ValueError("DINOv2 manifest checkpoint is missing")
    checkpoint = load_checkpoint(manifest.parent / checkpoint_name)
    registry_path = registry_path_for_fingerprint(
        default_dinov2_state_root(), checkpoint.fingerprint
    )
    feedback = HumanFeedbackStore(
        feedback_path_for_registry(registry_path),
        model_fingerprint=checkpoint.fingerprint,
        checkpoint_classes=checkpoint.classes,
        rejection=checkpoint.rejection,
        prototype_norm_power=checkpoint.prototype_norm_power,
    )
    center = checkpoint.feature_center.numpy().astype(np.float32, copy=False)
    return feedback, center


def _affected_learning_species(record) -> set[str]:
    return {
        str(species)
        for species in (record.positive_species, record.hard_negative_species)
        if species and species != "Unknown"
    }


def record_registry_species_feedback(
    classification_model_path: str,
    observation: FeedbackObservation,
    operation_id: str,
    confirmed_species: str,
):
    """Apply explicit human species confirmation to DINOv2 Registry evidence."""
    from .dinov2_registry_service import open_registry_for_model

    feedback, _center = _open_feedback_state(classification_model_path)
    registry = open_registry_for_model(classification_model_path)
    try:
        updated, previous_common, previous_scientific = registry.record_human_species(
            observation.embedding,
            common_name=confirmed_species,
            camera_id=observation.camera_id,
            captured_at=observation.captured_at,
            source_path=observation.source_path,
            bbox=observation.bbox,
            frame_index=observation.frame_index,
            timestamp_seconds=observation.timestamp_seconds,
            preferred_entry_id=observation.registry_id,
        )
        try:
            feedback.record_registry_feedback(
                observation.id,
                operation_id=operation_id,
                registration_id=updated.id,
                previous_common_name=previous_common,
                previous_scientific_name=previous_scientific,
                confirmed_species=confirmed_species,
            )
        except Exception:
            registry.restore_identity(
                updated.id,
                common_name=previous_common,
                scientific_name=previous_scientific,
            )
            raise
        return updated
    finally:
        registry.close()
        feedback.close()
