"""Backend persistence helpers for DINOv3 runtime observations.

This module owns the boundary between ephemeral classifier observations and the
fingerprint-scoped feedback/registry SQLite stores. The detector is drained
exactly once per inference batch; the same observations are then fanned out to
human-feedback persistence and SpeciesRegistry discovery/accumulation.
"""
from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
from typing import Iterable, Sequence

from system.config import SUPPORTED_VIDEO_EXTENSIONS
from system.dinov3.feedback import FeedbackObservation
from system.dinov3.events import camera_id_for_path

logger = logging.getLogger(__name__)


def _parse_capture_time(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    for candidate in (text, text.replace("Z", "+00:00")):
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            pass
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y:%m:%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _indexed_value(values: Sequence | None, index: int):
    if values is None or index < 0 or index >= len(values):
        return None
    return values[index]


def _media_kind(path: Path) -> str:
    return "video" if path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS else "image"


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
    """Persist one inference batch of DINOv3 observations.

    All observations with opaque ids are archived in ``HumanFeedbackStore``.
    Existing SpeciesRegistry behavior is preserved for rejected base observations
    and registry-matched candidates. Accepted checkpoint predictions are never
    inserted into SpeciesRegistry.
    """

    feedback = getattr(detector, "dinov3_feedback", None)
    registry = getattr(detector, "dinov3_registry", None)
    drain = getattr(detector, "drain_dinov3_observations", None)
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
                    "Ignoring out-of-range DINOv3 observation index: %s",
                    index,
                )
                continue

            source_path = Path(roots[index])
            item = items[index]
            captured_at = _parse_capture_time(getattr(item, "date_taken", None))
            camera_id = camera_id_for_path(source_path, camera_root)
            frame_index = _indexed_value(frame_indices, index)
            timestamp = _indexed_value(timestamp_seconds, index)

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
                            if getattr(observation, "best_known_species", "")
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

            if registry is None:
                continue
            if observation.source == "checkpoint" and observation.accepted:
                continue
            if observation.registry_id is not None:
                registry.record_observation(
                    int(observation.registry_id),
                    observation.embedding,
                    camera_id=camera_id,
                    captured_at=captured_at,
                    source_path=str(source_path),
                )
            else:
                registry.record_unknown(
                    observation.embedding,
                    camera_id=camera_id,
                    captured_at=captured_at,
                    source_path=str(source_path),
                )
        except Exception as exc:  # noqa: BLE001 - persistence is non-fatal to inference
            logger.warning("Failed to persist DINOv3 observation: %s", exc)
