"""Batched runtime persistence for DINOv3 observations."""
from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
import time
from typing import Sequence

from system.config import SUPPORTED_VIDEO_EXTENSIONS
from system.dinov3.events import camera_id_for_path
from system.dinov3.feedback import FeedbackObservation
from system.dinov3.registry_examples import (
    example_frame_cache,
    persist_feedback_observation_example,
)

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
    """Persist one inference batch with one feedback DB write when possible."""
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
    prepared: list[dict[str, object]] = []
    feedback_observations: list[FeedbackObservation] = []
    prepare_started = time.perf_counter()

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
            observation_id = str(
                getattr(observation, "observation_id", "") or ""
            )

            feedback_observation = None
            if feedback is not None and observation_id:
                feedback_observation = FeedbackObservation(
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
                feedback_observations.append(feedback_observation)

            prepared.append(
                {
                    "runtime": observation,
                    "feedback": feedback_observation,
                    "source_path": source_path,
                    "camera_id": camera_id,
                    "captured_at": captured_at,
                    "frame_index": frame_index,
                    "timestamp": timestamp,
                }
            )
        except Exception as exc:  # noqa: BLE001 - persistence is non-fatal
            logger.warning("Failed to prepare DINOv3 observation: %s", exc)

    db_started = time.perf_counter()
    if feedback is not None and feedback_observations:
        persist_many = getattr(feedback, "persist_observations", None)
        if callable(persist_many):
            try:
                persist_many(tuple(feedback_observations))
            except Exception as exc:  # noqa: BLE001 - retain per-row fallback
                logger.warning(
                    "Batch DINOv3 observation persistence failed; falling back: %s",
                    exc,
                )
                for observation in feedback_observations:
                    try:
                        feedback.persist_observation(observation)
                    except Exception as row_exc:  # noqa: BLE001
                        logger.warning(
                            "Failed to persist DINOv3 observation %s: %s",
                            observation.id,
                            row_exc,
                        )
        else:
            for observation in feedback_observations:
                try:
                    feedback.persist_observation(observation)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to persist DINOv3 observation %s: %s",
                        observation.id,
                        exc,
                    )
    db_elapsed = time.perf_counter() - db_started

    example_elapsed = 0.0
    registry_elapsed = 0.0
    with example_frame_cache():
        for context in prepared:
            runtime_observation = context["runtime"]
            feedback_observation = context["feedback"]
            source_path = context["source_path"]
            frame_index = context["frame_index"]
            timestamp = context["timestamp"]

            if feedback is not None and feedback_observation is not None:
                started = time.perf_counter()
                try:
                    persist_feedback_observation_example(
                        feedback.path,
                        feedback_observation.id,
                        source_path=source_path,
                        bbox=feedback_observation.bbox,
                        frame_index=(
                            int(frame_index) if frame_index is not None else None
                        ),
                        timestamp_seconds=(
                            float(timestamp) if timestamp is not None else None
                        ),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to persist DINOv3 feedback example %s: %s",
                        feedback_observation.id,
                        exc,
                    )
                example_elapsed += time.perf_counter() - started

            if registry is None:
                continue
            if runtime_observation.accepted:
                continue
            started = time.perf_counter()
            try:
                registry_kwargs = {
                    "camera_id": context["camera_id"],
                    "captured_at": context["captured_at"],
                    "source_path": str(source_path),
                    "bbox": tuple(
                        float(value) for value in runtime_observation.bbox
                    ),
                    "frame_index": (
                        int(frame_index) if frame_index is not None else None
                    ),
                    "timestamp_seconds": (
                        float(timestamp) if timestamp is not None else None
                    ),
                }
                # Runtime inference may collect Candidate evidence, but it must
                # never mutate an existing learned prototype. Only the explicit
                # human-feedback path owns that authority.
                registry.record_unknown(
                    runtime_observation.embedding,
                    candidate_kind=(
                        "new_mode_candidate"
                        if getattr(
                            runtime_observation,
                            "registry_action",
                            "candidate",
                        )
                        == "new_mode_candidate"
                        else "candidate"
                    ),
                    **registry_kwargs,
                )
            except Exception as exc:  # noqa: BLE001 - inference remains non-fatal
                logger.warning("Failed to persist DINOv3 Registry evidence: %s", exc)
            registry_elapsed += time.perf_counter() - started

    logger.info(
        (
            "DINOv3 persistence timing: observations=%d prepare=%.3fs db=%.3fs "
            "examples=%.3fs registry=%.3fs"
        ),
        len(prepared),
        db_started - prepare_started,
        db_elapsed,
        example_elapsed,
        registry_elapsed,
    )


setattr(persist_runtime_observations, "_neri_batched_runtime_persistence", True)
