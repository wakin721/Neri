"""Backend services for DINOv3 observation persistence and human feedback."""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from system.config import SUPPORTED_VIDEO_EXTENSIONS
from system.dinov3.feedback import (
    FeedbackObservation,
    HumanFeedbackStore,
    feedback_path_for_registry,
)
from system.dinov3.events import camera_id_for_path
from system.dinov3.registry import registry_path_for_fingerprint
from system.dinov3.state import default_dinov3_state_root

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
    """Persist one inference batch, draining classifier observations exactly once."""
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
                        frame_index=(int(frame_index) if frame_index is not None else None),
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
                    bbox=tuple(float(value) for value in observation.bbox),
                    frame_index=(int(frame_index) if frame_index is not None else None),
                    timestamp_seconds=(float(timestamp) if timestamp is not None else None),
                )
            else:
                registry.record_unknown(
                    observation.embedding,
                    camera_id=camera_id,
                    captured_at=captured_at,
                    source_path=str(source_path),
                    bbox=tuple(float(value) for value in observation.bbox),
                    frame_index=(int(frame_index) if frame_index is not None else None),
                    timestamp_seconds=(float(timestamp) if timestamp is not None else None),
                )
        except Exception as exc:  # noqa: BLE001 - inference persistence is non-fatal
            logger.warning("Failed to persist DINOv3 observation: %s", exc)


def _path_identity(value: str | Path) -> str:
    text = str(value or "").strip()
    try:
        return str(Path(text).expanduser().resolve()).replace("\\", "/").casefold()
    except Exception:
        return text.replace("\\", "/").casefold()


def _open_feedback_state(classification_model_path: str):
    """Open checkpoint-scoped feedback state without loading the DINO encoder."""
    from system.dinov3.checkpoint import load_checkpoint
    from system.dinov3.runtime import resolve_dinov3_manifest

    manifest = resolve_dinov3_manifest(classification_model_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    checkpoint_name = payload.get("checkpoint")
    if not isinstance(checkpoint_name, str) or not checkpoint_name.strip():
        raise ValueError("DINOv3 manifest checkpoint is missing")
    checkpoint = load_checkpoint(manifest.parent / checkpoint_name)
    registry_path = registry_path_for_fingerprint(
        default_dinov3_state_root(), checkpoint.fingerprint
    )
    feedback = HumanFeedbackStore(
        feedback_path_for_registry(registry_path),
        model_fingerprint=checkpoint.fingerprint,
        checkpoint_classes=checkpoint.classes,
        threshold=checkpoint.threshold,
    )
    feature_center = checkpoint.feature_center
    if feature_center is None:
        feedback.close()
        raise ValueError("DINOv3 checkpoint is missing feature_center")
    if hasattr(feature_center, "numpy"):
        feature_center = feature_center.numpy()
    center = np.asarray(feature_center, dtype=np.float32)
    return feedback, center


def _assign_registry_species(
    feedback: HumanFeedbackStore,
    classification_model_path: str,
    observation: FeedbackObservation,
    *,
    operation_id: str,
    confirmed_species: str,
):
    from .dinov3_registry_service import open_registry_for_model

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


def record_registry_species_feedback(
    classification_model_path: str,
    observation: FeedbackObservation,
    operation_id: str,
    confirmed_species: str,
):
    feedback, _feature_center = _open_feedback_state(classification_model_path)
    try:
        return _assign_registry_species(
            feedback,
            classification_model_path,
            observation,
            operation_id=operation_id,
            confirmed_species=confirmed_species,
        )
    finally:
        feedback.close()


def explain_feedback_observation(
    classification_model_path: str,
    observation_id: str,
) -> dict:
    from system.dinov3.classifier import DinoV3Classifier
    from .dinov3_registry_service import load_checkpoint_for_model, open_registry_for_model

    feedback, _feature_center = _open_feedback_state(classification_model_path)
    registry = open_registry_for_model(classification_model_path)
    try:
        observation = feedback.get_observation(observation_id)
        checkpoint = load_checkpoint_for_model(classification_model_path)
        classifier = DinoV3Classifier(checkpoint, feedback=feedback, registry=registry)
        result = classifier.explain_feature(observation.embedding)
        result["current_example_available"] = bool(
            observation.source_path and Path(observation.source_path).expanduser().is_file()
        )
        nearest_example = None
        nearest = result.get("nearest_species") or []
        if nearest:
            closest = nearest[0]
            registry_id = closest.get("registry_id")
            if registry_id is not None:
                try:
                    events = registry.list_events(int(registry_id))
                except KeyError:
                    events = []
                event = next((item for item in events if item.get("has_example")), None)
                if event is not None:
                    nearest_example = {
                        "kind": "registry",
                        "species": str(closest.get("name") or ""),
                        "registration_id": int(registry_id),
                        "event_id": int(event["id"]),
                    }
            if nearest_example is None:
                representative_id = feedback.representative_observation_id(
                    str(closest.get("name") or "")
                )
                if representative_id:
                    nearest_example = {
                        "kind": "observation",
                        "species": str(closest.get("name") or ""),
                        "observation_id": representative_id,
                    }
        result["nearest_example"] = nearest_example
        return result
    finally:
        registry.close()
        feedback.close()


def render_feedback_observation_example(
    classification_model_path: str,
    observation_id: str,
) -> bytes:
    from .dinov3_registry_service import render_media_example

    feedback, _feature_center = _open_feedback_state(classification_model_path)
    try:
        observation = feedback.get_observation(observation_id)
        return render_media_example(
            source_path=observation.source_path,
            bbox=observation.bbox,
            frame_index=observation.frame_index,
            timestamp_seconds=observation.timestamp_seconds,
        )
    finally:
        feedback.close()

def _raw_box_for_observation(detection_data: dict, observation_id: str) -> dict:
    boxes = detection_data.get("检测框")
    if not isinstance(boxes, list):
        raise KeyError(observation_id)
    for box in boxes:
        if not isinstance(box, dict):
            continue
        candidates = box.get("候选项")
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if (
                isinstance(candidate, dict)
                and str(candidate.get("observation_id") or "") == observation_id
            ):
                return box
    raise KeyError(observation_id)


def _recompute_detection_summary(detection_data: dict) -> None:
    counts: Counter[str] = Counter()
    boxes = detection_data.get("检测框")
    if isinstance(boxes, list):
        for box in boxes:
            if not isinstance(box, dict):
                continue
            species = str(box.get("物种") or "").strip()
            if species in {"", "空", "Unknown", "unknown"}:
                continue
            counts[species] += 1
    detection_data["物种名称"] = ",".join(counts.keys()) if counts else "空"
    detection_data["物种数量"] = (
        ",".join(str(counts[name]) for name in counts) if counts else "空"
    )


def _ensure_ecological_journal(feedback: HumanFeedbackStore) -> None:
    feedback._conn.execute(  # noqa: SLF001 - same SQLite ownership boundary
        """
        CREATE TABLE IF NOT EXISTS feedback_ecological_operations(
          operation_id TEXT NOT NULL,
          observation_id TEXT NOT NULL,
          input_path TEXT NOT NULL,
          file_path TEXT NOT NULL,
          detection_json TEXT NOT NULL,
          validated INTEGER,
          PRIMARY KEY(operation_id, observation_id)
        )
        """
    )
    feedback._conn.commit()  # noqa: SLF001


def _journal_ecological_state(
    feedback: HumanFeedbackStore,
    *,
    operation_id: str,
    observation_id: str,
    input_path: str,
    file_path: str,
    detection_data: dict,
    validated: bool | None,
) -> None:
    _ensure_ecological_journal(feedback)
    validated_value = None if validated is None else int(bool(validated))
    feedback._conn.execute(  # noqa: SLF001
        """
        INSERT OR IGNORE INTO feedback_ecological_operations(
          operation_id,observation_id,input_path,file_path,detection_json,validated
        ) VALUES(?,?,?,?,?,?)
        """,
        (
            operation_id,
            observation_id,
            input_path,
            file_path,
            json.dumps(detection_data, ensure_ascii=False),
            validated_value,
        ),
    )
    feedback._conn.commit()  # noqa: SLF001


def _affected_learning_species(record) -> set[str]:
    return {
        str(species)
        for species in (record.positive_species, record.hard_negative_species)
        if species and species != "Unknown"
    }


def apply_box_feedback(request):
    """Apply one explicit box-level ecological correction and learning event."""
    from . import services

    input_path = Path(request.input_path).expanduser().resolve()
    file_path = Path(request.file_path).expanduser().resolve()
    if not file_path.is_file():
        raise ValueError(f"文件不存在: {file_path}")
    if not input_path.exists():
        raise ValueError(f"输入路径不存在: {input_path}")

    feedback, feature_center = _open_feedback_state(request.classification_model_path)
    try:
        observation = feedback.get_observation(request.observation_id)
        if _path_identity(observation.source_path) != _path_identity(file_path):
            raise ValueError("DINOv3 observation does not belong to the selected file")

        current_item = services._reload_validation_item(file_path, input_path)
        detection_data = services._load_detection_data_for_path(
            file_path,
            services._detection_db_search_roots(input_path, None),
        )
        if not detection_data:
            raise KeyError(request.observation_id)
        target_box = _raw_box_for_observation(detection_data, request.observation_id)

        action = str(request.action)
        species_name = (
            str(request.species_name).strip() if request.species_name is not None else None
        )
        if action == "update":
            if not species_name:
                raise ValueError("species_name is required for update feedback")
            target_box["物种"] = species_name
        elif action == "empty":
            target_box["物种"] = "空"
        elif action not in {"correct", "unverified"}:
            raise ValueError(f"Unsupported DINOv3 feedback action: {action}")

        if action in {"update", "empty"}:
            _recompute_detection_summary(detection_data)
            _journal_ecological_state(
                feedback,
                operation_id=request.feedback_operation_id,
                observation_id=request.observation_id,
                input_path=str(input_path),
                file_path=str(file_path),
                detection_data=current_item.detection_data,
                validated=current_item.validated,
            )
            services._persist_validation_update(
                file_path,
                input_path,
                detection_data,
                current_item.validated,
            )

        registry_entry = None
        if (
            action == "update"
            and species_name
            and species_name not in feedback.checkpoint_classes
        ):
            registry_entry = _assign_registry_species(
                feedback,
                request.classification_model_path,
                observation,
                operation_id=request.feedback_operation_id,
                confirmed_species=species_name,
            )
            affected: set[str] = set()
        else:
            record = feedback.record_feedback(
                request.observation_id,
                operation_id=request.feedback_operation_id,
                action=action,
                confirmed_species=species_name,
            )
            affected = _affected_learning_species(record)
            for species in sorted(affected):
                feedback.recompute_species(species, feature_center)

        item = services._reload_validation_item(file_path, input_path)
        return {
            "item": item,
            "operation_id": request.feedback_operation_id,
            "affected_species": sorted(affected),
            "registry_id": registry_entry.id if registry_entry is not None else None,
        }
    finally:
        feedback.close()


def _restore_registry_assignments(registry, assignments) -> None:
    """Restore only identities still owned by the historical feedback operation."""
    for assignment in assignments:
        if not bool(assignment.get("identity_restore_allowed", True)):
            continue
        registration_id = int(assignment["registration_id"])
        try:
            current = registry.get(registration_id)
        except KeyError:
            continue
        if current.common_name != assignment["assigned_common_name"]:
            continue
        registry.restore_identity(
            registration_id,
            common_name=str(assignment["previous_common_name"]),
            scientific_name=str(assignment["previous_scientific_name"]),
        )


def revert_feedback_operation(request):
    """Revert both feedback learning state and ecological box edits for an operation."""
    from . import services

    feedback, feature_center = _open_feedback_state(request.classification_model_path)
    try:
        registry_assignments = feedback.registry_assignments_for_restore(
            request.feedback_operation_id
        )
        affected = feedback.revert_operation(request.feedback_operation_id)
        for species in sorted(affected):
            if species in feedback.checkpoint_classes:
                feedback.recompute_species(species, feature_center)

        if registry_assignments:
            from .dinov3_registry_service import open_registry_for_model

            registry = open_registry_for_model(request.classification_model_path)
            try:
                _restore_registry_assignments(registry, registry_assignments)
            finally:
                registry.close()

        _ensure_ecological_journal(feedback)
        rows = feedback._conn.execute(  # noqa: SLF001
            """
            SELECT * FROM feedback_ecological_operations
            WHERE operation_id=? ORDER BY observation_id
            """,
            (request.feedback_operation_id,),
        ).fetchall()
        for row in rows:
            input_path = Path(str(row["input_path"]))
            file_path = Path(str(row["file_path"]))
            if not file_path.is_file():
                continue
            before = json.loads(row["detection_json"])
            validated_raw = row["validated"]
            validated = None if validated_raw is None else bool(validated_raw)
            services._persist_validation_update(
                file_path,
                input_path,
                before,
                validated,
            )

        return {
            "operation_id": request.feedback_operation_id,
            "affected_species": sorted(affected),
        }
    finally:
        feedback.close()
