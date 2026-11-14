"""Deterministic DINOv3 observations: explanation, examples, and feedback.

The joint Neri1.1 model stores its training bank inside the checkpoint.  This
module reads that bank directly for the Species Validation feature-space view,
so explaining an observation never has to construct the heavy PyTorch ViT
encoder.
"""
from __future__ import annotations

import base64
from contextlib import closing
import json
import logging
from pathlib import Path
import sqlite3
import threading
from typing import Any, Mapping, Sequence

import numpy as np

from system.dino.state import default_state_root


logger = logging.getLogger(__name__)


_DINOV3_OBSERVATION_PREFIX = "dinov3:"
_UNKNOWN_SPECIES = {"", "空", "Unknown", "unknown", "拒识/Unknown"}
_BOX_METRIC_KEYS = (
    "ridge_score",
    "raw_ridge_score",
    "within_score",
    "box_known_score",
    "sequence_known_score",
)

_MODEL_CACHE: dict[tuple[str, int], tuple[Any, dict[str, np.ndarray]]] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def make_dinov3_observation_id(
    file_path: str | Path,
    box_index: int,
    bbox: Sequence[float] | None = None,
) -> str:
    """Return a stable, source-path-bound observation id."""
    values = list(bbox) if bbox is not None else []
    payload = {
        "p": str(file_path),
        "i": int(box_index),
        "b": [round(float(value), 2) for value in values[:4]],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    token = base64.urlsafe_b64encode(encoded).decode("ascii").rstrip("=")
    return f"{_DINOV3_OBSERVATION_PREFIX}{token}"


def is_dinov3_observation_id(observation_id: str | None) -> bool:
    return str(observation_id or "").startswith(_DINOV3_OBSERVATION_PREFIX)


def parse_dinov3_observation_id(observation_id: str) -> tuple[Path, int, list[float]]:
    try:
        text = str(observation_id or "")
        if not text.startswith(_DINOV3_OBSERVATION_PREFIX):
            raise ValueError("not a DINOv3 observation id")
        token = text[len(_DINOV3_OBSERVATION_PREFIX):]
        padded = token + "=" * (-len(token) % 4)
        decoded = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
        payload = json.loads(decoded)
        path = Path(str(payload["p"])).expanduser().resolve()
        index = int(payload.get("i", 0))
        bbox = [float(value) for value in payload.get("b", [])]
        return path, index, bbox
    except Exception as exc:  # noqa: BLE001 - invalid ids use KeyError.
        raise KeyError(observation_id) from exc


def is_dinov3_box_mapping(
    raw_box: Mapping[str, Any],
    detection_data: Mapping[str, Any] | None = None,
) -> bool:
    """Recognize a detection box produced by the Neri1.1 Ridge/Within path."""
    if not isinstance(raw_box, Mapping):
        return False
    if any(key in raw_box for key in _BOX_METRIC_KEYS):
        return True
    candidates = raw_box.get("候选项") or raw_box.get("candidates")
    if isinstance(candidates, list) and candidates:
        first = candidates[0]
        if isinstance(first, Mapping) and any(key in first for key in _BOX_METRIC_KEYS):
            return True
    return bool(isinstance(detection_data, Mapping) and detection_data.get("sequence_completed") is True)


def _unit(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm <= 1e-12:
        return np.zeros_like(array, dtype=np.float64)
    return array / norm


def _build_centroids(model: Any) -> dict[str, np.ndarray]:
    classes = [str(value) for value in (model.classes if model.classes is not None else [])]
    centroids: dict[str, np.ndarray] = {}
    updater = getattr(model, "within_updater", None)
    if updater is not None:
        try:
            bank, labels, _cameras = updater.bank()
            bank_array = np.asarray(bank, dtype=np.float64)
            label_array = np.asarray(labels)
            for name in classes:
                mask = label_array == name
                if np.any(mask):
                    centroids[name] = _unit(bank_array[mask].mean(axis=0))
        except Exception:  # noqa: BLE001 - fall back to ridge class means.
            centroids.clear()

    means = getattr(getattr(model, "ridge_head", None), "class_means_", None)
    if means is not None:
        mean_array = np.asarray(means, dtype=np.float64)
        for index, name in enumerate(classes):
            if name in centroids or index >= len(mean_array):
                continue
            centroids[name] = _unit(mean_array[index])
    return centroids


def _model_and_centroids(checkpoint_path: Path) -> tuple[Any, dict[str, np.ndarray]]:
    try:
        key = (str(checkpoint_path.resolve()), checkpoint_path.stat().st_mtime_ns)
    except OSError:
        key = (str(checkpoint_path), 0)
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is None:
            from system.dinov3.model import DinoV3RidgeWithinModel

            model = DinoV3RidgeWithinModel.load(checkpoint_path)
            cached = (model, _build_centroids(model))
            _MODEL_CACHE[key] = cached
        return cached


def _first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
    if not isinstance(mapping, Mapping):
        return None
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _box_bbox(raw_box: Mapping[str, Any]) -> list[float]:
    value = _first_present(raw_box, "边界框", "bbox", "box", "xyxy")
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return []
    try:
        return [float(item) for item in value[:4]]
    except (TypeError, ValueError):
        return []


def _path_key(value: str | Path) -> str:
    try:
        return str(Path(str(value)).expanduser().resolve()).replace("\\", "/").casefold()
    except Exception:  # noqa: BLE001 - identity comparison is best effort.
        return str(value or "").replace("\\", "/").casefold()


def _primary_candidate(raw_box: Mapping[str, Any]) -> Mapping[str, Any]:
    candidates = raw_box.get("候选项") or raw_box.get("candidates")
    if isinstance(candidates, list) and candidates and isinstance(candidates[0], Mapping):
        return candidates[0]
    return {}


def _locate_raw_box(
    boxes: list[Any],
    observation_id: str,
    decoded: tuple[Path, int, list[float]] | None,
) -> Mapping[str, Any]:
    for box in boxes:
        if not isinstance(box, Mapping):
            continue
        if str(box.get("observation_id") or "") == observation_id:
            return box
        candidates = box.get("候选项") or box.get("candidates")
        if isinstance(candidates, list):
            for candidate in candidates:
                if isinstance(candidate, Mapping) and str(candidate.get("observation_id") or "") == observation_id:
                    return box
    if decoded is not None:
        _path, box_index, bbox = decoded
        if 0 <= box_index < len(boxes) and isinstance(boxes[box_index], Mapping):
            return boxes[box_index]
        if bbox:
            best: Mapping[str, Any] | None = None
            best_distance: float | None = None
            for box in boxes:
                if not isinstance(box, Mapping):
                    continue
                other = _box_bbox(box)
                if len(other) < 4:
                    continue
                distance = sum(abs(left - right) for left, right in zip(bbox[:4], other))
                if best_distance is None or distance < best_distance:
                    best = box
                    best_distance = distance
            if best is not None:
                return best
    raise KeyError(observation_id)


def _candidate_entries(raw_box: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    entries: list[tuple[str, Mapping[str, Any]]] = []
    seen: set[str] = set()
    candidates = raw_box.get("候选项") or raw_box.get("candidates")
    if not isinstance(candidates, list):
        return entries
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        name = str(candidate.get("name") or "").strip()
        if not name or name in _UNKNOWN_SPECIES or name in seen:
            continue
        if _first_present(candidate, "within_score", "ridge_score") is None:
            continue
        seen.add(name)
        entries.append((name, candidate))
    return entries


def _candidate_score(candidate: Mapping[str, Any]) -> float:
    value = _first_present(candidate, "within_score", "ridge_score")
    return float(value) if value is not None else 0.5


def _choose_species(
    best_known: str,
    entries: list[tuple[str, Mapping[str, Any]]],
    centroids: dict[str, np.ndarray],
    classes: list[str],
) -> tuple[str, str]:
    first = best_known or (classes[0] if classes else (entries[0][0] if entries else "Unknown"))
    if first not in centroids and classes:
        first = classes[0]
    elif first not in centroids and entries:
        first = entries[0][0]

    second = next((name for name, _entry in entries if name != first and name in centroids), "")
    if not second:
        first_vector = centroids.get(first)
        if first_vector is not None and len(centroids) > 1:
            ranked = sorted(
                (
                    (float(np.dot(first_vector, vector)), name)
                    for name, vector in centroids.items()
                    if name != first
                ),
                reverse=True,
            )
            if ranked:
                second = ranked[0][1]
    if not second:
        second = first
    return first, second


def _second_score(
    raw_box: Mapping[str, Any],
    primary: Mapping[str, Any],
    entries: list[tuple[str, Mapping[str, Any]]],
    first: str,
    second: str,
    score1: float,
) -> float:
    margin_value = _first_present(raw_box, "within_margin")
    margin = float(margin_value) if margin_value is not None else 0.15
    for name, entry in entries:
        if name != first and name == second:
            within = _first_present(entry, "within_score")
            if within is not None:
                return float(within)
            if margin_value is not None:
                return score1 - margin
            ridge = _first_present(entry, "ridge_score")
            if ridge is not None:
                return float(ridge)
            return score1 - margin
    if second != first:
        return score1 - margin
    return _candidate_score(primary) if primary else score1 - margin


def _orthogonal_unit(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    best_vector: np.ndarray | None = None
    best_norm = 0.0
    for index in range(first.shape[0]):
        basis = np.zeros_like(first)
        basis[index] = 1.0
        candidate = basis - float(np.dot(basis, first)) * first - float(np.dot(basis, second)) * second
        norm = float(np.linalg.norm(candidate))
        if norm > best_norm:
            best_norm = norm
            best_vector = candidate / norm
    if best_vector is None or best_norm <= 1e-8:
        fallback = np.zeros_like(first)
        fallback[0] = 1.0
        return _unit(fallback - float(np.dot(fallback, first)) * first)
    return best_vector


def _build_projection(
    first: np.ndarray,
    second: np.ndarray,
    score1: float,
    score2: float,
    first_species: str,
    second_species: str,
    first_index: int,
    second_index: int,
    species: str,
) -> dict[str, Any]:
    unit_first = _unit(first)
    unit_second = _unit(second)
    u1 = unit_first
    raw_u2 = unit_second - float(np.dot(unit_second, u1)) * u1
    u2 = _unit(raw_u2)
    if float(np.linalg.norm(u2)) <= 1e-8:
        u2 = _orthogonal_unit(u1, np.zeros_like(u1))
    cross = float(np.dot(unit_second, u1))
    denominator = max(float(np.dot(unit_second, u2)), 1e-6)

    cos1 = float(np.clip(score1, -0.99, 0.99))
    cos2 = float(np.clip(score2, -0.99, 0.99))
    a1 = cos1
    a2 = (cos2 - cross * cos1) / denominator
    squared = a1 * a1 + a2 * a2
    if squared > 0.98:
        scale = float(np.sqrt(0.98 / squared))
        a1 *= scale
        a2 *= scale
    a3 = float(np.sqrt(max(0.0, 1.0 - a1 * a1 - a2 * a2)))
    u3 = _orthogonal_unit(u1, u2)
    current = a1 * u1 + a2 * u2 + a3 * u3

    origin = 0.5 * (unit_first + unit_second)
    axis_x = unit_second - unit_first
    axis_x = _unit(axis_x)
    if float(np.linalg.norm(axis_x)) <= 1e-8:
        axis_x = np.zeros_like(unit_first)
        axis_x[0] = 1.0
    delta = current - origin
    orthogonal = delta - float(np.dot(delta, axis_x)) * axis_x
    if float(np.linalg.norm(orthogonal)) <= 1e-8:
        orthogonal = u3 - float(np.dot(u3, axis_x)) * axis_x
    axis_y = _unit(orthogonal)
    if float(np.dot(delta, axis_y)) < 0:
        axis_y = -axis_y

    def project(vector: np.ndarray) -> tuple[float, float]:
        offset = vector - origin
        return float(np.dot(offset, axis_x)), float(np.dot(offset, axis_y))

    first_x, first_y = project(unit_first)
    second_x, second_y = project(unit_second)
    current_x, current_y = project(current)
    return {
        "method": "nearest_two_species_axis",
        "species": [first_species, second_species],
        "points": [
            {
                "kind": "prototype",
                "species": first_species,
                "source": "base",
                "prototype_index": first_index,
                "x": first_x,
                "y": first_y,
            },
            {
                "kind": "prototype",
                "species": second_species,
                "source": "base",
                "prototype_index": second_index,
                "x": second_x,
                "y": second_y,
            },
            {
                "kind": "current",
                "species": species,
                "source": "base",
                "prototype_index": first_index,
                "x": current_x,
                "y": current_y,
            },
        ],
    }


def _nearest_example(
    source_path: Path,
    box_index: int,
    species: str,
) -> dict[str, Any] | None:
    db_path = source_path.parent / "detections.db"
    if not db_path.is_file():
        return None
    try:
        with closing(sqlite3.connect(str(db_path))) as conn:
            rows = conn.execute(
                "SELECT image_filename, detection_json FROM detections LIMIT 200"
            ).fetchall()
    except Exception:  # noqa: BLE001 - examples are optional.
        return None

    source_key = _path_key(source_path)
    for filename, detection_json in rows:
        other_path = source_path.parent / Path(str(filename)).name
        if not other_path.is_file():
            continue
        same_file = _path_key(other_path) == source_key
        try:
            detection_data = json.loads(detection_json)
        except (TypeError, ValueError):
            continue
        boxes = detection_data.get("检测框") if isinstance(detection_data, dict) else None
        if not isinstance(boxes, list):
            continue
        for other_index, raw_box in enumerate(boxes):
            if not isinstance(raw_box, Mapping):
                continue
            if str(raw_box.get("物种") or "") != species:
                continue
            if same_file and other_index == box_index:
                continue
            return {
                "kind": "observation",
                "species": species,
                "observation_id": make_dinov3_observation_id(
                    other_path,
                    other_index,
                    _box_bbox(raw_box),
                ),
            }
    return None


def explain_dinov3_observation(
    classification_model_path: str,
    observation_id: str,
) -> dict[str, Any]:
    """Explain a Neri1.1 box without loading the DINOv3 ViT encoder."""
    from system.backend import services
    from system.dinov3.runtime import read_manifest, resolve_manifest_asset

    manifest_path = Path(classification_model_path).expanduser().resolve()
    payload = read_manifest(manifest_path)
    checkpoint_path = resolve_manifest_asset(manifest_path, payload["checkpoint"])
    model, centroids = _model_and_centroids(checkpoint_path)
    classes = [str(value) for value in (model.classes if model.classes is not None else [])]

    source_path, box_index, bbox = parse_dinov3_observation_id(observation_id)
    detection_data = services._load_detection_data_for_path(source_path, [source_path.parent])
    if not detection_data:
        raise KeyError(observation_id)
    boxes = detection_data.get("检测框")
    if not isinstance(boxes, list):
        raise KeyError(observation_id)
    raw_box = _locate_raw_box(boxes, observation_id, (source_path, box_index, bbox))
    primary = _primary_candidate(raw_box)

    species = str(raw_box.get("物种") or "Unknown")
    accepted = bool(raw_box.get("accepted", species not in _UNKNOWN_SPECIES))
    best_known = str(
        raw_box.get("best_known_species")
        or primary.get("best_known_species")
        or primary.get("predicted_species")
        or (species if accepted else "")
    )
    known_value = _first_present(raw_box, "known_score")
    if known_value is None:
        known_value = _first_present(primary, "known_score")
    known_score = float(known_value) if known_value is not None else 0.0
    threshold_value = _first_present(raw_box, "threshold")
    if threshold_value is None:
        threshold_value = _first_present(primary, "threshold")
    threshold = float(threshold_value) if threshold_value is not None else 0.0

    entries = _candidate_entries(raw_box)
    first_species, second_species = _choose_species(best_known, entries, centroids, classes)
    first_vector = centroids.get(first_species)
    second_vector = centroids.get(second_species)
    if first_vector is None:
        raise KeyError(observation_id)
    if second_vector is None:
        second_species = first_species
        second_vector = first_vector

    score1_value = _first_present(raw_box, "within_score")
    if score1_value is None:
        score1_value = _first_present(primary, "within_score")
    if score1_value is None:
        score1_value = _first_present(raw_box, "ridge_score")
    score1 = float(score1_value) if score1_value is not None else 0.5
    score2 = _second_score(raw_box, primary, entries, first_species, second_species, score1)

    first_index = classes.index(first_species) if first_species in classes else -1
    second_index = classes.index(second_species) if second_species in classes else -1
    projection = _build_projection(
        first_vector,
        second_vector,
        score1,
        score2,
        first_species,
        second_species,
        first_index,
        second_index,
        species,
    )

    nearest_species = [
        {
            "name": first_species,
            "nearest_prototype_index": first_index,
            "squared_distance": max(0.0, 2.0 - 2.0 * score1),
            "cosine_score": score1,
            "source": "base",
            "registry_id": None,
            "registration_status": None,
        },
        {
            "name": second_species,
            "nearest_prototype_index": second_index,
            "squared_distance": max(0.0, 2.0 - 2.0 * score2),
            "cosine_score": score2,
            "source": "base",
            "registry_id": None,
            "registration_status": None,
        },
    ]

    return {
        "species": species,
        "accepted": accepted,
        "best_known_species": best_known,
        "known_score": known_score,
        "threshold": threshold,
        "score_threshold": threshold,
        "nearest_prototype_index": first_index,
        "squared_distance": nearest_species[0]["squared_distance"],
        "class_margin": score1 - score2,
        "adjusted_distance_score": score1,
        "rejection": {"mode": "ridge_within"},
        "nearest_species": nearest_species,
        "projection": projection,
        "current_example_available": source_path.is_file(),
        "nearest_example": _nearest_example(source_path, box_index, first_species),
    }


def render_dinov3_observation_example(observation_id: str) -> bytes:
    from system.backend import services
    from system.backend.media_examples import render_media_example

    source_path, box_index, bbox = parse_dinov3_observation_id(observation_id)
    if not bbox or len(bbox) < 4:
        detection_data = services._load_detection_data_for_path(source_path, [source_path.parent])
        boxes = detection_data.get("检测框") if isinstance(detection_data, dict) else None
        if isinstance(boxes, list) and 0 <= box_index < len(boxes) and isinstance(boxes[box_index], Mapping):
            bbox = _box_bbox(boxes[box_index])
    return render_media_example(
        source_path=str(source_path),
        bbox=tuple(bbox[:4]),
        frame_index=None,
        timestamp_seconds=None,
    )


def _journal_path() -> Path:
    return default_state_root() / "dinov3_feedback_journal.sqlite3"


def _open_journal() -> sqlite3.Connection:
    path = _journal_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute(
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
    conn.commit()
    return conn


def _journal_state(
    *,
    operation_id: str,
    observation_id: str,
    input_path: Path,
    file_path: Path,
    detection_data: dict[str, Any],
    validated: bool | None,
) -> None:
    with closing(_open_journal()) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO feedback_ecological_operations("
            "operation_id,observation_id,input_path,file_path,detection_json,validated"
            ") VALUES(?,?,?,?,?,?)",
            (
                operation_id,
                observation_id,
                str(input_path),
                str(file_path),
                json.dumps(detection_data, ensure_ascii=False),
                None if validated is None else int(bool(validated)),
            ),
        )
        conn.commit()


def _find_target_box(
    boxes: list[Any],
    observation_id: str,
    decoded: tuple[Path, int, list[float]] | None,
) -> dict[str, Any]:
    for box in boxes:
        if not isinstance(box, dict):
            continue
        if str(box.get("observation_id") or "") == observation_id:
            return box
        candidates = box.get("候选项") or box.get("candidates")
        if isinstance(candidates, list):
            for candidate in candidates:
                if isinstance(candidate, dict) and str(candidate.get("observation_id") or "") == observation_id:
                    return box
    if decoded is not None:
        _path, box_index, bbox = decoded
        if 0 <= box_index < len(boxes) and isinstance(boxes[box_index], dict):
            return boxes[box_index]
        if bbox:
            best: dict[str, Any] | None = None
            best_distance: float | None = None
            for box in boxes:
                if not isinstance(box, dict):
                    continue
                other = _box_bbox(box)
                if len(other) < 4:
                    continue
                distance = sum(abs(left - right) for left, right in zip(bbox[:4], other))
                if best_distance is None or distance < best_distance:
                    best = box
                    best_distance = distance
            if best is not None:
                return best
    raise KeyError(observation_id)


def _set_candidate_name(target_box: dict[str, Any], name: str) -> None:
    candidates = target_box.get("候选项")
    if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
        candidates[0]["name"] = name


def _confirmed_empty_feedback(detection_data: dict[str, Any]) -> bool:
    annotation = detection_data.get("image_annotation")
    if isinstance(annotation, dict) and annotation.get("source") == "human" and annotation.get("species") not in {"", "空"}:
        # Excluding all detected boxes does not mean the annotated animals
        # elsewhere in this image disappeared.
        return False
    boxes = detection_data.get("检测框")
    return isinstance(boxes, list) and bool(boxes) and all(
        isinstance(box, dict)
        and box.get("物种") == "空"
        and box.get("feedback_status") == "confirmed"
        for box in boxes
    )


def _sync_empty_training_contribution(
    file_path: Path, detection_data: dict[str, Any], *, was_empty: bool,
) -> None:
    """Use the existing opt-in queue only for confirmed empty feedback."""
    confirmed_empty = _confirmed_empty_feedback(detection_data)
    if not was_empty and not confirmed_empty:
        return
    try:
        from system.training import get_queue

        get_queue().enqueue_async(file_path, detection_data, validated=confirmed_empty)
    except Exception as error:
        logger.warning("Training queue unavailable (%s)", type(error).__name__)


def apply_dinov3_box_feedback(request: Any) -> dict[str, Any]:
    """Persist a Neri1.1 box label locally and journal it for undo."""
    from system.backend import services
    from system.backend.dino_feedback_service import (
        _path_identity,
        _recompute_detection_summary,
    )

    input_path = Path(request.input_path).expanduser().resolve()
    file_path = Path(request.file_path).expanduser().resolve()
    if not file_path.is_file():
        raise ValueError(f"文件不存在: {file_path}")
    if not input_path.exists():
        raise ValueError(f"输入路径不存在: {input_path}")

    decoded: tuple[Path, int, list[float]] | None = None
    if is_dinov3_observation_id(request.observation_id):
        decoded = parse_dinov3_observation_id(request.observation_id)
        if _path_identity(decoded[0]) != _path_identity(file_path):
            raise ValueError("DINOv3 observation does not belong to the selected file")

    from system.backend.preview_fast import load_validation_index_for_filenames

    roots = [file_path.parent]
    detection_data = services._load_detection_data_for_path(
        file_path,
        roots,
        recursive=False,
    )
    if not detection_data:
        raise KeyError(request.observation_id)
    boxes = detection_data.get("检测框")
    if not isinstance(boxes, list):
        raise KeyError(request.observation_id)
    target_box = _find_target_box(boxes, request.observation_id, decoded)
    was_empty = _confirmed_empty_feedback(detection_data)
    db_paths = services._candidate_detection_dbs_for_roots(roots, recursive=False)
    validation_index = load_validation_index_for_filenames(db_paths, {file_path.name})
    current_validated = validation_index.get(file_path.name)

    target_box["observation_id"] = request.observation_id
    candidates = target_box.get("候选项")
    if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
        candidates[0]["observation_id"] = request.observation_id

    _journal_state(
        operation_id=request.feedback_operation_id,
        observation_id=request.observation_id,
        input_path=input_path,
        file_path=file_path,
        detection_data=detection_data,
        validated=current_validated,
    )

    action = str(request.action)
    species_name = str(request.species_name).strip() if request.species_name is not None else None
    if action == "update":
        if not species_name:
            raise ValueError("species_name is required for update feedback")
        target_box["物种"] = species_name
        target_box["feedback_status"] = "confirmed"
        _set_candidate_name(target_box, species_name)
    elif action == "empty":
        target_box["物种"] = "空"
        target_box["feedback_status"] = "confirmed"
        _set_candidate_name(target_box, "空")
    elif action == "correct":
        current = str(target_box.get("物种") or "").strip()
        if current in _UNKNOWN_SPECIES:
            best = str(target_box.get("best_known_species") or "").strip()
            if best:
                target_box["物种"] = best
                _set_candidate_name(target_box, best)
        target_box["feedback_status"] = "confirmed"
    elif action == "unverified":
        target_box.pop("feedback_status", None)
    else:
        raise ValueError(f"Unsupported DINOv3 feedback action: {action}")

    target_box["label_source"] = "box_annotation"
    target_box["learning_eligible"] = action in {"correct", "update"}
    target_box["learning_excluded"] = action == "unverified"
    annotation = detection_data.get("image_annotation")
    if action == "unverified" and isinstance(annotation, dict) and annotation.get("species") == target_box.get("物种"):
        target_box["feedback_status"] = "image_label"

    _recompute_detection_summary(detection_data)
    from system.backend.dinov3_learning import stage_feedback, _write_lock

    with _write_lock() as conn:
        stage_feedback(request.classification_model_path, input_path, file_path,
                       target_box, action, request.feedback_operation_id, connection=conn)
        services._persist_validation_update(
            file_path,
            input_path,
            detection_data,
            current_validated,
        )
    _sync_empty_training_contribution(file_path, detection_data, was_empty=was_empty)
    from system.backend.dinov3_learning import schedule_learning
    schedule_learning(request.classification_model_path)
    item = services._reload_validation_item(file_path, input_path)
    return {
        "item": item,
        "operation_id": request.feedback_operation_id,
        "affected_species": [],
        "registry_id": None,
    }


def revert_dinov3_feedback_operation(request: Any) -> dict[str, Any]:
    """Restore every journaled detection payload for one operation."""
    from system.backend import services

    model_path = getattr(request, "classification_model_path", None)
    if model_path:
        from system.backend.dinov3_learning import revert_staged_feedback
        revert_staged_feedback(model_path, request.feedback_operation_id)

    with closing(_open_journal()) as conn:
        rows = conn.execute(
            "SELECT * FROM feedback_ecological_operations WHERE operation_id=? ORDER BY observation_id",
            (request.feedback_operation_id,),
        ).fetchall()
        for row in rows:
            file_path = Path(str(row["file_path"]))
            input_path = Path(str(row["input_path"]))
            if not file_path.is_file():
                continue
            current_data = services._load_detection_data_for_path(
                file_path, [file_path.parent], recursive=False,
            )
            was_empty = _confirmed_empty_feedback(current_data or {})
            detection_data = json.loads(row["detection_json"])
            validated = row["validated"]
            services._persist_validation_update(
                file_path,
                input_path,
                detection_data,
                None if validated is None else bool(validated),
            )
            _sync_empty_training_contribution(file_path, detection_data, was_empty=was_empty)
        conn.execute(
            "DELETE FROM feedback_ecological_operations WHERE operation_id=?",
            (request.feedback_operation_id,),
        )
        conn.commit()
    if model_path:
        from system.backend.dinov3_learning import schedule_learning
        schedule_learning(model_path)
    return {"operation_id": request.feedback_operation_id, "affected_species": []}
