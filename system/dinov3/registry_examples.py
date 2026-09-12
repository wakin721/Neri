"""Persistent cropped examples for DINOv3 registry and feedback evidence."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

_EXAMPLE_SIZE = 320
_JPEG_QUALITY = 90


def _examples_root(store_path: str | Path) -> Path:
    return Path(store_path).expanduser().resolve().parent / "examples"


def registry_event_example_path(registry_path: str | Path, event_id: int) -> Path:
    return _examples_root(registry_path) / "registry" / f"event-{int(event_id)}.jpg"


def feedback_observation_example_path(
    feedback_path: str | Path,
    observation_id: str,
) -> Path:
    digest = hashlib.sha256(str(observation_id).encode("utf-8")).hexdigest()
    return _examples_root(feedback_path) / "observations" / f"{digest}.jpg"


def _read_frame(
    source_path: str | Path,
    *,
    frame_index: int | None = None,
    timestamp_seconds: float | None = None,
) -> np.ndarray | None:
    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        return None
    frame = cv2.imread(str(source))
    if frame is not None:
        return frame
    capture = cv2.VideoCapture(str(source))
    try:
        if frame_index is not None:
            capture.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
        elif timestamp_seconds is not None:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_seconds) * 1000.0)
        ok, frame = capture.read()
    finally:
        capture.release()
    return frame if ok and frame is not None else None


def _crop_example(frame: np.ndarray, bbox: Iterable[float]) -> np.ndarray | None:
    values = np.asarray(tuple(bbox), dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        return None
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in values)
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width <= 0 or box_height <= 0:
        return None
    pad_w = box_width * 0.1
    pad_h = box_height * 0.1
    left = max(0, int(np.floor(x1 - pad_w)))
    top = max(0, int(np.floor(y1 - pad_h)))
    right = min(width, int(np.ceil(x2 + pad_w)))
    bottom = min(height, int(np.ceil(y2 + pad_h)))
    if right <= left or bottom <= top:
        return None
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
    interpolation = (
        cv2.INTER_AREA
        if crop.shape[0] > _EXAMPLE_SIZE or crop.shape[1] > _EXAMPLE_SIZE
        else cv2.INTER_LINEAR
    )
    return cv2.resize(crop, (_EXAMPLE_SIZE, _EXAMPLE_SIZE), interpolation=interpolation)


def _encode_example(
    source_path: str | Path,
    bbox: Iterable[float],
    *,
    frame_index: int | None = None,
    timestamp_seconds: float | None = None,
) -> bytes | None:
    frame = _read_frame(
        source_path,
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
    )
    if frame is None:
        return None
    crop = _crop_example(frame, bbox)
    if crop is None:
        return None
    ok, encoded = cv2.imencode(
        ".jpg",
        crop,
        [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY],
    )
    return encoded.tobytes() if ok else None


def _write_example(path: Path, payload: bytes) -> bool:
    if not payload:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        temp.write_bytes(payload)
        temp.replace(path)
        return True
    except OSError:
        temp.unlink(missing_ok=True)
        return False


def _persist_example(
    path: Path,
    *,
    source_path: str | Path,
    bbox: Iterable[float],
    frame_index: int | None = None,
    timestamp_seconds: float | None = None,
) -> bool:
    if path.is_file() and path.stat().st_size > 0:
        return True
    payload = _encode_example(
        source_path,
        bbox,
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
    )
    return False if payload is None else _write_example(path, payload)


def persist_registry_event_example(
    registry_path: str | Path,
    event_id: int,
    *,
    source_path: str | Path,
    bbox: Iterable[float],
    frame_index: int | None = None,
    timestamp_seconds: float | None = None,
) -> bool:
    return _persist_example(
        registry_event_example_path(registry_path, event_id),
        source_path=source_path,
        bbox=bbox,
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
    )


def persist_feedback_observation_example(
    feedback_path: str | Path,
    observation_id: str,
    *,
    source_path: str | Path,
    bbox: Iterable[float],
    frame_index: int | None = None,
    timestamp_seconds: float | None = None,
) -> bool:
    return _persist_example(
        feedback_observation_example_path(feedback_path, observation_id),
        source_path=source_path,
        bbox=bbox,
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
    )


def store_feedback_observation_example(
    feedback_path: str | Path,
    observation_id: str,
    payload: bytes,
) -> bool:
    return _write_example(
        feedback_observation_example_path(feedback_path, observation_id),
        payload,
    )


def read_registry_event_example(
    registry_path: str | Path,
    event_id: int,
) -> bytes | None:
    path = registry_event_example_path(registry_path, event_id)
    try:
        payload = path.read_bytes()
    except OSError:
        return None
    return payload or None


def read_feedback_observation_example(
    feedback_path: str | Path,
    observation_id: str,
) -> bytes | None:
    path = feedback_observation_example_path(feedback_path, observation_id)
    try:
        payload = path.read_bytes()
    except OSError:
        return None
    return payload or None


def delete_registry_event_examples(
    registry_path: str | Path,
    event_ids: Iterable[int],
) -> None:
    for event_id in event_ids:
        registry_event_example_path(registry_path, int(event_id)).unlink(missing_ok=True)
