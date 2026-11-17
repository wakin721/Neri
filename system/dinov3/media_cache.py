"""Transactional, job-scoped video features for resumable mixed-media inference."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import closing
import hashlib
import io
import json
import logging
import math
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from .encoder import FEATURE_DIM
from .job_cache import _path_key, file_stamp
from .processor import (
    DinoV3CropRecord,
    DinoV3SequenceFinalizationError,
    _expected_detection_box_indices,
    _has_completed_empty_detection,
)
from .video import video_frame_key

logger = logging.getLogger(__name__)

_EXPECTED_TABLES: dict[str, tuple[set[str], list[str]]] = {
    "metadata": ({"signature"}, []),
    "videos": ({"path", "stamp", "item", "info"}, ["path"]),
    "video_frames": (
        {"path", "frame_order", "frame_index", "data", "crops", "features"},
        ["path", "frame_order"],
    ),
}

_SURFACE_SQLITE_PRIMARY_CODES = frozenset({
    sqlite3.SQLITE_PERM,
    sqlite3.SQLITE_ABORT,
    sqlite3.SQLITE_BUSY,
    sqlite3.SQLITE_LOCKED,
    sqlite3.SQLITE_NOMEM,
    sqlite3.SQLITE_READONLY,
    sqlite3.SQLITE_INTERRUPT,
    sqlite3.SQLITE_IOERR,
    sqlite3.SQLITE_FULL,
    sqlite3.SQLITE_CANTOPEN,
    sqlite3.SQLITE_CONSTRAINT,
    sqlite3.SQLITE_NOLFS,
    sqlite3.SQLITE_AUTH,
})

_CORRUPTION_SQLITE_PRIMARY_CODES = frozenset({
    sqlite3.SQLITE_CORRUPT,
    sqlite3.SQLITE_EMPTY,
    sqlite3.SQLITE_SCHEMA,
    sqlite3.SQLITE_FORMAT,
    sqlite3.SQLITE_NOTADB,
})

_SURFACE_ERROR_MARKERS = (
    "locked",
    "busy",
    "readonly",
    "read-only",
    "permission",
    "access denied",
    "access is denied",
    "unable to open",
    "disk i/o",
    "i/o error",
    "disk full",
    "database or disk is full",
    "no space",
    "interrupted",
    "authorization",
)

_CORRUPTION_ERROR_MARKERS = (
    "file is not a database",
    "not a database",
    "malformed",
    "corrupt",
    "no such table",
    "no such column",
    "has no column named",
    "database schema has changed",
    "columns but",
    "cannot modify",
    "view",
)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _valid_stamp(stamp: Any) -> list[int] | None:
    if not isinstance(stamp, (list, tuple)) or len(stamp) != 2:
        return None
    if not all(_is_int(part) and part >= 0 for part in stamp):
        return None
    return [int(stamp[0]), int(stamp[1])]


def _validate_info(info: Any) -> dict[str, Any] | None:
    if not isinstance(info, Mapping):
        return None
    points = info.get("frame_points")
    if not isinstance(points, (list, tuple)) or not points:
        return None
    if not all(_is_int(point) and point >= 0 for point in points):
        return None
    if not all(left < right for left, right in zip(points, points[1:])):
        return None
    fps = info.get("fps")
    if fps is not None:
        fps_val = _finite_float(fps)
        if fps_val is None or fps_val <= 0:
            return None
    for key in ("width", "height", "total_frames"):
        val = info.get(key)
        if val is not None and (not _is_int(val) or val < 0):
            return None
    normalized = dict(info)
    normalized["frame_points"] = [int(point) for point in points]
    return normalized


def _validate_frame_detection(
    video_path: str | Path,
    frame_index: int,
    data: Any,
    crops: Any,
) -> list[DinoV3CropRecord] | None:
    if not isinstance(data, Mapping):
        return None
    if data.get("error") or data.get("detection_completed") is not True:
        return None
    raw_indices = data.get("detection_box_indices")
    if isinstance(raw_indices, list) and not all(_is_int(idx) for idx in raw_indices):
        return None
    raw_count = data.get("detection_box_count")
    if raw_count is not None and not _is_int(raw_count):
        return None
    if isinstance(crops, (str, bytes, Mapping)) or not isinstance(crops, Sequence):
        return None
    crop_list = list(crops)
    frame_key = video_frame_key(video_path, frame_index)
    proxy = SimpleNamespace(path=frame_key, detection_data=dict(data))
    try:
        expected = _expected_detection_box_indices(proxy)
    except DinoV3SequenceFinalizationError:
        return None
    if expected is None:
        return None
    if any(not isinstance(crop, DinoV3CropRecord) or not _is_int(crop.box_index) for crop in crop_list):
        return None
    sorted_crops = sorted(crop_list, key=lambda crop: crop.box_index)
    if tuple(crop.box_index for crop in sorted_crops) != expected:
        return None
    if not sorted_crops:
        return [] if _has_completed_empty_detection(proxy) else None
    boxes = data.get("检测框")
    if boxes is not None and (not isinstance(boxes, list) or len(boxes) != len(sorted_crops)):
        return None

    allowed_paths = {_path_key(frame_key), _path_key(video_path)}
    for crop in sorted_crops:
        if _path_key(crop.image_path) not in allowed_paths:
            return None
        if (
            isinstance(crop.bbox, (str, bytes, Mapping))
            or not isinstance(crop.bbox, (Sequence, np.ndarray))
            or (isinstance(crop.bbox, np.ndarray) and crop.bbox.ndim != 1)
            or len(crop.bbox) != 4
        ):
            return None
        if any(_finite_float(coord) is None for coord in crop.bbox):
            return None
        if _finite_float(crop.detection_confidence) is None:
            return None
        if not isinstance(crop.initial_candidate, Mapping):
            return None
        if crop.detector_category not in {None, "animal", "person", "vehicle"}:
            return None
        emb = crop.raw_embedding
        if (
            not isinstance(emb, np.ndarray)
            or emb.shape != (FEATURE_DIM,)
            or emb.dtype.kind != "f"
            or not bool(np.isfinite(emb).all())
        ):
            return None
    return sorted_crops


def _validate_sorted_frames(
    video_path: str | Path,
    frames: Any,
    expected_points: list[int],
) -> list[tuple[int, int, dict[str, Any], list[DinoV3CropRecord]]] | None:
    if isinstance(frames, (str, bytes, Mapping)) or not isinstance(frames, Sequence):
        return None
    frame_list = list(frames)
    if len(frame_list) != len(expected_points):
        return None
    parsed: list[tuple[int, int, dict[str, Any], list[DinoV3CropRecord]]] = []
    seen_orders: set[int] = set()
    for entry in frame_list:
        if isinstance(entry, (str, bytes, Mapping)) or not isinstance(entry, Sequence) or len(entry) != 4:
            return None
        frame_order, frame_index, data, crops = entry
        if not _is_int(frame_order) or frame_order < 0 or frame_order in seen_orders:
            return None
        if not _is_int(frame_index) or frame_index < 0:
            return None
        seen_orders.add(int(frame_order))
        validated_crops = _validate_frame_detection(video_path, int(frame_index), data, crops)
        if validated_crops is None:
            return None
        parsed.append((int(frame_order), int(frame_index), dict(data), validated_crops))

    parsed.sort(key=lambda item: item[0])
    if [entry[1] for entry in parsed] != expected_points:
        return None
    return parsed


def _encode_frame_row(
    key: str,
    video_path: str | Path,
    frame_order: int,
    frame_index: int,
    data: dict[str, Any],
    crops: list[DinoV3CropRecord],
) -> tuple[str, int, int, str, str, bytes]:
    frame_key_norm = _path_key(video_frame_key(video_path, frame_index))
    features = (
        np.stack([crop.raw_embedding for crop in crops])
        if crops
        else np.empty((0, FEATURE_DIM), dtype=np.float32)
    )
    buffer = io.BytesIO()
    np.save(buffer, features, allow_pickle=False)
    crop_meta = [
        {
            "box_index": int(crop.box_index),
            "bbox": [float(coord) for coord in crop.bbox],
            "detection_confidence": float(crop.detection_confidence),
            "detector_category": crop.detector_category,
            "use_frame_key": _path_key(crop.image_path) == frame_key_norm,
            **(
                {"initial_candidate": dict(crop.initial_candidate)}
                if crop.initial_candidate
                else {}
            ),
        }
        for crop in crops
    ]
    return (
        key,
        frame_order,
        frame_index,
        json.dumps(data, ensure_ascii=False, allow_nan=False),
        json.dumps(crop_meta, ensure_ascii=False, allow_nan=False),
        buffer.getvalue(),
    )


def _is_corruption_or_schema_error(exc: sqlite3.Error) -> bool:
    """Return True only when the SQLite error indicates file corruption or schema mismatch."""
    errorcode = getattr(exc, "sqlite_errorcode", None)
    if errorcode is not None:
        if errorcode == sqlite3.SQLITE_IOERR_SHORT_READ:
            return True
        primary = errorcode & 0xFF
        if primary in _SURFACE_SQLITE_PRIMARY_CODES:
            return False
        if primary in _CORRUPTION_SQLITE_PRIMARY_CODES:
            return True
        if primary == sqlite3.SQLITE_ERROR:
            message = str(exc).lower()
            return any(marker in message for marker in _CORRUPTION_ERROR_MARKERS)
        return False

    if isinstance(exc, (sqlite3.IntegrityError, sqlite3.ProgrammingError, sqlite3.InterfaceError)):
        return False
    message = str(exc).lower()
    if any(marker in message for marker in _SURFACE_ERROR_MARKERS):
        return False
    if any(marker in message for marker in _CORRUPTION_ERROR_MARKERS):
        return True
    return type(exc) is sqlite3.DatabaseError


class DinoV3VideoCache:
    """SQLite-backed transactional cache for decoded video DINOv3 crop features."""

    VERSION = 2

    @staticmethod
    def path_for(state_path: Path, job_id: str) -> Path:
        name = hashlib.sha256(job_id.encode("utf-8")).hexdigest()
        return Path(state_path).parent / "dinov3_features" / f"{name}-video.sqlite3"

    def __init__(self, state_path: Path, job_id: str, signature: dict[str, Any]):
        self.path = self.path_for(state_path, job_id)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._signature = json.dumps(
            {"version": self.VERSION, **(signature or {})},
            sort_keys=True,
            ensure_ascii=False,
        )
        self._initialize_db()

    @staticmethod
    def _create_schema(conn: sqlite3.Connection) -> None:
        conn.execute("CREATE TABLE IF NOT EXISTS metadata(signature TEXT NOT NULL)")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS videos("
            "path TEXT PRIMARY KEY, stamp TEXT NOT NULL, item TEXT NOT NULL, info TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS video_frames("
            "path TEXT NOT NULL, frame_order INTEGER NOT NULL, frame_index INTEGER NOT NULL, "
            "data TEXT NOT NULL, crops TEXT NOT NULL, features BLOB NOT NULL, "
            "PRIMARY KEY(path, frame_order))"
        )

    @staticmethod
    def _has_valid_schema(conn: sqlite3.Connection) -> bool:
        existing_types = {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT name, type FROM sqlite_master WHERE name IN ('metadata', 'videos', 'video_frames')"
            ).fetchall()
        }
        if set(existing_types) != set(_EXPECTED_TABLES) or any(t != "table" for t in existing_types.values()):
            return False
        for table_name, (expected_cols, expected_pk) in _EXPECTED_TABLES.items():
            rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            actual_cols = {row[1] for row in rows}
            actual_pk = [row[1] for row in sorted((row for row in rows if row[5] > 0), key=lambda row: row[5])]
            if actual_cols != expected_cols or actual_pk != expected_pk:
                return False
        return True

    @staticmethod
    def _quick_check_ok(conn: sqlite3.Connection) -> bool:
        return conn.execute("PRAGMA quick_check(1)").fetchall() == [("ok",)]

    @staticmethod
    def _remove_cache_files(cache_path: Path) -> None:
        cache_path.unlink(missing_ok=True)
        for suffix in ("-wal", "-shm", "-journal"):
            cache_path.with_name(cache_path.name + suffix).unlink(missing_ok=True)

    def _rebuild_cache_file(self) -> None:
        self._remove_cache_files(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as conn, conn:
            self._create_schema(conn)
            conn.execute("INSERT INTO metadata VALUES(?)", (self._signature,))

    def _initialize_db(self) -> None:
        try:
            with closing(sqlite3.connect(self.path)) as conn:
                master_rows = conn.execute(
                    "SELECT name, type FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
                ).fetchall()
                if not master_rows:
                    with conn:
                        self._create_schema(conn)
                        conn.execute("INSERT INTO metadata VALUES(?)", (self._signature,))
                    return
                if not self._quick_check_ok(conn) or not self._has_valid_schema(conn):
                    raise sqlite3.DatabaseError("Invalid or corrupt video cache schema")
                previous = conn.execute("SELECT signature FROM metadata").fetchall()
                if len(previous) != 1 or previous[0][0] != self._signature:
                    with conn:
                        self._reset_tables(conn)
        except sqlite3.Error as exc:
            if not _is_corruption_or_schema_error(exc):
                raise
            logger.warning("Resetting corrupt DINOv3 video cache %s: %s", self.path, exc)
            self._rebuild_cache_file()

    def _reset_tables(self, conn: sqlite3.Connection) -> None:
        conn.execute("DELETE FROM video_frames")
        conn.execute("DELETE FROM videos")
        conn.execute("DELETE FROM metadata")
        conn.execute("INSERT INTO metadata VALUES(?)", (self._signature,))

    @staticmethod
    def _delete_video(conn: sqlite3.Connection, key: str) -> None:
        conn.execute("DELETE FROM video_frames WHERE path=?", (key,))
        conn.execute("DELETE FROM videos WHERE path=?", (key,))

    def save(
        self,
        item: Any,
        frames: Sequence[tuple[int, int, dict[str, Any], Sequence[DinoV3CropRecord]]],
        info: dict[str, Any],
        source_stamp: Sequence[int],
    ) -> bool:
        """Validate and atomically persist one completed video's frame detections and embeddings."""
        from system.backend.models import DetectionItem

        if not isinstance(item, DetectionItem) or item.error:
            return False
        if (item.detection_data or {}).get("error"):
            return False
        stamp = _valid_stamp(source_stamp)
        if stamp is None:
            return False
        clean_info = _validate_info(info)
        if clean_info is None:
            return False
        try:
            if file_stamp(item.path) != stamp:
                return False
            parsed_frames = _validate_sorted_frames(item.path, frames, clean_info["frame_points"])
            if parsed_frames is None:
                return False
            key = _path_key(item.path)
            stamp_json = json.dumps(stamp, allow_nan=False)
            item_json = item.model_dump_json()
            info_json = json.dumps(clean_info, ensure_ascii=False, allow_nan=False)
            frame_rows = [
                _encode_frame_row(key, item.path, order, index, data, crops)
                for order, index, data, crops in parsed_frames
            ]
            if file_stamp(item.path) != stamp:
                return False
        except (OSError, ValueError, TypeError):
            logger.warning("Cannot checkpoint DINOv3 video %s", getattr(item, "path", ""), exc_info=True)
            return False

        for attempt in range(2):
            try:
                with closing(sqlite3.connect(self.path)) as conn:
                    if not self._has_valid_schema(conn):
                        raise sqlite3.DatabaseError("Invalid or corrupt video cache schema")
                    sig_rows = conn.execute("SELECT signature FROM metadata").fetchall()
                    with conn:
                        if len(sig_rows) != 1 or sig_rows[0][0] != self._signature:
                            self._reset_tables(conn)
                        conn.execute("DELETE FROM video_frames WHERE path=?", (key,))
                        conn.execute(
                            "INSERT OR REPLACE INTO videos(path, stamp, item, info) VALUES(?,?,?,?)",
                            (key, stamp_json, item_json, info_json),
                        )
                        conn.executemany(
                            "INSERT INTO video_frames(path, frame_order, frame_index, data, crops, features) "
                            "VALUES(?,?,?,?,?,?)",
                            frame_rows,
                        )
                return True
            except sqlite3.Error as exc:
                if attempt == 0 and _is_corruption_or_schema_error(exc):
                    logger.warning("Resetting corrupt DINOv3 video cache %s during save: %s", self.path, exc)
                    self._rebuild_cache_file()
                    continue
                raise
        return True

    def restore(
        self,
        paths: Sequence[Path | str],
        progress_fn: Callable[[int, int], Any] | None = None,
    ) -> dict[str, tuple[Any, list[tuple[int, int, dict[str, Any], list[DinoV3CropRecord]]], dict[str, Any]]]:
        """Restore valid cached videos keyed by str(path); discard any corrupt or stale entry."""
        from system.backend.models import DetectionItem

        restored: dict[str, tuple[DetectionItem, list[tuple[int, int, dict[str, Any], list[DinoV3CropRecord]]], dict[str, Any]]] = {}
        invalid_keys: list[str] = []
        total = len(paths)

        try:
            with closing(sqlite3.connect(self.path)) as conn:
                if not self._has_valid_schema(conn) or not self._quick_check_ok(conn):
                    raise sqlite3.DatabaseError("Invalid or corrupt video cache schema")
                sig_rows = conn.execute("SELECT signature FROM metadata").fetchall()
                if len(sig_rows) != 1 or sig_rows[0][0] != self._signature:
                    with conn:
                        self._reset_tables(conn)
                    if progress_fn is not None:
                        progress_fn(total, total)
                    return {}

                for idx, path in enumerate(paths):
                    if progress_fn is not None:
                        progress_fn(idx, total)
                    key = _path_key(path)
                    video_row = conn.execute(
                        "SELECT stamp, item, info FROM videos WHERE path=?", (key,),
                    ).fetchone()
                    if video_row is None:
                        continue
                    try:
                        stamp = _valid_stamp(json.loads(video_row[0]))
                        if stamp is None or file_stamp(path) != stamp:
                            raise ValueError("Source video changed")
                        item = DetectionItem.model_validate_json(video_row[1])
                        if item.error or (item.detection_data or {}).get("error") or _path_key(item.path) != key:
                            raise ValueError("Invalid cached video item")
                        info = _validate_info(json.loads(video_row[2]))
                        if info is None:
                            raise ValueError("Invalid cached video info")

                        frame_rows = conn.execute(
                            "SELECT frame_order, frame_index, data, crops, features "
                            "FROM video_frames WHERE path=? ORDER BY frame_order ASC",
                            (key,),
                        ).fetchall()
                        decoded_frames = []
                        for frame_order, frame_index, data_json, crops_json, features_blob in frame_rows:
                            data = json.loads(data_json)
                            crops_meta = json.loads(crops_json)
                            if not isinstance(crops_meta, list):
                                raise ValueError("Invalid crop metadata")
                            features = np.load(io.BytesIO(features_blob), allow_pickle=False)
                            if not isinstance(features, np.ndarray):
                                raise ValueError("Invalid feature array")
                            if (
                                features.ndim != 2
                                or features.shape != (len(crops_meta), FEATURE_DIM)
                                or features.dtype.kind != "f"
                                or not bool(np.isfinite(features).all())
                            ):
                                raise ValueError("Incomplete or corrupt video frame features")
                            crops = []
                            for meta, feat in zip(crops_meta, features):
                                if not isinstance(meta, Mapping):
                                    raise ValueError("Invalid crop entry")
                                use_frame_key = meta.get("use_frame_key", True)
                                if not isinstance(use_frame_key, bool):
                                    raise ValueError("Invalid crop path flag")
                                crop_path = (
                                    video_frame_key(path, int(frame_index))
                                    if use_frame_key
                                    else str(path)
                                )
                                crops.append(
                                    DinoV3CropRecord(
                                        image_path=crop_path,
                                        box_index=meta["box_index"],
                                        bbox=tuple(meta["bbox"]),
                                        detection_confidence=meta["detection_confidence"],
                                        raw_embedding=feat.copy(),
                                        initial_candidate=dict(meta.get("initial_candidate") or {}),
                                        detector_category=meta.get("detector_category"),
                                    )
                                )
                            decoded_frames.append((frame_order, frame_index, data, crops))

                        validated = _validate_sorted_frames(path, decoded_frames, info["frame_points"])
                        if validated is None:
                            raise ValueError("Incomplete or mismatched video frames")
                    except (OSError, ValueError, TypeError, KeyError, EOFError):
                        invalid_keys.append(key)
                        logger.warning("Ignoring invalid DINOv3 video checkpoint: %s", path)
                        continue

                    restored[str(path)] = (item, validated, info)

                if invalid_keys:
                    with conn:
                        for key in invalid_keys:
                            self._delete_video(conn, key)
        except sqlite3.Error as exc:
            if not _is_corruption_or_schema_error(exc):
                raise
            logger.warning("Resetting corrupt DINOv3 video cache %s during restore: %s", self.path, exc)
            self._rebuild_cache_file()
            if progress_fn is not None:
                progress_fn(total, total)
            return {}

        if progress_fn is not None:
            progress_fn(total, total)
        return restored

    @classmethod
    def remove(cls, state_path: Path, job_id: str) -> None:
        cls._remove_cache_files(cls.path_for(state_path, job_id))
