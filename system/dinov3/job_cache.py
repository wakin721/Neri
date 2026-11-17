"""Transactional, job-scoped photo features for resumable sequence inference."""
from __future__ import annotations

from contextlib import closing
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import sqlite3

import numpy as np

from .processor import (
    DinoV3CropCollector, DinoV3CropRecord, _expected_detection_box_indices,
    _has_completed_empty_detection,
)

logger = logging.getLogger(__name__)


def _path_key(path):
    return os.path.normcase(os.path.abspath(str(path)))


def file_stamp(path):
    stat = Path(path).stat()
    return [stat.st_size, stat.st_mtime_ns]


class DinoV3JobCache:
    VERSION = 2

    @staticmethod
    def path_for(state_path: Path, job_id: str) -> Path:
        name = hashlib.sha256(job_id.encode("utf-8")).hexdigest()
        return state_path.parent / "dinov3_features" / f"{name}.sqlite3"

    def __init__(self, state_path: Path, job_id: str, signature: dict):
        self.path = self.path_for(state_path, job_id)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        signature = json.dumps(
            {"version": self.VERSION, **signature}, sort_keys=True, ensure_ascii=False,
        )
        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.execute("CREATE TABLE IF NOT EXISTS metadata(signature TEXT NOT NULL)")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS photos("
                "path TEXT PRIMARY KEY, stamp TEXT NOT NULL, result TEXT NOT NULL, "
                "crops TEXT NOT NULL, features BLOB NOT NULL)"
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(photos)")}
            if "content_digest" not in columns:
                conn.execute("ALTER TABLE photos ADD COLUMN content_digest TEXT")
            previous = conn.execute("SELECT signature FROM metadata").fetchone()
            if previous is None or previous[0] != signature:
                conn.execute("DELETE FROM photos")
                conn.execute("DELETE FROM metadata")
                conn.execute("INSERT INTO metadata VALUES(?)", (signature,))

    @staticmethod
    def _validate(item, crops):
        data = item.detection_data or {}
        expected = _expected_detection_box_indices(item)
        if item.error or data.get("detection_completed") is not True or expected is None:
            return False
        if tuple(c.box_index for c in crops) != expected:
            return False
        if not crops:
            return _has_completed_empty_detection(item)
        return all(
            _path_key(c.image_path) == _path_key(item.path)
            and len(c.bbox) == 4 and np.isfinite(c.bbox).all()
            and np.isfinite(c.detection_confidence)
            and c.raw_embedding.ndim == 1 and c.raw_embedding.size > 0
            and c.raw_embedding.dtype.kind == "f" and np.isfinite(c.raw_embedding).all()
            and c.detector_category in {None, "animal", "person", "vehicle"}
            for c in crops
        )

    def save_batch(self, items, collector, source_stamps):
        """Commit metadata, zero-box completion and full-precision features together."""
        if collector is None:
            return
        rows = []
        for item in items:
            key = _path_key(item.path)
            stamp = source_stamps.get(str(item.path))
            try:
                if stamp is None or file_stamp(item.path) != stamp:
                    continue
                crops = collector.get_crops_for_path(item.path)
                if not self._validate(item, crops):
                    continue
                features = (np.stack([c.raw_embedding for c in crops]) if crops
                            else np.empty((0, 0), dtype=np.float32))
                buffer = io.BytesIO()
                np.save(buffer, features, allow_pickle=False)
                metadata = [dict(box_index=c.box_index, bbox=list(c.bbox),
                                 detection_confidence=c.detection_confidence,
                                 detector_category=c.detector_category) for c in crops]
                rows.append((key, json.dumps(stamp), item.model_dump_json(),
                             json.dumps(metadata), buffer.getvalue(), collector.image_digest(item.path, stamp)))
            except (OSError, ValueError, TypeError):
                logger.warning("Cannot checkpoint DINOv3 photo %s", item.path, exc_info=True)
        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.executemany(
                "INSERT OR REPLACE INTO photos(path,stamp,result,crops,features,content_digest) VALUES(?,?,?,?,?,?)", rows,
            )

    def restore(self, files, collector: DinoV3CropCollector, *, progress_fn=None):
        from system.backend.models import DetectionItem

        restored = []
        invalid = []
        with closing(sqlite3.connect(self.path)) as conn:
            for index, path in enumerate(files):
                if progress_fn is not None:
                    progress_fn(index, len(files))
                key = _path_key(path)
                row = conn.execute(
                    "SELECT stamp,result,crops,features,content_digest FROM photos WHERE path=?", (key,),
                ).fetchone()
                if row is None:
                    continue
                try:
                    if json.loads(row[0]) != file_stamp(path):
                        raise ValueError("Source photo changed")
                    item = DetectionItem.model_validate_json(row[1])
                    metadata = json.loads(row[2])
                    features = np.load(io.BytesIO(row[3]), allow_pickle=False)
                    if features.ndim != 2 or len(features) != len(metadata):
                        raise ValueError("Incomplete cached features")
                    crops = [DinoV3CropRecord(
                        image_path=str(path), box_index=m["box_index"], bbox=tuple(m["bbox"]),
                        detection_confidence=m["detection_confidence"], raw_embedding=f.copy(),
                        detector_category=m.get("detector_category"),
                    ) for m, f in zip(metadata, features)]
                    if _path_key(item.path) != key or not self._validate(item, crops):
                        raise ValueError("Incomplete cached detection")
                except (OSError, ValueError, TypeError, KeyError, EOFError):
                    invalid.append((key,))
                    logger.warning("Ignoring invalid DINOv3 photo checkpoint: %s", path)
                    continue
                collector.record_crops(crops)
                if row[4]:
                    collector.record_image_digest(path, row[4], json.loads(row[0]))
                restored.append(item)
            with conn:
                conn.executemany("DELETE FROM photos WHERE path=?", invalid)
        if progress_fn is not None:
            progress_fn(len(files), len(files))
        return restored

    @classmethod
    def remove(cls, state_path: Path, job_id: str):
        cls.path_for(state_path, job_id).unlink(missing_ok=True)
