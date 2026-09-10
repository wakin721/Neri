"""Fingerprint-scoped SQLite registry for unknown and learned species."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sqlite3
from typing import Any, Callable

import numpy as np

from .events import EVENT_GAP_SECONDS
from .prototype_bank import PrototypeBank, PrototypeRecord
from .simple_shot import (
    build_prototype,
    cosine_similarity,
    deterministic_k_means,
    normalize_embedding,
)


class RegistryEntryNotFound(KeyError):
    pass


class RegistrationConditionError(ValueError):
    pass


@dataclass(frozen=True)
class RegistryEntry:
    id: int
    candidate_number: int
    status: str
    common_name: str
    scientific_name: str
    event_count: int
    camera_count: int
    prototype_count: int
    cluster_purity: float
    embedding_consistency: float
    conditions: dict[str, bool]
    can_register: bool
    display_name: str

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def registry_path_for_fingerprint(root: str | Path, fingerprint: str) -> Path:
    safe = fingerprint.lower()
    if len(safe) != 64 or any(c not in "0123456789abcdef" for c in safe):
        raise ValueError("model fingerprint must be a SHA-256 hex digest")
    return Path(root).expanduser().resolve() / "dinov3" / safe / "registry.sqlite3"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _blob(value: np.ndarray) -> bytes:
    return normalize_embedding(value).astype("<f4", copy=False).tobytes()


def _from_blob(value: bytes) -> np.ndarray:
    return normalize_embedding(np.frombuffer(value, dtype="<f4").copy())


class SpeciesRegistry:
    def __init__(
        self,
        path: str | Path,
        *,
        model_fingerprint: str,
        join_threshold: float = 0.82,
        registration_threshold: float = 0.78,
        purity_threshold: float = 0.9,
        consistency_threshold: float = 0.75,
        event_gap_seconds: int = EVENT_GAP_SECONDS,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.model_fingerprint = model_fingerprint
        self.join_threshold = float(join_threshold)
        self.registration_threshold = float(registration_threshold)
        self.purity_threshold = float(purity_threshold)
        self.consistency_threshold = float(consistency_threshold)
        self.event_gap_seconds = int(event_gap_seconds)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init()

    def _init(self) -> None:
        self._conn.executescript(
            """
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS metadata(
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS registrations(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_number INTEGER NOT NULL UNIQUE,
                status TEXT NOT NULL DEFAULT 'candidate',
                common_name TEXT NOT NULL DEFAULT '',
                scientific_name TEXT NOT NULL DEFAULT '',
                cluster_purity REAL NOT NULL DEFAULT 1.0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                registration_id INTEGER NOT NULL REFERENCES registrations(id) ON DELETE CASCADE,
                event_key TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                started_at TEXT,
                ended_at TEXT,
                timestamp_missing INTEGER NOT NULL DEFAULT 0,
                source_path TEXT NOT NULL,
                embedding_sum BLOB NOT NULL,
                embedding BLOB NOT NULL,
                sample_count INTEGER NOT NULL DEFAULT 1,
                UNIQUE(registration_id,event_key)
            );
            CREATE TABLE IF NOT EXISTS prototypes(
                registration_id INTEGER NOT NULL REFERENCES registrations(id) ON DELETE CASCADE,
                prototype_index INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                event_count INTEGER NOT NULL,
                PRIMARY KEY(registration_id,prototype_index)
            );
            """
        )
        existing_event_columns = {
            str(row["name"])
            for row in self._conn.execute("PRAGMA table_info(events)").fetchall()
        }
        for column_name, column_type in (
            ("box_x1", "REAL"),
            ("box_y1", "REAL"),
            ("box_x2", "REAL"),
            ("box_y2", "REAL"),
            ("frame_index", "INTEGER"),
            ("timestamp_seconds", "REAL"),
        ):
            if column_name not in existing_event_columns:
                self._conn.execute(
                    f"ALTER TABLE events ADD COLUMN {column_name} {column_type}"
                )
        row = self._conn.execute(
            "SELECT value FROM metadata WHERE key='model_fingerprint'"
        ).fetchone()
        if row is not None and row[0] != self.model_fingerprint:
            raise ValueError(
                "Registry model fingerprint does not match selected classifier"
            )
        self._conn.execute(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES('model_fingerprint',?)",
            (self.model_fingerprint,),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def _row(self, entry_id: int):
        row = self._conn.execute(
            "SELECT * FROM registrations WHERE id=?",
            (entry_id,),
        ).fetchone()
        if row is None:
            raise RegistryEntryNotFound(entry_id)
        return row

    def _create(self) -> int:
        number = int(
            self._conn.execute(
                "SELECT COALESCE(MAX(candidate_number),0)+1 FROM registrations"
            ).fetchone()[0]
        )
        now = _now()
        cursor = self._conn.execute(
            "INSERT INTO registrations(candidate_number,created_at,updated_at) VALUES(?,?,?)",
            (number, now, now),
        )
        self._conn.commit()
        return int(cursor.lastrowid)

    def _event_key(self, entry_id, camera, captured, source):
        if captured is None:
            return (
                hashlib.sha256(
                    f"{entry_id}|{camera}|missing|{Path(source).resolve()}".encode()
                ).hexdigest(),
                None,
                None,
                1,
            )
        text = captured.isoformat()
        previous = self._conn.execute(
            "SELECT event_key,started_at,ended_at FROM events "
            "WHERE registration_id=? AND camera_id=? AND ended_at IS NOT NULL "
            "ORDER BY ended_at DESC LIMIT 1",
            (entry_id, camera),
        ).fetchone()
        if previous is not None:
            gap = (captured - datetime.fromisoformat(previous["ended_at"])).total_seconds()
            if 0 <= gap < self.event_gap_seconds:
                return previous["event_key"], previous["started_at"], text, 0
        return (
            hashlib.sha256(f"{entry_id}|{camera}|{text}".encode()).hexdigest(),
            text,
            text,
            0,
        )

    def record_unknown(
        self,
        embedding,
        *,
        camera_id,
        captured_at,
        source_path,
        bbox=None,
        frame_index=None,
        timestamp_seconds=None,
    ):
        vector = normalize_embedding(embedding)
        matched = self.match(vector)
        entry_id = int(matched["id"]) if matched else self._create()
        return self.record_observation(
            entry_id,
            vector,
            camera_id=camera_id,
            captured_at=captured_at,
            source_path=source_path,
            bbox=bbox,
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
        )

    def record_observation(
        self,
        entry_id,
        embedding,
        *,
        camera_id,
        captured_at,
        source_path,
        bbox=None,
        frame_index=None,
        timestamp_seconds=None,
    ):
        self._row(entry_id)
        vector = normalize_embedding(embedding)
        bbox_values = None
        if bbox is not None:
            values = np.asarray(bbox, dtype=np.float64)
            if values.shape != (4,) or not np.isfinite(values).all():
                raise ValueError("bbox must contain four finite coordinates")
            bbox_values = tuple(float(value) for value in values)
        frame_value = None if frame_index is None else int(frame_index)
        timestamp_value = (
            None if timestamp_seconds is None else float(timestamp_seconds)
        )
        if timestamp_value is not None and not np.isfinite(timestamp_value):
            raise ValueError("timestamp_seconds must be finite")
        key, start, end, missing = self._event_key(
            entry_id,
            camera_id,
            captured_at,
            source_path,
        )
        old = self._conn.execute(
            "SELECT id,embedding_sum,sample_count FROM events "
            "WHERE registration_id=? AND event_key=?",
            (entry_id, key),
        ).fetchone()
        if old is None:
            raw = vector.astype("<f4").tobytes()
            self._conn.execute(
                "INSERT INTO events(registration_id,event_key,camera_id,started_at,ended_at,"
                "timestamp_missing,source_path,embedding_sum,embedding,box_x1,box_y1,box_x2,box_y2,"
                "frame_index,timestamp_seconds,sample_count) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)",
                (
                    entry_id,
                    key,
                    camera_id,
                    start,
                    end,
                    missing,
                    source_path,
                    raw,
                    raw,
                    *(bbox_values or (None, None, None, None)),
                    frame_value,
                    timestamp_value,
                ),
            )
        else:
            total = np.frombuffer(old["embedding_sum"], dtype="<f4").copy() + vector
            aggregate = normalize_embedding(total)
            self._conn.execute(
                "UPDATE events SET ended_at=COALESCE(?,ended_at),source_path=?,"
                "embedding_sum=?,embedding=?,sample_count=?,"
                "box_x1=COALESCE(?,box_x1),box_y1=COALESCE(?,box_y1),"
                "box_x2=COALESCE(?,box_x2),box_y2=COALESCE(?,box_y2),"
                "frame_index=COALESCE(?,frame_index),"
                "timestamp_seconds=COALESCE(?,timestamp_seconds) WHERE id=?",
                (
                    end,
                    source_path,
                    total.astype("<f4").tobytes(),
                    aggregate.astype("<f4").tobytes(),
                    int(old["sample_count"]) + 1,
                    *(bbox_values or (None, None, None, None)),
                    frame_value,
                    timestamp_value,
                    int(old["id"]),
                ),
            )
        self._conn.execute(
            "UPDATE registrations SET updated_at=? WHERE id=?",
            (_now(), entry_id),
        )
        self._conn.commit()
        self._refresh(entry_id)
        return self.get(entry_id)

    def set_identity(self, entry_id, *, common_name, scientific_name=""):
        self._row(entry_id)
        common = common_name.strip()
        if not common:
            raise ValueError("common_name is required")
        self._conn.execute(
            "UPDATE registrations SET common_name=?,scientific_name=?,updated_at=? WHERE id=?",
            (common, scientific_name.strip(), _now(), entry_id),
        )
        self._conn.commit()
        return self.get(entry_id)

    def set_cluster_purity(self, entry_id, value):
        self._row(entry_id)
        value = float(value)
        if not 0 <= value <= 1:
            raise ValueError("cluster purity must be between 0 and 1")
        self._conn.execute(
            "UPDATE registrations SET cluster_purity=?,updated_at=? WHERE id=?",
            (value, _now(), entry_id),
        )
        self._conn.commit()
        return self.get(entry_id)

    def delete(self, entry_id: int) -> None:
        self._row(entry_id)
        self._conn.execute("DELETE FROM registrations WHERE id=?", (entry_id,))
        self._conn.commit()

    def _event_rows(self, entry_id):
        return self._conn.execute(
            "SELECT * FROM events WHERE registration_id=? ORDER BY id",
            (entry_id,),
        ).fetchall()

    def _embeddings(self, entry_id) -> np.ndarray:
        rows = self._event_rows(entry_id)
        if not rows:
            return np.empty((0, 768), dtype=np.float32)
        return np.stack([_from_blob(row["embedding"]) for row in rows])

    def _prototypes(self, entry_id) -> np.ndarray:
        rows = self._conn.execute(
            "SELECT embedding FROM prototypes WHERE registration_id=? "
            "ORDER BY prototype_index",
            (entry_id,),
        ).fetchall()
        if not rows:
            return np.empty((0, 768), dtype=np.float32)
        return np.stack([_from_blob(row[0]) for row in rows])

    def _replace_prototypes(self, entry_id, prototypes, event_count) -> None:
        self._conn.execute(
            "DELETE FROM prototypes WHERE registration_id=?",
            (entry_id,),
        )
        for index, prototype in enumerate(prototypes):
            self._conn.execute(
                "INSERT INTO prototypes VALUES(?,?,?,?)",
                (entry_id, index, _blob(prototype), event_count),
            )
        self._conn.commit()

    @staticmethod
    def _status_prototype_limit(status: str) -> int:
        return {"candidate": 1, "provisional": 1, "confirmed": 2, "mature": 3}.get(
            status,
            1,
        )

    def _refresh(self, entry_id) -> None:
        row = self._row(entry_id)
        embeddings = self._embeddings(entry_id)
        count = len(embeddings)
        if count < 4:
            return

        cameras = int(
            self._conn.execute(
                "SELECT COUNT(DISTINCT camera_id) FROM events WHERE registration_id=?",
                (entry_id,),
            ).fetchone()[0]
        )
        status = str(row["status"])
        if status == "provisional" and count >= 10 and cameras >= 2:
            status = "confirmed"
        if status == "confirmed" and count >= 20 and cameras >= 3:
            status = "mature"

        # Registry joining remains in raw normalized embedding space. Reusing
        # the deterministic K<=3 splitter here improves long-lived matching,
        # while classifier overlay prototypes are computed separately in
        # centered feature space by prototype_bank().
        raw_prototypes = deterministic_k_means(
            embeddings,
            max_k=self._status_prototype_limit(status),
        )
        self._replace_prototypes(entry_id, raw_prototypes, count)

        if status != row["status"]:
            self._conn.execute(
                "UPDATE registrations SET status=?,updated_at=? WHERE id=?",
                (status, _now(), entry_id),
            )
            self._conn.commit()

    def get(self, entry_id) -> RegistryEntry:
        row = self._row(entry_id)
        events = int(
            self._conn.execute(
                "SELECT COUNT(*) FROM events WHERE registration_id=?",
                (entry_id,),
            ).fetchone()[0]
        )
        cameras = int(
            self._conn.execute(
                "SELECT COUNT(DISTINCT camera_id) FROM events WHERE registration_id=?",
                (entry_id,),
            ).fetchone()[0]
        )
        prototypes = self._prototypes(entry_id)
        embeddings = self._embeddings(entry_id)
        consistency = (
            float(np.quantile(np.max(embeddings @ prototypes.T, axis=1), 0.1))
            if len(embeddings) and len(prototypes)
            else 0.0
        )
        purity = float(row["cluster_purity"])
        common = str(row["common_name"])
        status = str(row["status"])
        conditions = {
            "events": events >= 4,
            "cluster_purity": purity >= self.purity_threshold,
            "embedding_consistency": consistency >= self.consistency_threshold,
            "identity": bool(common),
        }
        display = (
            f"未知物种 #{row['candidate_number']}"
            if status == "candidate"
            else (
                f"{common}（临时注册，待确认）"
                if status == "provisional"
                else (common or f"未知物种 #{row['candidate_number']}")
            )
        )
        return RegistryEntry(
            int(row["id"]),
            int(row["candidate_number"]),
            status,
            common,
            str(row["scientific_name"]),
            events,
            cameras,
            len(prototypes),
            purity,
            consistency,
            conditions,
            status == "candidate" and all(conditions.values()),
            display,
        )

    def list(self, *, status=None) -> list[RegistryEntry]:
        rows = self._conn.execute(
            "SELECT id FROM registrations"
            + (" WHERE status=?" if status else "")
            + " ORDER BY candidate_number",
            (status,) if status else (),
        ).fetchall()
        return [self.get(int(row[0])) for row in rows]

    def list_events(self, entry_id):
        self._row(entry_id)
        result = []
        for row in self._event_rows(entry_id):
            coordinates = (
                row["box_x1"],
                row["box_y1"],
                row["box_x2"],
                row["box_y2"],
            )
            bbox = (
                [float(value) for value in coordinates]
                if all(value is not None for value in coordinates)
                else None
            )
            source_path = str(row["source_path"])
            result.append(
                {
                    "id": int(row["id"]),
                    "event_key": str(row["event_key"]),
                    "source_path": source_path,
                    "camera_id": str(row["camera_id"]),
                    "started_at": row["started_at"],
                    "ended_at": row["ended_at"],
                    "timestamp_missing": bool(row["timestamp_missing"]),
                    "sample_count": int(row["sample_count"]),
                    "bbox": bbox,
                    "frame_index": (
                        int(row["frame_index"])
                        if row["frame_index"] is not None
                        else None
                    ),
                    "timestamp_seconds": (
                        float(row["timestamp_seconds"])
                        if row["timestamp_seconds"] is not None
                        else None
                    ),
                    "has_example": bool(source_path and bbox is not None),
                }
            )
        return result

    def prototype_bank(self, feature_center: np.ndarray) -> PrototypeBank:
        center = np.asarray(feature_center, dtype=np.float32)
        if center.shape != (768,) or not np.isfinite(center).all():
            raise ValueError("feature_center must be a finite 768-dimensional vector")

        formal: list[PrototypeRecord] = []
        provisional: list[PrototypeRecord] = []
        for entry in self.list():
            if entry.status == "candidate":
                continue
            embeddings = self._embeddings(entry.id)
            if len(embeddings) < 4:
                continue
            centered = embeddings - center[None, :]
            prototypes = deterministic_k_means(
                centered,
                max_k=self._status_prototype_limit(entry.status),
            )
            records = [
                PrototypeRecord(
                    species=entry.display_name,
                    embedding=prototype,
                    source="overlay",
                    registry_id=entry.id,
                    registration_status=entry.status,
                )
                for prototype in prototypes
            ]
            if entry.status == "provisional":
                provisional.extend(records)
            elif entry.status in {"confirmed", "mature"}:
                formal.extend(records)
        return PrototypeBank(tuple(formal), tuple(provisional))

    def _duplicate_formal_species(
        self,
        entry_id: int,
        formal_matcher: Callable[[np.ndarray], str | None],
    ) -> str | None:
        embeddings = self._embeddings(entry_id)
        if len(embeddings) < 4:
            return None
        matches = [formal_matcher(embedding) for embedding in embeddings]
        names = [str(match) for match in matches if match]
        if not names:
            return None
        species, count = Counter(names).most_common(1)[0]
        if count / len(embeddings) >= 0.8:
            return species
        return None

    def register(
        self,
        entry_id,
        *,
        formal_matcher: Callable[[np.ndarray], str | None] | None = None,
    ) -> RegistryEntry:
        detail = self.get(entry_id)
        if not detail.can_register:
            raise RegistrationConditionError(
                "Registration conditions not met: "
                + ", ".join(
                    key for key, value in detail.conditions.items() if not value
                )
            )
        if formal_matcher is not None:
            duplicate = self._duplicate_formal_species(entry_id, formal_matcher)
            if duplicate is not None:
                raise RegistrationConditionError(
                    f"Candidate already matches existing species: {duplicate}"
                )
        self._conn.execute(
            "UPDATE registrations SET status='provisional',updated_at=? WHERE id=?",
            (_now(), entry_id),
        )
        self._conn.commit()
        self._refresh(entry_id)
        return self.get(entry_id)

    def match(self, embedding):
        vector = normalize_embedding(embedding)
        best = None
        for entry in self.list():
            prototypes = self._prototypes(entry.id)
            if len(prototypes) == 0:
                events = self._embeddings(entry.id)
                if len(events) == 0:
                    continue
                prototypes = build_prototype(events)[None, :]
            score = float(np.max(cosine_similarity(vector, prototypes)))
            threshold = (
                self.join_threshold
                if entry.status == "candidate"
                else self.registration_threshold
            )
            if score >= threshold and (best is None or score > best[0]):
                best = (score, entry, threshold)
        if best is None:
            return None
        score, entry, threshold = best
        return {
            "id": entry.id,
            "display_name": entry.display_name,
            "status": entry.status,
            "score": score,
            "threshold": threshold,
            "accepted": entry.status in {"confirmed", "mature"},
        }
