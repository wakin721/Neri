"""Fingerprint-scoped human-feedback learning for DINOv3 classifiers.

The checkpoint remains immutable. Explicit box-level feedback is stored in a
separate SQLite database and is converted into reversible prototype generations
that can be composed with the checkpoint and SpeciesRegistry at inference time.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Iterable

import numpy as np

from .prototype_bank import PrototypeBank, PrototypeRecord
from .simple_shot import deterministic_k_means, normalize_embedding

DINO_DIM = 768
EVENT_GAP_SECONDS = 1800
_POSITIVE_COVERAGE_MIN = 0.80
_HARD_NEGATIVE_FALSE_ACCEPT_MAX = 0.20
_STATUS_RANK = {"collecting": 0, "provisional": 1, "confirmed": 2, "mature": 3}


def feedback_path_for_registry(registry_path: str | Path) -> Path:
    return Path(registry_path).expanduser().resolve().with_name("feedback.sqlite3")


def _feedback_table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def _ensure_registry_assignment_redirect_schema(conn: sqlite3.Connection) -> bool:
    if not _feedback_table_exists(conn, "feedback_registry_assignments"):
        return False
    columns = {
        str(row[1])
        for row in conn.execute(
            "PRAGMA table_info(feedback_registry_assignments)"
        ).fetchall()
    }
    if "identity_restore_allowed" not in columns:
        conn.execute(
            "ALTER TABLE feedback_registry_assignments "
            "ADD COLUMN identity_restore_allowed INTEGER NOT NULL DEFAULT 1"
        )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback_registry_assignment_redirects(
          operation_id TEXT NOT NULL,
          old_registration_id INTEGER NOT NULL,
          new_registration_id INTEGER NOT NULL,
          assigned_common_name TEXT NOT NULL,
          migrated_at TEXT NOT NULL,
          PRIMARY KEY(operation_id, old_registration_id)
        )
        """
    )
    return True


def redirect_registry_assignments_file(
    feedback_path: str | Path,
    redirects: dict[int, int],
) -> int:
    """Redirect legacy Registry assignment ids before duplicate entries are removed.

    Redirected assignments remain auditable but may no longer restore a Registry
    identity during undo, because that identity is now canonical for the merged
    survivor rather than owned by one historical feedback operation.
    """
    path = Path(feedback_path).expanduser().resolve()
    if not redirects or not path.is_file():
        return 0
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        if not _ensure_registry_assignment_redirect_schema(conn):
            conn.commit()
            return 0
        conn.commit()
        conn.execute("BEGIN IMMEDIATE")
        changed = 0
        migrated_at = datetime.now(timezone.utc).isoformat()
        for old_id, new_id in sorted(
            (int(old), int(new)) for old, new in redirects.items()
        ):
            rows = conn.execute(
                """
                SELECT operation_id,assigned_common_name
                FROM feedback_registry_assignments
                WHERE registration_id=?
                ORDER BY operation_id
                """,
                (old_id,),
            ).fetchall()
            for row in rows:
                operation_id = str(row["operation_id"])
                assigned_common_name = str(row["assigned_common_name"])
                conn.execute(
                    """
                    INSERT OR IGNORE INTO feedback_registry_assignment_redirects(
                      operation_id,old_registration_id,new_registration_id,
                      assigned_common_name,migrated_at
                    ) VALUES(?,?,?,?,?)
                    """,
                    (
                        operation_id,
                        old_id,
                        new_id,
                        assigned_common_name,
                        migrated_at,
                    ),
                )
                if old_id == new_id:
                    conn.execute(
                        """
                        UPDATE feedback_registry_assignments
                        SET identity_restore_allowed=0
                        WHERE operation_id=? AND registration_id=?
                        """,
                        (operation_id, old_id),
                    )
                else:
                    existing = conn.execute(
                        """
                        SELECT 1 FROM feedback_registry_assignments
                        WHERE operation_id=? AND registration_id=?
                        """,
                        (operation_id, new_id),
                    ).fetchone()
                    if existing is not None:
                        conn.execute(
                            """
                            UPDATE feedback_registry_assignments
                            SET identity_restore_allowed=0
                            WHERE operation_id=? AND registration_id=?
                            """,
                            (operation_id, new_id),
                        )
                        conn.execute(
                            """
                            DELETE FROM feedback_registry_assignments
                            WHERE operation_id=? AND registration_id=?
                            """,
                            (operation_id, old_id),
                        )
                    else:
                        conn.execute(
                            """
                            UPDATE feedback_registry_assignments
                            SET registration_id=?,identity_restore_allowed=0
                            WHERE operation_id=? AND registration_id=?
                            """,
                            (new_id, operation_id, old_id),
                        )
                changed += 1
        conn.commit()
        return changed
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@dataclass(frozen=True)
class FeedbackObservation:
    id: str
    source_path: str
    media_kind: str
    box_index: int
    frame_index: int | None
    timestamp_seconds: float | None
    bbox: tuple[float, float, float, float]
    camera_id: str
    captured_at: datetime | None
    predicted_species: str
    best_known_species: str | None
    accepted: bool
    prediction_source: str
    registry_id: int | None
    known_score: float
    threshold: float
    embedding: np.ndarray

    def __post_init__(self) -> None:
        value = normalize_embedding(self.embedding).astype(np.float32)
        value.setflags(write=False)
        object.__setattr__(self, "embedding", value)


@dataclass(frozen=True)
class FeedbackRecord:
    id: int
    observation_id: str
    operation_id: str
    feedback_type: str
    predicted_species: str
    confirmed_species: str | None
    positive_species: str | None
    hard_negative_species: str | None
    active: bool


@dataclass(frozen=True)
class SpeciesLearningState:
    species: str
    status: str
    positive_events: int
    prototype_count: int
    positive_coverage: float
    hard_negative_false_accept_rate: float
    needs_review: bool = False
    camera_count: int = 0
    active_generation: str | None = None


@dataclass(frozen=True)
class _EvidenceObservation:
    row_id: int
    observation_id: str
    source_path: str
    camera_id: str
    captured_at: datetime | None
    embedding: np.ndarray


@dataclass(frozen=True)
class _EvidenceEvent:
    key: str
    camera_id: str
    embedding: np.ndarray
    observation_ids: tuple[str, ...]


class HumanFeedbackStore:
    def __init__(
        self,
        path: str | Path,
        *,
        model_fingerprint: str,
        checkpoint_classes: Iterable[str],
        threshold: float,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.model_fingerprint = str(model_fingerprint)
        self.checkpoint_classes = tuple(str(item) for item in checkpoint_classes)
        self.threshold = float(threshold)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata(
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS observations(
              id TEXT PRIMARY KEY,
              payload TEXT NOT NULL,
              embedding BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS feedback_operations(
              operation_id TEXT PRIMARY KEY,
              reverted INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS human_feedback(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              observation_id TEXT NOT NULL,
              operation_id TEXT NOT NULL,
              feedback_type TEXT NOT NULL,
              predicted_species TEXT NOT NULL,
              confirmed_species TEXT,
              positive_species TEXT,
              hard_negative_species TEXT,
              active INTEGER NOT NULL DEFAULT 1,
              supersedes_id INTEGER
            );
            CREATE TABLE IF NOT EXISTS feedback_generations(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              generation_key TEXT NOT NULL UNIQUE,
              species TEXT NOT NULL,
              status TEXT NOT NULL,
              formal INTEGER NOT NULL,
              quality_passed INTEGER NOT NULL,
              positive_coverage REAL NOT NULL,
              hard_negative_false_accept_rate REAL NOT NULL,
              active INTEGER NOT NULL DEFAULT 0,
              created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_feedback_generations_species_active
              ON feedback_generations(species, active);
            CREATE TABLE IF NOT EXISTS feedback_generation_prototypes(
              generation_id INTEGER NOT NULL,
              prototype_index INTEGER NOT NULL,
              embedding BLOB NOT NULL,
              PRIMARY KEY(generation_id, prototype_index),
              FOREIGN KEY(generation_id) REFERENCES feedback_generations(id)
                ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS feedback_learning_states(
              species TEXT PRIMARY KEY,
              payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS feedback_registry_assignments(
              operation_id TEXT NOT NULL,
              registration_id INTEGER NOT NULL,
              previous_common_name TEXT NOT NULL DEFAULT '',
              previous_scientific_name TEXT NOT NULL DEFAULT '',
              assigned_common_name TEXT NOT NULL,
              identity_restore_allowed INTEGER NOT NULL DEFAULT 1,
              PRIMARY KEY(operation_id, registration_id)
            );
            CREATE TABLE IF NOT EXISTS feedback_registry_assignment_redirects(
              operation_id TEXT NOT NULL,
              old_registration_id INTEGER NOT NULL,
              new_registration_id INTEGER NOT NULL,
              assigned_common_name TEXT NOT NULL,
              migrated_at TEXT NOT NULL,
              PRIMARY KEY(operation_id, old_registration_id)
            );
            """
        )
        _ensure_registry_assignment_redirect_schema(self._conn)
        row = self._conn.execute(
            "SELECT value FROM metadata WHERE key='model_fingerprint'"
        ).fetchone()
        if row is not None and row[0] != self.model_fingerprint:
            self._conn.close()
            raise ValueError("feedback fingerprint mismatch")
        self._conn.execute(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES('model_fingerprint',?)",
            (self.model_fingerprint,),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def persist_observation(self, observation: FeedbackObservation) -> str:
        payload = {
            key: value
            for key, value in observation.__dict__.items()
            if key != "embedding"
        }
        payload["captured_at"] = (
            observation.captured_at.isoformat() if observation.captured_at else None
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO observations(id,payload,embedding) VALUES(?,?,?)",
            (
                observation.id,
                json.dumps(payload, ensure_ascii=False),
                observation.embedding.astype("<f4").tobytes(),
            ),
        )
        self._conn.commit()
        return observation.id

    def get_observation(self, observation_id: str) -> FeedbackObservation:
        row = self._conn.execute(
            "SELECT payload,embedding FROM observations WHERE id=?",
            (observation_id,),
        ).fetchone()
        if row is None:
            raise KeyError(observation_id)
        data = json.loads(row["payload"])
        if data.get("captured_at"):
            data["captured_at"] = datetime.fromisoformat(data["captured_at"])
        data["bbox"] = tuple(data["bbox"])
        data["embedding"] = np.frombuffer(row["embedding"], dtype="<f4").copy()
        return FeedbackObservation(**data)

    def record_feedback(
        self,
        observation_id: str,
        *,
        operation_id: str,
        action: str,
        confirmed_species: str | None = None,
    ) -> FeedbackRecord:
        observation = self.get_observation(observation_id)
        if action == "correct":
            if observation.predicted_species == "Unknown" or not observation.accepted:
                raise ValueError("explicit confirmed species required")
            kind = "correct"
            positive_species = observation.predicted_species
            hard_negative_species = None
            confirmed_species = confirmed_species or observation.predicted_species
        elif action == "update":
            if confirmed_species not in self.checkpoint_classes:
                raise ValueError("confirmed species must belong to checkpoint")
            kind = "corrected"
            positive_species = confirmed_species
            hard_negative_species = (
                observation.predicted_species
                if observation.accepted
                and observation.predicted_species != confirmed_species
                and observation.predicted_species in self.checkpoint_classes
                else None
            )
        elif action == "empty":
            kind = "empty"
            positive_species = None
            hard_negative_species = None
            confirmed_species = None
        else:
            kind = "ignored"
            positive_species = None
            hard_negative_species = None
            confirmed_species = None

        operation = self._conn.execute(
            "SELECT reverted FROM feedback_operations WHERE operation_id=?",
            (operation_id,),
        ).fetchone()
        if operation is not None and bool(operation["reverted"]):
            raise ValueError("feedback operation has already been reverted")
        self._conn.execute(
            "INSERT OR IGNORE INTO feedback_operations(operation_id,reverted) VALUES(?,0)",
            (operation_id,),
        )

        previous = self._conn.execute(
            "SELECT id FROM human_feedback WHERE observation_id=? AND active=1",
            (observation_id,),
        ).fetchone()
        supersedes_id = int(previous["id"]) if previous is not None else None
        if supersedes_id is not None:
            self._conn.execute(
                "UPDATE human_feedback SET active=0 WHERE id=?",
                (supersedes_id,),
            )
        cursor = self._conn.execute(
            """
            INSERT INTO human_feedback(
              observation_id,operation_id,feedback_type,predicted_species,
              confirmed_species,positive_species,hard_negative_species,active,
              supersedes_id
            ) VALUES(?,?,?,?,?,?,?,?,?)
            """,
            (
                observation_id,
                operation_id,
                kind,
                observation.predicted_species,
                confirmed_species,
                positive_species,
                hard_negative_species,
                1,
                supersedes_id,
            ),
        )
        self._conn.commit()
        return FeedbackRecord(
            id=int(cursor.lastrowid),
            observation_id=observation_id,
            operation_id=operation_id,
            feedback_type=kind,
            predicted_species=observation.predicted_species,
            confirmed_species=confirmed_species,
            positive_species=positive_species,
            hard_negative_species=hard_negative_species,
            active=True,
        )

    def record_registry_feedback(
        self,
        observation_id: str,
        *,
        operation_id: str,
        registration_id: int,
        previous_common_name: str,
        previous_scientific_name: str,
        confirmed_species: str,
    ) -> FeedbackRecord:
        """Record an auditable Registry assignment without training checkpoint classes."""
        observation = self.get_observation(observation_id)
        species = str(confirmed_species).strip()
        if not species or species in {"Unknown", "空"}:
            raise ValueError("confirmed registry species is required")
        operation = self._conn.execute(
            "SELECT reverted FROM feedback_operations WHERE operation_id=?",
            (operation_id,),
        ).fetchone()
        if operation is not None and bool(operation["reverted"]):
            raise ValueError("feedback operation has already been reverted")
        self._conn.execute(
            "INSERT OR IGNORE INTO feedback_operations(operation_id,reverted) VALUES(?,0)",
            (operation_id,),
        )
        self._conn.execute(
            """
            INSERT OR IGNORE INTO feedback_registry_assignments(
              operation_id,registration_id,previous_common_name,
              previous_scientific_name,assigned_common_name
            ) VALUES(?,?,?,?,?)
            """,
            (
                operation_id,
                int(registration_id),
                str(previous_common_name),
                str(previous_scientific_name),
                species,
            ),
        )

        previous = self._conn.execute(
            "SELECT id FROM human_feedback WHERE observation_id=? AND active=1",
            (observation_id,),
        ).fetchone()
        supersedes_id = int(previous["id"]) if previous is not None else None
        if supersedes_id is not None:
            self._conn.execute(
                "UPDATE human_feedback SET active=0 WHERE id=?",
                (supersedes_id,),
            )
        cursor = self._conn.execute(
            """
            INSERT INTO human_feedback(
              observation_id,operation_id,feedback_type,predicted_species,
              confirmed_species,positive_species,hard_negative_species,active,
              supersedes_id
            ) VALUES(?,?,?,?,?,?,?,?,?)
            """,
            (
                observation_id,
                operation_id,
                "registry",
                observation.predicted_species,
                species,
                None,
                None,
                1,
                supersedes_id,
            ),
        )
        self._conn.commit()
        return FeedbackRecord(
            id=int(cursor.lastrowid),
            observation_id=observation_id,
            operation_id=operation_id,
            feedback_type="registry",
            predicted_species=observation.predicted_species,
            confirmed_species=species,
            positive_species=None,
            hard_negative_species=None,
            active=True,
        )

    def registry_assignments(self, operation_id: str) -> list[dict[str, object]]:
        rows = self._conn.execute(
            """
            SELECT registration_id,previous_common_name,previous_scientific_name,
                   assigned_common_name,identity_restore_allowed
            FROM feedback_registry_assignments
            WHERE operation_id=? ORDER BY registration_id
            """,
            (operation_id,),
        ).fetchall()
        return [
            {
                "registration_id": int(row["registration_id"]),
                "previous_common_name": str(row["previous_common_name"]),
                "previous_scientific_name": str(row["previous_scientific_name"]),
                "assigned_common_name": str(row["assigned_common_name"]),
                "identity_restore_allowed": bool(row["identity_restore_allowed"]),
            }
            for row in rows
        ]

    def representative_observation_id(self, species: str) -> str | None:
        row = self._conn.execute(
            """
            SELECT h.observation_id
            FROM human_feedback h
            JOIN observations o ON o.id=h.observation_id
            WHERE h.active=1 AND h.positive_species=?
            ORDER BY h.id DESC LIMIT 1
            """,
            (species,),
        ).fetchone()
        return None if row is None else str(row["observation_id"])

    @staticmethod
    def _feedback_record(row: sqlite3.Row) -> FeedbackRecord:
        return FeedbackRecord(
            id=int(row["id"]),
            observation_id=str(row["observation_id"]),
            operation_id=str(row["operation_id"]),
            feedback_type=str(row["feedback_type"]),
            predicted_species=str(row["predicted_species"]),
            confirmed_species=row["confirmed_species"],
            positive_species=row["positive_species"],
            hard_negative_species=row["hard_negative_species"],
            active=bool(row["active"]),
        )

    def active_feedback(self, observation_id: str) -> FeedbackRecord | None:
        row = self._conn.execute(
            "SELECT * FROM human_feedback WHERE observation_id=? AND active=1",
            (observation_id,),
        ).fetchone()
        return None if row is None else self._feedback_record(row)

    def _restorable_predecessor(self, feedback_id: int | None) -> int | None:
        current = feedback_id
        while current is not None:
            row = self._conn.execute(
                """
                SELECT h.supersedes_id,o.reverted
                FROM human_feedback h
                JOIN feedback_operations o ON o.operation_id=h.operation_id
                WHERE h.id=?
                """,
                (current,),
            ).fetchone()
            if row is None:
                return None
            if not bool(row["reverted"]):
                return int(current)
            current = (
                int(row["supersedes_id"])
                if row["supersedes_id"] is not None
                else None
            )
        return None

    def revert_operation(self, operation_id: str) -> set[str]:
        operation = self._conn.execute(
            "SELECT reverted FROM feedback_operations WHERE operation_id=?",
            (operation_id,),
        ).fetchone()
        if operation is None:
            raise KeyError(operation_id)
        if bool(operation["reverted"]):
            return set()

        rows = self._conn.execute(
            "SELECT * FROM human_feedback WHERE operation_id=? ORDER BY id",
            (operation_id,),
        ).fetchall()
        affected: set[str] = set()
        for row in rows:
            for species in (
                row["predicted_species"],
                row["confirmed_species"],
                row["positive_species"],
                row["hard_negative_species"],
            ):
                if species and species != "Unknown":
                    affected.add(str(species))

        self._conn.execute(
            "UPDATE feedback_operations SET reverted=1 WHERE operation_id=?",
            (operation_id,),
        )
        for row in rows:
            if not bool(row["active"]):
                continue
            self._conn.execute(
                "UPDATE human_feedback SET active=0 WHERE id=?",
                (row["id"],),
            )
            predecessor = self._restorable_predecessor(
                int(row["supersedes_id"])
                if row["supersedes_id"] is not None
                else None
            )
            if predecessor is not None:
                self._conn.execute(
                    "UPDATE human_feedback SET active=1 WHERE id=?",
                    (predecessor,),
                )
        self._conn.commit()
        return affected

    @staticmethod
    def _timestamp_seconds(value: datetime) -> float:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        else:
            value = value.astimezone(timezone.utc)
        return value.timestamp()

    @staticmethod
    def _source_identity(source_path: str) -> str:
        return Path(source_path).expanduser().as_posix()

    def _evidence_observations(
        self,
        species: str,
        *,
        column: str,
    ) -> list[_EvidenceObservation]:
        if column not in {"positive_species", "hard_negative_species"}:
            raise ValueError("Unsupported feedback evidence column")
        rows = self._conn.execute(
            f"""
            SELECT h.id,h.observation_id,o.payload,o.embedding
            FROM human_feedback h
            JOIN observations o ON o.id=h.observation_id
            WHERE h.active=1 AND h.{column}=?
            ORDER BY h.id
            """,
            (species,),
        ).fetchall()
        result: list[_EvidenceObservation] = []
        for row in rows:
            payload = json.loads(row["payload"])
            captured_at = (
                datetime.fromisoformat(payload["captured_at"])
                if payload.get("captured_at")
                else None
            )
            result.append(
                _EvidenceObservation(
                    row_id=int(row["id"]),
                    observation_id=str(row["observation_id"]),
                    source_path=str(payload.get("source_path") or ""),
                    camera_id=str(payload.get("camera_id") or "unknown"),
                    captured_at=captured_at,
                    embedding=normalize_embedding(
                        np.frombuffer(row["embedding"], dtype="<f4").copy()
                    ),
                )
            )
        return result

    def _group_events(
        self,
        observations: Iterable[_EvidenceObservation],
    ) -> list[_EvidenceEvent]:
        by_camera: dict[str, list[_EvidenceObservation]] = {}
        for observation in observations:
            by_camera.setdefault(observation.camera_id, []).append(observation)

        events: list[_EvidenceEvent] = []
        for camera_id, items in sorted(by_camera.items()):
            missing: dict[str, list[_EvidenceObservation]] = {}
            timed: list[_EvidenceObservation] = []
            for item in items:
                if item.captured_at is None:
                    missing.setdefault(
                        self._source_identity(item.source_path), []
                    ).append(item)
                else:
                    timed.append(item)

            for identity, group in sorted(missing.items()):
                embedding = normalize_embedding(
                    np.stack([item.embedding for item in group]).mean(axis=0)
                )
                events.append(
                    _EvidenceEvent(
                        key=f"missing|{camera_id}|{identity}",
                        camera_id=camera_id,
                        embedding=embedding,
                        observation_ids=tuple(item.observation_id for item in group),
                    )
                )

            timed.sort(
                key=lambda item: (
                    self._timestamp_seconds(item.captured_at),
                    item.source_path,
                    item.row_id,
                )
            )
            cluster: list[_EvidenceObservation] = []
            previous_seconds: float | None = None
            for item in timed:
                current_seconds = self._timestamp_seconds(item.captured_at)
                if (
                    cluster
                    and previous_seconds is not None
                    and current_seconds - previous_seconds >= EVENT_GAP_SECONDS
                ):
                    events.append(self._make_timed_event(camera_id, cluster))
                    cluster = []
                cluster.append(item)
                previous_seconds = current_seconds
            if cluster:
                events.append(self._make_timed_event(camera_id, cluster))
        return events

    def _make_timed_event(
        self,
        camera_id: str,
        observations: list[_EvidenceObservation],
    ) -> _EvidenceEvent:
        first = observations[0]
        marker = self._timestamp_seconds(first.captured_at)
        embedding = normalize_embedding(
            np.stack([item.embedding for item in observations]).mean(axis=0)
        )
        return _EvidenceEvent(
            key=f"timed|{camera_id}|{marker:.6f}|{first.row_id}",
            camera_id=camera_id,
            embedding=embedding,
            observation_ids=tuple(item.observation_id for item in observations),
        )

    @staticmethod
    def _prototype_limit(status: str) -> int:
        if status == "mature":
            return 3
        if status == "confirmed":
            return 2
        return 1

    @staticmethod
    def _scores(centered: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
        if len(centered) == 0:
            return np.empty((0,), dtype=np.float32)
        deltas = centered[:, None, :] - prototypes[None, :, :]
        distances = np.einsum("nkd,nkd->nk", deltas, deltas, optimize=True)
        winner_indices = np.argmin(distances, axis=1)
        scores = np.empty((len(centered),), dtype=np.float32)
        for index, winner_index in enumerate(winner_indices):
            vector = centered[index]
            prototype = prototypes[int(winner_index)]
            denominator = float(np.linalg.norm(vector) * np.linalg.norm(prototype))
            scores[index] = (
                float(np.dot(vector, prototype) / denominator)
                if denominator > 0 and np.isfinite(denominator)
                else -1.0
            )
        return scores

    def _build_candidate(
        self,
        positive_events: list[_EvidenceEvent],
        negative_events: list[_EvidenceEvent],
        feature_center: np.ndarray,
        *,
        status: str,
    ) -> tuple[np.ndarray, float, float, bool]:
        center = np.asarray(feature_center, dtype=np.float32)
        if center.shape != (DINO_DIM,) or not np.isfinite(center).all():
            raise ValueError("Expected finite feature_center with shape (768,)")
        positive_centered = np.stack(
            [event.embedding - center for event in positive_events]
        ).astype(np.float32)
        prototypes = deterministic_k_means(
            positive_centered,
            max_k=self._prototype_limit(status),
        ).astype(np.float32)
        positive_scores = self._scores(positive_centered, prototypes)
        positive_coverage = float(np.mean(positive_scores >= self.threshold))

        if negative_events:
            negative_centered = np.stack(
                [event.embedding - center for event in negative_events]
            ).astype(np.float32)
            negative_scores = self._scores(negative_centered, prototypes)
            false_accept_rate = float(np.mean(negative_scores >= self.threshold))
        else:
            false_accept_rate = 0.0
        quality_passed = (
            positive_coverage >= _POSITIVE_COVERAGE_MIN
            and (
                not negative_events
                or false_accept_rate <= _HARD_NEGATIVE_FALSE_ACCEPT_MAX
            )
        )
        return prototypes, positive_coverage, false_accept_rate, quality_passed

    @staticmethod
    def _eligible_status(event_count: int, camera_count: int) -> str:
        if event_count >= 20 and camera_count >= 3:
            return "mature"
        if event_count >= 10 and camera_count >= 2:
            return "confirmed"
        if event_count >= 4:
            return "provisional"
        return "collecting"

    def _active_generation(self, species: str) -> sqlite3.Row | None:
        return self._conn.execute(
            """
            SELECT * FROM feedback_generations
            WHERE species=? AND active=1
            ORDER BY id DESC LIMIT 1
            """,
            (species,),
        ).fetchone()

    def _generation_prototypes(self, generation_id: int) -> tuple[np.ndarray, ...]:
        rows = self._conn.execute(
            """
            SELECT embedding FROM feedback_generation_prototypes
            WHERE generation_id=? ORDER BY prototype_index
            """,
            (generation_id,),
        ).fetchall()
        return tuple(
            np.frombuffer(row["embedding"], dtype="<f4").copy() for row in rows
        )

    def _activate_generation(
        self,
        species: str,
        *,
        status: str,
        prototypes: np.ndarray,
        quality_passed: bool,
        positive_coverage: float,
        false_accept_rate: float,
    ) -> sqlite3.Row:
        digest = hashlib.sha256()
        digest.update(self.model_fingerprint.encode("ascii", errors="ignore"))
        digest.update(species.encode("utf-8"))
        digest.update(status.encode("ascii"))
        digest.update(prototypes.astype("<f4").tobytes())
        generation_key = digest.hexdigest()
        row = self._conn.execute(
            "SELECT * FROM feedback_generations WHERE generation_key=?",
            (generation_key,),
        ).fetchone()
        if row is None:
            cursor = self._conn.execute(
                """
                INSERT INTO feedback_generations(
                  generation_key,species,status,formal,quality_passed,
                  positive_coverage,hard_negative_false_accept_rate,active,created_at
                ) VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    generation_key,
                    species,
                    status,
                    int(status in {"confirmed", "mature"}),
                    int(quality_passed),
                    float(positive_coverage),
                    float(false_accept_rate),
                    0,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            generation_id = int(cursor.lastrowid)
            for index, prototype in enumerate(prototypes):
                self._conn.execute(
                    """
                    INSERT INTO feedback_generation_prototypes(
                      generation_id,prototype_index,embedding
                    ) VALUES(?,?,?)
                    """,
                    (
                        generation_id,
                        index,
                        prototype.astype("<f4").tobytes(),
                    ),
                )
        else:
            generation_id = int(row["id"])
        self._conn.execute(
            "UPDATE feedback_generations SET active=0 WHERE species=?",
            (species,),
        )
        self._conn.execute(
            "UPDATE feedback_generations SET active=1 WHERE id=?",
            (generation_id,),
        )
        return self._conn.execute(
            "SELECT * FROM feedback_generations WHERE id=?",
            (generation_id,),
        ).fetchone()

    def _clear_generation(self, species: str) -> None:
        self._conn.execute(
            "UPDATE feedback_generations SET active=0 WHERE species=?",
            (species,),
        )

    def _save_state(self, state: SpeciesLearningState) -> SpeciesLearningState:
        self._conn.execute(
            "INSERT OR REPLACE INTO feedback_learning_states(species,payload) VALUES(?,?)",
            (
                state.species,
                json.dumps(
                    {
                        "species": state.species,
                        "status": state.status,
                        "positive_events": state.positive_events,
                        "prototype_count": state.prototype_count,
                        "positive_coverage": state.positive_coverage,
                        "hard_negative_false_accept_rate": state.hard_negative_false_accept_rate,
                        "needs_review": state.needs_review,
                        "camera_count": state.camera_count,
                        "active_generation": state.active_generation,
                    },
                    ensure_ascii=False,
                ),
            ),
        )
        self._conn.commit()
        return state

    def recompute_species(
        self,
        species: str,
        feature_center: np.ndarray,
    ) -> SpeciesLearningState:
        if species not in self.checkpoint_classes:
            raise ValueError("feedback learning only supports checkpoint species")
        positive_events = self._group_events(
            self._evidence_observations(species, column="positive_species")
        )
        negative_events = self._group_events(
            self._evidence_observations(species, column="hard_negative_species")
        )
        event_count = len(positive_events)
        camera_count = len({event.camera_id for event in positive_events})
        eligible_status = self._eligible_status(event_count, camera_count)

        if eligible_status == "collecting":
            self._clear_generation(species)
            return self._save_state(
                SpeciesLearningState(
                    species=species,
                    status="collecting",
                    positive_events=event_count,
                    prototype_count=0,
                    positive_coverage=0.0,
                    hard_negative_false_accept_rate=0.0,
                    needs_review=False,
                    camera_count=camera_count,
                    active_generation=None,
                )
            )

        try:
            candidate_prototypes, coverage, false_accept_rate, quality_passed = (
                self._build_candidate(
                    positive_events,
                    negative_events,
                    feature_center,
                    status=eligible_status,
                )
            )
        except (ValueError, FloatingPointError):
            candidate_prototypes = np.empty((0, DINO_DIM), dtype=np.float32)
            coverage = 0.0
            false_accept_rate = 1.0 if negative_events else 0.0
            quality_passed = False

        if eligible_status == "provisional":
            if len(candidate_prototypes) == 0:
                self._clear_generation(species)
                return self._save_state(
                    SpeciesLearningState(
                        species=species,
                        status="provisional",
                        positive_events=event_count,
                        prototype_count=0,
                        positive_coverage=coverage,
                        hard_negative_false_accept_rate=false_accept_rate,
                        needs_review=True,
                        camera_count=camera_count,
                        active_generation=None,
                    )
                )
            generation = self._activate_generation(
                species,
                status="provisional",
                prototypes=candidate_prototypes[:1],
                quality_passed=quality_passed,
                positive_coverage=coverage,
                false_accept_rate=false_accept_rate,
            )
            return self._save_state(
                SpeciesLearningState(
                    species=species,
                    status="provisional",
                    positive_events=event_count,
                    prototype_count=len(candidate_prototypes[:1]),
                    positive_coverage=coverage,
                    hard_negative_false_accept_rate=false_accept_rate,
                    needs_review=False,
                    camera_count=camera_count,
                    active_generation=str(generation["generation_key"]),
                )
            )

        if quality_passed and len(candidate_prototypes):
            generation = self._activate_generation(
                species,
                status=eligible_status,
                prototypes=candidate_prototypes,
                quality_passed=True,
                positive_coverage=coverage,
                false_accept_rate=false_accept_rate,
            )
            return self._save_state(
                SpeciesLearningState(
                    species=species,
                    status=eligible_status,
                    positive_events=event_count,
                    prototype_count=len(candidate_prototypes),
                    positive_coverage=coverage,
                    hard_negative_false_accept_rate=false_accept_rate,
                    needs_review=False,
                    camera_count=camera_count,
                    active_generation=str(generation["generation_key"]),
                )
            )

        active = self._active_generation(species)
        if (
            active is not None
            and bool(active["formal"])
            and _STATUS_RANK[str(active["status"])] <= _STATUS_RANK[eligible_status]
        ):
            prototypes = self._generation_prototypes(int(active["id"]))
            return self._save_state(
                SpeciesLearningState(
                    species=species,
                    status=str(active["status"]),
                    positive_events=event_count,
                    prototype_count=len(prototypes),
                    positive_coverage=coverage,
                    hard_negative_false_accept_rate=false_accept_rate,
                    needs_review=True,
                    camera_count=camera_count,
                    active_generation=str(active["generation_key"]),
                )
            )

        provisional_prototypes, provisional_coverage, provisional_far, _ = (
            self._build_candidate(
                positive_events,
                negative_events,
                feature_center,
                status="provisional",
            )
        )
        generation = self._activate_generation(
            species,
            status="provisional",
            prototypes=provisional_prototypes[:1],
            quality_passed=False,
            positive_coverage=provisional_coverage,
            false_accept_rate=provisional_far,
        )
        return self._save_state(
            SpeciesLearningState(
                species=species,
                status="provisional",
                positive_events=event_count,
                prototype_count=1,
                positive_coverage=coverage,
                hard_negative_false_accept_rate=false_accept_rate,
                needs_review=True,
                camera_count=camera_count,
                active_generation=str(generation["generation_key"]),
            )
        )

    def learning_state(self, species: str) -> SpeciesLearningState:
        row = self._conn.execute(
            "SELECT payload FROM feedback_learning_states WHERE species=?",
            (species,),
        ).fetchone()
        if row is None:
            return SpeciesLearningState(
                species=species,
                status="collecting",
                positive_events=0,
                prototype_count=0,
                positive_coverage=0.0,
                hard_negative_false_accept_rate=0.0,
            )
        return SpeciesLearningState(**json.loads(row["payload"]))

    def cluster_details(
        self,
        species: str,
        feature_center: np.ndarray,
    ) -> list[dict[str, object]]:
        """Describe current feedback clusters without exposing 768-D embeddings."""
        positive_events = self._group_events(
            self._evidence_observations(species, column="positive_species")
        )
        if not positive_events:
            return []

        center = feature_center
        if hasattr(center, "numpy"):
            center = center.numpy()
        center = np.asarray(center, dtype=np.float32)
        if center.shape != (DINO_DIM,) or not np.isfinite(center).all():
            raise ValueError("Expected finite feature_center with shape (768,)")

        state = self.learning_state(species)
        active_generation = self._active_generation(species)
        active = active_generation is not None
        if active_generation is not None:
            prototype_values = self._generation_prototypes(int(active_generation["id"]))
            prototypes = np.stack(prototype_values).astype(np.float32, copy=False)
        else:
            centered = np.stack(
                [event.embedding - center for event in positive_events]
            ).astype(np.float32)
            prototypes = deterministic_k_means(centered, max_k=1).astype(
                np.float32,
                copy=False,
            )

        event_vectors = np.stack(
            [event.embedding - center for event in positive_events]
        ).astype(np.float32)
        deltas = event_vectors[:, None, :] - prototypes[None, :, :]
        distances = np.einsum("nkd,nkd->nk", deltas, deltas, optimize=True)
        labels = np.argmin(distances, axis=1)
        result: list[dict[str, object]] = []
        for prototype_index in range(len(prototypes)):
            member_indices = np.flatnonzero(labels == prototype_index).tolist()
            if not member_indices:
                continue
            ordered = sorted(
                member_indices,
                key=lambda index: (float(distances[index, prototype_index]), index),
            )
            refs: list[dict[str, object]] = []
            seen: set[str] = set()
            for event_index in ordered:
                for observation_id in positive_events[event_index].observation_ids:
                    if observation_id in seen:
                        continue
                    seen.add(observation_id)
                    refs.append(
                        {
                            "kind": "observation",
                            "observation_id": observation_id,
                        }
                    )
                    if len(refs) >= 3:
                        break
                if len(refs) >= 3:
                    break
            member_distances = distances[member_indices, prototype_index]
            result.append(
                {
                    "id": f"feedback:{species}:{prototype_index}",
                    "label": (
                        f"Feedback Cluster #{prototype_index + 1}"
                        if active
                        else "反馈证据（尚未形成 prototype）"
                    ),
                    "source": "feedback" if active else "feedback_evidence",
                    "prototype_index": prototype_index,
                    "event_count": len(member_indices),
                    "camera_count": len(
                        {positive_events[index].camera_id for index in member_indices}
                    ),
                    "sample_count": sum(
                        len(positive_events[index].observation_ids)
                        for index in member_indices
                    ),
                    "mean_squared_distance": float(np.mean(member_distances)),
                    "active": active,
                    "learning_status": state.status,
                    "example_refs": refs,
                }
            )
        return result

    def prototype_bank(self, feature_center: np.ndarray) -> PrototypeBank:
        del feature_center  # Prototypes are persisted in checkpoint-centered space.
        rows = self._conn.execute(
            """
            SELECT g.id,g.species,g.status,g.formal
            FROM feedback_generations g
            WHERE g.active=1
            ORDER BY g.species,g.id
            """
        ).fetchall()
        formal: list[PrototypeRecord] = []
        provisional: list[PrototypeRecord] = []
        for row in rows:
            for embedding in self._generation_prototypes(int(row["id"])):
                record = PrototypeRecord(
                    species=str(row["species"]),
                    embedding=embedding,
                    source="feedback",
                    registration_status=str(row["status"]),
                )
                (formal if bool(row["formal"]) else provisional).append(record)
        return PrototypeBank(tuple(formal), tuple(provisional))
