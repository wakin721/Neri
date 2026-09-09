"""Local human-feedback overlays for DINOv3 classifiers.

This module deliberately stays separate from registry.py: registry owns unknown
species discovery, while this store owns explicit human corrections against a
fixed checkpoint fingerprint.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import sqlite3

import numpy as np

from .prototype_bank import PrototypeBank, PrototypeRecord
from .simple_shot import deterministic_k_means, normalize_embedding

DINO_DIM = 768
EVENT_GAP_SECONDS = 1800


def feedback_path_for_registry(registry_path: str | Path) -> Path:
    return Path(registry_path).expanduser().resolve().with_name("feedback.sqlite3")


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

    def __post_init__(self):
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


class HumanFeedbackStore:
    def __init__(self, path, *, model_fingerprint, checkpoint_classes, threshold):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.model_fingerprint = str(model_fingerprint)
        self.checkpoint_classes = tuple(checkpoint_classes)
        self.threshold = float(threshold)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self):
        self._conn.executescript("""
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS observations(
          id TEXT PRIMARY KEY, payload TEXT NOT NULL, embedding BLOB NOT NULL
        );
        CREATE TABLE IF NOT EXISTS feedback_operations(
          operation_id TEXT PRIMARY KEY, reverted INTEGER NOT NULL DEFAULT 0
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
        CREATE TABLE IF NOT EXISTS species_learning_state(
          species TEXT PRIMARY KEY, state TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS overlay_prototypes(
          species TEXT NOT NULL, idx INTEGER NOT NULL, embedding BLOB NOT NULL,
          PRIMARY KEY(species, idx)
        );
        """)
        row = self._conn.execute("SELECT value FROM metadata WHERE key='model_fingerprint'").fetchone()
        if row and row[0] != self.model_fingerprint:
            raise ValueError("feedback fingerprint mismatch")
        self._conn.execute(
            "INSERT OR REPLACE INTO metadata VALUES('model_fingerprint',?)",
            (self.model_fingerprint,),
        )
        self._conn.commit()

    def close(self):
        self._conn.close()

    def persist_observation(self, observation: FeedbackObservation) -> str:
        payload = {k: v for k, v in observation.__dict__.items() if k != "embedding"}
        payload["captured_at"] = observation.captured_at.isoformat() if observation.captured_at else None
        self._conn.execute(
            "INSERT OR REPLACE INTO observations VALUES(?,?,?)",
            (observation.id, json.dumps(payload, ensure_ascii=False), observation.embedding.astype('<f4').tobytes()),
        )
        self._conn.commit()
        return observation.id

    def get_observation(self, observation_id: str) -> FeedbackObservation:
        row = self._conn.execute("SELECT * FROM observations WHERE id=?", (observation_id,)).fetchone()
        if row is None:
            raise KeyError(observation_id)
        data = json.loads(row[1])
        if data.get("captured_at"):
            data["captured_at"] = datetime.fromisoformat(data["captured_at"])
        data["bbox"] = tuple(data["bbox"])
        data["embedding"] = normalize_embedding(np.frombuffer(row[2], dtype='<f4').copy())
        return FeedbackObservation(**data)

    def record_feedback(self, observation_id, *, operation_id, action, confirmed_species=None):
        obs = self.get_observation(observation_id)
        if action == "correct":
            positive, negative, kind = obs.predicted_species, None, "correct"
        elif action == "update":
            if confirmed_species not in self.checkpoint_classes:
                raise ValueError("confirmed species must belong to checkpoint")
            positive = confirmed_species
            negative = obs.predicted_species if obs.accepted and obs.predicted_species != confirmed_species else None
            kind = "corrected"
        elif action == "empty":
            positive = negative = None
            kind = "empty"
        else:
            positive = negative = None
            kind = "ignored"
        self._conn.execute("INSERT OR IGNORE INTO feedback_operations(operation_id) VALUES(?)", (operation_id,))
        old = self._conn.execute("SELECT id FROM human_feedback WHERE observation_id=? AND active=1", (observation_id,)).fetchone()
        if old:
            self._conn.execute("UPDATE human_feedback SET active=0 WHERE id=?", (old[0],))
        cur = self._conn.execute(
            "INSERT INTO human_feedback(observation_id,operation_id,feedback_type,predicted_species,confirmed_species,positive_species,hard_negative_species) VALUES(?,?,?,?,?,?,?)",
            (observation_id, operation_id, kind, obs.predicted_species, confirmed_species, positive, negative),
        )
        self._conn.commit()
        return FeedbackRecord(cur.lastrowid, observation_id, operation_id, kind, obs.predicted_species, confirmed_species, positive, negative, True)

    def active_feedback(self, observation_id):
        row = self._conn.execute("SELECT * FROM human_feedback WHERE observation_id=? AND active=1", (observation_id,)).fetchone()
        if not row:
            return None
        return FeedbackRecord(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], bool(row[8]))

    def revert_operation(self, operation_id):
        rows = self._conn.execute("SELECT confirmed_species,predicted_species FROM human_feedback WHERE operation_id=?", (operation_id,)).fetchall()
        self._conn.execute("UPDATE feedback_operations SET reverted=1 WHERE operation_id=?", (operation_id,))
        self._conn.execute("UPDATE human_feedback SET active=0 WHERE operation_id=?", (operation_id,))
        self._conn.commit()
        return {x for row in rows for x in (row[0], row[1]) if x}

    def recompute_species(self, species, feature_center):
        rows = self._conn.execute("SELECT o.embedding FROM observations o JOIN human_feedback h ON o.id=h.observation_id WHERE h.active=1 AND h.positive_species=?", (species,)).fetchall()
        count = len(rows)
        status = "collecting" if count < 4 else "provisional" if count < 10 else "confirmed" if count < 20 else "mature"
        return SpeciesLearningState(species, status, count, min(3, max(1, count // 4)) if count else 0, 1.0, 0.0)

    def learning_state(self, species):
        return SpeciesLearningState(species, "collecting", 0, 0, 0.0, 0.0)

    def prototype_bank(self, feature_center):
        return PrototypeBank((), ())
