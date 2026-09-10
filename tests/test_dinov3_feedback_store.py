from __future__ import annotations

from datetime import datetime
import sqlite3

import numpy as np
import pytest

from system.dinov3.feedback import (
    FeedbackObservation,
    HumanFeedbackStore,
    feedback_path_for_registry,
)

FP = "a" * 64


def vec(index: int) -> np.ndarray:
    value = np.zeros(768, dtype=np.float32)
    value[index] = 1.0
    return value


def make_observation(observation_id: str = "obs-1") -> FeedbackObservation:
    return FeedbackObservation(
        id=observation_id,
        source_path="C:/camera/IMG_0001.JPG",
        media_kind="image",
        box_index=0,
        frame_index=None,
        timestamp_seconds=None,
        bbox=(1.0, 2.0, 30.0, 40.0),
        camera_id="camera-a",
        captured_at=datetime(2026, 9, 9, 10, 0, 0),
        predicted_species="盘羊",
        best_known_species="盘羊",
        accepted=True,
        prediction_source="base",
        registry_id=None,
        known_score=0.8,
        threshold=0.31,
        embedding=vec(0),
    )


def test_feedback_path_is_registry_sibling(tmp_path):
    assert feedback_path_for_registry(tmp_path / "registry.sqlite3") == tmp_path / "feedback.sqlite3"


def test_persist_observation_round_trips_embedding_for_backend_only(tmp_path):
    store = HumanFeedbackStore(
        tmp_path / "feedback.sqlite3",
        model_fingerprint=FP,
        checkpoint_classes=("盘羊", "家牛"),
        threshold=0.31,
    )
    observation = make_observation()
    assert store.persist_observation(observation) == "obs-1"
    loaded = store.get_observation("obs-1")
    assert loaded.id == "obs-1"
    assert loaded.embedding.shape == (768,)
    assert np.allclose(loaded.embedding, vec(0))
    store.close()


def test_feedback_store_rejects_fingerprint_mismatch(tmp_path):
    path = tmp_path / "feedback.sqlite3"
    HumanFeedbackStore(
        path,
        model_fingerprint=FP,
        checkpoint_classes=("盘羊",),
        threshold=0.31,
    ).close()
    with pytest.raises(ValueError, match="fingerprint"):
        HumanFeedbackStore(
            path,
            model_fingerprint="b" * 64,
            checkpoint_classes=("盘羊",),
            threshold=0.31,
        )


def test_feedback_store_enables_wal_foreign_keys_and_integrity(tmp_path):
    path = tmp_path / "feedback.sqlite3"
    store = HumanFeedbackStore(
        path,
        model_fingerprint=FP,
        checkpoint_classes=("盘羊",),
        threshold=0.31,
    )
    assert store._conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    store.close()
