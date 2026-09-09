from __future__ import annotations

from datetime import timedelta

import numpy as np

from system.dinov3.feedback import HumanFeedbackStore
from tests.test_dinov3_feedback_store import FP, make_observation


def add_positive(store, index, *, camera, minutes):
    obs = make_observation(f"obs-{index}")
    obs = obs.__class__(
        **{
            **obs.__dict__,
            "camera_id": camera,
            "captured_at": obs.captured_at + timedelta(minutes=minutes),
            "source_path": f"C:/camera/{index}.JPG",
        }
    )
    store.persist_observation(obs)
    store.record_feedback(
        obs.id,
        operation_id=f"op-{index}",
        action="correct",
        confirmed_species="盘羊",
    )


def test_four_events_create_provisional_state(tmp_path):
    store = HumanFeedbackStore(
        tmp_path / "feedback.sqlite3",
        model_fingerprint=FP,
        checkpoint_classes=("盘羊", "家牛"),
        threshold=0.31,
    )
    for i in range(4):
        add_positive(store, i, camera="cam-a", minutes=i * 31)
    state = store.recompute_species("盘羊", np.zeros(768, dtype=np.float32))
    assert state.status == "provisional"
    assert state.prototype_count == 1
    store.close()


def test_ten_events_two_cameras_confirm(tmp_path):
    store = HumanFeedbackStore(
        tmp_path / "feedback.sqlite3",
        model_fingerprint=FP,
        checkpoint_classes=("盘羊", "家牛"),
        threshold=0.31,
    )
    for i in range(10):
        add_positive(store, i, camera=f"cam-{i % 2}", minutes=i * 31)
    state = store.recompute_species("盘羊", np.zeros(768, dtype=np.float32))
    assert state.status == "confirmed"
    assert state.positive_coverage >= 0.8
    store.close()


def test_twenty_events_three_cameras_mature(tmp_path):
    store = HumanFeedbackStore(
        tmp_path / "feedback.sqlite3",
        model_fingerprint=FP,
        checkpoint_classes=("盘羊", "家牛"),
        threshold=0.31,
    )
    for i in range(20):
        add_positive(store, i, camera=f"cam-{i % 3}", minutes=i * 31)
    state = store.recompute_species("盘羊", np.zeros(768, dtype=np.float32))
    assert state.status == "mature"
    assert 1 <= state.prototype_count <= 3
    store.close()
