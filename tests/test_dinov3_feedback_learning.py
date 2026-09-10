from __future__ import annotations

from datetime import timedelta
from hashlib import sha256

import numpy as np
import torch

from system.dinov3.checkpoint import load_checkpoint
from system.dinov3.feedback import HumanFeedbackStore
from tests.dinov3_multi_prototype_fixtures import make_multi_prototype_payload
from tests.test_dinov3_feedback_store import FP, make_observation, vec


def make_store(tmp_path):
    return HumanFeedbackStore(
        tmp_path / "feedback.sqlite3",
        model_fingerprint=FP,
        checkpoint_classes=("盘羊", "家牛"),
        threshold=0.31,
    )


def add_positive(store, index, *, camera, minutes=0, seconds=0, embedding=None):
    obs = make_observation(f"obs-{index}")
    obs = obs.__class__(
        **{
            **obs.__dict__,
            "camera_id": camera,
            "captured_at": obs.captured_at
            + timedelta(minutes=minutes, seconds=seconds),
            "source_path": f"C:/camera/{index}.JPG",
            "embedding": vec(0) if embedding is None else embedding,
        }
    )
    store.persist_observation(obs)
    store.record_feedback(
        obs.id,
        operation_id=f"op-{index}",
        action="correct",
        confirmed_species="盘羊",
    )
    return obs


def add_hard_negative(store, index, *, camera, minutes, embedding=None):
    obs = make_observation(f"neg-{index}")
    obs = obs.__class__(
        **{
            **obs.__dict__,
            "camera_id": camera,
            "captured_at": obs.captured_at + timedelta(minutes=minutes),
            "source_path": f"C:/camera/neg-{index}.JPG",
            "embedding": vec(0) if embedding is None else embedding,
        }
    )
    store.persist_observation(obs)
    store.record_feedback(
        obs.id,
        operation_id=f"neg-op-{index}",
        action="update",
        confirmed_species="家牛",
    )
    return obs


def test_four_events_create_assistive_provisional_overlay(tmp_path):
    store = make_store(tmp_path)
    for i in range(4):
        add_positive(store, i, camera="cam-a", minutes=i * 31)
    state = store.recompute_species("盘羊", np.zeros(768, dtype=np.float32))
    bank = store.prototype_bank(np.zeros(768, dtype=np.float32))
    assert state.status == "provisional"
    assert state.positive_events == 4
    assert state.prototype_count == 1
    assert len(bank.provisional) == 1
    assert bank.formal == ()
    assert bank.provisional[0].source == "feedback"
    store.close()


def test_ten_events_two_cameras_promote_when_quality_gate_passes(tmp_path):
    store = make_store(tmp_path)
    for i in range(10):
        add_positive(store, i, camera=f"cam-{i % 2}", minutes=i * 31)
    state = store.recompute_species("盘羊", np.zeros(768, dtype=np.float32))
    bank = store.prototype_bank(np.zeros(768, dtype=np.float32))
    assert state.status == "confirmed"
    assert state.camera_count == 2
    assert state.positive_coverage >= 0.80
    assert bank.provisional == ()
    assert 1 <= len(bank.formal) <= 2
    store.close()


def test_twenty_events_three_cameras_reach_mature(tmp_path):
    store = make_store(tmp_path)
    for i in range(20):
        add_positive(store, i, camera=f"cam-{i % 3}", minutes=i * 31)
    state = store.recompute_species("盘羊", np.zeros(768, dtype=np.float32))
    assert state.status == "mature"
    assert state.camera_count == 3
    assert 1 <= state.prototype_count <= 3
    store.close()


def test_event_gap_uses_strict_1800_second_boundary(tmp_path):
    store = make_store(tmp_path)
    add_positive(store, 0, camera="cam-a", seconds=0)
    add_positive(store, 1, camera="cam-a", seconds=1799)
    add_positive(store, 2, camera="cam-a", seconds=3599)
    state = store.recompute_species("盘羊", np.zeros(768, dtype=np.float32))
    assert state.positive_events == 2
    assert state.status == "collecting"
    store.close()


def test_missing_timestamps_group_by_source_path_identity(tmp_path):
    store = make_store(tmp_path)
    for index, source in enumerate(("same.jpg", "same.jpg", "other.jpg")):
        obs = make_observation(f"missing-{index}")
        obs = obs.__class__(
            **{
                **obs.__dict__,
                "captured_at": None,
                "camera_id": "cam-a",
                "source_path": f"C:/camera/{source}",
            }
        )
        store.persist_observation(obs)
        store.record_feedback(
            obs.id,
            operation_id=f"missing-op-{index}",
            action="correct",
            confirmed_species="盘羊",
        )
    state = store.recompute_species("盘羊", np.zeros(768, dtype=np.float32))
    assert state.positive_events == 2
    store.close()


def test_hard_negative_false_accept_rate_blocks_confirmation(tmp_path):
    store = make_store(tmp_path)
    for i in range(10):
        add_positive(store, i, camera=f"cam-{i % 2}", minutes=i * 31)
    for i in range(3):
        add_hard_negative(
            store,
            i,
            camera=f"neg-cam-{i}",
            minutes=400 + i * 31,
            embedding=vec(0),
        )
    state = store.recompute_species("盘羊", np.zeros(768, dtype=np.float32))
    bank = store.prototype_bank(np.zeros(768, dtype=np.float32))
    assert state.status == "provisional"
    assert state.needs_review is True
    assert state.hard_negative_false_accept_rate > 0.20
    assert bank.formal == ()
    assert len(bank.provisional) == 1
    store.close()


def test_failed_generation_preserves_last_known_good_confirmed_generation(tmp_path):
    store = make_store(tmp_path)
    for i in range(10):
        add_positive(store, i, camera=f"cam-{i % 2}", minutes=i * 31)
    confirmed = store.recompute_species("盘羊", np.zeros(768, dtype=np.float32))
    assert confirmed.status == "confirmed"
    assert confirmed.active_generation is not None
    generation = confirmed.active_generation
    prototypes_before = tuple(
        record.embedding.copy()
        for record in store.prototype_bank(np.zeros(768, dtype=np.float32)).formal
    )

    for i in range(3):
        add_hard_negative(
            store,
            i,
            camera=f"neg-cam-{i}",
            minutes=400 + i * 31,
            embedding=vec(0),
        )
    failed = store.recompute_species("盘羊", np.zeros(768, dtype=np.float32))
    prototypes_after = tuple(
        record.embedding.copy()
        for record in store.prototype_bank(np.zeros(768, dtype=np.float32)).formal
    )

    assert failed.status == "confirmed"
    assert failed.needs_review is True
    assert failed.active_generation == generation
    assert len(prototypes_after) == len(prototypes_before)
    assert all(
        np.allclose(before, after)
        for before, after in zip(prototypes_before, prototypes_after, strict=True)
    )
    store.close()


def test_feedback_learning_never_mutates_checkpoint_bytes(tmp_path):
    checkpoint_path = tmp_path / "multi_prototype.pt"
    torch.save(
        make_multi_prototype_payload(
            classes=("盘羊", "家牛"),
            threshold=0.31,
        ),
        checkpoint_path,
    )
    checkpoint = load_checkpoint(checkpoint_path)
    checksum_before = sha256(checkpoint_path.read_bytes()).hexdigest()
    fingerprint_before = checkpoint.fingerprint

    store = HumanFeedbackStore(
        tmp_path / "feedback.sqlite3",
        model_fingerprint=checkpoint.fingerprint,
        checkpoint_classes=checkpoint.classes,
        threshold=checkpoint.threshold,
    )
    for index in range(10):
        obs = make_observation(f"immutable-{index}")
        obs = obs.__class__(
            **{
                **obs.__dict__,
                "camera_id": f"cam-{index % 2}",
                "captured_at": obs.captured_at + timedelta(minutes=index * 31),
                "source_path": f"C:/camera/immutable-{index}.JPG",
                "embedding": vec(0),
            }
        )
        store.persist_observation(obs)
        store.record_feedback(
            obs.id,
            operation_id=f"immutable-op-{index}",
            action="correct",
            confirmed_species="盘羊",
        )

    state = store.recompute_species("盘羊", np.zeros(768, dtype=np.float32))
    bank = store.prototype_bank(np.zeros(768, dtype=np.float32))
    assert state.status == "confirmed"
    assert bank.formal
    store.close()

    checksum_after = sha256(checkpoint_path.read_bytes()).hexdigest()
    reloaded = load_checkpoint(checkpoint_path)
    assert checksum_after == checksum_before
    assert reloaded.fingerprint == fingerprint_before
