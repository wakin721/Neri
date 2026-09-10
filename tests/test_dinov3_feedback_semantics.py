from __future__ import annotations

import pytest

from system.dinov3.feedback import HumanFeedbackStore
from tests.test_dinov3_feedback_store import FP, make_observation


def _store(tmp_path):
    return HumanFeedbackStore(
        tmp_path / "feedback.sqlite3",
        model_fingerprint=FP,
        checkpoint_classes=("盘羊", "家牛"),
        threshold=0.31,
    )


def test_correct_creates_positive_only(tmp_path):
    store = _store(tmp_path)
    store.persist_observation(make_observation())
    record = store.record_feedback(
        "obs-1",
        operation_id="op-1",
        action="correct",
        confirmed_species="盘羊",
    )
    assert record.positive_species == "盘羊"
    assert record.hard_negative_species is None
    store.close()


def test_accepted_correction_creates_positive_and_hard_negative(tmp_path):
    store = _store(tmp_path)
    store.persist_observation(make_observation())
    record = store.record_feedback(
        "obs-1",
        operation_id="op-1",
        action="update",
        confirmed_species="家牛",
    )
    assert record.positive_species == "家牛"
    assert record.hard_negative_species == "盘羊"
    store.close()


def test_rejected_unknown_correction_has_no_species_hard_negative(tmp_path):
    obs = make_observation()
    obs = obs.__class__(
        **{
            **obs.__dict__,
            "predicted_species": "Unknown",
            "best_known_species": "盘羊",
            "accepted": False,
        }
    )
    store = _store(tmp_path)
    store.persist_observation(obs)
    record = store.record_feedback(
        "obs-1",
        operation_id="op-1",
        action="update",
        confirmed_species="家牛",
    )
    assert record.positive_species == "家牛"
    assert record.hard_negative_species is None
    store.close()


def test_correct_rejects_unknown_without_explicit_species(tmp_path):
    obs = make_observation()
    obs = obs.__class__(
        **{
            **obs.__dict__,
            "predicted_species": "Unknown",
            "accepted": False,
        }
    )
    store = _store(tmp_path)
    store.persist_observation(obs)
    with pytest.raises(ValueError, match="explicit confirmed species"):
        store.record_feedback("obs-1", operation_id="op-1", action="correct")
    store.close()


def test_update_rejects_species_outside_checkpoint(tmp_path):
    store = _store(tmp_path)
    store.persist_observation(make_observation())
    with pytest.raises(ValueError, match="checkpoint"):
        store.record_feedback(
            "obs-1", operation_id="op-1", action="update", confirmed_species="新物种"
        )
    store.close()


def test_unverified_is_ignored_and_not_learning_evidence(tmp_path):
    store = _store(tmp_path)
    store.persist_observation(make_observation())
    record = store.record_feedback(
        "obs-1", operation_id="op-1", action="unverified"
    )
    assert record.feedback_type == "ignored"
    assert record.positive_species is None
    assert record.hard_negative_species is None
    store.close()


def test_revert_restores_superseded_feedback(tmp_path):
    store = _store(tmp_path)
    store.persist_observation(make_observation())
    store.record_feedback(
        "obs-1",
        operation_id="op-1",
        action="correct",
        confirmed_species="盘羊",
    )
    store.record_feedback(
        "obs-1",
        operation_id="op-2",
        action="update",
        confirmed_species="家牛",
    )
    assert store.active_feedback("obs-1").confirmed_species == "家牛"
    assert store.revert_operation("op-2") == {"盘羊", "家牛"}
    restored = store.active_feedback("obs-1")
    assert restored is not None
    assert restored.operation_id == "op-1"
    assert restored.positive_species == "盘羊"
    store.close()


def test_revert_unknown_operation_raises_key_error(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(KeyError):
        store.revert_operation("missing")
    store.close()
