from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from system.dinov3.dual_overlay import (
    MultiDualHumanFeedbackStore,
    MultiDualSpeciesRegistry,
    multi_dual_feedback_path_for_registry,
)
from system.dinov3.feedback import _EvidenceEvent
from system.dinov3.rejection import MultiDualRejectionConfig


FP = "e" * 64
BASE = datetime(2026, 9, 10, 0, 0, 0)


def _unit(index: int = 0) -> np.ndarray:
    value = np.zeros(768, dtype=np.float32)
    value[index] = 1.0
    return value


def _rejection() -> MultiDualRejectionConfig:
    return MultiDualRejectionConfig(
        cosine_threshold=0.3,
        squared_distance_threshold=0.5,
    )


def test_dual_registry_overlay_prototypes_use_full_cl2n(tmp_path):
    registry = MultiDualSpeciesRegistry(
        tmp_path / "registry.sqlite3",
        model_fingerprint=FP,
        consistency_threshold=0.5,
    )
    entry = None
    for index in range(4):
        if entry is None:
            entry = registry.record_unknown(
                _unit(0),
                camera_id="cam-0",
                captured_at=BASE + timedelta(hours=index),
                source_path=f"{index}.jpg",
            )
        else:
            entry = registry.record_observation(
                entry.id,
                _unit(0),
                camera_id="cam-0",
                captured_at=BASE + timedelta(hours=index),
                source_path=f"{index}.jpg",
            )
    entry = registry.set_identity(entry.id, common_name="new-species")
    registry.register(entry.id)

    center = np.zeros(768, dtype=np.float32)
    center[0] = 0.25
    bank = registry.prototype_bank(center)

    assert len(bank.provisional) == 1
    assert np.allclose(bank.provisional[0].embedding, _unit(0))
    assert np.linalg.norm(bank.provisional[0].embedding) == np.float32(1.0)
    registry.close()


def test_dual_feedback_candidate_prototypes_use_full_cl2n(tmp_path):
    store = MultiDualHumanFeedbackStore(
        tmp_path / "feedback.sqlite3",
        model_fingerprint=FP,
        checkpoint_classes=("A", "B"),
        rejection=_rejection(),
    )
    positive = [
        _EvidenceEvent(
            key=f"event-{index}",
            camera_id="cam-0",
            embedding=_unit(0),
            observation_ids=(f"obs-{index}",),
        )
        for index in range(4)
    ]
    center = np.zeros(768, dtype=np.float32)
    center[0] = 0.25

    prototypes, coverage, false_accept_rate, quality_passed = store._build_candidate(
        positive,
        [],
        center,
        status="provisional",
    )

    assert prototypes.shape == (1, 768)
    assert np.allclose(prototypes[0], _unit(0))
    assert coverage == 1.0
    assert false_accept_rate == 0.0
    assert quality_passed is True
    store.close()


def test_dual_feedback_uses_separate_store_path_and_calibration_scope():
    generic = multi_dual_feedback_path_for_registry("/tmp/registry.sqlite3")
    first = multi_dual_feedback_path_for_registry(
        "/tmp/registry.sqlite3",
        _rejection(),
    )
    second = multi_dual_feedback_path_for_registry(
        "/tmp/registry.sqlite3",
        MultiDualRejectionConfig(
            cosine_threshold=0.31,
            squared_distance_threshold=0.5,
        ),
    )

    assert generic.name == "feedback_multi_dual.sqlite3"
    assert first.name.startswith("feedback_multi_dual_")
    assert first.suffix == ".sqlite3"
    assert first != second
