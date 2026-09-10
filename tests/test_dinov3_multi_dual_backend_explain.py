from __future__ import annotations

import pytest

from system.backend import dinov3_feedback_service, dinov3_registry_service
from system.dinov3.checkpoint import load_checkpoint
from system.dinov3.dual_overlay import (
    MultiDualHumanFeedbackStore,
    multi_dual_feedback_path_for_registry,
)
from system.dinov3.rejection import MultiDualRejectionConfig
from system.dinov3.registry import registry_path_for_fingerprint
from tests.test_dinov3_feedback_store import make_observation
from tests.test_dinov3_multi_dual_rejection import _write_runtime_fixture


def _persist_dual_observation(tmp_path, observation_id: str):
    rejection = MultiDualRejectionConfig(
        cosine_threshold=0.42,
        squared_distance_threshold=0.55,
    )
    manifest, checkpoint_path = _write_runtime_fixture(
        tmp_path,
        rejection=rejection.as_dict(),
    )
    checkpoint = load_checkpoint(checkpoint_path)
    state_root = tmp_path / "state"
    registry_path = registry_path_for_fingerprint(state_root, checkpoint.fingerprint)
    feedback = MultiDualHumanFeedbackStore(
        multi_dual_feedback_path_for_registry(registry_path, rejection),
        model_fingerprint=checkpoint.fingerprint,
        checkpoint_classes=checkpoint.classes,
        rejection=rejection,
    )
    try:
        feedback.persist_observation(make_observation(observation_id))
    finally:
        feedback.close()
    return manifest, state_root, rejection


def test_backend_feedback_state_reopens_multi_dual_generation(monkeypatch, tmp_path):
    manifest, state_root, _rejection = _persist_dual_observation(
        tmp_path,
        "obs-dual-state",
    )
    monkeypatch.setattr(
        dinov3_feedback_service,
        "default_dinov3_state_root",
        lambda: state_root,
    )

    feedback, _feature_center = dinov3_feedback_service._open_feedback_state(
        str(manifest)
    )
    try:
        assert isinstance(feedback, MultiDualHumanFeedbackStore)
        assert feedback.get_observation("obs-dual-state").id == "obs-dual-state"
    finally:
        feedback.close()


def test_backend_explanation_reopens_multi_dual_feedback_and_geometry(
    monkeypatch, tmp_path
):
    manifest, state_root, _rejection = _persist_dual_observation(
        tmp_path,
        "obs-dual-explain",
    )
    monkeypatch.setattr(
        dinov3_feedback_service,
        "default_dinov3_state_root",
        lambda: state_root,
    )
    monkeypatch.setattr(
        dinov3_registry_service,
        "default_dinov3_state_root",
        lambda: state_root,
    )

    explanation = dinov3_feedback_service.explain_feedback_observation(
        str(manifest),
        "obs-dual-explain",
    )

    assert explanation["rejection"] == {
        "mode": "multi_dual",
        "cosine_threshold": pytest.approx(0.42),
        "squared_distance_threshold": pytest.approx(0.55),
    }
    assert explanation["threshold"] == pytest.approx(0.42)
    assert explanation["squared_distance"] is not None
