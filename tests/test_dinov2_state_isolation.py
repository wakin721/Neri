from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from system.dinov2.checkpoint import validate_checkpoint
from system.dinov2.feedback import HumanFeedbackStore
from system.dinov2.registry import SpeciesRegistry, registry_path_for_fingerprint
from tests.dinov2_fixtures import make_dinov2_payload


def _unit(index: int) -> np.ndarray:
    value = np.zeros(768, dtype=np.float32)
    value[index] = 1.0
    return value


def test_registry_path_uses_dinov2_fingerprint_namespace(tmp_path):
    path = registry_path_for_fingerprint(tmp_path, "a" * 64)
    assert path == (tmp_path / "dinov2" / ("a" * 64) / "registry.sqlite3").resolve()
    assert "dinov3" not in path.parts


class RegistryHarness(SpeciesRegistry):
    def __init__(self, embeddings):
        self._test_embeddings = np.asarray(embeddings, dtype=np.float32)

    def list(self):
        return [SimpleNamespace(id=7, status="provisional", display_name="Fox")]

    def _embeddings(self, entry_id):
        assert entry_id == 7
        return self._test_embeddings

    @staticmethod
    def _status_prototype_limit(status):
        return 1


def test_registry_overlay_rebuilds_prototype_in_cl2n_space():
    embeddings = np.stack([
        _unit(0),
        (_unit(0) * 0.98 + _unit(1) * 0.20) / np.linalg.norm(_unit(0) * 0.98 + _unit(1) * 0.20),
        (_unit(0) * 0.97 + _unit(2) * 0.24) / np.linalg.norm(_unit(0) * 0.97 + _unit(2) * 0.24),
        (_unit(0) * 0.96 + _unit(3) * 0.28) / np.linalg.norm(_unit(0) * 0.96 + _unit(3) * 0.28),
    ]).astype(np.float32)
    center = np.zeros(768, dtype=np.float32)
    center[0] = 0.2
    registry = RegistryHarness(embeddings)

    bank = registry.prototype_bank(center)

    transformed = embeddings - center[None, :]
    transformed /= np.linalg.norm(transformed, axis=1, keepdims=True)
    expected = transformed.mean(axis=0)
    assert len(bank.provisional) == 1
    assert len(bank.formal) == 0
    assert bank.provisional[0].species == "Fox"
    assert np.allclose(bank.provisional[0].embedding, expected, atol=1e-7)


def test_feedback_candidate_is_built_in_cl2n_space():
    checkpoint = validate_checkpoint(make_dinov2_payload())
    store = HumanFeedbackStore.__new__(HumanFeedbackStore)
    store.rejection = checkpoint.rejection
    store.threshold = checkpoint.rejection.cosine_threshold
    events = [SimpleNamespace(embedding=_unit(0)) for _ in range(4)]
    center = np.zeros(768, dtype=np.float32)
    center[1] = 0.2

    prototypes, coverage, false_accept_rate, quality = store._build_candidate(
        events,
        [],
        center,
        status="provisional",
    )

    transformed = _unit(0) - center
    transformed /= np.linalg.norm(transformed)
    assert prototypes.shape == (1, 768)
    assert np.allclose(prototypes[0], transformed, atol=1e-7)
    assert coverage == 1.0
    assert false_accept_rate == 0.0
    assert quality is True


def test_registry_overlay_applies_checkpoint_prototype_norm_power():
    embeddings = np.stack([
        _unit(0),
        (_unit(0) + _unit(1)) / np.sqrt(2.0),
        (_unit(0) + _unit(2)) / np.sqrt(2.0),
        (_unit(0) + _unit(3)) / np.sqrt(2.0),
    ]).astype(np.float32)
    registry = RegistryHarness(embeddings)
    registry.prototype_norm_power = 1.0

    bank = registry.prototype_bank(np.zeros(768, dtype=np.float32))

    assert len(bank.provisional) == 1
    assert np.linalg.norm(bank.provisional[0].embedding) == pytest.approx(1.0, abs=1e-7)


def test_feedback_candidate_applies_checkpoint_prototype_norm_power():
    checkpoint = validate_checkpoint(make_dinov2_payload())
    store = HumanFeedbackStore.__new__(HumanFeedbackStore)
    store.rejection = checkpoint.rejection
    store.threshold = checkpoint.rejection.cosine_threshold
    store.prototype_norm_power = 1.0
    values = [
        _unit(0),
        (_unit(0) + _unit(1)) / np.sqrt(2.0),
        (_unit(0) + _unit(2)) / np.sqrt(2.0),
        (_unit(0) + _unit(3)) / np.sqrt(2.0),
    ]
    events = [SimpleNamespace(embedding=value.astype(np.float32)) for value in values]

    prototypes, *_ = store._build_candidate(
        events,
        [],
        np.zeros(768, dtype=np.float32),
        status="provisional",
    )

    assert np.linalg.norm(prototypes[0]) == pytest.approx(1.0, abs=1e-7)


class FeedbackClusterHarness(HumanFeedbackStore):
    def __init__(self, events):
        self._events = events

    def _evidence_observations(self, species, *, column):
        assert column == "positive_species"
        return [object()]

    def _group_events(self, observations):
        return self._events

    def _active_generation(self, species):
        return None

    def learning_state(self, species):
        return SimpleNamespace(status="collecting")


def test_feedback_cluster_details_use_cl2n_geometry():
    raw = [
        _unit(0),
        (_unit(0) + _unit(1)) / np.sqrt(2.0),
        (_unit(0) + _unit(2)) / np.sqrt(2.0),
        (_unit(0) + _unit(3)) / np.sqrt(2.0),
    ]
    events = [
        SimpleNamespace(
            embedding=value.astype(np.float32),
            camera_id=f"cam-{index}",
            observation_ids=(f"obs-{index}",),
        )
        for index, value in enumerate(raw)
    ]
    store = FeedbackClusterHarness(events)
    center = np.zeros(768, dtype=np.float32)
    center[0] = 0.2

    details = store.cluster_details("A", center)

    transformed = np.stack(raw).astype(np.float32) - center[None, :]
    transformed /= np.linalg.norm(transformed, axis=1, keepdims=True)
    prototype = transformed.mean(axis=0)
    expected = np.mean(np.sum((transformed - prototype[None, :]) ** 2, axis=1))
    assert len(details) == 1
    assert details[0]["mean_squared_distance"] == pytest.approx(float(expected), abs=1e-7)
    assert details[0]["event_count"] == 4
