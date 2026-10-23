from types import SimpleNamespace

import numpy as np
import pytest

from system.backend.rejection_clusters import _resolve, cluster_assignment, persist_cluster_assignments
from system.dinov2.feedback_base import FeedbackObservation, HumanFeedbackStore
from system.dinov2.registry import SpeciesRegistry


def unit(index=0):
    value = np.zeros(768, dtype=np.float32)
    value[index] = 1
    return value


@pytest.fixture
def stores(tmp_path):
    fingerprint = "c" * 64
    registry = SpeciesRegistry(tmp_path / "registry.sqlite3", model_fingerprint=fingerprint)
    feedback = HumanFeedbackStore(tmp_path / "feedback.sqlite3", model_fingerprint=fingerprint,
                                 checkpoint_classes=("A", "B"), threshold=0.5)
    try:
        yield registry, feedback
    finally:
        feedback.close()
        registry.close()


def observe(feedback, oid, vector, *, accepted=False):
    feedback.persist_observation(FeedbackObservation(
        id=oid, source_path=f"{oid}.jpg", media_kind="image", box_index=0,
        frame_index=None, timestamp_seconds=None, bbox=(0, 0, 16, 16),
        camera_id="c", captured_at=None, predicted_species="Unknown",
        best_known_species="A", accepted=accepted, prediction_source="base",
        registry_id=None, known_score=0.1, threshold=0.5, embedding=vector,
    ))


def test_old_observations_match_real_clusters_and_links_survive_reopen(stores, monkeypatch):
    registry, feedback = stores
    first = registry.record_unknown(unit(), camera_id="c", captured_at=None, source_path="a.jpg")
    second = registry.record_unknown(unit(1), camera_id="c", captured_at=None, source_path="b.jpg")
    for oid, vec, accepted in [("a", unit(), False), ("b", unit(1), False),
                               ("noise", unit(2), False), ("known", unit(), True)]:
        observe(feedback, oid, vec, accepted=accepted)
    before = registry._conn.total_changes
    result = _resolve(feedback, registry._conn, ["a", "b", "noise", "known", "missing"])
    assert set(result) == {"a", "b"}
    assert result["a"] == cluster_assignment(feedback.model_fingerprint, first)
    assert result["b"] == cluster_assignment(feedback.model_fingerprint, second)
    assert result["a"]["cluster_id"] != result["b"]["cluster_id"]
    assert registry._conn.total_changes == before
    from system.dinov2 import registry_match_index
    monkeypatch.setattr(registry_match_index, "RegistryMatchIndex",
                        lambda *args: pytest.fail("cached links must not rematch feature vectors"))
    reopened = HumanFeedbackStore(feedback.path, model_fingerprint=feedback.model_fingerprint,
                                 checkpoint_classes=("A", "B"), threshold=0.5)
    try:
        assert _resolve(reopened, registry._conn, ["a", "b"]) == result
    finally:
        reopened.close()


def test_deleted_cluster_is_unassigned_and_other_models_are_rejected(stores):
    registry, feedback = stores
    entry = registry.record_unknown(unit(), camera_id="c", captured_at=None, source_path="a.jpg")
    observe(feedback, "a", unit())
    persist_cluster_assignments(feedback, {"a": cluster_assignment(feedback.model_fingerprint, entry)})
    registry.delete(entry.id)
    assert _resolve(feedback, registry._conn, ["a"]) == {}
    feedback.model_fingerprint = "d" * 64
    with pytest.raises(ValueError, match="fingerprint"):
        _resolve(feedback, registry._conn, ["a"])


def test_service_opens_registry_read_only_and_closes_feedback(stores, monkeypatch):
    from system.backend.rejection_clusters import resolve_rejection_clusters
    registry, feedback = stores
    entry = registry.record_unknown(unit(), camera_id="c", captured_at=None, source_path="a.jpg")
    observe(feedback, "a", unit())
    def open_feedback(_):
        return HumanFeedbackStore(feedback.path, model_fingerprint=feedback.model_fingerprint,
                                  checkpoint_classes=("A", "B"), threshold=0.5), None
    monkeypatch.setattr("system.backend.dinov2_feedback_service._open_feedback_state", open_feedback)
    before = registry._conn.total_changes
    assert resolve_rejection_clusters("model.neri.json", ["a"])["a"] == cluster_assignment(feedback.model_fingerprint, entry)
    assert registry._conn.total_changes == before


def test_runtime_persistence_records_cluster_without_invalidating_registry_cache(stores, tmp_path, monkeypatch):
    from system.backend.dinov2_persistence_fast import persist_runtime_observations
    from system.dinov2.classifier import DinoV2Observation
    registry, feedback = stores
    monkeypatch.setattr("system.backend.dinov2_persistence_fast.persist_feedback_observation_example", lambda *a, **kw: True)
    observation = DinoV2Observation(
        result_index=0, box_index=0, embedding=unit(), accepted=False, species="Unknown",
        source="base", registry_id=None, registration_status=None, known_score=0.1,
        threshold=0.5, detection_confidence=0.9, observation_id="new", bbox=(0, 0, 16, 16),
    )
    detector = SimpleNamespace(dinov2_feedback=feedback, dinov2_registry=registry,
                               drain_dinov2_observations=lambda: (observation,))
    persist_runtime_observations(detector, [tmp_path / "new.jpg"], [SimpleNamespace(date_taken=None)], tmp_path)
    result = _resolve(feedback, registry._conn, ["new"])
    assert result["new"]["candidate_number"] == 1
    index = next(iter(registry._match_indexes.values()))
    assert registry.match(unit(), statuses={"candidate"}, candidate_kinds={"candidate"})["id"] == 1
    assert next(iter(registry._match_indexes.values())) is index


def test_batch_matching_keeps_thresholds_and_id_ties(stores):
    from system.dinov2.registry_match_index import RegistryMatchIndex
    registry, _ = stores
    for vector in [unit(), unit(1), unit()]:
        eid = registry._create()
        registry.record_observation(eid, vector, camera_id="c", captured_at=None, source_path=f"{eid}.jpg")
    index = RegistryMatchIndex(registry._conn, ("candidate",))
    matches = index.match_batch(np.stack([unit(), unit(1), unit(2)]), 0.82)
    assert [row["id"] if row else None for row in matches] == [1, 2, None]


def test_bulk_api_returns_cluster_ids_and_bounds_request_size(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from system.backend.dinov2_feedback_api import dinov2_feedback_router
    calls = []
    monkeypatch.setattr("system.backend.rejection_clusters.resolve_rejection_clusters",
                        lambda model, ids: calls.append((model, ids)) or {"a": {"cluster_id": "registry:fp:1:0"}})
    app = FastAPI()
    app.include_router(dinov2_feedback_router())
    client = TestClient(app)
    response = client.post("/api/dinov2/feedback/rejection-clusters",
                           json={"classification_model_path": "model.neri.json", "observation_ids": ["a"]})
    assert response.status_code == 200
    assert response.json()["a"]["cluster_id"] == "registry:fp:1:0"
    assert calls == [("model.neri.json", ["a"])]
    assert client.post("/api/dinov2/feedback/rejection-clusters", json={
        "classification_model_path": "model.neri.json", "observation_ids": ["a"] * 10001,
    }).status_code == 422
