from types import SimpleNamespace
from datetime import datetime, timezone

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
    assert result["a"] == {**cluster_assignment(feedback.model_fingerprint, first), "density_reason": "insufficient_events"}
    assert result["b"] == {**cluster_assignment(feedback.model_fingerprint, second), "density_reason": "insufficient_events"}
    assert result["a"]["cluster_id"] != result["b"]["cluster_id"]
    assert registry._conn.total_changes == before
    from system.dinov2 import registry_match_index
    monkeypatch.setattr(registry_match_index, "RegistryMatchIndex",
                        lambda *args: pytest.fail("cached links must not rematch feature vectors"))
    monkeypatch.setattr("system.backend.rejection_density.rejected_event_labels",
                        lambda *args: pytest.fail("unchanged cached density must not be recomputed"))
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


def test_burst_is_noise_and_cached_link_promotes_after_independent_events(stores, monkeypatch):
    registry, feedback = stores
    # Same pool context as discovery: allow_single_cluster=False requires
    # separable density groups, not a forced cluster for one homogeneous pool.
    for index in range(10):
        registry.record_unknown(unit(1), camera_id="background", captured_at=None, source_path=f"background-{index}.jpg")
    # Four photos in the same trigger still contribute just one event.
    for index in range(4):
        entry = registry.record_unknown(unit(), camera_id="c", captured_at=datetime(2026, 10, 1, 12, 0, index, tzinfo=timezone.utc),
                                        source_path=f"burst-{index}.jpg")
        observe(feedback, f"burst-{index}", unit())
    ids = [f"burst-{index}" for index in range(4)]
    result = _resolve(feedback, registry._conn, ids)
    link = result[ids[0]]
    assert all(value == link for value in result.values())
    assert link["event_count"] == 1
    assert link["min_cluster_events"] == 5
    assert link["is_noise"] is True
    # Retain candidate identity/evidence; refresh cached links without re-matching.
    from system.dinov2 import registry_match_index
    monkeypatch.setattr(registry_match_index, "RegistryMatchIndex",
                        lambda *args: pytest.fail("support changes must not rematch cached links"))
    for day in range(2, 6):
        registry.record_observation(entry.id, unit(), camera_id="c", captured_at=datetime(2026, 10, day, 12, tzinfo=timezone.utc),
                                    source_path=f"event-{day}.jpg")
        current = _resolve(feedback, registry._conn, ids)[ids[0]]
        assert current["cluster_id"] == link["cluster_id"]
        assert current["event_count"] == day
        assert current["is_noise"] is (day < 5)
        assert current["density_verified"] is (day >= 5)


def test_small_candidate_is_not_reassigned_to_a_supported_neighbor(stores):
    registry, feedback = stores
    larger = registry._create()
    neighbor = unit() + unit(1) * 0.2
    for index in range(4):
        registry.record_observation(larger, neighbor, camera_id="c", captured_at=None, source_path=f"large-{index}.jpg")
    small = registry._create()
    registry.record_observation(small, unit(), camera_id="c", captured_at=None, source_path="small.jpg")
    observe(feedback, "small", unit())
    result = _resolve(feedback, registry._conn, ["small"])["small"]
    assert result["registry_id"] == small
    assert result["is_noise"] is True


@pytest.mark.parametrize("legacy_registry", [False, True])
def test_enough_events_are_still_noise_until_density_passes_and_cache_tracks_features(stores, legacy_registry, monkeypatch):
    registry, feedback = stores
    if legacy_registry:
        with registry._conn:
            registry._conn.execute("DELETE FROM metadata WHERE key='rejection_density_revision_v1'")
    for index in range(10):
        first = registry.record_unknown(unit(), camera_id="c", captured_at=None, source_path=f"first-{index}.jpg")
    observe(feedback, "first", unit())
    isolated = _resolve(feedback, registry._conn, ["first"])["first"]
    assert isolated["event_count"] == 10
    assert isolated["is_noise"] is True  # allow_single_cluster=False, not a size-only check
    assert isolated["density_reason"] == "density_noise"
    for index in range(10):
        second = registry.record_unknown(unit(1), camera_id="c", captured_at=None, source_path=f"second-{index}.jpg")
    supported = _resolve(feedback, registry._conn, ["first"])["first"]
    assert supported["is_noise"] is False
    assert supported["density_verified"] is True
    assert supported["cluster_id"] == isolated["cluster_id"]
    with monkeypatch.context() as cached:
        cached.setattr("system.backend.rejection_density.rejected_event_labels",
                       lambda *args: pytest.fail("unchanged supported density must reuse its persisted result"))
        assert _resolve(feedback, registry._conn, ["first"])["first"] == supported
    # Same event IDs/counts, different geometry. Both revision-based and legacy
    # caches must re-evaluate actual features instead of keeping old support.
    with registry._conn:
        registry._conn.execute("UPDATE events SET embedding=? WHERE registration_id=?",
                               (unit().astype('<f4').tobytes(), second.id))
    assert _resolve(feedback, registry._conn, ["first"])["first"]["is_noise"] is True
    assert registry.get(first.id).event_count == 10


def test_small_view_does_not_trigger_full_pool_density(stores, monkeypatch):
    registry, feedback = stores
    for index in range(10):
        registry.record_unknown(unit(1), camera_id="c", captured_at=None, source_path=f"large-{index}.jpg")
    registry.record_unknown(unit(), camera_id="c", captured_at=None, source_path="small.jpg")
    observe(feedback, "small", unit())
    monkeypatch.setattr("system.backend.rejection_density.candidate_density_support",
                        lambda *args: pytest.fail("undersized candidates need no expensive full-pool fit"))
    result = _resolve(feedback, registry._conn, ["small"])["small"]
    assert result["event_count"] == 1
    assert result["is_noise"] is True


def test_candidate_mixing_density_groups_is_not_presented_as_one_coherent_cluster(stores):
    registry, feedback = stores
    eid = registry._create()
    for axis in range(2):
        for index in range(6):
            registry.record_observation(eid, unit(axis), camera_id="c", captured_at=None,
                                        source_path=f"{axis}-{index}.jpg")
    observe(feedback, "mixed", unit())
    persist_cluster_assignments(feedback, {"mixed": cluster_assignment(feedback.model_fingerprint, registry.get(eid))})
    result = _resolve(feedback, registry._conn, ["mixed"])["mixed"]
    assert result["event_count"] == 12
    assert result["is_noise"] is True
    assert result["density_reason"] == "multiple_density_groups"


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
    assert resolve_rejection_clusters("model.neri.json", ["a"])["a"] == {
        **cluster_assignment(feedback.model_fingerprint, entry), "density_reason": "insufficient_events",
    }
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


@pytest.fixture
def learning_cluster(tmp_path, monkeypatch):
    from tests.test_dinov2_memory import _memory_model
    from system.dinov2.checkpoint import load_checkpoint
    from system.backend.dinov2_registry_service import open_registry_for_model, _open_candidate_feedback
    from system.backend import services
    from system.backend.dinov2_feedback_api import DinoV2ClusterFeedbackRequest
    monkeypatch.setenv("NERI_DINOV2_STATE_DIR", str(tmp_path / "state"))
    model, _ = _memory_model(tmp_path)
    checkpoint = load_checkpoint(model)
    root = tmp_path / "media"
    root.mkdir()
    registry = open_registry_for_model(str(model))
    feedback = _open_candidate_feedback(registry, checkpoint)
    entry = None
    for index in range(10):
        path = root / f"image-{index}.jpg"
        path.write_bytes(b"source pixels must not be decoded")
        entry = registry.record_unknown(unit(4), camera_id=f"c-{index % 2}",
                                        captured_at=None, source_path=str(path), bbox=(0, 0, 16, 16))
    for index in range(10):
        registry.record_unknown(unit(6), camera_id="background", captured_at=None, source_path=f"background-{index}.jpg")
    other = registry.record_unknown(unit(5), camera_id="other", captured_at=None, source_path="other.jpg")
    path = root / "image-0.jpg"
    for oid, vector in [("view-a", unit(4)), ("view-b", unit(5))]:
        feedback.persist_observation(FeedbackObservation(
            id=oid, source_path=str(path), media_kind="image", box_index=0,
            frame_index=None, timestamp_seconds=None, bbox=(0, 0, 16, 16),
            camera_id="c-0", captured_at=None, predicted_species="Unknown",
            best_known_species="A", accepted=False, prediction_source="base",
            registry_id=None, known_score=0.1, threshold=1, embedding=vector,
        ))
    persist_cluster_assignments(feedback, {
        "view-a": cluster_assignment(checkpoint.fingerprint, entry),
        "view-b": cluster_assignment(checkpoint.fingerprint, other),
    })
    data = {"物种名称": "拒识/Unknown", "物种数量": "2", "检测框": [
        {"物种": "拒识/Unknown", "边界框": [0, 0, 16, 16], "observation_id": oid, "候选项": []}
        for oid in ("view-a", "view-b")
    ]}
    services._persist_validation_update(path, root, data, None)
    request = DinoV2ClusterFeedbackRequest(input_path=str(root), classification_model_path=str(model),
        cluster_id=cluster_assignment(checkpoint.fingerprint, entry)["cluster_id"],
        observation_ids=["view-a"], species_name="A", feedback_operation_id="cluster-learn-A")
    try:
        yield SimpleNamespace(model=model, checkpoint=checkpoint, registry=registry, feedback=feedback,
                              root=root, path=path, request=request, entry=entry, other=other)
    finally:
        feedback.close()
        registry.close()


def test_cluster_confirmation_changes_head_predictions_and_undo_restores_them(learning_cluster):
    from system.backend.cluster_learning import confirm_rejection_cluster
    from system.backend.dinov2_feedback_service import revert_feedback_operation
    from system.dinov2.memory_classifier import MemoryDinoV2Classifier
    case = learning_cluster
    classifier = MemoryDinoV2Classifier(case.checkpoint, feedback=case.feedback)
    assert not classifier.classify_features(unit(4)[None, :])[0].accepted
    original_events = case.registry.get(case.entry.id).event_count
    result = confirm_rejection_cluster(case.request)
    assert result["feature_events"] == 10
    assert result["learning"]["formal"] and result["learning"]["active"]
    assert len(case.feedback.memory_bank().formal) == 10
    prediction = classifier.classify_features(unit(4)[None, :])[0]
    assert prediction.accepted and prediction.species == "A"
    boxes = result["items"][0]["detection_boxes"]
    assert boxes[0]["species"] == "A" and boxes[0]["feedback_status"] == "confirmed"
    assert boxes[1]["species"] == "拒识/Unknown"
    assert not result["items"][0]["validated"]
    count = case.feedback._conn.execute("SELECT COUNT(*) FROM human_feedback").fetchone()[0]
    assert confirm_rejection_cluster(case.request) == result
    assert case.feedback._conn.execute("SELECT COUNT(*) FROM human_feedback").fetchone()[0] == count
    assert case.registry.get(case.entry.id).event_count == original_events
    revert_feedback_operation(case.request)
    assert not classifier.classify_features(unit(4)[None, :])[0].accepted
    from system.backend import services
    restored = services._reload_validation_item(case.path, case.root)
    assert all(box["species"] == "拒识/Unknown" for box in restored.detection_boxes)
    with pytest.raises(ValueError, match="已撤回"):
        confirm_rejection_cluster(case.request)


def test_cluster_confirmation_rejects_other_clusters_and_models(learning_cluster):
    from system.backend.cluster_learning import confirm_rejection_cluster
    case = learning_cluster
    request = case.request.model_copy(update={"observation_ids": ["view-b"]})
    with pytest.raises(ValueError, match="不全部属于"):
        confirm_rejection_cluster(request)
    request = case.request.model_copy(update={"cluster_id": case.request.cluster_id.replace(case.checkpoint.fingerprint, "f" * 64)})
    with pytest.raises(ValueError, match="不一致"):
        confirm_rejection_cluster(request)
    assert not case.feedback.memory_bank().formal


def test_noise_candidate_cannot_apply_cluster_learning(learning_cluster):
    from system.backend.cluster_learning import confirm_rejection_cluster
    case = learning_cluster
    request = case.request.model_copy(update={
        "cluster_id": cluster_assignment(case.checkpoint.fingerprint, case.other)["cluster_id"],
        "observation_ids": ["view-b"],
    })
    with pytest.raises(ValueError, match="不足 5 个独立事件"):
        confirm_rejection_cluster(request)
    assert not case.feedback.memory_bank().formal
    assert case.feedback._conn.execute("SELECT COUNT(*) FROM human_feedback").fetchone()[0] == 0
    from system.backend import services
    restored = services._reload_validation_item(case.path, case.root)
    assert all(box["species"] == "拒识/Unknown" for box in restored.detection_boxes)


def test_density_failure_blocks_learning_even_with_enough_events(learning_cluster):
    from system.backend.cluster_learning import confirm_rejection_cluster
    case = learning_cluster
    # Remove the second dense mode: the remaining pool has no supported split.
    with case.registry._conn:
        case.registry._conn.execute("DELETE FROM registrations WHERE id NOT IN (?,?)", (case.entry.id, case.other.id))
    with pytest.raises(ValueError, match="HDBSCAN 密度"):
        confirm_rejection_cluster(case.request)
    assert not case.feedback.memory_bank().formal
    assert case.feedback._conn.execute("SELECT COUNT(*) FROM human_feedback").fetchone()[0] == 0


def test_learning_uses_verified_snapshot_while_runtime_adds_an_unverified_event(learning_cluster, monkeypatch):
    from system.backend.cluster_learning import confirm_rejection_cluster
    from system.backend import dinov2_registry_service
    case = learning_cluster
    writer = SpeciesRegistry(case.registry.path, model_fingerprint=case.checkpoint.fingerprint)
    original = dinov2_registry_service._candidate_feedback_observations

    def with_runtime_write(registry, entry_id, *, checkpoint):
        writer.record_observation(entry_id, unit(7), camera_id="later", captured_at=None, source_path="arriving-later.jpg")
        return original(registry, entry_id, checkpoint=checkpoint)

    monkeypatch.setattr(dinov2_registry_service, "_candidate_feedback_observations", with_runtime_write)
    try:
        result = confirm_rejection_cluster(case.request)
    finally:
        writer.close()
    assert result["feature_events"] == 10
    assert case.registry.get(case.entry.id).event_count == 11
    assert case.feedback._conn.execute("SELECT COUNT(*) FROM human_feedback").fetchone()[0] == 10


def test_changing_cluster_label_removes_old_learning_and_undo_reactivates_it(learning_cluster):
    from system.backend.cluster_learning import confirm_rejection_cluster
    from system.backend.dinov2_feedback_service import revert_feedback_operation
    from system.dinov2.memory_classifier import MemoryDinoV2Classifier
    case = learning_cluster
    confirm_rejection_cluster(case.request)
    changed = case.request.model_copy(update={"species_name": "B", "feedback_operation_id": "cluster-learn-B"})
    confirm_rejection_cluster(changed)
    classifier = MemoryDinoV2Classifier(case.checkpoint, feedback=case.feedback)
    assert classifier.classify_features(unit(4)[None, :])[0].species == "B"
    assert all(example.species == "B" for example in case.feedback.memory_bank().formal)
    revert_feedback_operation(changed)
    assert classifier.classify_features(unit(4)[None, :])[0].species == "A"
    assert all(example.species == "A" for example in case.feedback.memory_bank().formal)


def test_new_species_cluster_stays_candidate_until_registration_and_can_undo(learning_cluster):
    from system.backend.cluster_learning import confirm_rejection_cluster
    from system.backend.dinov2_feedback_service import revert_feedback_operation
    case = learning_cluster
    request = case.request.model_copy(update={"species_name": "新物种", "feedback_operation_id": "cluster-new"})
    result = confirm_rejection_cluster(request)
    assert result["learning"]["status"] == "candidate"
    assert not result["learning"]["active"]
    entry = case.registry.get(case.entry.id)
    assert entry.common_name == "新物种" and entry.status == "candidate"
    assert not case.registry.memory_bank().formal
    revert_feedback_operation(request)
    assert not case.registry.get(case.entry.id).common_name


def test_cluster_learning_rolls_back_if_local_result_write_fails(learning_cluster, monkeypatch):
    from system.backend.cluster_learning import confirm_rejection_cluster
    from system.backend import services
    case = learning_cluster
    original = services._persist_validation_update
    calls = []
    def fail_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("simulated disk failure")
        return original(*args, **kwargs)
    monkeypatch.setattr(services, "_persist_validation_update", fail_once)
    with pytest.raises(OSError, match="disk failure"):
        confirm_rejection_cluster(case.request)
    assert not case.feedback.memory_bank().formal
    assert all(box["species"] == "拒识/Unknown" for box in services._reload_validation_item(case.path, case.root).detection_boxes)


def test_cluster_box_update_keeps_video_counts_as_per_frame_maxima():
    from system.backend.dinov2_feedback_service import _recompute_detection_summary
    data = {"检测框": [{"物种": "A", "frame_index": frame} for frame in (0, 1, 2)]}
    _recompute_detection_summary(data)
    assert data["物种名称"] == "A" and data["物种数量"] == "1"
