from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from system.dinov2.registry import SpeciesRegistry


def _unit(index=0):
    value = np.zeros(768, dtype=np.float32)
    value[index] = 1
    return value


def test_batch_matching_reuses_snapshot_and_sees_new_candidates(tmp_path):
    registry = SpeciesRegistry(tmp_path / "registry.sqlite3", model_fingerprint="a" * 64)
    sql = []
    registry._conn.set_trace_callback(sql.append)
    try:
        with registry.matching_batch():
            first = registry.record_unknown(_unit(), camera_id="c", captured_at=None, source_path="a.jpg")
            second = registry.record_unknown(_unit(), camera_id="c", captured_at=None, source_path="b.jpg")
            third = registry.record_unknown(_unit(1), camera_id="c", captured_at=None, source_path="d.jpg")
            assert first.id == second.id
            assert third.id != first.id
            assert registry.match(_unit(), statuses={"candidate"})["id"] == first.id
            assert registry.match(_unit(), statuses=set()) is None
            assert registry.match(_unit(), statuses={"candidate"}, exclude_entry_ids={first.id}) is None
        # One filtered full snapshot for record_unknown; later writes refresh just their entry.
        snapshots = [query for query in sql if "FROM registrations r WHERE r.status IN" in query
                     and query.startswith("SELECT r.id")]
        assert len(snapshots) == 2  # record_unknown's kind filter and the explicit unfiltered match
    finally:
        registry.close()


def test_batch_matching_detects_updates_from_other_connection(tmp_path):
    path = tmp_path / "registry.sqlite3"
    registry = SpeciesRegistry(path, model_fingerprint="b" * 64)
    other = SpeciesRegistry(path, model_fingerprint="b" * 64)
    try:
        with registry.matching_batch():
            assert registry.match(_unit()) is None
            entry = other.record_unknown(_unit(), camera_id="c", captured_at=None, source_path="a.jpg")
            assert registry.match(_unit())["id"] == entry.id
    finally:
        other.close()
        registry.close()


def test_matching_reuses_snapshot_across_batches_and_nested_contexts(tmp_path):
    registry = SpeciesRegistry(tmp_path / "registry.sqlite3", model_fingerprint="d" * 64)
    sql = []
    registry._conn.set_trace_callback(sql.append)
    try:
        for batch in range(3):
            with registry.matching_batch():
                with registry.matching_batch():
                    entry = registry.record_unknown(
                        _unit(), camera_id="c", captured_at=None, source_path=f"{batch}.jpg",
                    )
                assert registry.match(_unit(), statuses={"candidate"},
                                      candidate_kinds={"candidate"})["id"] == entry.id
        snapshots = [query for query in sql if query.startswith("SELECT r.id")]
        assert len(snapshots) == 1
        assert entry.event_count == 3
    finally:
        registry.close()


@pytest.mark.parametrize("external", [False, True])
def test_matching_cache_sees_identity_changes_and_deletions_between_batches(tmp_path, external):
    path = tmp_path / "registry.sqlite3"
    registry = SpeciesRegistry(path, model_fingerprint="e" * 64)
    writer = SpeciesRegistry(path, model_fingerprint="e" * 64) if external else registry
    try:
        entry = registry.record_unknown(_unit(), camera_id="c", captured_at=None, source_path="a.jpg")
        with registry.matching_batch():
            assert registry.match(_unit())["id"] == entry.id
        writer.set_identity(entry.id, common_name="Changed")
        with registry.matching_batch():
            assert registry.match(_unit())["display_name"] == "Changed"
        writer.delete(entry.id)
        with registry.matching_batch():
            assert registry.match(_unit()) is None
    finally:
        if external:
            writer.close()
        registry.close()


def test_memory_bank_reads_only_registered_evidence_in_existing_order(tmp_path, monkeypatch):
    from system.dinov2.memory_bank import MemoryBank, MemoryExample
    from system.dinov2.registry import _from_blob

    registry = SpeciesRegistry(tmp_path / "registry.sqlite3", model_fingerprint="f" * 64)
    try:
        # Include named/unnamed candidates, all learned states, and an entry
        # with insufficient evidence. Candidate pixels/vectors must stay cold.
        for index, (status, name, count) in enumerate([
            ("candidate", "Candidate", 5), ("provisional", "P", 4),
            ("confirmed", "C", 5), ("mature", "", 4),
            ("confirmed", "Too few", 3), ("candidate", "", 1),
        ]):
            entry_id = registry._create()
            for event in range(count):
                registry.record_observation(entry_id, _unit(index), camera_id=f"c-{event}",
                                            captured_at=None, source_path=f"{index}-{event}.jpg")
            registry._conn.execute("UPDATE registrations SET status=?,common_name=? WHERE id=?",
                                   (status, name, entry_id))
            registry._conn.commit()
        formal, provisional = [], []
        for entry in registry.list():
            if entry.status == "candidate" or entry.event_count < 4:
                continue
            destination = provisional if entry.status == "provisional" else formal
            destination.extend(MemoryExample(
                species=entry.display_name, embedding=_from_blob(row["embedding"]),
                camera_id=str(row["camera_id"]), source="overlay", registry_id=entry.id,
                registration_status=entry.status,
            ) for row in registry._event_rows(entry.id))
        expected = MemoryBank(tuple(formal), tuple(provisional))
        monkeypatch.setattr(registry, "list", lambda **kwargs: pytest.fail("must not load the UI list"))
        bank = registry.memory_bank()
        for actual, previous in ((bank.formal, expected.formal), (bank.provisional, expected.provisional)):
            assert len(actual) == len(previous)
            for sample, old in zip(actual, previous):
                assert (sample.species, sample.camera_id, sample.registry_id, sample.registration_status) == (
                    old.species, old.camera_id, old.registry_id, old.registration_status,
                )
                np.testing.assert_array_equal(sample.embedding, old.embedding)
        # No candidate entry has to be deserialized, even if its old blob is corrupt.
        registry._conn.execute("UPDATE events SET embedding=x'00' WHERE registration_id IN "
                               "(SELECT id FROM registrations WHERE status='candidate')")
        registry._conn.commit()
        assert len(registry.memory_bank().formal) == len(expected.formal)
    finally:
        registry.close()


def test_batch_and_sequential_candidate_routing_agree(tmp_path):
    sequential = SpeciesRegistry(tmp_path / "sequential.sqlite3", model_fingerprint="c" * 64)
    batched = SpeciesRegistry(tmp_path / "batched.sqlite3", model_fingerprint="c" * 64)
    vectors = [_unit(0), _unit(1), _unit(0) + 0.2 * _unit(2), _unit(2), _unit(1)]
    def record(store):
        return [store.record_unknown(vector, camera_id="c", captured_at=None, source_path=f"{i}.jpg").id
                for i, vector in enumerate(vectors)]
    try:
        expected = record(sequential)
        with batched.matching_batch():
            assert record(batched) == expected
        assert [(e.id, e.event_count) for e in batched.list()] == [
            (e.id, e.event_count) for e in sequential.list()
        ]
    finally:
        sequential.close()
        batched.close()


def _assert_fresh_index_scores(index, registry, vector):
    from system.dinov2.registry_match_index import RegistryMatchIndex
    expected = list(RegistryMatchIndex(registry._conn, index.statuses, index.candidate_kinds).scores(vector))
    actual = list(index.scores(vector))
    assert [row["id"] for row, _ in actual] == [row["id"] for row, _ in expected]
    np.testing.assert_array_equal([score for _, score in actual], [score for _, score in expected])
    assert [row for row, _ in actual] == [row for row, _ in expected]


def test_match_matrix_updates_and_appends_without_full_bank_copies(tmp_path):
    from system.dinov2.registry_match_index import RegistryMatchIndex
    registry = SpeciesRegistry(tmp_path / "registry.sqlite3", model_fingerprint="1" * 64)
    try:
        first = registry.record_unknown(_unit(), camera_id="c", captured_at=None, source_path="first.jpg")
        index = RegistryMatchIndex(registry._conn, ("candidate",), ("candidate",))
        _assert_fresh_index_scores(index, registry, _unit())
        storage = index._storage
        registry.record_observation(first.id, _unit() + _unit(1) * 0.1,
                                    camera_id="c", captured_at=None, source_path="update.jpg")
        index.refresh(first.id)
        assert index._storage is storage
        _assert_fresh_index_scores(index, registry, _unit(1))
        for number in range(2, 22):
            entry_id = registry._create()
            registry.record_observation(entry_id, _unit(number), camera_id="c",
                                        captured_at=None, source_path=f"{number}.jpg")
            index.refresh(entry_id)
            if number < 16:
                assert index._storage is storage
            _assert_fresh_index_scores(index, registry, _unit(number))
        # Reserve growth must preserve earlier evidence as well as new candidates.
        assert index._storage is not storage
        _assert_fresh_index_scores(index, registry, _unit())
        registry.set_identity(first.id, common_name="Renamed")
        index.refresh(first.id)
        _assert_fresh_index_scores(index, registry, _unit())
        registry.delete(first.id)
        index.refresh(first.id)
        _assert_fresh_index_scores(index, registry, _unit())
    finally:
        registry.close()


def test_match_matrix_keeps_filters_and_rebuilds_changed_prototype_offsets(tmp_path):
    from system.dinov2.registry_match_index import RegistryMatchIndex
    registry = SpeciesRegistry(tmp_path / "registry.sqlite3", model_fingerprint="2" * 64)
    try:
        first = registry.record_unknown(_unit(), camera_id="c", captured_at=None, source_path="a.jpg")
        other = registry.record_unknown(_unit(1), camera_id="c", captured_at=None,
                                        source_path="b.jpg", candidate_kind="new_mode_candidate")
        index = RegistryMatchIndex(registry._conn, ("candidate",), ("candidate",))
        _assert_fresh_index_scores(index, registry, _unit())
        storage = index._storage
        index.refresh(other.id)
        assert index._storage is storage
        # Move from one prototype to two, then update both in place.
        registry._replace_prototypes(first.id, np.stack([_unit(), _unit(2)]), 4)
        registry._conn.commit()
        index.refresh(first.id)
        _assert_fresh_index_scores(index, registry, _unit(2))
        storage = index._storage
        registry._replace_prototypes(first.id, np.stack([_unit(3), _unit(4)]), 4)
        registry._conn.commit()
        index.refresh(first.id)
        assert index._storage is storage
        _assert_fresh_index_scores(index, registry, _unit(4))
        # A previously excluded, smaller ID must return in the original ID order.
        registry._conn.execute("UPDATE registrations SET candidate_kind='new_mode_candidate' WHERE id=?", (first.id,))
        registry._conn.commit()
        index.refresh(first.id)
        assert list(index.scores(_unit())) == []
        registry._conn.execute("UPDATE registrations SET candidate_kind='candidate' WHERE id=?", (other.id,))
        registry._conn.commit()
        index.refresh(other.id)
        _assert_fresh_index_scores(index, registry, _unit(1))
        registry._conn.execute("UPDATE registrations SET candidate_kind='candidate' WHERE id=?", (first.id,))
        registry._conn.commit()
        index.refresh(first.id)
        _assert_fresh_index_scores(index, registry, _unit(3))
    finally:
        registry.close()


def test_match_score_filter_preserves_float_threshold_boundaries_and_ties(tmp_path):
    from system.dinov2.registry_match_index import RegistryMatchIndex
    registry = SpeciesRegistry(tmp_path / "registry.sqlite3", model_fingerprint="3" * 64)
    try:
        for _ in range(2):
            entry_id = registry._create()
            registry.record_observation(entry_id, _unit() + _unit(1) * 0.5, camera_id="c",
                                        captured_at=None, source_path=f"{entry_id}.jpg")
        index = RegistryMatchIndex(registry._conn)
        scores = list(index.scores(_unit()))
        score = scores[0][1]
        assert len(list(index.scores(_unit(), minimum_score=score))) == 2
        assert list(index.scores(_unit(), minimum_score=np.nextafter(score, np.inf))) == []
        registry.join_threshold = registry.registration_threshold = score
        assert registry.match(_unit())["id"] == scores[0][0]["id"]
        assert registry.match(_unit(), exclude_entry_ids={scores[0][0]["id"]})["id"] == scores[1][0]["id"]
        registry.join_threshold = registry.registration_threshold = float(np.nextafter(score, np.inf))
        assert registry.match(_unit()) is None
    finally:
        registry.close()


def test_feedback_example_reuses_preloaded_bgr_frame(tmp_path, monkeypatch):
    from system.backend.dinov2_persistence_fast import persist_runtime_observations
    from system.dinov2.registry_examples import feedback_observation_example_path

    source = tmp_path / "image.jpg"
    source.write_bytes(b"invalid image: must use the decoded frame")
    observation = SimpleNamespace(
        result_index=0, box_index=0, observation_id="one", embedding=_unit(),
        accepted=True, species="A", source="checkpoint", registry_id=None,
        known_score=0.9, threshold=0.5, bbox=(0, 0, 32, 32), best_known_species="A",
    )
    persisted = []
    store = SimpleNamespace(path=tmp_path / "feedback.sqlite3",
                            persist_observations=lambda values: persisted.extend(values))
    detector = SimpleNamespace(dinov2_feedback=store, dinov2_registry=None,
                               drain_dinov2_observations=lambda: (observation,))
    monkeypatch.setattr(cv2, "imread", lambda *args: pytest.fail("frame must not be decoded again"))
    frame = np.full((32, 32, 3), [14, 100, 200], dtype=np.uint8)
    persist_runtime_observations(
        detector, [source], [SimpleNamespace(date_taken=None)], tmp_path,
        preloaded_data=([0], [frame], [frame[:, :, ::-1]]),
    )
    assert len(persisted) == 1
    example = feedback_observation_example_path(store.path, "one")
    decoded = cv2.imdecode(np.fromfile(example, np.uint8), cv2.IMREAD_COLOR)
    assert np.max(np.abs(decoded[160, 160].astype(int) - [14, 100, 200])) < 4


@pytest.mark.parametrize("device,enabled,expected", [("cuda", True, True), ("cuda", False, False), ("cpu", True, False)])
def test_dinov2_inference_honors_precision_option(device, enabled, expected):
    from system.dinov2.image_processor import ImageProcessor

    processor = ImageProcessor(None)
    processor._legacy = SimpleNamespace(_determine_device=lambda enabled: ("cpu", False),
                                        _sync_device=lambda device: None)
    processor.model = lambda *args, **kwargs: [SimpleNamespace(boxes=None)]
    encoder = SimpleNamespace(device=device, use_fp16=False)
    processor.load_dinov2_classifier(SimpleNamespace(encoder=encoder))
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    processor.detect_batch_species(["image.jpg"], use_fp16=enabled,
                                    preloaded_data=([0], [frame], [frame]))
    assert encoder.use_fp16 is expected
