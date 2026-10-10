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
