"""Equivalence and persistence guarantees for sequence aggregation optimizations."""
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from system import detection_db
from system.backend.job_state_store import JobStateStore
from system.backend.models import DetectionItem, JobState, JobSummary
from system.dinov3 import context
from system.dinov3 import processor
from system.dinov3.classifier import DinoV3Prediction
from system.dinov3.job_cache import DinoV3JobCache, file_stamp
from system.dinov3.processor import (
    DinoV3CropCollector, DinoV3CropRecord, _rejection_event_groups, read_image_with_digest,
)


def _scan_history(index, query):
    """Original exhaustive protocol, kept independent of the range index."""
    config = index.config
    when = index.times[query]
    if when is None or config.max_history == 0:
        return []
    camera, sequence = index.cameras[query], index.sequences[query]
    visible = [i for i in index._by_camera.get(camera, [])
               if when - config.days * 86400 <= index.times[i] < when]
    seen = set(index.identities[query])
    selected = []
    short_limit = min(config.short_term, config.max_history)
    if short_limit:
        for i in reversed(visible):
            if index.sequences[i] == sequence and seen.isdisjoint(index.identities[i]):
                selected.append(i)
                seen.update(index.identities[i])
                if len(selected) == short_limit:
                    break
    groups = {}
    for i in visible:
        if index.sequences[i] != sequence and index.identities[query].isdisjoint(index.identities[i]):
            groups.setdefault(index.sequences[i], []).append(i)
    ordered = sorted(groups, key=lambda seq: index._sequence_ranks[camera, seq])
    for seq in ordered:
        if len(selected) == config.max_history:
            break
        candidates = [i for i in groups[seq] if seen.isdisjoint(index.identities[i])]
        if candidates:
            row = min(candidates, key=lambda i: (index._ranks[i], i))
            selected.append(row)
            seen.update(index.identities[row])
    return selected


@pytest.mark.parametrize("short,max_history", [(0, 0), (0, 8), (3, 8), (9, 5)])
@pytest.mark.parametrize("rank_ties", [False, True])
@pytest.mark.parametrize("days", [0.1, 7.0])
def test_index_matches_exhaustive_history(monkeypatch, short, max_history, rank_ties, days):
    rng = np.random.default_rng(15112026)
    records = [dict(camera_id=f"camera-{rng.integers(3)}", seq_id=f"event-{rng.integers(7)}",
                    datetime=None if i % 17 == 0 else 1700000000 + int(rng.integers(20)) * 86400,
                    content_hash=f"hash-{rng.integers(20)}", path=f"path-{rng.integers(30)}")
               for i in range(150)]
    if rank_ties:
        monkeypatch.setattr(context.FastCausalHistoryIndex, "_rank", lambda *args: b"tie")
    config = context.HistoryConfig(days=days, max_history=max_history, short_term=short)
    # Duplicated candidate indices, missing times and queries outside the
    # candidate pool must retain the same behavior as exhaustive selection.
    index = context.FastCausalHistoryIndex(records, config, candidate_indices=list(range(0, 150, 2)) * 2)
    expected = np.full((150, max_history), -1, dtype=np.int64)
    for query in range(150):
        rows = _scan_history(index, query)
        expected[query, :len(rows)] = rows
    np.testing.assert_array_equal(index.build().indices, expected)
    queries = [149, 2, 50, 2]
    np.testing.assert_array_equal(index.build(queries).indices, expected[queries])


def test_history_adaptation_preserves_raw_vector_pooling():
    rng = np.random.default_rng(15)
    raw = rng.normal(size=(18, 24)).astype(np.float32) * np.arange(1, 19, dtype=np.float32)[:, None]
    raw[3] = 0
    queries = [15, 8, 12, 1]
    history = np.array([[0, 1, 3, -1], [5, 6, -1, -1], [-1, -1, -1, -1], [0, 3, 9, 10]])
    expected = raw[queries].copy()
    for row, query in enumerate(queries):
        candidates = history[row][history[row] >= 0]
        if not len(candidates):
            continue
        h = raw[candidates]
        logits = context.unit(h) @ context.unit(raw[query]) / 0.1
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        expected[row] = context.unit(0.9 * raw[query] + 0.1 * (weights @ h))
    actual = context.adapt_with_history(raw, history, queries, chunk_size=2)
    np.testing.assert_allclose(actual, expected, atol=2e-7)


def test_grouping_never_formats_feature_arrays_and_does_not_mutate_inputs(monkeypatch):
    class Feature(np.ndarray):
        def __str__(self):
            raise AssertionError("a feature vector is not a text field")

    feature = np.ones(768, dtype=np.float32).view(Feature)
    records = [dict(camera_id="camera", path=f"{i}.jpg", raw_embedding=feature,
                    date_taken=1700000000 + i) for i in range(4)]
    original_extract = context._extract_unit_embedding
    calls = []

    def extract(record):
        calls.append(record["path"])
        return original_extract(record)

    monkeypatch.setattr(context, "_extract_unit_embedding", extract)
    grouped = context.group_job_events(records)
    assert len({r["track_id"] for r in grouped}) == 1
    assert len(calls) == len(records)
    assert all("datetime" not in record and "track_id" not in record for record in records)


@pytest.mark.parametrize("nested", [False, True])
def test_rejection_group_reuse_matches_directory_regrouping(tmp_path, nested):
    records = []
    for i in range(8):
        folder = tmp_path / "site" / (f"camera-{i % 2}" if nested else "camera")
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{i}.jpg"
        records.append(dict(path=str(path), camera_id=context.camera_id_for_path(path, tmp_path),
                            date_taken=None if i == 7 else 1700000000 + i * 900,
                            box_index=0, raw_embedding=np.ones(16, dtype=np.float32)))
    original = context.group_job_events(records)
    expected = context.group_job_events([{**r, "camera_id": context.camera_id_for_path(r["path"])} for r in records])
    actual = _rejection_event_groups(original)
    fields = ("camera_id", "path", "event_id", "track_id", "sequence_id", "datetime")
    key = lambda rows: sorted(tuple(row[f] for f in fields) for row in rows)
    assert key(actual) == key(expected)


def test_decode_digest_survives_checkpoint_and_invalidates_on_change(monkeypatch, tmp_path):
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "image.png"
    pixels = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
    path.write_bytes(cv2.imencode(".png", pixels)[1].tobytes())
    collector = DinoV3CropCollector()
    np.testing.assert_array_equal(read_image_with_digest(path, collector), cv2.imread(str(path)))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert collector.image_digest(path) == digest
    collector.record_crop(DinoV3CropRecord(str(path), 0, (1, 2, 10, 12), .9, np.ones(16, np.float32)))
    item = DetectionItem(filename=path.name, path=str(path), file_type="image",
                         detection_data={"detection_completed": True, "detection_box_indices": [0]})
    cache = DinoV3JobCache(tmp_path / "state.json", "job", {})
    cache.save_batch([item], collector, {str(path): file_stamp(path)})
    restored = DinoV3CropCollector()
    assert len(cache.restore([path], restored)) == 1
    assert restored.image_digest(path) == digest

    def no_reread(*args):
        raise AssertionError("cached source bytes must not be read again")

    with monkeypatch.context() as patch:
        patch.setattr(processor, "_image_file_digest", no_reread)
        predictions = [DinoV3Prediction("deer", True, "deer", 1.0, .1, np.ones(16), (), .8, .7)]
        classifier = SimpleNamespace(backend="dinov3", display_name="Neri1.2preview",
                                     classify_features=lambda *a, **kw: predictions)
        detector = SimpleNamespace(dinov3_classifier=classifier, _dinov3_collector=restored)
        finished = processor.finalize_photo_job(detector, [item], tmp_path)
        assert finished[0].detection_data["sequence_completed"] is True

    cache.restore([path], restored)
    # Crop retries must not discard fingerprints obtained by preloading.
    restored.discard_paths([path])
    assert restored.image_digest(path) == digest
    path.write_bytes(b"changed source file")
    assert restored.image_digest(path) is None
    restored.clear()
    assert not restored._image_digests


def test_old_feature_cache_migrates_without_discarding_features(tmp_path):
    path = tmp_path / "image.jpg"
    path.write_bytes(b"fixture")
    collector = DinoV3CropCollector()
    collector.record_crop(DinoV3CropRecord(str(path), 0, (1, 2, 10, 12), .9, np.ones(16, np.float32)))
    item = DetectionItem(filename=path.name, path=str(path), file_type="image",
                         detection_data={"detection_completed": True, "detection_box_indices": [0]})
    cache = DinoV3JobCache(tmp_path / "state.json", "job", {})
    cache.save_batch([item], collector, {str(path): file_stamp(path)})
    with closing(sqlite3.connect(cache.path)) as conn, conn:
        conn.execute("ALTER TABLE photos DROP COLUMN content_digest")
    reopened = DinoV3JobCache(tmp_path / "state.json", "job", {})
    restored = DinoV3CropCollector()
    assert len(reopened.restore([path], restored)) == 1
    assert len(restored.get_crops_for_path(path)) == 1
    assert restored.image_digest(path) is None


def test_media_batch_uses_one_connection_and_rolls_back_both_tables(monkeypatch, tmp_path):
    database = str(tmp_path / "detections.db")
    detection_db.upsert_media_batch(database, [("a", "a.jpg", {"value": "old"})], [("a.jpg", True)])
    with closing(sqlite3.connect(database)) as conn, conn:
        conn.execute("CREATE TRIGGER reject_validation BEFORE UPDATE ON validation "
                     "BEGIN SELECT RAISE(ABORT, 'simulated failure'); END")
    original_connect = sqlite3.connect
    connections = []

    def connect(*args, **kwargs):
        connections.append(args[0])
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(detection_db.sqlite3, "connect", connect)
    with pytest.raises(sqlite3.IntegrityError, match="simulated failure"):
        detection_db.upsert_media_batch(database, [("a", "a.jpg", {"value": "new"})], [("a.jpg", False)])
    assert connections == [database]
    assert detection_db.get_detection(database, "a") == {"value": "old"}
    assert detection_db.get_all_validation(database) == {"a.jpg": True}


def test_job_corrections_update_only_changed_rows_and_trim_tail(tmp_path):
    now = datetime.now(timezone.utc).isoformat()
    store = JobStateStore(tmp_path / "state.json")
    items = [DetectionItem(filename=f"{i}.jpg", path=f"{i}.jpg", file_type="image") for i in range(3)]
    job = JobSummary(id="job", state=JobState.RUNNING, input_dir=str(tmp_path),
                     created_at=now, updated_at=now, results=items)
    store.save({"job": job}, {})
    with closing(sqlite3.connect(store.path)) as conn, conn:
        conn.execute("CREATE TABLE writes(kind TEXT, position INTEGER)")
        conn.execute("CREATE TRIGGER track_updates AFTER UPDATE ON results "
                     "BEGIN INSERT INTO writes VALUES('update', new.position); END")
        conn.execute("CREATE TRIGGER track_deletes AFTER DELETE ON results "
                     "BEGIN INSERT INTO writes VALUES('delete', old.position); END")
    replaced = [item.model_copy() for item in items]
    replaced[0] = replaced[0].model_copy(update={"species": ["corrected"]})
    job = job.model_copy(update={"results": replaced})
    store.save({"job": job}, {})
    with closing(sqlite3.connect(store.path)) as conn:
        assert conn.execute("SELECT * FROM writes").fetchall() == [("update", 0)]
    job = job.model_copy(update={"results": replaced[:1]})
    store.save({"job": job}, {})
    results = store.load()["jobs"]["job"]["results"]
    assert len(results) == 1 and results[0]["species"] == ["corrected"]
