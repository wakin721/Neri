"""Exercise publication, replay calibration, persistence and retry behavior."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from system.backend import dinov3_learning as learning
from system.dinov3.deployment import publish_model
from system.dinov3.encoder import sha256
from system.dinov3.model import DinoV3RidgeWithinModel
from system.dinov3.observation_support import make_dinov3_observation_id
from system.dinov3.runtime import read_manifest, resolve_manifest_asset, load_dinov3_model
from tests.test_dinov3_deployment import deployed
from tests.test_dinov3_box_explanation_and_feedback import _write_image


@pytest.fixture
def ready(deployed, tmp_path, monkeypatch):
    manifest, model, raw = deployed
    monkeypatch.setenv("NERI_DINO_STATE_DIR", str(tmp_path / "state"))
    rng = np.random.default_rng(135)
    unknown = rng.normal(size=(8, 768))
    unknown /= np.linalg.norm(unknown, axis=1, keepdims=True)
    cal = dict(features=np.r_[raw, unknown], labels=np.asarray(["A"]*8+["B"]*8+["U"]*8),
               cameras=np.asarray(["cal"]*24), sequences=np.asarray([f"cal{i}" for i in range(24)]),
               contents=np.asarray([f"calcontent{i}:0" for i in range(24)]))
    payload = read_manifest(manifest)
    with threadpool_limits(limits=1):
        model.calibrate(cal["features"], cal["labels"], cal["cameras"], cal["sequences"], contents=cal["contents"])
        publish_model(model, manifest, existing_manifest=payload, calibration_data=cal)

    class Encoder:
        calls = 0
        def __init__(self, *args, **kwargs):
            pass
        def encode(self, crops, **kwargs):
            type(self).calls += 1
            values = np.zeros((len(crops), 768), dtype=np.float32)
            for i, crop in enumerate(crops):
                values[i, 0 if crop.mean() < 70 else 2] = 1
            return values
    import system.dinov3.encoder
    monkeypatch.setattr(system.dinov3.encoder, "DinoV3Encoder", Encoder)
    return manifest, model, cal, Encoder


def stage(manifest, source, species="C", op="op"):
    bbox = [20, 20, 140, 100]
    box = {"物种": species, "边界框": bbox,
           "observation_id": make_dinov3_observation_id(source, 0, bbox)}
    learning.stage_feedback(manifest, source.parent, source, box, "update", op)
    return box


def loaded(manifest):
    payload = read_manifest(manifest)
    return DinoV3RidgeWithinModel.load(resolve_manifest_asset(manifest, payload["checkpoint"]))


def test_new_and_old_species_update_reload_and_retry(ready, tmp_path):
    manifest, original, _cal, Encoder = ready
    old, new = tmp_path/"old.jpg", tmp_path/"new.jpg"
    _write_image(old, 40)
    _write_image(new, 130)
    stage(manifest, old, "A", "old")
    stage(manifest, new, "C", "new")
    stage(manifest, new, "C", "new-again")
    assert learning.learning_status(manifest)["species_counts"] == {"A": 1, "C": 1}
    thresholds = learning.learning_status(manifest)['rejection']
    assert thresholds['box_threshold'] == loaded(manifest).box_threshold
    assert thresholds['sequence_threshold'] == loaded(manifest).threshold
    assert thresholds['direction'] == 'higher_is_known'
    before = read_manifest(manifest)
    result = learning.update_model(manifest)
    assert result["updated_count"] == 2
    assert result["species_counts"] == {"A": 1, "C": 1}
    updated = loaded(manifest)
    assert set(updated.classes) == {"A", "B", "C"}
    assert updated.is_calibrated
    np.testing.assert_array_equal(updated.origin_, original.origin_)
    np.testing.assert_array_equal(updated.transform_, original.transform_)
    assert len(updated.metadata["human_learning_samples"]) == 2
    assert read_manifest(manifest)["checkpoint"] != before["checkpoint"]
    assert resolve_manifest_asset(manifest, before["checkpoint"]).exists()
    assert learning.learning_status(manifest)["pending_count"] == 0
    stable = manifest.read_bytes()
    assert learning.update_model(manifest)["updated_count"] == 0
    assert manifest.read_bytes() == stable
    assert Encoder.calls == 1
    vector = np.zeros((1, 768), dtype=np.float32)
    vector[0, 2] = 1
    runtime = load_dinov3_model(manifest, encoder_factory=lambda *a, **kw: SimpleNamespace())
    assert runtime.classifier.classify_features(vector)[0].best_known_species == "C"


def test_same_content_deduplicates_across_paths_and_after_publication(ready, tmp_path):
    manifest, _model, _cal, _encoder = ready
    a, b = tmp_path/"a.jpg", tmp_path/"b.jpg"
    _write_image(a, 130)
    b.write_bytes(a.read_bytes())
    stage(manifest, a, op="a")
    stage(manifest, b, op="b")
    assert learning.update_model(manifest)["updated_count"] == 1
    c = tmp_path/"c.jpg"
    c.write_bytes(a.read_bytes())
    stage(manifest, c, op="c")
    before = manifest.read_bytes()
    assert learning.update_model(manifest)["updated_count"] == 0
    assert manifest.read_bytes() == before


def test_pending_label_change_and_undo(ready, tmp_path):
    manifest, *_ = ready
    source = tmp_path/"a.jpg"
    _write_image(source)
    box = stage(manifest, source, "A", "initial")
    stage(manifest, source, "C", "corrected")
    learning.revert_staged_feedback(manifest, "corrected")
    assert learning.learning_status(manifest)["species_counts"] == {"A": 1}
    for action in ("empty", "unverified"):
        learning.stage_feedback(manifest, tmp_path, source, box, action, action)
        assert learning.learning_status(manifest)["pending_count"] == 0
        learning.revert_staged_feedback(manifest, action)
        assert learning.learning_status(manifest)["pending_count"] == 1
    learning.revert_staged_feedback(manifest, "initial")
    assert learning.learning_status(manifest)["pending_count"] == 0


@pytest.mark.parametrize("failure", ["encoder", "calibration", "publication", "changed", "conflict", "calibration_hash"])
def test_failure_preserves_model_and_pending(ready, tmp_path, monkeypatch, failure):
    manifest, _model, _cal, Encoder = ready
    source = tmp_path/"a.jpg"
    _write_image(source, 130)
    stage(manifest, source)
    before = manifest.read_bytes()
    if failure == "encoder":
        monkeypatch.setattr(Encoder, "encode", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("encoder failed")))
    elif failure == "calibration":
        monkeypatch.setattr(learning, "_recalibrate", lambda *a: (_ for _ in ()).throw(ValueError("no unknown")))
    elif failure == "publication":
        import system.dinov3.deployment
        monkeypatch.setattr(system.dinov3.deployment, "publish_model", lambda *a, **kw: (_ for _ in ()).throw(OSError("disk full")))
    elif failure == "changed":
        source.write_bytes(b"changed")
    elif failure == "conflict":
        other = tmp_path/"other.jpg"
        other.write_bytes(source.read_bytes())
        stage(manifest, other, "A", "other")
    else:
        payload = read_manifest(manifest)
        resolve_manifest_asset(manifest, payload["calibration"]).write_bytes(b"corrupt")
    with pytest.raises((ValueError, RuntimeError, OSError)):
        learning.update_model(manifest)
    assert manifest.read_bytes() == before
    assert learning.learning_status(manifest)["pending_count"] == (2 if failure == "conflict" else 1)


def test_calibration_import_checks_provenance_and_preserves_full_context(ready, tmp_path):
    manifest, _model, cal, _encoder = ready
    before = manifest.read_bytes()
    file = tmp_path/"external.npz"
    np.savez_compressed(file, **cal, encoder_sha256=np.asarray("0"*64))
    with pytest.raises(ValueError, match="SHA-256"):
        learning.import_calibration(manifest, file)
    assert manifest.read_bytes() == before
    data = {**cal, "adapted_features": cal["features"].copy()}
    np.savez_compressed(file, **data, encoder_sha256=np.asarray(read_manifest(manifest)["encoder_sha256"]))
    assert learning.import_calibration(manifest, file)["calibration_available"]
    payload = read_manifest(manifest)
    archive = resolve_manifest_asset(manifest, payload["calibration"])
    assert sha256(archive) == payload["calibration_sha256"]
    with np.load(archive, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["adapted_features"], data["adapted_features"])


def test_new_species_cannot_consume_last_auxiliary_unknown_class(ready, tmp_path):
    manifest, *_ = ready
    source = tmp_path/"u.jpg"
    _write_image(source, 130)
    stage(manifest, source, "U")
    before = manifest.read_bytes()
    with pytest.raises(ValueError, match="unknown"):
        learning.update_model(manifest)
    assert manifest.read_bytes() == before
    assert learning.learning_status(manifest)["pending_count"] == 1


def test_automatic_feedback_relabels_and_reverts_after_learning(ready, tmp_path):
    from system.dinov3.observation_support import apply_dinov3_box_feedback, revert_dinov3_feedback_operation
    from tests.test_dinov3_box_explanation_and_feedback import _write_detection_db, _box
    manifest, *_ = ready
    source = tmp_path / "auto.jpg"
    _write_image(source, 130)
    _write_detection_db(tmp_path, [(source, {"检测框": [_box()], "sequence_completed": True})])
    observation = make_dinov3_observation_id(source, 0, [10, 10, 80, 80])

    def await_worker():
        key = (learning._key(manifest), str(learning.default_state_root()))
        assert learning._WORKERS[key]["idle"].wait(30), "automatic learning did not finish"
        status = learning.learning_status(manifest)
        assert status["state"] == "ready", status
        assert status["pending_count"] == 0
        return status

    def confirm(name, operation):
        return apply_dinov3_box_feedback(SimpleNamespace(
            input_path=str(tmp_path), file_path=str(source), classification_model_path=str(manifest),
            observation_id=observation, action="update", species_name=name, feedback_operation_id=operation))

    confirm("C", "first")
    assert await_worker()["learned_species_counts"] == {"C": 1}
    assert "C" in loaded(manifest).classes
    confirm("B", "correction")
    assert await_worker()["learned_species_counts"] == {"B": 1}
    assert set(loaded(manifest).classes) == {"A", "B"}
    revert_dinov3_feedback_operation(SimpleNamespace(classification_model_path=str(manifest), feedback_operation_id="correction"))
    assert await_worker()["learned_species_counts"] == {"C": 1}
    revert_dinov3_feedback_operation(SimpleNamespace(classification_model_path=str(manifest), feedback_operation_id="first"))
    assert await_worker()["learned_count"] == 0
    assert set(loaded(manifest).classes) == {"A", "B"}


@pytest.mark.parametrize("phase", ["encoding", "publication"])
def test_annotations_remain_writable_while_encoding_and_stale_update_never_publishes(ready, tmp_path, monkeypatch, phase):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    manifest, _model, _cal, Encoder = ready
    source = tmp_path / "race.jpg"
    _write_image(source, 130)
    stage(manifest, source, "C", "first")
    entered, release = threading.Event(), threading.Event()
    original = Encoder.encode
    def slow_encode(self, *args, **kwargs):
        entered.set()
        assert release.wait(10)
        return original(self, *args, **kwargs)
    if phase == "encoding":
        monkeypatch.setattr(Encoder, "encode", slow_encode)
    else:
        save = DinoV3RidgeWithinModel.save
        def slow_save(self, *args, **kwargs):
            entered.set()
            assert release.wait(10)
            return save(self, *args, **kwargs)
        monkeypatch.setattr(DinoV3RidgeWithinModel, "save", slow_save)
    before = manifest.read_bytes()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(learning.update_model, manifest)
        assert entered.wait(10)
        try:
            stage(manifest, source, "B", "newer")
        finally:
            release.set()
        assert future.result(timeout=20)["superseded"]
    assert manifest.read_bytes() == before
    assert learning.learning_status(manifest)["species_counts"] == {"B": 1}


def test_bookkeeping_failure_after_publication_keeps_active_generation_usable(ready):
    from contextlib import contextmanager
    manifest, model, cal, _encoder = ready
    @contextmanager
    def broken_commit():
        yield
        raise OSError("simulated SQLite commit failure")
    before = manifest.read_bytes()
    with pytest.raises(OSError, match="SQLite"):
        publish_model(model, manifest, existing_manifest=read_manifest(manifest),
                      calibration_data=cal, publication_guard=broken_commit)
    assert manifest.read_bytes() != before
    assert loaded(manifest).is_calibrated
