from pathlib import Path
from contextlib import closing
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from system.backend import services
from system.backend.models import CreateJobRequest, DetectionItem, JobState, JobSummary, ProcessingOptions
from system.dinov3.classifier import DinoV3Prediction
from system.dinov3.job_cache import DinoV3JobCache, file_stamp
from system.dinov3.processor import DinoV3CropCollector, DinoV3CropRecord


def photo(path, *, empty=False):
    return DetectionItem(filename=path.name, path=str(path), file_type="jpg", detection_data={
        "detection_completed": True, "detection_box_count": 0 if empty else 1,
        "detection_box_indices": [] if empty else [0],
        "检测框": [] if empty else [{"边界框": [1, 2, 30, 40]}],
        "sequence_pending": True,
    })


def crop(path):
    return DinoV3CropRecord(str(path), 0, (1, 2, 30, 40), 0.9,
                            np.array([0.123456789012345, 0.9, 0.3], dtype=np.float64))


def test_cache_restores_full_precision_and_empty_photos_after_reopening(tmp_path):
    paths = [tmp_path / name for name in ("animal.jpg", "empty.jpg")]
    for p in paths:
        p.write_bytes(b"fixture")
    collector = DinoV3CropCollector()
    collector.record_crop(crop(paths[0]))
    signature = {"model": "v1"}
    cache = DinoV3JobCache(tmp_path / "state.json", "one", signature)
    cache.save_batch([photo(paths[0]), photo(paths[1], empty=True)], collector,
                     {str(p): file_stamp(p) for p in paths})
    collector.clear()
    reopened = DinoV3JobCache(tmp_path / "state.json", "one", signature)
    restored = reopened.restore(paths, collector)
    assert [item.path for item in restored] == list(map(str, paths))
    actual = collector.get_all_crops()[0]
    assert actual.raw_embedding.dtype == np.float64
    np.testing.assert_array_equal(actual.raw_embedding, crop(paths[0]).raw_embedding)
    assert collector.get_crops_for_path(paths[1]) == []
    other = DinoV3JobCache(tmp_path / "state.json", "another-job", signature)
    assert other.restore(paths, DinoV3CropCollector()) == []


@pytest.mark.parametrize("invalidate", ["source", "model", "blob", "indices", "incomplete"])
def test_invalid_features_are_not_treated_as_completed(tmp_path, invalidate):
    path = tmp_path / "animal.jpg"
    path.write_bytes(b"fixture")
    collector = DinoV3CropCollector()
    collector.record_crop(crop(path))
    cache = DinoV3JobCache(tmp_path / "state.json", "one", {"model": "v1"})
    item = photo(path)
    if invalidate == "incomplete":
        collector.clear()
    cache.save_batch([item], collector, {str(path): file_stamp(path)})
    if invalidate == "source":
        path.write_bytes(b"different contents")
    elif invalidate == "model":
        cache = DinoV3JobCache(tmp_path / "state.json", "one", {"model": "v2"})
    elif invalidate == "blob":
        with closing(sqlite3.connect(cache.path)) as conn, conn:
            conn.execute("UPDATE photos SET features=?", (b"incomplete",))
    elif invalidate == "indices":
        with closing(sqlite3.connect(cache.path)) as conn, conn:
            conn.execute('UPDATE photos SET crops=?', ('[{"box_index":1,"bbox":[1,2,30,40],"detection_confidence":0.9}]',))
    restored_collector = DinoV3CropCollector()
    assert cache.restore([path], restored_collector) == []
    assert restored_collector.get_all_crops() == []


@pytest.mark.parametrize("cancel_phase,invalidate", [
    ("photos", None), ("classification", None), ("history", None), ("write", None),
    ("classification", "source"), ("classification", "blob"),
])
def test_resume_after_restart_reuses_features_and_reports_sequence_progress(
    monkeypatch, tmp_path, cancel_phase, invalidate,
):
    legacy = services._legacy
    state_path = tmp_path / "state.json"
    for module in (services, legacy):
        monkeypatch.setattr(module, "job_state_path", lambda: state_path)
    paths = [tmp_path / name for name in ("1.jpg", "2.jpg", "3.jpg")]
    for path in paths:
        path.write_bytes(path.name.encode())
    monkeypatch.setattr(legacy, "_validate_dino_job_options", lambda *a: None)
    monkeypatch.setattr(legacy, "_start_batch_log_session", lambda *a: None)
    monkeypatch.setattr(legacy, "_resolve_job_inputs", lambda *a, **kw: (tmp_path, paths))
    monkeypatch.setattr(legacy, "_resolve_classification_model_path", lambda p: Path(p))
    monkeypatch.setattr(legacy, "_dino_manifest_payload", lambda p: {
        "backend": "dinov3", "display_name": "Neri1.2preview",
    })
    monkeypatch.setattr(legacy, "load_media_indexes", lambda *a, **kw: ({}, {}))
    monkeypatch.setattr(legacy, "_save_detection_data_batch", lambda *a, **kw: None)
    monkeypatch.setattr(legacy, "_export_results", lambda *a, **kw: None)
    detected, classifications, updates = [], [], []
    manager = services.ProcessingJobManager()
    should_cancel = True

    class Classifier:
        backend = "dinov3"
        display_name = "Neri1.2preview"
        model_fingerprint = "fixture"
        model = SimpleNamespace(inference_recipe="production_dual")

        def classify_features(self, raw_features, **kwargs):
            classifications.append(raw_features.copy())
            if should_cancel and cancel_phase == "classification":
                manager._cancelled_jobs.add("job")
            return [DinoV3Prediction(
                species="deer", accepted=True, best_known_species="deer", known_score=1.0,
                threshold=0.1, embedding=f, candidates=(), ridge_score=0.8, within_score=0.7,
            ) for f in raw_features]

    detectors = []

    def load_detector(*args):
        detector = SimpleNamespace(dinov3_classifier=Classifier(), _dinov3_collector=DinoV3CropCollector())
        detectors.append(detector)
        return detector

    def detect(detector, batch, batch_items, *args, **kwargs):
        detected.extend(batch)
        output = []
        for p in batch:
            empty = p == paths[-1]
            if not empty:
                detector._dinov3_collector.record_crop(crop(p))
            output.append(photo(p, empty=empty))
        if should_cancel and cancel_phase == "photos":
            manager._cancelled_jobs.add("job")
        return output

    monkeypatch.setattr(legacy, "_load_detector", load_detector)
    monkeypatch.setattr(legacy, "_detect_image_batch", detect)
    request = CreateJobRequest(input_dir=str(tmp_path), options=ProcessingOptions(
        enable_detection=True, model_path="fake.pt", classification_model_path="joint.neri.json",
        batch_size=1, video_mode="skip"))
    now = services.utc_now()
    manager._jobs["job"] = JobSummary(id="job", state=JobState.QUEUED, input_dir=str(tmp_path),
                                      created_at=now, updated_at=now)
    manager._job_requests["job"] = request
    original_mutate = manager._mutate_job

    def mutate(*args, **kwargs):
        original_mutate(*args, **kwargs)
        stage = kwargs.get("sequence_stage")
        if should_cancel and ((cancel_phase == "history" and stage == "整合历史特征")
                              or (cancel_phase == "write" and stage == "写入序列结果")):
            manager._cancelled_jobs.add("job")

    monkeypatch.setattr(manager, "_mutate_job", mutate)
    restarted = None
    try:
        manager._run_job("job", request)
        assert manager.get_job("job").state == JobState.CANCELLED
        cache_path = DinoV3JobCache.path_for(state_path, "job")
        assert cache_path.exists()
        before = list(detected)
        if invalidate == "source":
            paths[0].write_bytes(b"updated source photo")
        elif invalidate == "blob":
            with closing(sqlite3.connect(cache_path)) as conn, conn:
                conn.execute("UPDATE photos SET features=? WHERE path LIKE ?", (b"broken", "%1.jpg"))
        should_cancel = False
        restarted = services.ProcessingJobManager()
        restarted_mutate = restarted._mutate_job

        def record(*args, **kwargs):
            if kwargs.get("sequence_stage"):
                updates.append(kwargs.copy())
            return restarted_mutate(*args, **kwargs)

        monkeypatch.setattr(restarted, "_mutate_job", record)
        restarted.resume_job("job")
        restarted._executor.shutdown(wait=True)
        finished = restarted.get_job("job")
        assert finished.state == JobState.COMPLETED, finished.error
        assert detected == paths + ([paths[0]] if invalidate else [])
        if cancel_phase != "photos":
            assert before == paths
            assert detectors[0]._dinov3_collector.get_all_crops() == []
        assert finished.processed == 3 and finished.sequence_stage is None
        assert all(i.detection_data["sequence_completed"] for i in finished.results)
        assert [i.path for i in finished.results] == list(map(str, paths))
        assert not cache_path.exists()
        assert {u["sequence_stage"] for u in updates} >= {
            "计算照片指纹", "构建因果历史", "整合历史特征", "全序列分类", "聚类拒识目标", "写入序列结果",
        }
        assert all(u["message"].startswith("Neri1.2preview序列汇总：") for u in updates)
        assert any(u["sequence_total"] > 0 and u["sequence_processed"] == u["sequence_total"]
                   for u in updates)
        np.testing.assert_array_equal(classifications[-1], np.stack([crop(p).raw_embedding for p in paths[:2]]))
        # Restored execution must make exactly the same sequence decisions as
        # aggregating the complete, uninterrupted set of features.
        from system.dinov3.processor import finalize_photo_job
        reference_detector = load_detector()
        reference_detector._dinov3_collector.record_crops([crop(p) for p in paths[:2]])
        reference = finalize_photo_job(
            reference_detector, [photo(p, empty=p == paths[-1]) for p in paths], tmp_path,
            apply_data_fn=legacy._apply_detection_data, job_id="job",
        )
        assert [i.detection_data for i in finished.results] == [i.detection_data for i in reference]
    finally:
        manager._executor.shutdown()
        if restarted is not None:
            restarted._executor.shutdown()


@pytest.mark.parametrize("clear_all", [False, True])
def test_deleting_paused_jobs_removes_their_feature_cache(monkeypatch, tmp_path, clear_all):
    state_path = tmp_path / "state.json"
    for module in (services, services._legacy):
        monkeypatch.setattr(module, "job_state_path", lambda: state_path)
    cache = DinoV3JobCache(state_path, "job", {})
    manager = services.ProcessingJobManager()
    now = services.utc_now()
    manager._jobs["job"] = JobSummary(id="job", state=JobState.CANCELLED, input_dir=str(tmp_path),
                                      created_at=now, updated_at=now)
    try:
        if clear_all:
            manager.clear_jobs()
        else:
            manager.delete_job("job")
        assert not cache.path.exists()
    finally:
        manager._executor.shutdown()
