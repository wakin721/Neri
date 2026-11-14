"""Bad source files must not block sequence aggregation for healthy photos."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from system.backend import services
from system.backend.models import CreateJobRequest, DetectionItem, JobState, JobSummary, ProcessingOptions
from system.dino.image_processor import ImageProcessor
from system.dinov3.classifier import DinoV3Prediction
from system.dinov3.processor import (
    DinoV3CropCollector, DinoV3CropRecord, finalize_photo_job,
    ensure_dinov3_sequences_completed, DinoV3SequenceFinalizationError,
)


@pytest.mark.parametrize("bad_bytes", [b"", b"not a JPEG"])
def test_real_decoder_bad_file_does_not_fail_healthy_photo_job(monkeypatch, tmp_path, bad_bytes):
    good = tmp_path / "good.jpg"
    bad = tmp_path / "bad.jpg"
    Image.new("RGB", (24, 24), "gray").save(good)
    bad.write_bytes(bad_bytes)
    manager, request, exported = _job_fixture(monkeypatch, tmp_path, [good, bad])
    try:
        manager._run_job("job", request, [])
        job = manager.get_job("job")
        assert job.state == JobState.COMPLETED, job.error
        assert job.processed == job.total == 2
        assert job.failed_count == 1
        assert "1 个文件失败" in job.message
        by_name = {item.filename: item for item in job.results}
        assert by_name[good.name].error is None
        assert by_name[good.name].detection_data["sequence_completed"] is True
        assert "无法读取或解码" in by_name[bad.name].error
        assert by_name[bad.name].detection_data["sequence_status"] == "failed"
        assert by_name[bad.name].detection_data["sequence_completed"] is False
        assert by_name[bad.name].detection_data["sequence_pending"] is False
        assert [item.path for item in exported] == [str(good)]
        assert bad.read_bytes() == bad_bytes
        # Reload the actual SQLite checkpoint, not just in-memory counters.
        manager._save_state_unlocked()
        restored = services.ProcessingJobManager()
        try:
            assert restored.get_job("job").failed_count == 1
            assert restored.get_job("job").results[1].error
        finally:
            restored._executor.shutdown()
    finally:
        manager._executor.shutdown()


def test_all_unreadable_photos_report_failure_and_never_export_as_empty(monkeypatch, tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"")
    manager, request, exported = _job_fixture(monkeypatch, tmp_path, [bad])
    try:
        manager._run_job("job", request, [])
        job = manager.get_job("job")
        assert job.state == JobState.FAILED
        assert job.failed_count == 1
        assert "全部文件处理失败" in job.error
        assert job.results[0].detection_data["sequence_status"] == "failed"
        assert not exported
    finally:
        manager._executor.shutdown()


def test_failed_photo_is_excluded_from_features_without_losing_its_error(tmp_path):
    good = tmp_path / "good.jpg"
    good.write_bytes(b"source identity")
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"")
    feature = np.ones(768, dtype=np.float32) / np.sqrt(768)
    collector = DinoV3CropCollector()
    collector.record_crop(DinoV3CropRecord(str(good), 0, (1, 2, 12, 18), .9, feature))
    classified = []

    def classify(features, **kwargs):
        classified.append(features.copy())
        return [DinoV3Prediction(
            species="horse", accepted=True, best_known_species="horse",
            known_score=1., threshold=0., embedding=feature, candidates=(),
            ridge_score=.8, within_score=.7,
        )]

    detector = SimpleNamespace(
        dinov3_classifier=SimpleNamespace(backend="dinov3", classify_features=classify),
        _dinov3_collector=collector,
    )
    items = [
        DetectionItem(filename=bad.name, path=str(bad), file_type="jpg", error="zero-byte file"),
        DetectionItem(filename=good.name, path=str(good), file_type="jpg", detection_data={
            "detection_completed": True, "detection_box_indices": [0], "detection_box_count": 1,
        }),
    ]
    saved = []
    completed = finalize_photo_job(
        detector, items, tmp_path, apply_data_fn=services._legacy._apply_detection_data,
        save_batch_fn=lambda batch, root: saved.extend(batch),
    )
    assert len(classified) == 1 and classified[0].shape == (1, 768)
    assert completed[0].error == "zero-byte file"
    assert completed[0].detection_data["sequence_status"] == "failed"
    assert completed[1].detection_data["sequence_completed"] is True
    assert completed[1].species == ["horse"]
    assert {path.name for path, _ in saved} == {good.name, bad.name}
    ensure_dinov3_sequences_completed(completed)


def test_unexplained_missing_features_still_fail_instead_of_being_skipped(tmp_path):
    detector = SimpleNamespace(
        dinov3_classifier=SimpleNamespace(backend="dinov3"),
        _dinov3_collector=DinoV3CropCollector(),
    )
    item = DetectionItem(filename="lost.jpg", path=str(tmp_path / "lost.jpg"), file_type="jpg",
                         detection_data={"detection_completed": True, "detection_box_indices": [0]})
    with pytest.raises(DinoV3SequenceFinalizationError, match="crop collection is incomplete"):
        finalize_photo_job(detector, [item], tmp_path)


def test_retry_failed_job_selects_only_errors_and_preserves_parent_and_export(monkeypatch, tmp_path):
    good = tmp_path / "good.jpg"
    bad = tmp_path / "bad.jpg"
    good.write_bytes(b"valid")
    bad.write_bytes(b"")
    manager, request, _ = _job_fixture(monkeypatch, tmp_path, [good, bad])
    try:
        parent = manager._jobs["job"].model_copy(update={
            "state": JobState.COMPLETED, "failed_count": 1,
            "results": [
                DetectionItem(filename=good.name, path=str(good), file_type="jpg"),
                DetectionItem(filename=bad.name, path=str(bad), file_type="jpg", error="bad file"),
            ],
        })
        manager._jobs["job"] = parent
        manager._job_requests["job"] = request.model_copy(update={"output_dir": str(tmp_path / "export")})
        captured = []
        monkeypatch.setattr(manager, "create_job", lambda req: captured.append(req) or "retry")
        assert manager.retry_failed_job("job") == "retry"
        assert captured[0].input_paths == [str(bad)]
        assert captured[0].output_dir is None
        assert captured[0].options == request.options
        assert manager._jobs["job"] is parent
    finally:
        manager._executor.shutdown()


def _job_fixture(monkeypatch, tmp_path, photos):
    legacy = services._legacy
    for module in (services, legacy):
        monkeypatch.setattr(module, "job_state_path", lambda: tmp_path / "state.json")
    monkeypatch.setattr(legacy, "_validate_dino_job_options", lambda *a: None)
    monkeypatch.setattr(legacy, "_start_batch_log_session", lambda *a: None)
    monkeypatch.setattr(legacy, "_resolve_job_inputs", lambda *a, **kw: (tmp_path, photos))
    monkeypatch.setattr(legacy, "_resolve_classification_model_path", Path)
    monkeypatch.setattr(legacy, "_dino_manifest_payload", lambda *a: {"backend": "dinov3"})
    monkeypatch.setattr(legacy, "load_media_indexes", lambda *a, **kw: ({}, {}))
    monkeypatch.setattr(legacy, "_build_metadata_item", lambda path: DetectionItem(
        filename=path.name, path=str(path), file_type="jpg", size_bytes=path.stat().st_size,
    ))
    monkeypatch.setattr(legacy, "_persist_dino_observations", lambda *a, **kw: None)
    monkeypatch.setattr(legacy, "_selected_inference_class_ids", lambda *a: None)
    monkeypatch.setattr(legacy, "_save_detection_data_batch", lambda *a: None)
    exported = []
    monkeypatch.setattr(legacy, "_export_results", lambda out, items: exported.extend(items))
    detector = ImageProcessor(None)
    # Real disk reading and decoding, frozen lightweight detector output.
    underlying = detector._ensure_legacy()
    detector.model = lambda images, **kw: [SimpleNamespace(boxes=[], names={}) for _ in images]
    detector.load_dino_classifier(SimpleNamespace(backend="dinov3"))
    monkeypatch.setattr(underlying, "_determine_device", lambda *a: ("cpu", False))
    monkeypatch.setattr(underlying, "_sync_device", lambda *a: None)
    monkeypatch.setattr(legacy, "_load_detector", lambda *a: detector)
    request = CreateJobRequest(input_dir=str(tmp_path), options=ProcessingOptions(
        enable_detection=True, model_path="detector.pt", classification_model_path="joint.neri.json",
        batch_size=2, video_mode="skip",
    ))
    manager = services.ProcessingJobManager()
    now = services.utc_now()
    manager._jobs["job"] = JobSummary(id="job", state=JobState.QUEUED, input_dir=str(tmp_path),
                                    created_at=now, updated_at=now)
    manager._job_requests["job"] = request
    return manager, request, exported
