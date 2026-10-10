from pathlib import Path
import sqlite3

import pytest

from system.backend import services
from system.backend.models import CreateJobRequest, DetectionItem, JobState, JobSummary, ProcessingOptions


def test_device_stat_error_is_not_treated_as_a_deleted_photo(monkeypatch, tmp_path):
    path = tmp_path / "photo.jpg"
    error = OSError(5, "device I/O error", str(path))
    error.winerror = 1117
    monkeypatch.setattr(Path, "stat", lambda *args, **kwargs: (_ for _ in ()).throw(error))
    with pytest.raises(services.MediaStorageError, match="1117|device I/O error"):
        services._checked_media_file(path)


def test_actually_deleted_photo_can_still_be_skipped(tmp_path):
    assert not services._checked_media_file(tmp_path / "deleted.jpg")


def test_scan_failure_is_not_returned_as_a_partial_success(monkeypatch, tmp_path):
    def failed_walk(path, *, onerror):
        yield str(path), [], ["first.jpg"]
        onerror(OSError(5, "device I/O error", str(path / "camera")))
    monkeypatch.setattr(services._legacy.os, "walk", failed_walk)
    with pytest.raises(services.MediaStorageError):
        list(services._iter_supported_files(tmp_path))


def test_database_write_failure_stops_batch_and_resumes_unsaved_photo(monkeypatch, tmp_path):
    legacy = services._legacy
    state_path = tmp_path / "state.json"
    for module in (services, legacy):
        monkeypatch.setattr(module, "job_state_path", lambda: state_path)
    photos = [tmp_path / "first.jpg", tmp_path / "second.jpg"]
    for photo in photos:
        photo.write_bytes(b"fixture")
    monkeypatch.setattr(legacy, "_validate_dinov2_job_options", lambda *args: None)
    monkeypatch.setattr(legacy, "_start_batch_log_session", lambda *args, **kwargs: None)
    monkeypatch.setattr(legacy, "_resolve_job_inputs", lambda *args, **kwargs: (tmp_path, photos))
    monkeypatch.setattr(legacy, "_preview_detection_db_roots", lambda *args, **kwargs: [])
    monkeypatch.setattr(legacy, "_load_detection_index", lambda *args, **kwargs: {})
    monkeypatch.setattr(legacy, "_build_metadata_item", lambda path: DetectionItem(
        filename=path.name, path=str(path), file_type="jpg"))
    monkeypatch.setattr(legacy, "_serialize_detector_output", lambda *args: {})
    monkeypatch.setattr(legacy, "_persist_dinov2_observations", lambda *args, **kwargs: None)
    monkeypatch.setattr(legacy, "_selected_inference_class_ids", lambda *args: None)
    monkeypatch.setattr(legacy, "_export_results", lambda *args: None)
    detected = []
    class Detector:
        def detect_batch_species(self, paths, **kwargs):
            detected.extend(paths)
            return [{} for _ in paths]
    monkeypatch.setattr(legacy, "_load_detector", lambda *args: Detector())
    from system import detection_db
    writes = []
    disconnected = True
    monkeypatch.setattr(detection_db, "init_db", lambda *args: None)
    monkeypatch.setattr(detection_db, "upsert_validation_bulk", lambda *args: None)
    def save(path, payloads):
        if disconnected and payloads[0][1] == "second.jpg":
            raise sqlite3.OperationalError("unable to open database file")
        writes.extend(row[1] for row in payloads)
    monkeypatch.setattr(detection_db, "upsert_detections_bulk", save)
    request = CreateJobRequest(input_dir=str(tmp_path), options=ProcessingOptions(
        enable_detection=True, model_path="fake.pt", batch_size=1, thread_count=1, video_mode="skip"))
    now = services.utc_now()
    manager = services.ProcessingJobManager()
    restarted = None
    try:
        manager._jobs["job"] = JobSummary(id="job", state=JobState.QUEUED,
            input_dir=str(tmp_path), created_at=now, updated_at=now)
        manager._job_requests["job"] = request
        manager._run_job("job", request, [])
        failed = manager.get_job("job")
        assert failed.state == JobState.FAILED
        assert failed.processed == 1
        assert failed.total == 2
        assert [item.filename for item in failed.results] == ["first.jpg"]
        assert "unable to open database file" in failed.error
        assert "second.jpg" not in writes
        restarted = services.ProcessingJobManager()
        saved = restarted.get_job("job")
        assert saved.processed == 1
        assert saved.state == JobState.FAILED
        disconnected = False
        detected.clear()
        restarted._run_job("job", request, saved.results)
        finished = restarted.get_job("job")
        assert finished.state == JobState.COMPLETED
        assert finished.processed == 2
        assert detected == [str(photos[1])]
        assert writes == ["first.jpg", "second.jpg"]
    finally:
        manager._executor.shutdown()
        if restarted is not None:
            restarted._executor.shutdown()


def test_input_resolution_device_failure_reports_storage_error(monkeypatch, tmp_path):
    legacy = services._legacy
    for module in (services, legacy):
        monkeypatch.setattr(module, "job_state_path", lambda: tmp_path / "state.json")
    monkeypatch.setattr(legacy, "_start_batch_log_session", lambda *args, **kwargs: None)
    def fail_resolve(*args, **kwargs):
        error = OSError(5, "device I/O error", str(tmp_path))
        error.winerror = 1117
        raise error
    monkeypatch.setattr(legacy, "_resolve_job_inputs", fail_resolve)
    request = CreateJobRequest(input_dir=str(tmp_path), options=ProcessingOptions(enable_detection=False))
    manager = services.ProcessingJobManager()
    now = services.utc_now()
    manager._jobs["job"] = JobSummary(id="job", state=JobState.QUEUED,
        input_dir=str(tmp_path), created_at=now, updated_at=now)
    manager._job_requests["job"] = request
    try:
        manager._run_job("job", request, [])
        job = manager.get_job("job")
        assert job.state == JobState.FAILED
        assert job.processed == 0
        assert "device I/O error" in job.error
        assert job.message != "处理失败"
    finally:
        manager._executor.shutdown()
