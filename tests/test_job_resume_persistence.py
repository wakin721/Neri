from pathlib import Path
import json

import pytest

from system.backend import services
from system.backend.models import (
    CreateJobRequest,
    DetectionItem,
    JobState,
    JobSummary,
    ProcessingOptions,
)


def _request(input_dir: Path) -> CreateJobRequest:
    return CreateJobRequest(
        input_dir=str(input_dir),
        options=ProcessingOptions(
            enable_detection=True,
            model_path="fake.pt",
            batch_size=1,
            thread_count=1,
            video_mode="skip",
        ),
    )


def _job(job_id: str, input_dir: Path) -> JobSummary:
    now = services.utc_now()
    return JobSummary(
        id=job_id,
        state=JobState.QUEUED,
        input_dir=str(input_dir),
        created_at=now,
        updated_at=now,
    )


def _patch_state_path(monkeypatch, state_path: Path) -> None:
    resolver = lambda: state_path
    monkeypatch.setattr(services, "job_state_path", resolver)
    monkeypatch.setattr(services._legacy, "job_state_path", resolver)


def test_completed_image_batch_is_checkpointed_while_job_is_running(
    monkeypatch,
    tmp_path,
):
    state_path = tmp_path / "job_state.json"
    _patch_state_path(monkeypatch, state_path)

    image_path = tmp_path / "camera-01" / "0001.jpg"
    image_path.parent.mkdir()
    image_path.write_bytes(b"image")

    request = _request(tmp_path)
    job_id = "restart-resume"
    manager = services.ProcessingJobManager(max_workers=1)
    manager._jobs[job_id] = _job(job_id, tmp_path)
    manager._job_requests[job_id] = request

    item = DetectionItem(
        filename=image_path.name,
        path=str(image_path),
        file_type="image",
    )

    class FakeDetector:
        pass

    legacy = services._legacy
    monkeypatch.setattr(legacy, "_validate_dinov2_job_options", lambda *args: None)
    monkeypatch.setattr(legacy, "_start_batch_log_session", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        legacy,
        "_resolve_job_inputs",
        lambda request, cancelled=None: (tmp_path, [image_path]),
    )
    monkeypatch.setattr(legacy, "_preview_detection_db_roots", lambda *args, **kwargs: [])
    monkeypatch.setattr(legacy, "_load_detection_index", lambda *args, **kwargs: {})
    monkeypatch.setattr(legacy, "_load_detector", lambda *args, **kwargs: FakeDetector())
    monkeypatch.setattr(legacy, "_build_metadata_item", lambda path: item)
    monkeypatch.setattr(
        legacy,
        "_detect_image_batch",
        lambda detector, batch, batch_items, request, input_path, preloaded_data=None, inference_batcher=None: [item],
    )
    monkeypatch.setattr(legacy, "_export_results", lambda *args, **kwargs: None)

    snapshots: list[tuple[JobState, int]] = []
    original_save = manager._save_state_unlocked

    def capture_save() -> None:
        current = manager._jobs[job_id]
        snapshots.append((current.state, current.processed))
        original_save()

    monkeypatch.setattr(manager, "_save_state_unlocked", capture_save)

    try:
        manager._run_job(job_id, request, [])
    finally:
        manager._executor.shutdown(wait=False, cancel_futures=True)

    assert (JobState.RUNNING, 1) in snapshots


def test_failed_state_checkpoint_does_not_corrupt_previous_resume_state(
    monkeypatch,
    tmp_path,
):
    state_path = tmp_path / "job_state.json"
    _patch_state_path(monkeypatch, state_path)

    manager = services.ProcessingJobManager(max_workers=1)
    job_id = "atomic-state"
    manager._jobs[job_id] = _job(job_id, tmp_path)
    manager._job_requests[job_id] = _request(tmp_path)

    manager._save_state_unlocked()
    manager._jobs[job_id] = manager._jobs[job_id].model_copy(update={"processed": 1})
    original_dumps = json.dumps

    def fail_dumps(payload, *args, **kwargs):
        if isinstance(payload, dict) and payload.get("filename") == "fail.jpg":
            raise OSError("simulated interrupted checkpoint")
        return original_dumps(payload, *args, **kwargs)

    manager._jobs[job_id].results.append(DetectionItem(filename="fail.jpg", path="fail.jpg", file_type="image"))
    monkeypatch.setattr(json, "dumps", fail_dumps)

    try:
        manager._save_state_unlocked()
    finally:
        manager._executor.shutdown(wait=False, cancel_futures=True)

    saved = manager._state_store.load()["jobs"][job_id]
    assert saved["processed"] == 0
    assert saved["results"] == []


def test_legacy_history_migrates_once_and_incremental_progress_survives_restart(monkeypatch, tmp_path):
    state_path = tmp_path / "job_state.json"
    _patch_state_path(monkeypatch, state_path)
    job = _job("legacy", tmp_path)
    job.results = [DetectionItem(filename="old.jpg", path="old.jpg", file_type="image")]
    job.processed = 1
    state_path.write_text(json.dumps({"jobs": {job.id: job.model_dump()},
                                     "requests": {job.id: _request(tmp_path).model_dump()}}), encoding="utf-8")
    original_json = state_path.read_bytes()
    manager = services.ProcessingJobManager()
    try:
        next_item = DetectionItem(filename="new.jpg", path="new.jpg", file_type="image")
        manager._mutate_job(job.id, state=JobState.RUNNING, processed=2,
                            results=[*manager._jobs[job.id].results, next_item])
        restarted = services.ProcessingJobManager()
        try:
            saved = restarted.get_job(job.id)
            assert saved.state == JobState.CANCELLED
            assert saved.processed == 2
            assert [item.filename for item in saved.results] == ["old.jpg", "new.jpg"]
            restarted.clear_jobs()
            assert restarted._state_store.load() == {"jobs": {}, "requests": {}}
        finally:
            restarted._executor.shutdown()
    finally:
        manager._executor.shutdown()
    assert state_path.read_bytes() == original_json


def test_progress_serializes_only_new_results(monkeypatch, tmp_path):
    _patch_state_path(monkeypatch, tmp_path / "job_state.json")
    manager = services.ProcessingJobManager()
    manager._jobs["history"] = _job("history", tmp_path)
    manager._jobs["history"].results = [DetectionItem(filename="old.jpg", path="old.jpg", file_type="image")]
    manager._jobs["active"] = _job("active", tmp_path)
    manager._job_requests["active"] = _request(tmp_path)
    manager._save_state_unlocked()
    old = manager._jobs["history"].results[0]
    original_dump = DetectionItem.model_dump
    calls = []

    def record_dump(item, *args, **kwargs):
        assert item is not old, "history must not be serialized for an active batch"
        calls.append(item.filename)
        return original_dump(item, *args, **kwargs)

    monkeypatch.setattr(DetectionItem, "model_dump", record_dump)
    try:
        first = DetectionItem(filename="first.jpg", path="first.jpg", file_type="image")
        second = DetectionItem(filename="second.jpg", path="second.jpg", file_type="image")
        manager._mutate_job("active", processed=1, results=[first])
        manager._mutate_job("active", processed=2, results=[first, second])
        assert calls == ["first.jpg", "second.jpg"]
        assert len(manager._state_store.load()["jobs"]["active"]["results"]) == 2
        # Replacing an earlier result also persists corrections, without duplicate positions.
        corrected = first.model_copy(update={"species": ["corrected"]})
        manager._mutate_job("active", results=[corrected, second])
        saved_results = manager._state_store.load()["jobs"]["active"]["results"]
        assert len(saved_results) == 2
        assert saved_results[0]["species"] == ["corrected"]
    finally:
        manager._executor.shutdown()


def test_checkpoint_rebuilds_after_software_cache_is_removed(monkeypatch, tmp_path):
    _patch_state_path(monkeypatch, tmp_path / "job_state.json")
    manager = services.ProcessingJobManager()
    try:
        manager._jobs["history"] = _job("history", tmp_path)
        manager._jobs["history"].results = [DetectionItem(filename="a.jpg", path="a.jpg", file_type="image")]
        manager._save_state_unlocked()
        manager._state_store.path.unlink()
        manager._save_state_unlocked()
        assert len(manager._state_store.load()["jobs"]["history"]["results"]) == 1
    finally:
        manager._executor.shutdown()


@pytest.mark.parametrize("stage", ["_validate_dinov2_job_options", "_start_batch_log_session"])
def test_startup_failure_releases_job_and_checkpoints_error(monkeypatch, tmp_path, stage):
    _patch_state_path(monkeypatch, tmp_path / "job_state.json")
    manager = services.ProcessingJobManager()
    request = _request(tmp_path)
    manager._jobs["startup"] = _job("startup", tmp_path)
    manager._job_requests["startup"] = request

    def fail(*args, **kwargs):
        raise ValueError("startup fixture failure")

    monkeypatch.setattr(services._legacy, stage, fail)
    try:
        manager._run_job("startup", request, [])
        job = manager.get_job("startup")
        assert job.state == JobState.FAILED
        assert not job.active
        assert job.error == "startup fixture failure"
        assert manager._state_store.load()["jobs"]["startup"]["state"] == "failed"
    finally:
        manager._executor.shutdown()


@pytest.mark.parametrize("cancelled", [False, True])
def test_fast_video_interruption_keeps_unpublished_completed_results(monkeypatch, tmp_path, cancelled):
    _patch_state_path(monkeypatch, tmp_path / "job_state.json")
    legacy = services._legacy
    paths = [tmp_path / f"video-{index}.mp4" for index in range(3)]
    for path in paths:
        path.write_bytes(b"video fixture")
    request = _request(tmp_path)
    request.options.video_mode = "fast"
    manager = services.ProcessingJobManager()
    manager._jobs["videos"] = _job("videos", tmp_path)
    manager._job_requests["videos"] = request
    monkeypatch.setattr(legacy, "_start_batch_log_session", lambda *args: None)
    monkeypatch.setattr(legacy, "_resolve_job_inputs", lambda *args, **kwargs: (tmp_path, paths))
    monkeypatch.setattr(legacy, "load_media_indexes", lambda *args, **kwargs: ({}, {}))
    monkeypatch.setattr(legacy, "_load_detector", lambda *args: object())
    monkeypatch.setattr(legacy, "_build_metadata_item", lambda path: DetectionItem(
        filename=path.name, path=str(path), file_type="video"))
    monkeypatch.setattr(legacy.time, "monotonic", lambda: 1000.0)

    def interrupt_batch(detector, video_paths, items, *args, on_item_ready, **kwargs):
        on_item_ready(0, items[0])
        on_item_ready(1, items[1])
        # The second completion is held back by the 200 ms UI throttle.
        assert manager.get_job("videos").processed == 1
        if cancelled:
            manager._cancelled_jobs.add("videos")
            raise RuntimeError("用户取消任务")
        raise services.MediaStorageError(paths[2], "保存检测结果", OSError("disk disconnected"))

    monkeypatch.setattr(legacy, "_detect_video_fast_batch", interrupt_batch)
    try:
        manager._run_job("videos", request, [])
        job = manager.get_job("videos")
        assert job.state == (JobState.CANCELLED if cancelled else JobState.FAILED)
        assert not job.active
        assert job.processed == 2
        assert [item.path for item in job.results] == [str(path) for path in paths[:2]]
        saved = manager._state_store.load()["jobs"]["videos"]
        assert saved["processed"] == 2
        assert len(saved["results"]) == 2
    finally:
        manager._executor.shutdown()
